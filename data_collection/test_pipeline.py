"""Offline end-to-end verification using synthetic fixtures only."""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import csv
import importlib.util
import io
import json
import os
import pandas as pd
import pprint
import subprocess
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from datetime import date, datetime, timezone
from pathlib import Path
from unittest.mock import patch
from common import (
    CODE_DIR, digest, load_spec, make_company, make_matchers, read_config, read_csv, read_json,
    write_json,
)
from news import acquire_pool, build_library, cache_path as news_cache, match_aliases, pool_tasks
from prices import acquire_prices
from run_pipeline import execute, export_dataset


# Test pipeline

TEST_PIPELINE_HELP = """Validate the full pipeline offline using synthetic records in temporary directories.

Run: python test_pipeline.py
Do not access real accounts or news, or leave data and __pycache__ in the release
directory.
"""

class FixtureProvider:
    def query(self, api_name, **params):
        if api_name in ("news", "major_news"):
            key = "datetime" if api_name == "news" else "pub_time"
            day = params["start_date"][:10]
            names = "FixtureChina FixtureUS FixtureHK FixtureJapan"
            records = [{key: day + f" {8+i:02d}:00:00", "title": names + f" COVID item {i}",
                "content": "Synthetic fixture text " * (15+i)} for i in range(3)]
            return pd.DataFrame([row for row in records if params["start_date"] <= row[key] <= params["end_date"]],
                columns=[key, "title", "content"])
        return pd.DataFrame([{"ts_code": params["ts_code"], "trade_date": params["start_date"],
            "open": 10.0, "high": 12.0, "low": 9.0, "close": 11.0, "vol": 2.0}])

    def close(self):
        pass


class FixtureResponse:
    def __init__(self, symbol):
        self.symbol = symbol

    def raise_for_status(self):
        pass

    def json(self):
        currency = "JPY" if self.symbol.endswith(".T") else ("HKD" if self.symbol.endswith(".HK") else "USD")
        tz = "Asia/Tokyo" if currency == "JPY" else ("Asia/Hong_Kong" if currency == "HKD" else "America/New_York")
        return {"chart": {"result": [{"meta": {"exchangeTimezoneName": tz, "currency": currency},
            "timestamp": [int(datetime(2020, 1, 3, 13, tzinfo=timezone.utc).timestamp())],
            "indicators": {"quote": [{"open": [10], "high": [12], "low": [9], "close": [11], "volume": [200]}],
                           "adjclose": [{"adjclose": [10.5]}]}}]}}


class FixtureSession:
    def __init__(self):
        self.headers = {}

    def get(self, url, **kwargs):
        return FixtureResponse(url.rsplit("/", 1)[-1])

    def close(self):
        pass


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="python_only_pipeline_test_")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name) / "work"
        self.config = Path(self.temporary.name) / "config.py"
        defaults = read_config(CODE_DIR / "config.py")
        settings = defaults["SETTINGS"] | {"news_sources": ["fixture"], "major_sources": ["fixtureMajor"],
            "news_row_limits": {"news": 2, "major_news": 2},
            "request_gap_seconds": 0, "price_request_gap_seconds": 0, "retries": 1}
        events = [{"event_id": "covid19_first_wave", "pre_start_date": "2020-01-03",
            "event_start_date": "2020-01-04", "event_end_date": "2020-01-04", "post_end_date": "2020-01-05"}]
        companies = [
            {"market": "CSI300", "ticker": "000001", "company_name": "FixtureChina", "exchange": "SZ"},
            {"market": "SP500", "ticker": "FIXUS", "company_name": "FixtureUS"},
            {"market": "HSTECH", "ticker": "00002", "company_name": "FixtureHK"},
            {"market": "Nikkei225", "ticker": "7203", "company_name": "FixtureJapan"},
            {"market": "SP500", "ticker": "VOID", "company_name": "FixtureNoCoverage"}]
        self.config.write_text("\n".join(key + " = " + pprint.pformat(value) for key,value in
            (("SETTINGS", settings), ("EVENTS", events), ("COMPANIES", companies))), encoding="utf-8")
        self.settings, self.events, self.companies = load_spec(self.config)

    def seed(self):
        with patch("news.tushare_client", return_value=FixtureProvider()),\
             patch("prices.tushare_client", return_value=FixtureProvider()),\
             patch("requests.Session", side_effect=FixtureSession), redirect_stdout(io.StringIO()):
            paths = acquire_pool(self.root, self.settings, self.events)
            build = build_library(self.root, self.settings, self.events, self.companies, paths)
            acquire_prices(self.root, self.settings, self.events, self.companies)
        return paths, build

    def test_end_to_end_and_offline_resume(self):
        self.seed()
        no_network = AssertionError("offline processing attempted network access")
        with patch("requests.sessions.Session.request", side_effect=no_network),\
             patch("news.tushare_client", side_effect=no_network),\
             patch("prices.tushare_client", side_effect=no_network),\
             patch.dict(os.environ, {"TUSHARE_TOKEN": "", "TUSHARE_NEWS_TOKEN": ""}),\
             redirect_stdout(io.StringIO()):
            execute(self.root, self.settings, self.events, self.companies, from_cache=True)
        pointer = read_json(self.root / "metadata/latest_dataset.json")
        benchmark = self.root / pointer["benchmark_path"]
        prices = read_csv(benchmark / "data/price_ohlcv.csv")
        news = read_csv(benchmark / "data/source_news_text.csv")
        firm_days = read_csv(benchmark / "data/firm_day.csv")
        self.assertEqual(len(prices), 5)
        self.assertEqual(len(news), 36)  # 3 days * 3 distinct articles * 4 matched companies; 2 sources deduped.
        self.assertEqual(len(firm_days), 15)
        self.assertEqual(len({row["source_news_id"] for row in news}), len(news))
        self.assertEqual(next(row["volume"] for row in prices if row["market"] == "CSI300"), "200.0")
        self.assertTrue(all(row["matched_news_count"] == "0" for row in firm_days if row["ticker"] == "VOID"))
        weekend = [row for row in firm_days if row["date"] == "2020-01-04"]
        self.assertTrue(all(row["has_price"] == "0" and row["close"] == "" for row in weekend))
        self.assertTrue(any(row["has_news"] == "1" for row in weekend))
        self.assertTrue(all(row["is_pre_event_input"] == ("1" if row["date"] == "2020-01-03" else "0") for row in firm_days))
        self.assertTrue((benchmark / "views/by_event_market/covid19_first_wave/CSI300/news/000001/2020-01-04/representative_news.csv").is_file())

    def test_missing_and_corrupt_pool_are_failures(self):
        paths, _ = self.seed()
        paths[0].unlink()
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            with self.assertRaises(RuntimeError):
                acquire_pool(self.root, self.settings, self.events, from_cache=True)
        checkpoint = read_json(self.root / "metadata/tushare_raw_pool_checkpoint.json")
        self.assertEqual(checkpoint["failed_tasks"], 1)
        task = next(pool_tasks(self.settings, self.events))
        write_json(news_cache(self.root, task), dict(task, records=[], row_count=0, records_sha256="bad"), compressed=True)
        with redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            with self.assertRaises(RuntimeError):
                acquire_pool(self.root, self.settings, self.events, from_cache=True)

    def test_alias_boundaries_validity_and_ambiguity(self):
        company = make_company({"market": "SP500", "ticker": "CAT", "company_name": "FixtureCompany",
            "aliases": "AliasBrand", "membership_start": "2020-01-04"})
        settings = self.settings | {"alias_records": [{"market": "SP500", "ticker": "CAT", "alias": "AliasBrand",
            "confidence": 0.9, "valid_from": "2020-01-04", "valid_to": "2020-01-05"}]}
        matcher = make_matchers([company], settings)["SP500"]
        hits = match_aliases(title="concatenate numbers 2020", content="", matcher=matcher, match_date=date(2020,1,3))
        self.assertFalse(hits)
        self.assertFalse(match_aliases(title="AliasBrand", content="", matcher=matcher, match_date=date(2020,1,3)))
        self.assertTrue(match_aliases(title="AliasBrand", content="", matcher=matcher, match_date=date(2020,1,4)))
        # Index membership does not suppress company news in the pre-event observation window.
        self.assertTrue(match_aliases(title="FixtureCompany", content="", matcher=matcher, match_date=date(2020,1,3)))
        other = make_company({"market": "SP500", "ticker": "OTHER", "company_name": "OtherFixture", "aliases": "AliasBrand"})
        matcher = make_matchers([company, other], self.settings)["SP500"]
        self.assertTrue(all(hit.ambiguous for hit in match_aliases(title="AliasBrand", content="", matcher=matcher)))

    def test_changed_pool_refuses_stale_export(self):
        paths, _ = self.seed()
        payload = read_json(paths[0])
        payload["records"][0]["content"] += " New fixture revision."
        payload["records_sha256"] = digest(payload["records"])
        write_json(paths[0], payload, compressed=True)
        with self.assertRaises(ValueError):
            export_dataset(self.root, self.settings, self.events, self.companies)

    @unittest.skipUnless(importlib.util.find_spec("pyarrow"), "optional pyarrow is not installed")
    def test_parquet_schema_and_csv_parity(self):
        self.seed()
        with redirect_stdout(io.StringIO()):
            benchmark = export_dataset(self.root, self.settings, self.events, self.companies, parquet=True)
        prices = pd.read_parquet(benchmark / "data/price_ohlcv.parquet")
        news = pd.read_parquet(benchmark / "data/source_news_text.parquet")
        firm_days = pd.read_parquet(benchmark / "data/firm_day.parquet")
        self.assertEqual(len(prices), len(read_csv(benchmark / "data/price_ohlcv.csv")))
        self.assertEqual(len(news), 36)
        self.assertEqual(len(firm_days), 15)
        self.assertTrue(pd.api.types.is_numeric_dtype(prices["volume"]))
        self.assertIn("000001", set(prices["ticker"]))
        self.assertTrue(firm_days.loc[firm_days["date"] == "2020-01-04", "close"].isna().all())

    def test_isolated_cli_from_cache(self):
        self.seed()
        clean = dict(os.environ, TUSHARE_TOKEN="", TUSHARE_NEWS_TOKEN="", PYTHONIOENCODING="utf-8")
        result = subprocess.run([sys.executable, str(CODE_DIR / "run_pipeline.py"),
            "--config", str(self.config), "--work-dir", str(self.root), "--from-cache"],
            cwd=self.temporary.name, env=clean, capture_output=True, text=True, encoding="utf-8")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(read_json(self.root / "metadata/pipeline_run.json")["status"], "complete")


if __name__ == "__main__":
    unittest.main(verbosity=2)

