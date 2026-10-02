"""Full acquisition and processing workflow, event export and configuration setup."""
from __future__ import annotations

import sys
sys.dont_write_bytecode = True

import argparse
import pprint
from collections import defaultdict
from common import (
    CODE_DIR, DEFAULT_CONFIG, MARKETS, RequestRunner, add_pipeline_args, add_symbol_args,
    digest, load_symbols, make_matchers, matching_build, now, period, read_config, read_csv,
    read_json, run_cli, spec_from_args, tushare_client, write_csv, write_json, yahoo_symbol,
)
from prices import PRICE_FIELDS, acquire_prices, epoch, load_price_build
from news import (
    acquire_pool, build_candidate, build_daily_pool, build_library,
    build_matcher_for_companies, classify_content, classify_event_relevance, classify_risk,
)
from pathlib import Path
from datetime import date, timedelta
from urllib.parse import quote


# Build event dataset

BUILD_EVENT_DATASET_HELP = """Company news and normalized prices -> event slices, firm-day alignment, CSV/Parquet.

Use the frozen price_ohlcv and source_news_text schemas. Cover the full calendar
observation window, retain weekend news, and leave missing prices unfilled.
Keep companies with no matched news. news_query_complete indicates successful
queries for the selected sources, rather than complete coverage of all news.
"""

NEWS_FIELDS = ["source_news_id", "event_id", "market", "ticker", "company_name", "date",
               "source_name", "url_hash", "title", "content"]
QUALITY_FIELDS = ["source_news_id", "event_id", "market", "ticker", "date", "period",
    "match_alias", "alignment_score", "content_availability", "company_match_strength",
    "event_relevance", "manual_risk_flag", "is_analysis_candidate"]
FIRM_DAY_FIELDS = ["event_id", "market", "ticker", "company_name", "date", "period",
    "is_pre_event_input", "news_query_complete", "matched_news_count", "has_news",
    "has_price", "open", "high", "low", "close", "volume", "currency"]


def write_table(path, fields, rows, parquet):
    write_csv(path.with_suffix(".csv"), fields, rows)
    if parquet:
        import pandas as pd
        frame = pd.DataFrame(rows, columns=fields)
        for field in ("open", "high", "low", "close", "volume"):
            if field in frame:
                frame[field] = pd.to_numeric(frame[field], errors="raise").astype("float64")
        temporary = path.with_suffix(".parquet.tmp")
        frame.to_parquet(temporary, index=False, engine="pyarrow")
        temporary.replace(path.with_suffix(".parquet"))


def active_for_event(company, event):
    day = event["event_start_date"]
    return ((not company.membership_start or company.membership_start <= day)
            and (not company.membership_end or day <= company.membership_end))


def export_dataset(root, settings, events, companies, *, parquet=False):
    if parquet:
        try:
            import pyarrow
        except ImportError as exc:
            raise RuntimeError("Parquet 导出需先运行 python -m pip install pyarrow") from exc
    build, manifest = matching_build(root, settings, events, companies)
    price_build = load_price_build(root, events, companies)
    price_rows = read_csv(price_build / "price_ohlcv.csv")
    price_lookup = {(row["market"], row["ticker"], row["date"]): row for row in price_rows}
    generation = digest([manifest["build_id"], price_build.name, "dataset-schema-v1"])[:24]
    benchmark = root / "datasets" / generation / "benchmark"
    by_market = defaultdict(list)
    for company in companies:
        by_market[company.market].append(company)
    matchers = make_matchers(companies, settings)
    news_lookup = defaultdict(list)
    for market, group in by_market.items():
        matcher = matchers[market]
        news_root = build / "raw" / market / "company_news" / "news"
        for directory in sorted(news_root.glob("*/*")):
            if not directory.is_dir():
                continue
            bad_json = []
            candidates = [build_candidate(news_root, directory, path, bad_json, matcher)
                for path in sorted(directory.glob("*.json")) if path.name != "representative_news.json"]
            if bad_json:
                raise ValueError("公司库有损坏新闻 JSON，拒绝导出")
            candidates = [candidate for candidate in candidates if candidate is not None]
            pool, _ = build_daily_pool(candidates, settings["min_alignment_score"])
            news_lookup[(market, directory.parent.name, directory.name)] = [
                candidate for candidate in pool if candidate.alignment_score >= settings["min_alignment_score"]]
    all_news, all_quality, all_firm_days, indexes, coverage = [], [], [], [], []
    from datetime import date, timedelta
    for event in events:
        event_id = event["event_id"]
        for market, group in sorted(by_market.items()):
            selected = [company for company in group if active_for_event(company, event)]
            view = benchmark / "views" / "by_event_market" / event_id / market
            news, quality, firm_days, selected_prices = [], [], [], []
            for company in selected:
                news_days = price_days = count = 0
                day = date.fromisoformat(event["pre_start_date"])
                end = date.fromisoformat(event["post_end_date"])
                while day <= end:
                    day_text = day.isoformat()
                    day_period = period(event, day_text)
                    pool = news_lookup.get((market, company.ticker, day_text), [])
                    target = view / "news" / company.ticker / day_text
                    if pool:
                        write_csv(target / "news.csv", ["time", "source", "title", "content"], [candidate.csv_row() for candidate in pool])
                        write_csv(target / "representative_news.csv", ["time", "source", "title", "content"], [pool[0].csv_row()])
                        news_days += 1
                    for candidate in pool:
                        row_id = digest([event_id, market, company.ticker, candidate.article_id])
                        row = {"source_news_id": row_id, "event_id": event_id, "market": market, "ticker": company.ticker,
                            "company_name": company.company_name, "date": day_text, "source_name": candidate.source,
                            "url_hash": digest(candidate.url) if candidate.url else "",
                            "title": candidate.title, "content": candidate.content}
                        news.append(row)
                        strength = "direct_company" if candidate.payload["match_location"] == "title" else "alias_or_renamed_company"
                        availability = classify_content(candidate.title, candidate.content, "")
                        relevance = classify_event_relevance(event_id, "main_tushare", candidate.title, candidate.content, settings["content_scan_chars"])
                        risk = classify_risk(source_group="main_tushare", content_availability=availability,
                            company_match_strength=strength, company_match_reason="company_alias", event_relevance=relevance)
                        quality.append({"source_news_id": row_id, "event_id": event_id, "market": market,
                            "ticker": company.ticker, "date": day_text, "period": day_period,
                            "match_alias": candidate.payload["match_alias"], "alignment_score": candidate.alignment_score,
                            "content_availability": availability, "company_match_strength": strength,
                            "event_relevance": relevance, "manual_risk_flag": risk, "is_analysis_candidate": int(risk != "exclude_candidate")})
                        # Generate article files only in the user's work directory; the code package contains no article text.
                        write_json(target / (candidate.article_id + ".json"), {**candidate.payload, "event_id": event_id, "period": day_period})
                        count += 1
                    price = price_lookup.get((market, company.ticker, day_text))
                    if price:
                        selected_prices.append(price)
                        price_days += 1
                    firm_days.append({"event_id": event_id, "market": market, "ticker": company.ticker,
                        "company_name": company.company_name, "date": day_text, "period": day_period,
                        "is_pre_event_input": int(day_period == "pre_event"), "news_query_complete": 1,
                        "matched_news_count": len(pool), "has_news": int(bool(pool)), "has_price": int(price is not None),
                        **{field: price.get(field, "") if price else "" for field in ("open", "high", "low", "close", "volume")},
                        "currency": company.currency})
                    day += timedelta(days=1)
                coverage.append({"event_id": event_id, "market": market, "ticker": company.ticker,
                    "company_name": company.company_name, "observed_news_days": news_days,
                    "observed_price_days": price_days, "matched_news_count": count,
                    "news_query_status": "selected_sources_complete", "universe_mode":
                    "membership_at_event_start" if company.membership_start or company.membership_end else "observed_company_universe"})
            write_table(view / "source_news_text", NEWS_FIELDS, news, parquet)
            write_table(view / "price_ohlcv", PRICE_FIELDS, selected_prices, parquet)
            write_table(view / "firm_day", FIRM_DAY_FIELDS, firm_days, parquet)
            write_csv(view / "source_news_quality.csv", QUALITY_FIELDS, quality)
            write_json(view / "slice_manifest.json", {"event": event, "market": market,
                "company_count": len(selected), "news_rows": len(news), "price_rows": len(selected_prices),
                "company_build_id": manifest["build_id"], "price_build_id": price_build.name})
            indexes.append({"event_id": event_id, "market": market, "company_count": len(selected),
                            "news_rows": len(news), "price_rows": len(selected_prices),
                            "view_path": str(view.relative_to(benchmark))})
            all_news.extend(news)
            all_quality.extend(quality)
            all_firm_days.extend(firm_days)
    # Keep the main price table unique by market/company/date instead of duplicating rows by event.
    selected_keys = {(row["market"], row["ticker"], row["date"]) for row in all_firm_days}
    canonical_prices = [row for row in price_rows if (row["market"], row["ticker"], row["date"]) in selected_keys]
    write_table(benchmark / "data" / "price_ohlcv", PRICE_FIELDS, canonical_prices, parquet)
    write_table(benchmark / "data" / "source_news_text", NEWS_FIELDS, all_news, parquet)
    write_table(benchmark / "data" / "firm_day", FIRM_DAY_FIELDS, all_firm_days, parquet)
    write_csv(benchmark / "data" / "source_news_quality.csv", QUALITY_FIELDS, all_quality)
    write_csv(benchmark / "views" / "by_event_market" / "manifest.csv",
        ["event_id", "market", "company_count", "news_rows", "price_rows", "view_path"], indexes)
    write_csv(benchmark / "data" / "coverage.csv", ["event_id", "market", "ticker", "company_name",
        "observed_news_days", "observed_price_days", "matched_news_count", "news_query_status", "universe_mode"], coverage)
    write_json(benchmark / "manifest.json", {"generated_at": now(), "generation": generation,
        "company_build_id": manifest["build_id"], "price_build_id": price_build.name, "events": events,
        "company_count": len(companies), "news_rows": len(all_news), "price_rows": len(canonical_prices),
        "firm_day_rows": len(all_firm_days), "volume_unit": "shares", "price_adjustment": "unadjusted",
        "source_news_scope": "Tushare selected sources, company alias matched",
        "join_policy": "calendar_day_no_price_imputation",
        "prediction_input_policy": "filter is_pre_event_input=1; do not use during/post rows as pre-event inputs"})
    write_json(root / "metadata" / "latest_dataset.json", {"generation": generation,
        "benchmark_path": str(benchmark.relative_to(root))})
    print(f"最终数据集: {benchmark}")
    return benchmark


def export_dataset_main():
    parser = argparse.ArgumentParser(description=BUILD_EVENT_DATASET_HELP)
    add_pipeline_args(parser)
    parser.add_argument("--parquet", action="store_true")
    args = parser.parse_args()
    root, settings, events, companies = spec_from_args(args)
    export_dataset(root, settings, events, companies, parquet=args.parquet)
    return 0


# Create company config

CREATE_COMPANY_CONFIG_HELP = """Fetch names for user-supplied symbols and generate a Python pipeline configuration.

Do not include the project's existing company universe. Save generated
configurations outside the code directory. Fetch A-share, Hong Kong, and US names
from Tushare, and Japanese names from Yahoo chart metadata.
Users must review the generated configuration and supply historical index
membership and additional aliases, including names in other languages.
"""

def create_config_main():
    parser = argparse.ArgumentParser(description=CREATE_COMPANY_CONFIG_HELP)
    parser.add_argument("--market", choices=MARKETS, required=True)
    add_symbol_args(parser)
    parser.add_argument("--output-config", type=Path, required=True)
    parser.add_argument("--price-source", choices=["yahoo", "tushare"])
    args = parser.parse_args()
    symbols = load_symbols(parser, args)
    target = args.output_config.resolve()
    if target == CODE_DIR or CODE_DIR in target.parents:
        parser.error("生成配置需保存在代码发布目录之外，避免把自己的名单混进代码包")
    if target.suffix != ".py" or target.exists():
        parser.error("输出需为尚不存在的 .py 文件")
    defaults = read_config(DEFAULT_CONFIG)
    companies = []
    source = args.price_source or ("tushare" if args.market == "CSI300" else "yahoo")
    if args.market == "Nikkei225" and source != "yahoo":
        parser.error("日股使用 Yahoo")
    pro = None
    import requests
    runner = RequestRunner(1, 3)
    try:
        with requests.Session() as session:
            session.headers.update({"User-Agent": "Mozilla/5.0"})
            for symbol in symbols:
                if args.market == "Nikkei225":
                    ticker = symbol.removesuffix(".T")
                    price_symbol = yahoo_symbol(args.market, ticker)
                    def fetch():
                        response = session.get("https://query1.finance.yahoo.com/v8/finance/chart/" + quote(price_symbol, safe=""),
                            params={"period1": epoch(date.today() - timedelta(days=14)), "period2": epoch(date.today() + timedelta(days=1)), "interval": "1d"},
                            timeout=30)
                        response.raise_for_status()
                        payload = response.json()
                        if payload.get("chart", {}).get("error"):
                            raise ValueError("Yahoo 公司信息查询失败")
                        meta = payload["chart"]["result"][0]["meta"]
                        return meta.get("longName") or meta.get("shortName")
                    name = runner(fetch)
                    exchange = ""
                else:
                    if pro is None:
                        pro = tushare_client(30)
                    endpoint = {"CSI300": "stock_basic", "HSTECH": "hk_basic", "SP500": "us_basic"}[args.market]
                    query_symbol = symbol
                    if args.market == "CSI300":
                        if not symbol.endswith((".SH", ".SZ", ".BJ")):
                            parser.error("A股代码必须带 .SH/.SZ/.BJ")
                        ticker, exchange = symbol.split(".")
                    elif args.market == "HSTECH":
                        ticker = f"{int(symbol.split('.')[0]):05d}"
                        query_symbol, exchange = ticker + ".HK", "HK"
                    else:
                        ticker, exchange = symbol, ""
                    frame = runner(lambda: pro.query(endpoint, ts_code=query_symbol))
                    candidates = frame[frame["ts_code"].astype(str).str.upper() == query_symbol]
                    if candidates.empty:
                        raise ValueError(f"无法获取 {symbol} 的公司名，请手动配置（退市公司可用历史身份信息）")
                    item = candidates.iloc[0]
                    name = item.get("name") or item.get("enname") or item.get("fullname")
                    price_symbol = yahoo_symbol(args.market, ticker, exchange) if source == "yahoo" else query_symbol
                if not isinstance(name, str) or not name.strip():
                    raise ValueError(f"{symbol}: 接口未返回公司名，请手动填写")
                companies.append({"market": args.market, "ticker": ticker, "company_name": name,
                    "aliases": "", "exchange": exchange, "price_source": source,
                    "price_symbol": price_symbol, "currency": MARKETS[args.market]})
    finally:
        if pro is not None:
            pro.close()
    text = '"""Your locally generated company configuration; review names and add aliases."""\n'
    for key, value in (("SETTINGS", defaults["SETTINGS"]), ("EVENTS", defaults["EVENTS"]), ("COMPANIES", companies)):
        text += "\n" + key + " = " + pprint.pformat(value, sort_dicts=False, width=110) + "\n"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text, encoding="utf-8")
    print(f"已生成 {target}；请补充跨语言别名并确认研究公司范围")
    return 0


# Run pipeline

RUN_PIPELINE_HELP = """Shared raw news pool -> company daily pools -> prices -> event slices -> final tables.

Stages: all/pool/company/prices/dataset. Processing stages can use cached data
entirely offline. This entry point covers Yahoo/Tushare and does not fetch the
project's supplemental SEC, HKEX, TDnet, or GDELT sources.
"""

def execute(root, settings, events, companies, stage="all", from_cache=False, refresh=False, parquet=False):
    status = {"started_at": now(), "stage": stage, "status": "running"}
    report = root / "metadata" / "pipeline_run.json"
    write_json(report, status)
    try:
        if stage in ("all", "pool"):
            paths = acquire_pool(root, settings, events, from_cache=from_cache, refresh=refresh)
        if stage == "company":
            paths = acquire_pool(root, settings, events, from_cache=True)
        if stage in ("all", "company"):
            build_library(root, settings, events, companies, paths)
        if stage in ("all", "prices"):
            acquire_prices(root, settings, events, companies, from_cache=from_cache, refresh=refresh)
        if stage in ("all", "dataset"):
            export_dataset(root, settings, events, companies, parquet=parquet)
        status.update(status="complete", finished_at=now())
    except Exception as exc:
        from common import safe_error
        status.update(status="error", finished_at=now(), error=safe_error(exc))
        write_json(report, status)
        raise
    write_json(report, status)


def run_main():
    parser = argparse.ArgumentParser(description=RUN_PIPELINE_HELP + "\nAdditional commands: create-config and usage.", formatter_class=argparse.RawDescriptionHelpFormatter)
    add_pipeline_args(parser)
    parser.add_argument("--stage", choices=["all", "pool", "company", "prices", "dataset"], default="all")
    parser.add_argument("--from-cache", action="store_true", help="只使用完整缓存，不联网且不需要 Token")
    parser.add_argument("--refresh", action="store_true", help="重新抓取已成功的原始新闻/股价片段")
    parser.add_argument("--parquet", action="store_true")
    args = parser.parse_args()
    if args.from_cache and args.refresh:
        parser.error("from-cache 和 refresh 不能同时使用")
    root, settings, events, companies = spec_from_args(args, require_companies=args.stage != "pool")
    execute(root, settings, events, companies, args.stage, args.from_cache, args.refresh, args.parquet)
    return 0


# Usage

USAGE_HELP = """Print usage instructions for the Python-only release without downloading data."""

HELP = r'''
public_data_download -- complete Yahoo Finance / Tushare acquisition and processing

Release Python source only. Do not bundle company lists, raw news/prices, tokens,
caches, or generated results. The original alias matching, daily-pool scoring,
source priorities, and representative selection rules are consolidated in news.py.
This entry point covers Yahoo prices and Tushare prices/news. SEC, HKEX, TDnet,
GDELT, and portal supplements are outside this workflow.
News quality and event relevance are heuristic flags, not human ground truth.
Yahoo/Tushare alone cannot reconstruct all existing companies, supplementary
news, or historical index memberships.

File layout: prices.py, news.py, common.py, config.py, run_pipeline.py, test_pipeline.py.
Print this guide with: python run_pipeline.py usage

1. Environment and tokens
   Python 3.10+:
   python -m pip install requests pandas tzdata
   Optional Parquet support:
   python -m pip install pyarrow

   PowerShell, using your own token:
   $env:TUSHARE_TOKEN = Read-Host 'Enter your own Tushare Token'
   Bash / macOS / Linux:
   read -rs TUSHARE_TOKEN; export TUSHARE_TOKEN
   Set TUSHARE_NEWS_TOKEN for separate news credentials; otherwise TUSHARE_TOKEN
   is used. The scripts do not read the project's .env, save tokens, or accept
   command-line tokens.
   Configure your own proxy using HTTPS_PROXY / HTTP_PROXY.
   Tushare news and some price endpoints require separate access permissions.

2. Prepare your company configuration
   Run inside the code directory, for example:
   python run_pipeline.py create-config --market CSI300 --symbols 600519.SH 000001.SZ --output-config ../downloaded_data/my_config.py

   Fetch names for your own symbols and generate a configuration outside the
   code release. Examples: HSTECH 00700.HK, SP500 AAPL, Nikkei225 7203.T.
   Alternatively, use --symbols-file with your own text file, one code per line.
   API names do not form a complete historical identity table. Review them and
   add aliases for Chinese/Japanese names, former names, and brands.
   Combine markets in one COMPANIES list or supply your own --companies-csv.

   Alternatively, copy config.py outside the release and edit literals:
   SETTINGS: providers, request gaps, thresholds, optional raw-pool date range.
   EVENTS: observation/event windows; defaults use the newer frozen field schema.
   COMPANIES: empty by default, without the project's existing company universe.
   Required company fields: market/ticker/company_name.
   Recommended fields: aliases/price_source/price_symbol/currency/exchange.
   price_source: yahoo or tushare; Japanese prices use yahoo.
   Defaults: CSI300 uses Tushare; other markets use Yahoo.
   Default currencies: CSI300=CNY, HSTECH=HKD, SP500=USD, Nikkei225=JPY.
   membership_start/end constrain membership at the event start; empty values
   indicate an observed-company universe. Membership dates do not suppress
   company news in the pre-event observation window.

   Set explicit alias validity in SETTINGS["alias_records"] or --aliases-csv:
   market,ticker,alias,alias_type,confidence,valid_from,valid_to
   Explicit records override automatic validity for the same company and alias.
   confidence must be between 0 and 1. Numeric-only, overly short, and known
   ambiguous aliases are rejected. Articles matching only a shared ambiguous
   alias are not automatically assigned to every company.

3. Run the complete workflow
   python run_pipeline.py --config ../downloaded_data/my_config.py --work-dir ../downloaded_data/pipeline
   Add --parquet to export both CSV and Parquet.
   Add --event-ids covid19_first_wave to process selected events.
   Add --companies-csv ../downloaded_data/companies.csv to override COMPANIES.
   Use the company fields listed above; market is required.
   Preserve symbols as strings, including leading zeros.

   Stages:
   (1) Shared compressed news/major_news pool for all markets.
   (2) HTML cleaning, alias matching, and a company news library.
   (3) Cross-source daily deduplication, quality scores, representatives, audits.
   (4) Yahoo/Tushare price acquisition, date normalization, and volume conversion.
   (5) Observation-window slices labeled pre_event/during_event/post_event.
   (6) News/price long tables, firm-day alignment, quality and coverage reports.

   Split news windows recursively when responses reach the row limit. Fail if
   even a single second remains saturated; do not mark truncated data complete.
   Reuse successful fragments automatically. Rerun the same command to resume.
   --refresh fetches completed news/price fragments again.
   --from-cache uses complete caches only, without network access or tokens.
   These two flags cannot be combined.
   Empty responses have an explicit empty status; they do not prove no news
   existed across all sources.

4. Run individual stages
   python run_pipeline.py --config ../downloaded_data/my_config.py --stage pool
   python run_pipeline.py --config ../downloaded_data/my_config.py --stage company
   python run_pipeline.py --config ../downloaded_data/my_config.py --stage prices
   python run_pipeline.py --config ../downloaded_data/my_config.py --stage dataset
   The default work directory is the same as above. Pass the same --work-dir
   explicitly to each command when using another location.
   The pool stage does not require a company list.
   Set both raw_pool_start/raw_pool_end in SETTINGS to accumulate a longer pool.
   Without these dates, fetch event observation windows only. A longer pool must
   cover every observation window to be processed.
   company reads the raw pool; dataset reads the company and price libraries.
   Individual stage entries are news.py pool, news.py company, prices.py build,
   and run_pipeline.py --stage dataset. Use the same configuration and work directory.
   After changing companies, aliases, events, or the raw pool, rebuild the
   dependent company/prices/dataset stages. Stale builds are rejected.

5. Outputs, defaulting to downloaded_data/pipeline/ beside the code directory
   tushare/{news,major_news}/{source}/*.json.gz
       Shared raw pool with original API fields and content hashes.
   metadata/tushare_raw_pool_checkpoint.json
   metadata/price_checkpoint.json
   metadata/pipeline_run.json
       Fragment and workflow statuses, including success/empty/error, without tokens.
   builds/{build_id}/raw/{market}/company_news/news/{ticker}/{date}/
       Article JSON, news.csv, and representative_news.json/csv.
   builds/{build_id}/metadata/
       Companies/aliases, daily-pool statistics, quality scores, build fingerprints.
   prices/raw_cache/
       Raw price responses and normalized records.
   prices/normalized/{fingerprint}/
       Unified/by-market prices, source/unit audits, and separate Yahoo adj_close.
   datasets/{generation}/benchmark/data/
       price_ohlcv.csv
       source_news_text.csv
       firm_day.csv
       source_news_quality.csv
       coverage.csv
   datasets/{generation}/benchmark/views/by_event_market/{event_id}/{market}/
       Event/market tables, company-day news pools, representatives, slice manifests.
   metadata/latest_dataset.json
       Pointer to the latest dataset, using work-directory-relative paths.
   Changed configurations or pools receive separate builds; existing results
   are not deleted.

   price_ohlcv key: market/ticker/date
   Fields: market,ticker,date,open,high,low,close,volume,currency
   source_news_text fields:
   source_news_id,event_id,market,ticker,company_name,date,source_name,url_hash,title,content
   Leave url_hash empty when Tushare provides no URL; do not invent a URL/hash.
   firm_day includes the full calendar window and companies with no matched news.
   Missing prices remain empty without forward filling.
   OHLC is unadjusted. Convert A-share Tushare lots to shares by multiplying by
   100; other sources use shares. Yahoo adj_close does not replace close.
   Do not substitute successor-company prices for delisted-company prices.
   For pre-event prediction, filter firm_day to is_pre_event_input=1.
   during_event/post_event rows describe impact and are not pre-event inputs.
   news_query_complete refers to selected sources, not complete news coverage.
   Daily alignment uses original provider dates. Without a news time zone, do
   not infer whether an article arrived before or after the market session.

6. Standalone download entry points
   python prices.py yahoo --symbols AAPL MSFT 0700.HK 7203.T --start-date 2020-01-01 --end-date 2020-03-31
   python prices.py tushare --market cn --symbols 600519.SH --start-date 2020-01-01 --end-date 2020-03-31
   python news.py download --api news --sources sina cls --start-date 2020-01-01 --end-date 2020-01-02 --resume
   These commands download only. Use run_pipeline.py for matching and slicing.

7. Offline verification and release
   python test_pipeline.py
   Generate synthetic records in temporary directories to validate the workflow,
   cache reuse, unit conversion, deduplication, weekend news, zero-news companies,
   alias validity, and error handling.
   Tests do not access the network, real tokens, or the original project data.

   Share only the .py files in this directory. Keep your configurations,
   downloaded_data, caches, reports, datasets, and credentials outside the package.
   The scripts forbid data outputs inside the code release and disable local
   bytecode caches.

Official API documentation:
https://tushare.pro/document/1?doc_id=130
https://tushare.pro/document/2?doc_id=143
https://tushare.pro/document/2?doc_id=195
https://tushare.pro/document/2?doc_id=27
https://tushare.pro/document/2?doc_id=192
https://tushare.pro/document/2?doc_id=254
'''


def main():
    """Run the complete workflow, generate a company config or print usage."""
    if len(sys.argv) > 1 and sys.argv[1] == "usage":
        sys.argv.pop(1)
        parser = argparse.ArgumentParser(description="Print the complete usage guide")
        parser.parse_args()
        print(HELP)
        return 0
    if len(sys.argv) > 1 and sys.argv[1] == "create-config":
        sys.argv.pop(1)
        return create_config_main()
    return run_main()


if __name__ == "__main__":
    run_cli(main)
