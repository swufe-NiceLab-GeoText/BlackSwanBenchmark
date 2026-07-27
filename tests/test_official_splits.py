from __future__ import annotations

import json
import unittest
from pathlib import Path

import pandas as pd

from blackswan_tasks.labels import iter_task_frames
from blackswan_tasks.schema import validate_base_labels
from blackswan_tasks.splits import load_split


ROOT = Path(__file__).resolve().parents[1]
SPLIT_ROOT = ROOT / "splits"


def synthetic_labels_for_official_splits() -> pd.DataFrame:
    random_split = load_split(SPLIT_ROOT / "random_firm_split_seed42.json")
    events = list(random_split["train"]["events"])
    markets = list(random_split["train"]["markets"])
    selected: dict[str, set[str]] = {market: set() for market in markets}
    for part in ["train", "validation", "test"]:
        by_market = random_split[part]["tickers_by_market"]
        for market in markets:
            selected[market].add(str(by_market[market][0]))

    rows: list[dict[str, object]] = []
    for event_index, event_id in enumerate(events):
        for market_index, market in enumerate(markets):
            for ticker_index, ticker in enumerate(sorted(selected[market])):
                rows.append(
                    {
                        "event_id": event_id,
                        "market": market,
                        "ticker": ticker,
                        "event_return": 0.01 * (event_index - market_index + ticker_index),
                        "label_available_flag": True,
                    }
                )
    return validate_base_labels(pd.DataFrame(rows))


class OfficialSplitTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.labels = synthetic_labels_for_official_splits()

    def test_random_firm_split(self) -> None:
        split = load_split(SPLIT_ROOT / "random_firm_split_seed42.json")
        bundles = iter_task_frames("random_firm_split_seed42", split, self.labels)
        self.assertEqual(len(bundles), 1)
        self.assertFalse(bundles[0].train.empty)
        self.assertFalse(bundles[0].validation.empty)
        self.assertFalse(bundles[0].test.empty)

    def test_leave_one_event_out(self) -> None:
        split = load_split(SPLIT_ROOT / "leave_one_event_out.json")
        bundles = iter_task_frames("leave_one_event_out", split, self.labels)
        self.assertEqual(len(bundles), 4)
        self.assertTrue(all(not bundle.train.empty for bundle in bundles))
        self.assertTrue(all(bundle.validation.empty for bundle in bundles))
        self.assertTrue(all(not bundle.test.empty for bundle in bundles))

    def test_leave_one_market_out(self) -> None:
        split = load_split(SPLIT_ROOT / "leave_one_market_out.json")
        bundles = iter_task_frames("leave_one_market_out", split, self.labels)
        self.assertEqual(len(bundles), 4)
        self.assertTrue(all(not bundle.train.empty for bundle in bundles))
        self.assertTrue(all(bundle.validation.empty for bundle in bundles))
        self.assertTrue(all(not bundle.test.empty for bundle in bundles))


if __name__ == "__main__":
    unittest.main()
