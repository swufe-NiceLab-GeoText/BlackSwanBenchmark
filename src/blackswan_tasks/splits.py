"""Fixed split parsing and sample filtering."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd


def load_split(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    with source.open(encoding="utf-8") as handle:
        split = json.load(handle)
    if not isinstance(split, dict):
        raise ValueError("split JSON must contain an object")
    return split


def fold_specs(split_name: str, split: dict[str, Any]) -> list[dict[str, Any]]:
    folds = split.get("folds")
    if folds is not None:
        if not isinstance(folds, list) or not folds:
            raise ValueError("split folds must be a non-empty list")
        return list(folds)
    if "train" not in split or "test" not in split:
        raise ValueError("single-fold split must contain train and test")
    return [
        {
            "fold_id": f"{split_name}__test",
            "train": split["train"],
            "validation": split.get("validation", {"events": [], "markets": []}),
            "test": split["test"],
        }
    ]


def filter_partition(frame: pd.DataFrame, spec: dict[str, Any]) -> pd.DataFrame:
    events = spec.get("events", [])
    markets = spec.get("markets", [])
    if not events or not markets:
        return frame.iloc[0:0].copy()

    mask = frame["event_id"].isin(events) & frame["market"].isin(markets)
    tickers_by_market = spec.get("tickers_by_market")
    if tickers_by_market:
        ticker_mask = pd.Series(False, index=frame.index)
        for market, tickers in tickers_by_market.items():
            ticker_mask |= frame["market"].eq(market) & frame["ticker"].isin(tickers)
        mask &= ticker_mask
    return frame.loc[mask].copy().reset_index(drop=True)

