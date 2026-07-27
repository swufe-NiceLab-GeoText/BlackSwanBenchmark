"""Split-aware runtime target construction for T1, T2, and T3."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from .schema import GROUP_COLUMNS
from .splits import filter_partition, fold_specs


DEFAULT_HIGH_IMPACT_QUANTILE = 0.90
LABEL_PROTOCOL_VERSION = "split_part_market_return_v2"


@dataclass(frozen=True)
class RuntimeTaskFrames:
    split_name: str
    fold_id: str
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    threshold: float
    quantile: float
    label_protocol_version: str = LABEL_PROTOCOL_VERSION


def attach_abnormal_returns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    if out.empty:
        return out
    out["event_return"] = pd.to_numeric(out["event_return"], errors="coerce")
    out["market_return"] = out.groupby(GROUP_COLUMNS, dropna=False)["event_return"].transform("mean")
    out["abnormal_return"] = out["event_return"] - out["market_return"]
    out["abs_abnormal_return"] = out["abnormal_return"].abs()
    out["label_protocol_version"] = LABEL_PROTOCOL_VERSION
    return out


def attach_task_targets(frame: pd.DataFrame, threshold: float) -> pd.DataFrame:
    out = attach_abnormal_returns(frame)
    if out.empty:
        out["high_impact_label"] = pd.Series(dtype="int8")
        out["ranking_target"] = pd.Series(dtype="float64")
        return out
    out["high_impact_label"] = (out["abs_abnormal_return"] >= threshold).astype("int8")
    out["ranking_target"] = out["abs_abnormal_return"].astype(float)
    return out


def iter_task_frames(
    split_name: str,
    split: dict[str, Any],
    base_labels: pd.DataFrame,
    quantile: float = DEFAULT_HIGH_IMPACT_QUANTILE,
) -> list[RuntimeTaskFrames]:
    if not 0.0 < quantile < 1.0:
        raise ValueError("high-impact quantile must be strictly between 0 and 1")

    bundles: list[RuntimeTaskFrames] = []
    for fold in fold_specs(split_name, split):
        train_base = filter_partition(base_labels, fold["train"])
        if train_base.empty:
            raise ValueError(f"{fold['fold_id']} has no training labels")
        train_targets = attach_abnormal_returns(train_base)
        threshold_values = train_targets["abs_abnormal_return"].dropna()
        if threshold_values.empty:
            raise ValueError(f"{fold['fold_id']} has no valid training values for Q90")
        threshold = float(threshold_values.quantile(quantile))

        bundles.append(
            RuntimeTaskFrames(
                split_name=split_name,
                fold_id=str(fold["fold_id"]),
                train=attach_task_targets(train_base, threshold),
                validation=attach_task_targets(
                    filter_partition(
                        base_labels,
                        fold.get("validation", {"events": [], "markets": []}),
                    ),
                    threshold,
                ),
                test=attach_task_targets(filter_partition(base_labels, fold["test"]), threshold),
                threshold=threshold,
                quantile=float(quantile),
            )
        )
    return bundles

