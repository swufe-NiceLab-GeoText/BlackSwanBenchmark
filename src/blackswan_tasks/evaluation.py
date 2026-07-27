"""Unified evaluation of T1, T2, and T3 predictions."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .labels import DEFAULT_HIGH_IMPACT_QUANTILE, iter_task_frames
from .metrics import classification_metrics, ranking_metrics_by_event_market, regression_metrics
from .schema import KEY_COLUMNS, validate_base_labels, validate_predictions


def _join_test_predictions(test: pd.DataFrame, predictions: pd.DataFrame, fold_id: str) -> pd.DataFrame:
    joined = test.merge(predictions, on=KEY_COLUMNS, how="left", validate="one_to_one")
    missing = joined[["t1_score", "t2_prediction"]].isna().any(axis=1)
    if missing.any():
        examples = joined.loc[missing, KEY_COLUMNS].head(5).to_dict("records")
        raise ValueError(f"{fold_id} is missing predictions for {int(missing.sum())} rows: {examples}")
    joined["t3_score"] = joined["t2_prediction"].abs()
    return joined


def _metric_rows(
    bundle: Any,
    evaluated: pd.DataFrame,
) -> list[dict[str, object]]:
    t1 = classification_metrics(evaluated["high_impact_label"], evaluated["t1_score"])
    t2 = regression_metrics(evaluated["abnormal_return"], evaluated["t2_prediction"])
    t3, n_groups = ranking_metrics_by_event_market(evaluated)

    rows: list[dict[str, object]] = []
    for task, metrics in [("T1", t1), ("T2", t2), ("T3", t3)]:
        for metric, value in metrics.items():
            rows.append(
                {
                    "split_name": bundle.split_name,
                    "fold_id": bundle.fold_id,
                    "task": task,
                    "metric": metric,
                    "value": value,
                    "n_samples": int(len(evaluated)),
                    "n_groups": n_groups if task == "T3" else pd.NA,
                    "high_impact_quantile": bundle.quantile,
                    "high_impact_threshold": bundle.threshold,
                    "label_protocol_version": bundle.label_protocol_version,
                    "t3_source": "abs(t2_prediction)" if task == "T3" else "",
                    "t3_grouping": "event_id+market" if task == "T3" else "",
                }
            )
    return rows


def evaluate_predictions(
    base_labels: pd.DataFrame,
    predictions: pd.DataFrame,
    split: dict[str, Any],
    split_name: str,
    quantile: float = DEFAULT_HIGH_IMPACT_QUANTILE,
) -> pd.DataFrame:
    labels = validate_base_labels(base_labels)
    predicted = validate_predictions(predictions)
    rows: list[dict[str, object]] = []
    for bundle in iter_task_frames(split_name, split, labels, quantile=quantile):
        if bundle.test.empty:
            raise ValueError(f"{bundle.fold_id} has no test labels")
        evaluated = _join_test_predictions(bundle.test, predicted, bundle.fold_id)
        rows.extend(_metric_rows(bundle, evaluated))
    return pd.DataFrame(rows)


def write_metrics(metrics: pd.DataFrame, path: str | Path) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.suffix.lower() == ".parquet":
        metrics.to_parquet(target, index=False)
    elif target.suffix.lower() == ".csv":
        metrics.to_csv(target, index=False)
    else:
        raise ValueError("metric output must use .csv or .parquet")

