"""Fold-aware multimodal reference training for T1, T2, and T3."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .evaluation import evaluate_predictions, write_metrics
from .labels import DEFAULT_HIGH_IMPACT_QUANTILE, iter_task_frames
from .models import build_task_estimator, predict_t1_score
from .schema import KEY_COLUMNS


@dataclass(frozen=True)
class TrainingResult:
    predictions: pd.DataFrame
    metrics: pd.DataFrame
    folds: pd.DataFrame


def _numeric_feature_columns(features: pd.DataFrame) -> list[str]:
    columns = [column for column in features.columns if column not in KEY_COLUMNS]
    non_numeric = [column for column in columns if not pd.api.types.is_numeric_dtype(features[column])]
    if non_numeric:
        raise ValueError(f"event feature frame contains non-numeric model columns: {non_numeric[:10]}")
    if not columns:
        raise ValueError("event feature frame contains no model features")
    return columns


def _join_features(targets: pd.DataFrame, features: pd.DataFrame, feature_columns: list[str], context: str) -> pd.DataFrame:
    joined = targets.merge(features, on=KEY_COLUMNS, how="left", validate="one_to_one", indicator=True)
    missing = joined["_merge"].ne("both")
    if missing.any():
        examples = joined.loc[missing, KEY_COLUMNS].head(5).to_dict("records")
        raise ValueError(f"{context} has {int(missing.sum())} samples without features: {examples}")
    return joined.drop(columns="_merge")


def train_reference_model(
    features: pd.DataFrame,
    base_labels: pd.DataFrame,
    split: dict[str, Any],
    split_name: str,
    t1_config: dict[str, Any],
    t2_config: dict[str, Any],
    seed: int = 42,
    quantile: float = DEFAULT_HIGH_IMPACT_QUANTILE,
) -> TrainingResult:
    feature_columns = _numeric_feature_columns(features)
    prediction_parts: list[pd.DataFrame] = []
    fold_rows: list[dict[str, object]] = []

    for bundle in iter_task_frames(split_name, split, base_labels, quantile=quantile):
        train = _join_features(bundle.train, features, feature_columns, f"{bundle.fold_id} train")
        test = _join_features(bundle.test, features, feature_columns, f"{bundle.fold_id} test")
        if test.empty:
            raise ValueError(f"{bundle.fold_id} has no test samples")

        x_train = train[feature_columns].astype(float)
        x_test = test[feature_columns].astype(float)
        y_t1 = train["high_impact_label"].to_numpy(dtype=int)
        y_t2 = train["abnormal_return"].to_numpy(dtype=float)

        t1_estimator = build_task_estimator("T1", t1_config, seed, y_t1)
        t2_estimator = build_task_estimator("T2", t2_config, seed, y_t2)
        t1_estimator.fit(x_train, y_t1)
        t2_estimator.fit(x_train, y_t2)

        fold_predictions = test[KEY_COLUMNS].copy()
        fold_predictions["t1_score"] = np.clip(predict_t1_score(t1_estimator, x_test), 0.0, 1.0)
        fold_predictions["t2_prediction"] = np.asarray(t2_estimator.predict(x_test), dtype=float)
        fold_predictions["fold_id"] = bundle.fold_id
        prediction_parts.append(fold_predictions)
        fold_rows.append(
            {
                "split_name": split_name,
                "fold_id": bundle.fold_id,
                "seed": seed,
                "n_train": len(train),
                "n_validation": len(bundle.validation),
                "n_test": len(test),
                "n_features": len(feature_columns),
                "t1_model": t1_config["model"],
                "t2_model": t2_config["model"],
                "high_impact_quantile": bundle.quantile,
                "high_impact_threshold": bundle.threshold,
            }
        )

    predictions = pd.concat(prediction_parts, ignore_index=True)
    duplicated = predictions.duplicated(KEY_COLUMNS, keep=False)
    if duplicated.any():
        examples = predictions.loc[duplicated, KEY_COLUMNS + ["fold_id"]].head(5).to_dict("records")
        raise ValueError(f"test predictions overlap across folds: {examples}")
    predictions = predictions.sort_values(KEY_COLUMNS).reset_index(drop=True)
    metrics = evaluate_predictions(base_labels, predictions, split, split_name, quantile=quantile)
    return TrainingResult(predictions=predictions, metrics=metrics, folds=pd.DataFrame(fold_rows))


def _write_frame(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix.lower() == ".parquet":
        frame.to_parquet(path, index=False)
    elif path.suffix.lower() == ".csv":
        frame.to_csv(path, index=False)
    else:
        raise ValueError(f"unsupported output format for {path}; use .csv or .parquet")


def write_training_result(
    result: TrainingResult,
    output_dir: str | Path,
    manifest: dict[str, Any],
) -> None:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    _write_frame(result.predictions, target / "predictions.parquet")
    write_metrics(result.metrics, target / "metrics.csv")
    _write_frame(result.folds, target / "folds.csv")
    (target / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
