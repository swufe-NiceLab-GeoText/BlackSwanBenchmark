"""Shared schemas and validation helpers."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


KEY_COLUMNS = ["event_id", "market", "ticker"]
GROUP_COLUMNS = ["event_id", "market"]
BASE_LABEL_COLUMNS = KEY_COLUMNS + ["event_return", "label_available_flag"]
PREDICTION_COLUMNS = KEY_COLUMNS + ["t1_score", "t2_prediction"]


def read_frame(path: str | Path) -> pd.DataFrame:
    source = Path(path)
    if not source.exists():
        raise FileNotFoundError(source)
    if source.is_dir():
        parquet_files = sorted(source.rglob("*.parquet"))
        if not parquet_files:
            raise ValueError(f"Parquet dataset directory contains no .parquet files: {source}")
        return pd.read_parquet(source)
    suffix = source.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(source)
    if suffix in {".csv", ".txt"}:
        return pd.read_csv(source, dtype={column: str for column in KEY_COLUMNS})
    raise ValueError(f"unsupported table format {suffix!r}; use CSV or Parquet")


def require_columns(frame: pd.DataFrame, columns: list[str], context: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{context} is missing required columns: {missing}")


def validate_unique_keys(frame: pd.DataFrame, context: str) -> None:
    if frame[KEY_COLUMNS].isna().any().any():
        raise ValueError(f"{context} contains null sample keys")
    duplicated = frame.duplicated(KEY_COLUMNS, keep=False)
    if duplicated.any():
        examples = frame.loc[duplicated, KEY_COLUMNS].head(5).to_dict("records")
        raise ValueError(f"{context} contains duplicate sample keys: {examples}")


def coerce_availability(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values):
        return values.fillna(False).astype(bool)
    normalized = values.astype("string").str.strip().str.lower()
    mapped = normalized.map(
        {
            "true": True,
            "1": True,
            "yes": True,
            "false": False,
            "0": False,
            "no": False,
        }
    )
    invalid = values.notna() & mapped.isna()
    if invalid.any():
        bad = sorted(values[invalid].astype(str).unique().tolist())[:5]
        raise ValueError(f"invalid label_available_flag values: {bad}")
    return mapped.fillna(False).astype(bool)


def validate_base_labels(frame: pd.DataFrame) -> pd.DataFrame:
    require_columns(frame, BASE_LABEL_COLUMNS, "base labels")
    if "high_impact_label" in frame.columns:
        raise ValueError("base labels must not materialize high_impact_label")
    out = frame.copy()
    validate_unique_keys(out, "base labels")
    out["label_available_flag"] = coerce_availability(out["label_available_flag"])
    out["event_return"] = pd.to_numeric(out["event_return"], errors="coerce")
    available = out["label_available_flag"]
    if out.loc[available, "event_return"].isna().any():
        raise ValueError("available base labels contain non-numeric event_return values")
    return out.loc[available].sort_values(KEY_COLUMNS).reset_index(drop=True)


def validate_predictions(frame: pd.DataFrame) -> pd.DataFrame:
    require_columns(frame, PREDICTION_COLUMNS, "predictions")
    out = frame[PREDICTION_COLUMNS].copy()
    validate_unique_keys(out, "predictions")
    for column in ["t1_score", "t2_prediction"]:
        out[column] = pd.to_numeric(out[column], errors="coerce")
        values = out[column].to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"predictions contain non-finite {column} values")
    if not out["t1_score"].between(0.0, 1.0, inclusive="both").all():
        raise ValueError("t1_score must be within [0, 1]")
    return out.sort_values(KEY_COLUMNS).reset_index(drop=True)
