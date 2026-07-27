"""Ready-to-run multimodal data loading and feature construction."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .schema import KEY_COLUMNS, read_frame, require_columns, validate_base_labels


DAILY_KEY_COLUMNS = KEY_COLUMNS + ["date"]

PRICE_COLUMNS = [
    "days_to_event_start",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "price_daily_return",
    "price_trading_day_flag",
    "price_missing_flag",
    "close_ffill",
]

SOURCE_NEWS_COLUMNS = [
    "source_news_count",
    "source_count",
    "title_char_count",
    "content_char_count",
    "avg_title_length",
    "avg_content_length",
    "title_available_count",
    "content_available_count",
    "event_source_news_count",
    "event_source_active_day_count",
    "source_unavailable_flag",
    "source_available_zero_news_flag",
    "source_news_missing_flag",
]

GDELT_FIRM_COLUMNS = [
    "gdelt_news_count",
    "gdelt_source_count",
    "gdelt_source_entropy",
    "avg_gdelt_tone",
    "min_gdelt_tone",
    "max_gdelt_tone",
    "positive_news_count",
    "negative_news_count",
    "neutral_news_count",
    "tone_missing_count",
    "positive_news_share",
    "negative_news_share",
    "gdelt_theme_count",
    "gdelt_entity_count",
    "gdelt_news_count_3d",
    "gdelt_news_count_7d",
    "gdelt_news_count_14d",
    "gdelt_negative_count_3d",
    "gdelt_negative_count_7d",
    "gdelt_negative_share_7d",
    "gdelt_avg_tone_3d",
    "gdelt_avg_tone_7d",
    "gdelt_min_tone_7d",
    "gdelt_news_burst_3d_vs_14d",
    "gdelt_negative_burst_3d_vs_14d",
    "gdelt_tone_drop_3d_vs_14d",
    "gdelt_theme_war_conflict_count",
    "gdelt_theme_war_conflict_share",
    "gdelt_theme_sanction_count",
    "gdelt_theme_sanction_share",
    "gdelt_theme_energy_count",
    "gdelt_theme_energy_share",
    "gdelt_theme_supply_chain_trade_count",
    "gdelt_theme_supply_chain_trade_share",
    "gdelt_theme_financial_stress_count",
    "gdelt_theme_financial_stress_share",
    "gdelt_theme_policy_regulation_count",
    "gdelt_theme_policy_regulation_share",
    "gdelt_theme_disaster_health_count",
    "gdelt_theme_disaster_health_share",
    "gdelt_theme_protest_unrest_count",
    "gdelt_theme_protest_unrest_share",
    "gdelt_firm_coverage_missing_flag",
    "gdelt_firm_row_missing_flag",
]

GDELT_MARKET_COLUMNS = [
    "eligible_firm_count",
    "covered_firm_count",
    "covered_firm_share",
    "market_gdelt_firm_mention_count",
    "market_gdelt_unique_article_count",
    "avg_market_gdelt_tone_equal_weighted",
    "avg_market_gdelt_tone_news_weighted",
    "min_market_firm_tone",
    "negative_firm_count",
    "negative_firm_share",
    "market_negative_news_count",
    "market_negative_news_share",
    "market_source_count",
    "market_source_entropy",
    "market_theme_count",
    "market_entity_count",
    "market_gdelt_mentions_3d",
    "market_gdelt_mentions_7d",
    "market_gdelt_mentions_14d",
    "market_negative_news_3d",
    "market_negative_news_7d",
    "market_negative_share_7d",
    "market_covered_firm_share_7d",
    "market_tone_drop_3d_vs_14d",
    "market_negative_burst_3d_vs_14d",
    "market_coverage_burst_3d_vs_14d",
    "market_theme_war_conflict_count",
    "market_theme_war_conflict_share",
    "market_theme_sanction_count",
    "market_theme_sanction_share",
    "market_theme_energy_count",
    "market_theme_energy_share",
    "market_theme_supply_chain_trade_count",
    "market_theme_supply_chain_trade_share",
    "market_theme_financial_stress_count",
    "market_theme_financial_stress_share",
    "market_theme_policy_regulation_count",
    "market_theme_policy_regulation_share",
    "market_theme_disaster_health_count",
    "market_theme_disaster_health_share",
    "market_theme_protest_unrest_count",
    "market_theme_protest_unrest_share",
    "gdelt_market_coverage_missing_flag",
    "gdelt_market_row_missing_flag",
]

FEATURE_SET_COLUMNS = {
    "S0": PRICE_COLUMNS,
    "S1": PRICE_COLUMNS + SOURCE_NEWS_COLUMNS,
    "S2": PRICE_COLUMNS,
    "S3": PRICE_COLUMNS + GDELT_FIRM_COLUMNS,
    "S4": PRICE_COLUMNS + GDELT_MARKET_COLUMNS,
    "S5": PRICE_COLUMNS + GDELT_FIRM_COLUMNS + GDELT_MARKET_COLUMNS,
    "S6": PRICE_COLUMNS + SOURCE_NEWS_COLUMNS + GDELT_FIRM_COLUMNS + GDELT_MARKET_COLUMNS,
}

TEXT_EMBEDDING_FEATURE_SETS = {"S2", "S6"}
REPEATED_DAILY_COLUMNS = {"event_source_news_count", "event_source_active_day_count", "eligible_firm_count"}


@dataclass(frozen=True)
class ReadyToRunPaths:
    daily_features: Path
    text_embeddings: Path
    labels: Path


@dataclass(frozen=True)
class SequenceBatch:
    keys: pd.DataFrame
    values: np.ndarray
    mask: np.ndarray
    feature_names: tuple[str, ...]


def _resolve_file(data_dir: Path, names: tuple[str, ...], collection_name: str) -> Path:
    roots = [data_dir, data_dir / "ready_to_run"]
    for root in roots:
        for name in names:
            candidate = root / name
            if candidate.exists():
                return candidate
    collection_roots = [data_dir / "data", data_dir]
    for root in collection_roots:
        candidate = root / collection_name
        if candidate.is_dir() and any(candidate.glob("*.parquet")):
            return candidate
    expected_paths = [root / name for root in roots for name in names]
    expected_paths.extend(root / collection_name for root in collection_roots)
    expected = ", ".join(str(path) for path in expected_paths)
    raise FileNotFoundError(f"none of the expected data files exists: {expected}")


def resolve_ready_to_run_paths(data_dir: str | Path) -> ReadyToRunPaths:
    root = Path(data_dir)
    return ReadyToRunPaths(
        daily_features=_resolve_file(root, ("daily_feature_frame.parquet",), "daily_features"),
        text_embeddings=_resolve_file(root, ("source_news_text_embeddings.parquet",), "text_embeddings"),
        labels=_resolve_file(
            root,
            ("price_event_outcomes.parquet", "price_event_labels.parquet"),
            "outcomes",
        ),
    )


def feature_set_uses_text_embeddings(feature_set: str) -> bool:
    return feature_set.upper() in TEXT_EMBEDDING_FEATURE_SETS


def _embedding_columns(frame: pd.DataFrame) -> list[str]:
    columns = [column for column in frame.columns if column.startswith("emb_")]
    try:
        return sorted(columns, key=lambda column: int(column.removeprefix("emb_")))
    except ValueError as exc:
        raise ValueError("embedding columns must use numeric emb_<index> names") from exc


def _feature_columns(feature_set: str, frame: pd.DataFrame) -> list[str]:
    normalized = feature_set.upper()
    if normalized not in FEATURE_SET_COLUMNS:
        raise ValueError(f"unknown feature set {feature_set!r}; choose from {sorted(FEATURE_SET_COLUMNS)}")
    columns = [column for column in FEATURE_SET_COLUMNS[normalized] if column in frame.columns]
    if not columns:
        raise ValueError(f"daily feature frame has no columns for feature set {normalized}")
    return columns


def load_daily_features(path: str | Path) -> pd.DataFrame:
    frame = read_frame(path)
    require_columns(frame, DAILY_KEY_COLUMNS + ["event_start_date"], "daily feature frame")
    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"], errors="raise")
    out["event_start_date"] = pd.to_datetime(out["event_start_date"], errors="raise")
    if out.duplicated(DAILY_KEY_COLUMNS).any():
        raise ValueError("daily feature frame contains duplicate event-market-ticker-date keys")
    if (out["date"] >= out["event_start_date"]).any():
        raise ValueError("daily feature frame contains event-window or post-event rows")
    return out.sort_values(DAILY_KEY_COLUMNS).reset_index(drop=True)


def load_text_embeddings(path: str | Path, daily: pd.DataFrame) -> pd.DataFrame:
    frame = read_frame(path)
    require_columns(frame, DAILY_KEY_COLUMNS + ["text_doc_count", "embedding_dim"], "text embeddings")
    out = frame.copy()
    out["date"] = pd.to_datetime(out["date"], errors="raise")
    if out.duplicated(DAILY_KEY_COLUMNS).any():
        raise ValueError("text embeddings contain duplicate event-market-ticker-date keys")
    embedding_columns = _embedding_columns(out)
    if not embedding_columns:
        raise ValueError("text embeddings contain no emb_* columns")
    dimensions = pd.to_numeric(out["embedding_dim"], errors="coerce").dropna().unique()
    if len(dimensions) != 1 or int(dimensions[0]) != len(embedding_columns):
        raise ValueError("embedding_dim does not match the number of emb_* columns")

    event_dates = daily[["event_id", "event_start_date"]].drop_duplicates()
    checked = out[["event_id", "date"]].merge(event_dates, on="event_id", how="left", validate="many_to_one")
    if checked["event_start_date"].isna().any():
        raise ValueError("text embeddings contain an unknown event_id")
    if (checked["date"] >= checked["event_start_date"]).any():
        raise ValueError("text embeddings contain event-window or post-event rows")
    return out.sort_values(DAILY_KEY_COLUMNS).reset_index(drop=True)


def load_ready_to_run_labels(path: str | Path) -> pd.DataFrame:
    return validate_base_labels(read_frame(path))


def _should_sum(column: str) -> bool:
    if column in REPEATED_DAILY_COLUMNS:
        return False
    return column == "volume" or column.endswith("_count") or column.endswith("_flag")


def aggregate_daily_features(daily: pd.DataFrame, feature_set: str) -> pd.DataFrame:
    selected = _feature_columns(feature_set, daily)
    work = daily[DAILY_KEY_COLUMNS + selected].copy()
    for column in selected:
        work[column] = pd.to_numeric(work[column], errors="coerce")

    aggregations: dict[str, list[str]] = {}
    for column in selected:
        functions = ["mean", "std", "min", "max", "last"]
        if _should_sum(column):
            functions.append("sum")
        aggregations[column] = functions

    grouped = work.groupby(KEY_COLUMNS, sort=False)
    output = grouped.agg(aggregations)
    output.columns = [f"{column}__{function}" for column, function in output.columns]
    output = output.join(grouped.size().rename("pre_event_day_count")).reset_index()
    std_columns = [column for column in output.columns if column.endswith("__std")]
    output[std_columns] = output[std_columns].fillna(0.0)
    return output


def aggregate_text_embeddings(embeddings: pd.DataFrame) -> pd.DataFrame:
    embedding_columns = _embedding_columns(embeddings)
    work = embeddings[KEY_COLUMNS + ["text_doc_count"] + embedding_columns].copy()
    work["text_doc_count"] = pd.to_numeric(work["text_doc_count"], errors="coerce").fillna(0.0)
    for column in embedding_columns:
        work[column] = pd.to_numeric(work[column], errors="coerce")
    grouped = work.groupby(KEY_COLUMNS, sort=False)
    output = grouped[embedding_columns].mean()
    output = output.join(grouped["text_doc_count"].sum().rename("text_embedding_doc_count"))
    output["text_embedding_available_flag"] = 1.0
    return output.reset_index()


def build_event_feature_frame(
    daily: pd.DataFrame,
    labels: pd.DataFrame,
    feature_set: str,
    embeddings: pd.DataFrame | None = None,
) -> pd.DataFrame:
    normalized = feature_set.upper()
    features = aggregate_daily_features(daily, normalized)
    if feature_set_uses_text_embeddings(normalized):
        if embeddings is None:
            raise ValueError(f"feature set {normalized} requires source-news text embeddings")
        features = features.merge(aggregate_text_embeddings(embeddings), on=KEY_COLUMNS, how="left", validate="one_to_one")
        features["text_embedding_doc_count"] = features["text_embedding_doc_count"].fillna(0.0)
        features["text_embedding_available_flag"] = features["text_embedding_available_flag"].fillna(0.0)

    label_keys = validate_base_labels(labels)[KEY_COLUMNS]
    aligned = label_keys.merge(features, on=KEY_COLUMNS, how="left", validate="one_to_one", indicator=True)
    missing = aligned["_merge"].ne("both")
    if missing.any():
        examples = aligned.loc[missing, KEY_COLUMNS].head(5).to_dict("records")
        raise ValueError(f"{int(missing.sum())} labeled samples have no pre-event features: {examples}")
    return aligned.drop(columns="_merge").sort_values(KEY_COLUMNS).reset_index(drop=True)


def load_event_feature_frame(data_dir: str | Path, feature_set: str = "S6") -> tuple[pd.DataFrame, pd.DataFrame]:
    paths = resolve_ready_to_run_paths(data_dir)
    daily = load_daily_features(paths.daily_features)
    labels = load_ready_to_run_labels(paths.labels)
    embeddings = None
    if feature_set_uses_text_embeddings(feature_set):
        embeddings = load_text_embeddings(paths.text_embeddings, daily)
    return build_event_feature_frame(daily, labels, feature_set, embeddings), labels


def build_sequence_batch(
    daily: pd.DataFrame,
    labels: pd.DataFrame,
    feature_set: str,
    embeddings: pd.DataFrame | None = None,
    max_length: int = 90,
) -> SequenceBatch:
    if max_length <= 0:
        raise ValueError("max_length must be positive")
    normalized = feature_set.upper()
    selected = _feature_columns(normalized, daily)
    frame = daily[DAILY_KEY_COLUMNS + selected].copy()
    if feature_set_uses_text_embeddings(normalized):
        if embeddings is None:
            raise ValueError(f"feature set {normalized} requires source-news text embeddings")
        embedding_columns = _embedding_columns(embeddings)
        text = embeddings[DAILY_KEY_COLUMNS + ["text_doc_count"] + embedding_columns].copy()
        text = text.rename(columns={"text_doc_count": "text_embedding_doc_count"})
        text["text_embedding_available_flag"] = 1.0
        frame = frame.merge(text, on=DAILY_KEY_COLUMNS, how="left", validate="one_to_one")
        frame["text_embedding_doc_count"] = frame["text_embedding_doc_count"].fillna(0.0)
        frame["text_embedding_available_flag"] = frame["text_embedding_available_flag"].fillna(0.0)
        selected += ["text_embedding_doc_count", "text_embedding_available_flag"] + embedding_columns

    for column in selected:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.sort_values(DAILY_KEY_COLUMNS)
    groups = {key: group for key, group in frame.groupby(KEY_COLUMNS, sort=False)}
    keys = validate_base_labels(labels)[KEY_COLUMNS].sort_values(KEY_COLUMNS).reset_index(drop=True)
    values = np.full((len(keys), max_length, len(selected)), np.nan, dtype=np.float32)
    mask = np.zeros((len(keys), max_length), dtype=bool)
    for index, key_values in enumerate(keys.itertuples(index=False, name=None)):
        group = groups.get(tuple(key_values))
        if group is None or group.empty:
            raise ValueError(f"labeled sample has no pre-event sequence: {tuple(key_values)}")
        sequence = group[selected].tail(max_length).to_numpy(dtype=np.float32)
        length = len(sequence)
        values[index, -length:, :] = sequence
        mask[index, -length:] = True
    return SequenceBatch(keys=keys, values=values, mask=mask, feature_names=tuple(selected))
