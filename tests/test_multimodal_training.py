from __future__ import annotations

import json
import unittest
from pathlib import Path

import pandas as pd

from blackswan_tasks.data import (
    _embedding_columns,
    build_event_feature_frame,
    build_sequence_batch,
    load_daily_features,
    load_event_feature_frame,
    resolve_ready_to_run_paths,
)
from blackswan_tasks.training import train_reference_model


ROOT = Path(__file__).resolve().parents[1]


def synthetic_daily(labels: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for sample_index, sample in enumerate(labels.itertuples(index=False)):
        start = pd.Timestamp("2024-01-10") if sample.event_id == "demo_train" else pd.Timestamp("2024-02-10")
        for day_index, date in enumerate(pd.date_range(start - pd.Timedelta(days=3), periods=3)):
            rows.append(
                {
                    "event_id": sample.event_id,
                    "market": sample.market,
                    "ticker": sample.ticker,
                    "date": date,
                    "event_start_date": start,
                    "days_to_event_start": (start - date).days,
                    "close": 100.0 + sample_index + day_index,
                    "volume": 1000.0 + 10 * sample_index,
                    "price_daily_return": 0.001 * (sample_index - day_index),
                    "source_news_count": float((sample_index + day_index) % 3),
                    "source_news_missing_flag": 0.0,
                    "gdelt_news_count": float(sample_index + day_index),
                    "avg_gdelt_tone": float(day_index - 1),
                    "market_gdelt_unique_article_count": float(20 + day_index),
                    "avg_market_gdelt_tone_equal_weighted": 0.25 * day_index,
                }
            )
    return pd.DataFrame(rows)


def synthetic_embeddings(daily: pd.DataFrame) -> pd.DataFrame:
    text = daily.iloc[::2][["event_id", "market", "ticker", "date"]].copy()
    text["text_doc_count"] = 1
    text["embedding_dim"] = 2
    text["emb_000"] = 0.1
    text["emb_001"] = -0.2
    return text


class MultimodalTrainingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.labels = pd.read_csv(ROOT / "examples" / "demo_base_labels.csv", dtype={"ticker": str})
        cls.daily = synthetic_daily(cls.labels)
        cls.embeddings = synthetic_embeddings(cls.daily)
        cls.split = json.loads((ROOT / "examples" / "demo_split.json").read_text())

    def test_event_features_align_all_three_modalities(self) -> None:
        features = build_event_feature_frame(self.daily, self.labels, "S6", self.embeddings)
        self.assertEqual(len(features), len(self.labels))
        self.assertIn("close__last", features.columns)
        self.assertIn("source_news_count__sum", features.columns)
        self.assertIn("gdelt_news_count__sum", features.columns)
        self.assertIn("market_gdelt_unique_article_count__sum", features.columns)
        self.assertIn("emb_000", features.columns)
        self.assertIn("text_embedding_available_flag", features.columns)

    def test_sequence_builder_preserves_relative_time_and_mask(self) -> None:
        batch = build_sequence_batch(self.daily, self.labels, "S6", self.embeddings, max_length=4)
        self.assertEqual(batch.values.shape[0], len(self.labels))
        self.assertEqual(batch.values.shape[1], 4)
        self.assertTrue((batch.mask.sum(axis=1) == 3).all())
        self.assertIn("emb_001", batch.feature_names)

    def test_post_event_daily_row_is_rejected(self) -> None:
        invalid = self.daily.copy()
        invalid.loc[0, "date"] = invalid.loc[0, "event_start_date"]
        with self.assertRaisesRegex(ValueError, "event-window or post-event"):
            load_daily_features_from_frame(invalid)

    def test_reference_training_produces_all_task_metrics(self) -> None:
        features = build_event_feature_frame(self.daily, self.labels, "S6", self.embeddings)
        t1_config = {"task": "T1", "model": "linear", "params": {"max_iter": 500}}
        t2_config = {"task": "T2", "model": "linear", "params": {"alpha": 1.0}}
        result = train_reference_model(
            features,
            self.labels,
            self.split,
            "demo",
            t1_config,
            t2_config,
            seed=42,
        )
        self.assertEqual(set(result.metrics["task"]), {"T1", "T2", "T3"})
        self.assertEqual(len(result.predictions), 6)
        self.assertTrue(result.predictions["t1_score"].between(0.0, 1.0).all())

    def test_huggingface_shard_layout_is_loaded_directly(self) -> None:
        from tempfile import TemporaryDirectory

        with TemporaryDirectory() as directory:
            root = Path(directory)
            hf_data = root / "data"
            for name in ["daily_features", "text_embeddings", "outcomes"]:
                (hf_data / name).mkdir(parents=True)
            self.daily.to_parquet(hf_data / "daily_features" / "full--demo--00000.parquet", index=False)
            self.embeddings.to_parquet(
                hf_data / "text_embeddings" / "full--demo--00000.parquet",
                index=False,
            )
            self.labels.to_parquet(hf_data / "outcomes" / "full--demo--00000.parquet", index=False)

            paths = resolve_ready_to_run_paths(root)
            self.assertEqual(paths.daily_features.name, "daily_features")
            self.assertEqual(paths.text_embeddings.name, "text_embeddings")
            self.assertEqual(paths.labels.name, "outcomes")
            features, labels = load_event_feature_frame(root, "S6")
            self.assertEqual(len(features), len(self.labels))
            self.assertEqual(len(labels), len(self.labels))
            self.assertIn("emb_000", features.columns)

    def test_embedding_columns_are_sorted_by_numeric_index(self) -> None:
        frame = pd.DataFrame(columns=["emb_1000", "emb_101", "emb_100", "metadata"])
        self.assertEqual(_embedding_columns(frame), ["emb_100", "emb_101", "emb_1000"])


def load_daily_features_from_frame(frame: pd.DataFrame) -> pd.DataFrame:
    from tempfile import TemporaryDirectory

    with TemporaryDirectory() as directory:
        path = Path(directory) / "daily.parquet"
        frame.to_parquet(path, index=False)
        return load_daily_features(path)


if __name__ == "__main__":
    unittest.main()
