from __future__ import annotations

import json
import unittest
from pathlib import Path

import pandas as pd

from blackswan_tasks.evaluation import evaluate_predictions
from blackswan_tasks.labels import attach_abnormal_returns, iter_task_frames
from blackswan_tasks.schema import validate_base_labels


ROOT = Path(__file__).resolve().parents[1]


class ThreeTaskFrameworkTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.labels = pd.read_csv(ROOT / "examples" / "demo_base_labels.csv", dtype={"ticker": str})
        cls.predictions = pd.read_csv(
            ROOT / "examples" / "demo_predictions.csv",
            dtype={"ticker": str},
        )
        cls.split = json.loads((ROOT / "examples" / "demo_split.json").read_text())

    def test_train_only_q90_and_greater_equal_boundary(self) -> None:
        labels = validate_base_labels(self.labels)
        bundle = iter_task_frames("demo", self.split, labels)[0]
        expected = bundle.train["abs_abnormal_return"].quantile(0.90)
        self.assertAlmostEqual(bundle.threshold, expected)
        from blackswan_tasks.labels import attach_task_targets

        boundary = pd.DataFrame(
            {
                "event_id": ["boundary", "boundary"],
                "market": ["DEMO", "DEMO"],
                "ticker": ["B01", "B02"],
                "event_return": [-1.0, 1.0],
            }
        )
        rebuilt = attach_task_targets(boundary, threshold=1.0)
        self.assertTrue(rebuilt["high_impact_label"].eq(1).all())

    def test_abnormal_return_is_partition_centered(self) -> None:
        test = self.labels[self.labels["event_id"].eq("demo_test")]
        targets = attach_abnormal_returns(test)
        self.assertAlmostEqual(float(targets["abnormal_return"].mean()), 0.0, places=12)

    def test_all_three_tasks_are_evaluated(self) -> None:
        metrics = evaluate_predictions(
            self.labels,
            self.predictions,
            self.split,
            split_name="demo",
        )
        self.assertEqual(set(metrics["task"]), {"T1", "T2", "T3"})
        self.assertEqual(len(metrics), 6)
        t3 = metrics[metrics["task"].eq("T3")]
        self.assertTrue(t3["t3_source"].eq("abs(t2_prediction)").all())
        self.assertTrue(t3["t3_grouping"].eq("event_id+market").all())
        self.assertTrue(t3["n_groups"].eq(1).all())

    def test_materialized_t1_label_is_rejected(self) -> None:
        invalid = self.labels.copy()
        invalid["high_impact_label"] = 0
        with self.assertRaisesRegex(ValueError, "must not materialize"):
            validate_base_labels(invalid)


if __name__ == "__main__":
    unittest.main()
