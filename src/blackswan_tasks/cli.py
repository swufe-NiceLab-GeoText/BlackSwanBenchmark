"""Command-line interface for the three-task evaluator."""

from __future__ import annotations

import argparse
from pathlib import Path

from .evaluation import evaluate_predictions, write_metrics
from .labels import DEFAULT_HIGH_IMPACT_QUANTILE
from .schema import read_frame
from .splits import load_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate BlackSwanNewsBench T1/T2/T3 predictions")
    parser.add_argument("--labels", required=True, help="Base-label CSV or Parquet")
    parser.add_argument("--predictions", required=True, help="Prediction CSV or Parquet")
    parser.add_argument("--split", required=True, help="Split JSON")
    parser.add_argument("--split-name", default=None, help="Optional result split name")
    parser.add_argument("--quantile", type=float, default=DEFAULT_HIGH_IMPACT_QUANTILE)
    parser.add_argument("--output", required=True, help="Output .csv or .parquet")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split_path = Path(args.split)
    split = load_split(split_path)
    split_name = args.split_name or str(split.get("split_type") or split_path.stem)
    metrics = evaluate_predictions(
        base_labels=read_frame(args.labels),
        predictions=read_frame(args.predictions),
        split=split,
        split_name=split_name,
        quantile=args.quantile,
    )
    write_metrics(metrics, args.output)
    print(metrics.to_string(index=False))
    print(f"wrote {Path(args.output)}")


if __name__ == "__main__":
    main()

