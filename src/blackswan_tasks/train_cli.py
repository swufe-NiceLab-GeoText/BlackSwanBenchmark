"""Command-line training entry point for the multimodal reference baseline."""

from __future__ import annotations

import argparse
import importlib.metadata
from pathlib import Path

from .data import load_event_feature_frame, resolve_ready_to_run_paths
from .labels import DEFAULT_HIGH_IMPACT_QUANTILE
from .models import load_task_config
from .splits import load_split
from .training import train_reference_model, write_training_result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train and evaluate a BlackSwanNewsBench multimodal reference model")
    parser.add_argument("--data-dir", default="data", help="Directory containing ready-to-run Parquet files")
    parser.add_argument("--split", required=True, help="Official split JSON")
    parser.add_argument("--split-name", default=None, help="Optional result split name")
    parser.add_argument("--feature-set", default="S6", choices=["S0", "S1", "S2", "S3", "S4", "S5", "S6"])
    parser.add_argument("--t1-config", required=True, help="Independent T1 model configuration JSON")
    parser.add_argument("--t2-config", required=True, help="Independent T2 model configuration JSON")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--quantile", type=float, default=DEFAULT_HIGH_IMPACT_QUANTILE)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split_path = Path(args.split)
    split = load_split(split_path)
    split_name = args.split_name or str(split.get("split_type") or split_path.stem)
    features, labels = load_event_feature_frame(args.data_dir, args.feature_set)
    t1_config = load_task_config(args.t1_config, "T1")
    t2_config = load_task_config(args.t2_config, "T2")
    resolved_paths = resolve_ready_to_run_paths(args.data_dir)
    result = train_reference_model(
        features=features,
        base_labels=labels,
        split=split,
        split_name=split_name,
        t1_config=t1_config,
        t2_config=t2_config,
        seed=args.seed,
        quantile=args.quantile,
    )
    configured_models = sorted({t1_config["model"], t2_config["model"]})
    runner_name = "local_reference_" + "_and_".join(configured_models)
    manifest = {
        "framework": "BlackSwanNewsBench_three_task_framework",
        "runner": runner_name,
        "reporting_status": "local baseline",
        "data_dir": str(Path(args.data_dir)),
        "resolved_data": {
            "daily_features": str(resolved_paths.daily_features),
            "text_embeddings": str(resolved_paths.text_embeddings),
            "outcomes": str(resolved_paths.labels),
        },
        "split": str(split_path),
        "split_name": split_name,
        "feature_set": args.feature_set,
        "seed": args.seed,
        "quantile": args.quantile,
        "t1_config": t1_config,
        "t2_config": t2_config,
        "t3_source": "abs(t2_prediction)",
        "t3_grouping": "event_id+market",
        "dependencies": {
            package: importlib.metadata.version(package)
            for package in ["numpy", "pandas", "pyarrow", "scikit-learn", "scipy"]
        },
    }
    if t1_config["model"] == "lightgbm" or t2_config["model"] == "lightgbm":
        manifest["dependencies"]["lightgbm"] = importlib.metadata.version("lightgbm")
    write_training_result(result, args.output_dir, manifest)
    print(result.metrics.to_string(index=False))
    print(f"wrote {Path(args.output_dir)}")


if __name__ == "__main__":
    main()
