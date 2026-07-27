"""Local reference estimators for the model-agnostic task framework."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.dummy import DummyClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


SUPPORTED_REFERENCE_MODELS = {"linear", "early_fusion_mlp", "lightgbm"}


def load_task_config(path: str | Path, expected_task: str) -> dict[str, Any]:
    source = Path(path)
    with source.open(encoding="utf-8") as handle:
        config = json.load(handle)
    if not isinstance(config, dict):
        raise ValueError(f"task config must be a JSON object: {source}")
    task = str(config.get("task", expected_task)).upper()
    if task != expected_task.upper():
        raise ValueError(f"{source} declares task {task}, expected {expected_task.upper()}")
    model = str(config.get("model", "")).lower()
    if model not in SUPPORTED_REFERENCE_MODELS:
        raise ValueError(f"unsupported local reference model {model!r}; choose from {sorted(SUPPORTED_REFERENCE_MODELS)}")
    params = config.get("params", {})
    if not isinstance(params, dict):
        raise ValueError(f"params must be a JSON object: {source}")
    return {"task": task, "model": model, "params": dict(params)}


def _normalize_params(params: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(params)
    hidden = normalized.get("hidden_layer_sizes")
    if isinstance(hidden, list):
        normalized["hidden_layer_sizes"] = tuple(int(value) for value in hidden)
    return normalized


def build_task_estimator(
    task: str,
    config: dict[str, Any],
    seed: int,
    y_train: np.ndarray,
) -> Pipeline:
    normalized_task = task.upper()
    model_name = str(config["model"]).lower()
    params = _normalize_params(config.get("params", {}))

    if normalized_task == "T1" and np.unique(y_train).size < 2:
        model: BaseEstimator = DummyClassifier(strategy="constant", constant=int(y_train[0]))
    elif model_name == "linear" and normalized_task == "T1":
        defaults = {"max_iter": 2000, "class_weight": "balanced", "random_state": seed}
        defaults.update(params)
        model = LogisticRegression(**defaults)
    elif model_name == "linear" and normalized_task == "T2":
        defaults = {"alpha": 1.0}
        defaults.update(params)
        model = Ridge(**defaults)
    elif model_name == "early_fusion_mlp" and normalized_task == "T1":
        defaults = {
            "hidden_layer_sizes": (128, 64),
            "activation": "relu",
            "alpha": 1e-4,
            "batch_size": 128,
            "learning_rate_init": 1e-3,
            "max_iter": 200,
            "early_stopping": True,
            "validation_fraction": 0.1,
            "n_iter_no_change": 12,
            "random_state": seed,
        }
        defaults.update(params)
        model = MLPClassifier(**defaults)
    elif model_name == "early_fusion_mlp" and normalized_task == "T2":
        defaults = {
            "hidden_layer_sizes": (128, 64),
            "activation": "relu",
            "alpha": 1e-4,
            "batch_size": 128,
            "learning_rate_init": 1e-3,
            "max_iter": 200,
            "early_stopping": True,
            "validation_fraction": 0.1,
            "n_iter_no_change": 12,
            "random_state": seed,
        }
        defaults.update(params)
        model = MLPRegressor(**defaults)
    elif model_name == "lightgbm" and normalized_task in {"T1", "T2"}:
        try:
            from lightgbm import LGBMClassifier, LGBMRegressor
        except ImportError as exc:  # pragma: no cover - depends on optional runtime package
            raise RuntimeError("lightgbm model requested but lightgbm is not installed") from exc
        defaults = {
            "n_estimators": 50,
            "learning_rate": 0.03,
            "num_leaves": 31,
            "subsample": 0.85,
            "colsample_bytree": 0.85,
            "random_state": seed,
            "n_jobs": -1,
            "verbose": -1,
        }
        defaults.update(params)
        model = LGBMClassifier(**defaults) if normalized_task == "T1" else LGBMRegressor(**defaults)
    else:
        raise ValueError(f"unsupported task/model combination: {normalized_task}/{model_name}")

    if model_name == "lightgbm":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median").set_output(transform="pandas")),
                ("model", model),
            ]
        )

    return Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
            ("model", model),
        ]
    )


def predict_t1_score(estimator: Pipeline, features: Any) -> np.ndarray:
    probabilities = estimator.predict_proba(features)
    classes = np.asarray(estimator.named_steps["model"].classes_)
    positive = np.flatnonzero(classes == 1)
    if positive.size == 0:
        return np.zeros(len(features), dtype=float)
    return np.asarray(probabilities[:, int(positive[0])], dtype=float)
