"""Official metrics used by the three benchmark tasks."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from .schema import GROUP_COLUMNS


def safe_spearman(y_true: pd.Series, y_pred: pd.Series) -> float:
    valid = y_true.notna() & y_pred.notna()
    truth = y_true[valid].astype(float)
    score = y_pred[valid].astype(float)
    if len(truth) < 2 or truth.nunique() < 2 or score.nunique() < 2:
        return math.nan
    return float(truth.corr(score, method="spearman"))


def classification_metrics(y_true: pd.Series, y_score: pd.Series) -> dict[str, float]:
    valid = y_true.notna() & y_score.notna()
    truth = y_true[valid].astype(int)
    score = y_score[valid].astype(float)
    if truth.empty or truth.nunique() < 2:
        return {"auprc": math.nan, "auroc": math.nan}
    return {
        "auprc": float(average_precision_score(truth, score)),
        "auroc": float(roc_auc_score(truth, score)),
    }


def regression_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    valid = y_true.notna() & y_pred.notna()
    truth = y_true[valid].astype(float)
    pred = y_pred[valid].astype(float)
    return {
        "mae": float((pred - truth).abs().mean()) if not truth.empty else math.nan,
        "spearman": safe_spearman(truth, pred),
    }


def ndcg_at_10pct(y_true: pd.Series, y_score: pd.Series) -> float:
    valid = y_true.notna() & y_score.notna()
    truth = y_true[valid].astype(float)
    score = y_score[valid].astype(float)
    if truth.empty:
        return math.nan
    k = max(1, int(math.ceil(len(truth) * 0.10)))
    relevance = truth.rank(method="average", pct=True).to_numpy(dtype=float)
    scores = score.to_numpy(dtype=float)
    predicted_order = np.argsort(-scores)[:k]
    ideal_order = np.argsort(-relevance)[:k]

    def dcg(indices: np.ndarray) -> float:
        gains = relevance[indices]
        discounts = np.log2(np.arange(2, len(indices) + 2))
        return float((gains / discounts).sum())

    ideal = dcg(ideal_order)
    return dcg(predicted_order) / ideal if ideal > 0 else math.nan


def ranking_metrics_by_event_market(
    frame: pd.DataFrame,
    target_column: str = "ranking_target",
    prediction_column: str = "t3_score",
) -> tuple[dict[str, float], int]:
    grouped_metrics: list[dict[str, float]] = []
    for _, group in frame.groupby(GROUP_COLUMNS, sort=False, dropna=False):
        if group.empty:
            continue
        grouped_metrics.append(
            {
                "ndcg_at_10pct": ndcg_at_10pct(group[target_column], group[prediction_column]),
                "spearman": safe_spearman(group[target_column], group[prediction_column]),
            }
        )
    if not grouped_metrics:
        return {"ndcg_at_10pct": math.nan, "spearman": math.nan}, 0

    output: dict[str, float] = {}
    for metric in ["ndcg_at_10pct", "spearman"]:
        values = np.asarray([item[metric] for item in grouped_metrics], dtype=float)
        output[metric] = math.nan if np.isnan(values).all() else float(np.nanmean(values))
    return output, len(grouped_metrics)

