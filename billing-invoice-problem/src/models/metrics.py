"""Evaluation metrics for the two-part dollar-churn model (Steps 5-6).

Business-aligned, not just accuracy/AUC. Dollar-weighted metrics answer the
question the credit team actually cares about: of the revenue that churned, how
much did we catch by acting on the highest-risk customers?
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    mean_absolute_error,
    roc_auc_score,
)


def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """ROC-AUC (rank quality). Single-class input -> nan."""
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def pr_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:
    """Precision-Recall AUC (average precision). Better than ROC under imbalance."""
    y_true = np.asarray(y_true)
    if len(np.unique(y_true)) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def brier(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Brier score (calibration + refinement); lower is better."""
    return float(brier_score_loss(np.asarray(y_true), np.asarray(y_prob)))


def mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Mean absolute error (dollar severity)."""
    return float(mean_absolute_error(np.asarray(y_true), np.asarray(y_pred)))


def dollar_recall_at_k(
    dollars_actual: np.ndarray, score: np.ndarray, k_frac: float
) -> float:
    """Share of actual churned dollars captured in the top ``k_frac`` by score.

    Rank customers by predicted risk/expected-loss, take the top ``k_frac``,
    and report the fraction of total actual dollar churn they account for.
    """
    dollars_actual = np.asarray(dollars_actual, dtype=float)
    score = np.asarray(score, dtype=float)
    total = dollars_actual.sum()
    if total <= 0 or len(score) == 0:
        return float("nan")
    n = len(score)
    k = max(1, int(np.ceil(k_frac * n)))
    top_idx = np.argsort(-score)[:k]
    return float(dollars_actual[top_idx].sum() / total)


def pct_dollars_captured(
    dollars_actual: np.ndarray, score: np.ndarray, k_frac: float
) -> float:
    """Alias of :func:`dollar_recall_at_k`, expressed as a percentage."""
    return dollar_recall_at_k(dollars_actual, score, k_frac) * 100.0


def calibration_bins(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10
) -> dict:
    """Reliability-diagram data: mean predicted vs observed per probability bin."""
    y_true = np.asarray(y_true, dtype=float)
    y_prob = np.asarray(y_prob, dtype=float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(y_prob, edges[1:-1]), 0, n_bins - 1)
    mean_pred, obs_rate, counts = [], [], []
    for b in range(n_bins):
        m = idx == b
        counts.append(int(m.sum()))
        if m.any():
            mean_pred.append(float(y_prob[m].mean()))
            obs_rate.append(float(y_true[m].mean()))
        else:
            mean_pred.append(float("nan"))
            obs_rate.append(float("nan"))
    return {"mean_predicted": mean_pred, "observed_rate": obs_rate, "count": counts}
