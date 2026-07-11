"""Tests for evaluation metrics (Step 5)."""

from __future__ import annotations

import numpy as np

from src.models import metrics as M


def test_roc_auc_perfect_and_degenerate() -> None:
    y = np.array([0, 0, 1, 1])
    assert M.roc_auc(y, np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert np.isnan(M.roc_auc(np.array([1, 1, 1]), np.array([0.1, 0.2, 0.3])))


def test_brier_bounds() -> None:
    y = np.array([0, 1])
    assert M.brier(y, np.array([0.0, 1.0])) == 0.0
    assert M.brier(y, np.array([1.0, 0.0])) == 1.0


def test_dollar_recall_at_k_ranks_by_score() -> None:
    # Big-dollar churner has the highest score -> top 20% captures most dollars.
    dollars = np.array([1000.0, 10.0, 5.0, 2.0, 1.0])
    score = np.array([0.9, 0.1, 0.1, 0.1, 0.1])
    # top 20% of 5 = 1 customer (the 1000). total = 1018.
    assert abs(M.dollar_recall_at_k(dollars, score, 0.2) - 1000.0 / 1018.0) < 1e-9


def test_dollar_recall_handles_zero_total() -> None:
    assert np.isnan(M.dollar_recall_at_k(np.zeros(3), np.array([1.0, 2, 3]), 0.5))


def test_pct_dollars_captured_is_percentage() -> None:
    dollars = np.array([100.0, 0.0])
    score = np.array([1.0, 0.0])
    assert M.pct_dollars_captured(dollars, score, 0.5) == 100.0


def test_calibration_bins_shapes() -> None:
    y = np.array([0, 0, 1, 1, 1])
    p = np.array([0.1, 0.2, 0.6, 0.8, 0.9])
    out = M.calibration_bins(y, p, n_bins=5)
    assert len(out["mean_predicted"]) == 5
    assert sum(out["count"]) == 5
