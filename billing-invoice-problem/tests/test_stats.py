"""Tests for significance testing (Step 6)."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import roc_auc_score

from src.models.stats import delong_roc_test, paired_bootstrap_delta


def test_delong_aucs_match_sklearn() -> None:
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    p_base = rng.random(500)
    p_treat = rng.random(500)
    res = delong_roc_test(y, p_base, p_treat)
    assert abs(res["auc_base"] - roc_auc_score(y, p_base)) < 1e-3
    assert abs(res["auc_treatment"] - roc_auc_score(y, p_treat)) < 1e-3


def test_delong_detects_clear_improvement() -> None:
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, 800)
    p_base = rng.random(800)                       # random -> AUC ~0.5
    p_treat = np.clip(y + rng.normal(0, 0.3, 800), 0, 1)  # informative
    res = delong_roc_test(y, p_base, p_treat)
    assert res["delta"] > 0
    assert res["significant"]


def test_paired_bootstrap_no_difference_is_insignificant() -> None:
    # Identical base and treatment -> delta 0, not significant.
    y = np.array([0, 1] * 100)
    score = np.random.default_rng(3).random(200)

    def compute(idx):
        from src.models.metrics import pr_auc
        return pr_auc(y[idx], score[idx]), pr_auc(y[idx], score[idx])

    res = paired_bootstrap_delta(compute, len(y), n_boot=100, seed=0)
    assert res["delta"] == 0.0
    assert not res["significant"]


def test_paired_bootstrap_detects_improvement() -> None:
    rng = np.random.default_rng(5)
    y = rng.integers(0, 2, 600)
    base = rng.random(600)
    treat = np.clip(y + rng.normal(0, 0.2, 600), 0, 1)

    def compute(idx):
        from src.models.metrics import pr_auc
        return pr_auc(y[idx], base[idx]), pr_auc(y[idx], treat[idx])

    res = paired_bootstrap_delta(compute, len(y), n_boot=200, seed=0)
    assert res["delta"] > 0
    assert res["significant"]
