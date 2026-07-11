"""Tests for PSI drift monitoring (Step 8)."""

from __future__ import annotations

import numpy as np

from src.serving.monitoring import drift_label, population_stability_report, psi


def test_psi_zero_for_identical_distributions() -> None:
    rng = np.random.default_rng(0)
    x = rng.normal(size=5000)
    assert psi(x, x.copy()) < 0.01
    assert drift_label(psi(x, x.copy())) == "stable"


def test_psi_detects_large_shift() -> None:
    rng = np.random.default_rng(1)
    ref = rng.normal(0, 1, 5000)
    shifted = rng.normal(3, 1, 5000)  # mean shifted by 3 sigma
    val = psi(ref, shifted)
    assert val > 0.25
    assert drift_label(val) == "significant"


def test_population_report_flags_and_recommends() -> None:
    rng = np.random.default_rng(2)
    ref = {"a": rng.normal(0, 1, 2000), "b": rng.normal(0, 1, 2000)}
    cur = {"a": rng.normal(0, 1, 2000), "b": rng.normal(4, 1, 2000)}
    rep = population_stability_report(ref, cur)
    assert rep["features"]["b"]["drift"] == "significant"
    assert rep["action"] == "retrain"
