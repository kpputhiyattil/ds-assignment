"""
Tests for src/models/evaluate.py

Pure unit tests — no real data, no model training.
CVResult stubs are constructed directly from numpy arrays.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.evaluate import (
    regression_metrics,
    lift_table,
    quality_band_metrics,
    compare_models,
    oof_residual_summary,
)
from src.models.train import CVResult, FoldResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fold(fold: int, mae: float = 0.1, rmse: float = 0.15,
               r2: float = 0.5, spearman: float = 0.7, n_val: int = 5) -> FoldResult:
    return FoldResult(fold=fold, mae=mae, rmse=rmse, r2=r2, spearman=spearman, n_val=n_val)


def _make_cvresult(
    model_type: str = "catboost",
    n: int = 20,
    seed: int = 0,
) -> CVResult:
    rng = np.random.default_rng(seed)
    y_true = rng.uniform(-0.2, 0.2, n)
    y_pred = y_true + rng.normal(0, 0.05, n)   # near-perfect predictions
    return CVResult(
        model_type=model_type,
        oof_predictions=y_pred,
        oof_true=y_true,
        landlord_ids=np.array([f"L{i}" for i in range(n)]),
        fold_results=[
            _make_fold(0, mae=0.05, rmse=0.07, r2=0.9, spearman=0.95),
            _make_fold(1, mae=0.06, rmse=0.08, r2=0.88, spearman=0.93),
            _make_fold(2, mae=0.04, rmse=0.06, r2=0.92, spearman=0.97),
        ],
        final_model=None,
        feature_importance=pd.DataFrame({"feature": ["a", "b"], "importance": [0.6, 0.4]}),
    )


# ---------------------------------------------------------------------------
# regression_metrics
# ---------------------------------------------------------------------------

class TestRegressionMetrics:

    def test_returns_dict(self):
        m = regression_metrics(np.array([1, 2, 3]), np.array([1, 2, 3]))
        assert isinstance(m, dict)

    def test_required_keys_present(self):
        m = regression_metrics(np.array([1, 2, 3]), np.array([1.1, 2.1, 3.1]))
        for key in ("mae", "rmse", "r2", "spearman", "n"):
            assert key in m

    def test_perfect_predictions_zero_error(self):
        y = np.linspace(0, 1, 10)
        m = regression_metrics(y, y)
        assert m["mae"]  == pytest.approx(0.0, abs=1e-10)
        assert m["rmse"] == pytest.approx(0.0, abs=1e-10)

    def test_perfect_predictions_r2_one(self):
        y = np.linspace(0, 1, 10)
        m = regression_metrics(y, y)
        assert m["r2"] == pytest.approx(1.0, abs=1e-8)

    def test_perfect_ranking_spearman_one(self):
        y = np.arange(10, dtype=float)
        m = regression_metrics(y, y + 5.0)  # same rank, different scale
        assert m["spearman"] == pytest.approx(1.0, abs=1e-8)

    def test_reversed_ranking_spearman_minus_one(self):
        y = np.arange(10, dtype=float)
        m = regression_metrics(y, -y)
        assert m["spearman"] == pytest.approx(-1.0, abs=1e-8)

    def test_mae_non_negative(self):
        rng = np.random.default_rng(42)
        y_true = rng.uniform(0, 1, 50)
        y_pred = rng.uniform(0, 1, 50)
        m = regression_metrics(y_true, y_pred)
        assert m["mae"] >= 0.0

    def test_rmse_non_negative(self):
        rng = np.random.default_rng(42)
        y_true = rng.uniform(0, 1, 50)
        y_pred = rng.uniform(0, 1, 50)
        m = regression_metrics(y_true, y_pred)
        assert m["rmse"] >= 0.0

    def test_n_correct(self):
        m = regression_metrics(np.array([1, 2, 3]), np.array([1, 2, 3]))
        assert m["n"] == 3

    def test_prefix_applied_to_keys(self):
        m = regression_metrics(np.array([1, 2, 3]), np.array([1, 2, 3]), prefix="oof_")
        assert "oof_mae" in m
        assert "mae" not in m

    def test_nan_pairs_excluded(self):
        y_true = np.array([1.0, np.nan, 3.0, 4.0])
        y_pred = np.array([1.0, 2.0,   np.nan, 4.0])
        m = regression_metrics(y_true, y_pred)
        assert m["n"] == 2  # only rows where both are non-NaN

    def test_all_nan_returns_nan_metrics(self):
        m = regression_metrics(np.array([np.nan]), np.array([np.nan]))
        assert np.isnan(m["mae"])
        assert m["n"] == 0

    def test_single_pair_no_crash(self):
        m = regression_metrics(np.array([0.5]), np.array([0.6]))
        assert np.isfinite(m["mae"])

    def test_accepts_series(self):
        s_true = pd.Series([1.0, 2.0, 3.0])
        s_pred = pd.Series([1.1, 2.1, 3.1])
        m = regression_metrics(s_true, s_pred)
        assert m["n"] == 3


# ---------------------------------------------------------------------------
# lift_table
# ---------------------------------------------------------------------------

class TestLiftTable:

    def test_returns_dataframe(self):
        y = np.linspace(-0.2, 0.2, 20)
        lt = lift_table(y, y + 0.01)
        assert isinstance(lt, pd.DataFrame)

    def test_required_columns(self):
        y = np.linspace(-0.2, 0.2, 20)
        lt = lift_table(y, y + 0.01)
        for col in ("bin", "n", "mean_pred", "mean_actual", "lift"):
            assert col in lt.columns

    def test_n_bins_rows(self):
        y = np.linspace(-0.2, 0.2, 30)
        lt = lift_table(y, y, n_bins=5)
        assert len(lt) == 5

    def test_default_10_bins(self):
        y = np.linspace(0, 1, 30)
        lt = lift_table(y, y)
        assert len(lt) == 10

    def test_total_n_equals_input_size(self):
        n = 27
        y = np.linspace(0, 1, n)
        lt = lift_table(y, y, n_bins=5)
        assert lt["n"].sum() == n

    def test_bin_1_has_highest_mean_pred(self):
        """When prediction is perfect, bin 1 (top predicted) has highest actual."""
        y = np.linspace(0, 1, 40)
        lt = lift_table(y, y, n_bins=5)
        assert lt.loc[lt["bin"] == 1, "mean_pred"].iloc[0] >= lt.loc[lt["bin"] == 5, "mean_pred"].iloc[0]

    def test_nan_pairs_excluded(self):
        y_true = np.array([np.nan, 1.0, 2.0, 3.0])
        y_pred = np.array([1.0, np.nan, 2.0, 3.0])
        lt = lift_table(y_true, y_pred, n_bins=2)
        # Only 2 valid pairs remain
        assert lt["n"].sum() == 2

    def test_single_sample_no_crash(self):
        lt = lift_table(np.array([0.5]), np.array([0.4]), n_bins=5)
        assert len(lt) >= 1

    def test_empty_returns_empty_df(self):
        lt = lift_table(np.array([np.nan]), np.array([np.nan]))
        assert len(lt) == 0


# ---------------------------------------------------------------------------
# quality_band_metrics
# ---------------------------------------------------------------------------

class TestQualityBandMetrics:

    def test_returns_dict(self):
        y = np.linspace(0, 1, 20)
        qm = quality_band_metrics(y, y)
        assert isinstance(qm, dict)

    def test_required_keys(self):
        y = np.linspace(0, 1, 20)
        qm = quality_band_metrics(y, y)
        for key in ("top_capture_rate", "bottom_capture_rate",
                    "n_true_top", "n_true_bottom", "n_pred_top", "n_pred_bottom"):
            assert key in qm

    def test_perfect_predictions_top_capture_one(self):
        y = np.linspace(0, 1, 20)
        qm = quality_band_metrics(y, y)
        assert qm["top_capture_rate"] == pytest.approx(1.0, abs=1e-6)

    def test_perfect_predictions_bottom_capture_one(self):
        y = np.linspace(0, 1, 20)
        qm = quality_band_metrics(y, y)
        assert qm["bottom_capture_rate"] == pytest.approx(1.0, abs=1e-6)

    def test_reversed_predictions_low_capture(self):
        """Reversed predictions should have 0 capture at extremes for large n."""
        y = np.linspace(0, 1, 20)
        qm = quality_band_metrics(y, -y)
        assert qm["top_capture_rate"] == pytest.approx(0.0, abs=1e-6)
        assert qm["bottom_capture_rate"] == pytest.approx(0.0, abs=1e-6)

    def test_capture_rates_between_0_and_1(self):
        rng = np.random.default_rng(7)
        y = rng.uniform(0, 1, 50)
        qm = quality_band_metrics(y, rng.uniform(0, 1, 50))
        assert 0.0 <= qm["top_capture_rate"] <= 1.0
        assert 0.0 <= qm["bottom_capture_rate"] <= 1.0

    def test_all_nan_returns_nan(self):
        qm = quality_band_metrics(np.array([np.nan]), np.array([np.nan]))
        assert np.isnan(qm["top_capture_rate"])

    def test_n_true_top_positive(self):
        y = np.linspace(0, 1, 20)
        qm = quality_band_metrics(y, y)
        assert qm["n_true_top"] > 0
        assert qm["n_true_bottom"] > 0


# ---------------------------------------------------------------------------
# compare_models
# ---------------------------------------------------------------------------

class TestCompareModels:

    def test_returns_dataframe(self):
        results = {
            "catboost":     _make_cvresult("catboost"),
            "random_forest": _make_cvresult("random_forest"),
        }
        df = compare_models(results)
        assert isinstance(df, pd.DataFrame)

    def test_one_row_per_model(self):
        results = {
            "catboost":     _make_cvresult("catboost"),
            "xgboost":      _make_cvresult("xgboost"),
            "lightgbm":     _make_cvresult("lightgbm"),
            "random_forest": _make_cvresult("random_forest"),
        }
        df = compare_models(results)
        assert len(df) == 4

    def test_model_type_column_present(self):
        results = {"rf": _make_cvresult("rf")}
        df = compare_models(results)
        assert "model_type" in df.columns
        assert df["model_type"].iloc[0] == "rf"

    def test_metric_columns_present(self):
        results = {"catboost": _make_cvresult("catboost")}
        df = compare_models(results)
        for col in ("mae_mean", "mae_std", "rmse_mean", "r2_mean", "spearman_mean"):
            assert col in df.columns

    def test_sorted_by_mae_ascending(self):
        r1 = _make_cvresult("low_mae")
        r2 = _make_cvresult("high_mae")
        # Override fold results so r1 has lower MAE
        r1.fold_results[0] = _make_fold(0, mae=0.01)
        r1.fold_results[1] = _make_fold(1, mae=0.01)
        r1.fold_results[2] = _make_fold(2, mae=0.01)
        r2.fold_results[0] = _make_fold(0, mae=0.5)
        r2.fold_results[1] = _make_fold(1, mae=0.5)
        r2.fold_results[2] = _make_fold(2, mae=0.5)
        df = compare_models({"low_mae": r1, "high_mae": r2})
        assert df["model_type"].iloc[0] == "low_mae"

    def test_n_folds_column(self):
        results = {"catboost": _make_cvresult("catboost")}
        df = compare_models(results)
        assert df["n_folds"].iloc[0] == 3

    def test_n_landlords_column(self):
        results = {"catboost": _make_cvresult("catboost", n=20)}
        df = compare_models(results)
        assert df["n_landlords"].iloc[0] == 20

    def test_empty_results_returns_empty_df(self):
        df = compare_models({})
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 0


# ---------------------------------------------------------------------------
# oof_residual_summary
# ---------------------------------------------------------------------------

class TestOofResidualSummary:

    def test_returns_dataframe(self):
        result = _make_cvresult()
        df = oof_residual_summary(result)
        assert isinstance(df, pd.DataFrame)

    def test_columns_present(self):
        result = _make_cvresult()
        df = oof_residual_summary(result)
        for col in ("landlord_id", "true_score", "oof_pred", "residual", "abs_error"):
            assert col in df.columns

    def test_row_count_equals_n_landlords(self):
        result = _make_cvresult(n=20)
        df = oof_residual_summary(result)
        assert len(df) == 20

    def test_sorted_by_abs_error_descending(self):
        result = _make_cvresult()
        df = oof_residual_summary(result)
        assert df["abs_error"].is_monotonic_decreasing

    def test_residual_equals_pred_minus_true(self):
        result = _make_cvresult()
        df = oof_residual_summary(result)
        np.testing.assert_allclose(
            df["residual"].values,
            (df["oof_pred"] - df["true_score"]).values,
            atol=1e-10,
        )

    def test_abs_error_non_negative(self):
        result = _make_cvresult()
        df = oof_residual_summary(result)
        assert (df["abs_error"] >= 0).all()
