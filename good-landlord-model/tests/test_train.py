"""
Tests for src/models/train.py

Uses a synthetic feature matrix with 30 landlords so GroupKFold(5) is valid
for all model types.  All models run with fast/tiny hyperparameters to keep
the test suite quick.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.train import (
    CVResult,
    FoldResult,
    ALL_MODEL_TYPES,
    train_cv,
    train_all,
    _resolve_model_cfg,
    _resolve_cv_folds,
    _encode_sklearn,
    _CatEncoder,
    _NumImputer,
)


# ---------------------------------------------------------------------------
# Synthetic data helpers
# ---------------------------------------------------------------------------

N_LANDLORDS = 30
RNG = np.random.default_rng(0)


def _make_matrix(n: int = N_LANDLORDS, seed: int = 0) -> pd.DataFrame:
    """
    Synthetic feature matrix: numeric + categorical features, AdjustedScore target.
    Matches the shape produced by build_feature_matrix.
    """
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "LandLordID":             [f"L{i:03d}" for i in range(n)],
        "LandlordAge":            rng.uniform(1, 50, n),
        "Area":                   rng.uniform(100, 5000, n),
        "TotalPopulationAround":  rng.uniform(1000, 100_000, n),
        "ActiveCompanies":        rng.integers(0, 20, n).astype(float),
        "PopulationDensity":      rng.uniform(1, 200, n),
        "AreaPerCompany":         rng.uniform(10, 500, n),
        "PopulationPerCompany":   rng.uniform(50, 2000, n),
        "PortfolioSize":          rng.integers(1, 10, n).astype(float),
        "PortfolioActiveRate":    rng.uniform(0, 1, n),
        "PortfolioMeanBudget":    rng.uniform(500, 5000, n),
        "PortfolioStdBudget":     rng.uniform(100, 1000, n),
        "PortfolioMeanSales":     rng.uniform(1000, 20000, n),
        "PortfolioMeanClients":   rng.uniform(10, 200, n),
        "PortfolioMeanRetention": rng.uniform(0, 1, n),
        "PreferredIndustry":      rng.choice(["Retail", "Tech", "Food", None], n),
        "OriginCity":             rng.choice(["Tel Aviv", "Haifa", "Jerusalem"], n),
        "OriginCountry":          ["IL"] * n,
        "AdjustedScore":          rng.uniform(-0.2, 0.2, n),
        "TenantCount":            rng.integers(1, 10, n).astype(float),
    })


def _make_matrix_with_nan(n: int = N_LANDLORDS) -> pd.DataFrame:
    """Matrix where some numeric values are NaN."""
    m = _make_matrix(n)
    idx = np.random.default_rng(1).choice(n, size=n // 5, replace=False)
    m.loc[idx, "LandlordAge"] = np.nan
    m.loc[idx, "PortfolioStdBudget"] = np.nan
    return m


NUMERIC_COLS = [
    "LandlordAge", "Area", "TotalPopulationAround", "ActiveCompanies",
    "PopulationDensity", "AreaPerCompany", "PopulationPerCompany",
    "PortfolioSize", "PortfolioActiveRate", "PortfolioMeanBudget",
    "PortfolioStdBudget", "PortfolioMeanSales", "PortfolioMeanClients",
    "PortfolioMeanRetention",
]
CAT_COLS = ["PreferredIndustry", "OriginCity", "OriginCountry"]


# Tiny cfg for fast tests
FAST_CFG = {
    "seed": 42,
    "landlord_model": {
        "cv_folds": 3,
        "catboost": {
            "iterations": 5,
            "depth": 2,
            "learning_rate": 0.1,
            "loss_function": "RMSE",
            "eval_metric": "MAE",
            "random_seed": 42,
            "verbose": False,
        },
        "xgboost": {
            "n_estimators": 5,
            "max_depth": 2,
            "learning_rate": 0.1,
            "random_state": 42,
        },
        "lightgbm": {
            "n_estimators": 5,
            "max_depth": 2,
            "learning_rate": 0.1,
            "random_state": 42,
            "verbose": -1,
        },
        "random_forest": {
            "n_estimators": 5,
            "max_depth": 3,
            "random_state": 42,
        },
    },
}


# ---------------------------------------------------------------------------
# Config helper tests
# ---------------------------------------------------------------------------

class TestConfigHelpers:

    def test_resolve_cv_folds_from_cfg(self):
        cfg = {"landlord_model": {"cv_folds": 7}}
        assert _resolve_cv_folds(cfg) == 7

    def test_resolve_cv_folds_default(self):
        assert _resolve_cv_folds(None) == 5

    def test_resolve_model_cfg_extracts_section(self):
        cfg = {"landlord_model": {"catboost": {"iterations": 200}}}
        assert _resolve_model_cfg(cfg, "catboost")["iterations"] == 200

    def test_resolve_model_cfg_missing_returns_empty(self):
        assert _resolve_model_cfg({}, "catboost") == {}

    def test_resolve_model_cfg_none_returns_empty(self):
        assert _resolve_model_cfg(None, "xgboost") == {}


# ---------------------------------------------------------------------------
# Encoding helpers
# ---------------------------------------------------------------------------

class TestEncoding:

    def test_num_imputer_fills_nan(self):
        df_tr = pd.DataFrame({"a": [1.0, np.nan, 3.0]})
        df_vl = pd.DataFrame({"a": [np.nan, 2.0, 4.0]})
        imp = _NumImputer()
        out_tr = imp.fit_transform(df_tr, ["a"])
        out_vl = imp.transform(df_vl)
        assert not np.isnan(out_tr).any()
        assert not np.isnan(out_vl).any()

    def test_num_imputer_empty_cols(self):
        df = pd.DataFrame({"a": [1.0, 2.0]})
        imp = _NumImputer()
        out = imp.fit_transform(df, [])
        assert out.shape == (2, 0)

    def test_cat_encoder_handles_none(self):
        df_tr = pd.DataFrame({"c": ["A", None, "B"]})
        df_vl = pd.DataFrame({"c": [None, "A", "C"]})
        enc = _CatEncoder()
        out_tr = enc.fit_transform(df_tr, ["c"])
        out_vl = enc.transform(df_vl)
        assert not np.isnan(out_tr).any()
        # Unknown "C" → -1 (no NaN)
        assert not np.isnan(out_vl).any()

    def test_cat_encoder_empty_cols(self):
        df = pd.DataFrame({"a": [1, 2]})
        enc = _CatEncoder()
        out = enc.fit_transform(df, [])
        assert out.shape == (2, 0)

    def test_encode_sklearn_shapes(self):
        m = _make_matrix(10)
        X_tr, X_vl, names = _encode_sklearn(
            m.iloc[:7], m.iloc[7:], NUMERIC_COLS, CAT_COLS
        )
        assert X_tr.shape == (7, len(NUMERIC_COLS) + len(CAT_COLS))
        assert X_vl.shape == (3, len(NUMERIC_COLS) + len(CAT_COLS))
        assert len(names) == len(NUMERIC_COLS) + len(CAT_COLS)


# ---------------------------------------------------------------------------
# train_cv — shared checks across model types
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model_type", ALL_MODEL_TYPES)
class TestTrainCV:

    def test_returns_cvresult(self, model_type):
        m = _make_matrix()
        result = train_cv(m, NUMERIC_COLS, CAT_COLS, model_type=model_type, cfg=FAST_CFG, n_splits=3)
        assert isinstance(result, CVResult)

    def test_model_type_stored(self, model_type):
        m = _make_matrix()
        result = train_cv(m, NUMERIC_COLS, CAT_COLS, model_type=model_type, cfg=FAST_CFG, n_splits=3)
        assert result.model_type == model_type

    def test_oof_length_equals_n_landlords(self, model_type):
        m = _make_matrix()
        result = train_cv(m, NUMERIC_COLS, CAT_COLS, model_type=model_type, cfg=FAST_CFG, n_splits=3)
        assert len(result.oof_predictions) == len(m)
        assert len(result.oof_true) == len(m)

    def test_no_nan_in_oof_predictions(self, model_type):
        m = _make_matrix()
        result = train_cv(m, NUMERIC_COLS, CAT_COLS, model_type=model_type, cfg=FAST_CFG, n_splits=3)
        assert not np.isnan(result.oof_predictions).any()

    def test_fold_results_count(self, model_type):
        m = _make_matrix()
        result = train_cv(m, NUMERIC_COLS, CAT_COLS, model_type=model_type, cfg=FAST_CFG, n_splits=3)
        assert len(result.fold_results) == 3

    def test_fold_results_type(self, model_type):
        m = _make_matrix()
        result = train_cv(m, NUMERIC_COLS, CAT_COLS, model_type=model_type, cfg=FAST_CFG, n_splits=3)
        for fr in result.fold_results:
            assert isinstance(fr, FoldResult)

    def test_fold_mae_non_negative(self, model_type):
        m = _make_matrix()
        result = train_cv(m, NUMERIC_COLS, CAT_COLS, model_type=model_type, cfg=FAST_CFG, n_splits=3)
        for fr in result.fold_results:
            assert fr.mae >= 0.0

    def test_fold_rmse_non_negative(self, model_type):
        m = _make_matrix()
        result = train_cv(m, NUMERIC_COLS, CAT_COLS, model_type=model_type, cfg=FAST_CFG, n_splits=3)
        for fr in result.fold_results:
            assert fr.rmse >= 0.0

    def test_landlord_ids_stored(self, model_type):
        m = _make_matrix()
        result = train_cv(m, NUMERIC_COLS, CAT_COLS, model_type=model_type, cfg=FAST_CFG, n_splits=3)
        assert len(result.landlord_ids) == len(m)

    def test_final_model_not_none(self, model_type):
        m = _make_matrix()
        result = train_cv(m, NUMERIC_COLS, CAT_COLS, model_type=model_type, cfg=FAST_CFG, n_splits=3)
        assert result.final_model is not None

    def test_feature_importance_shape(self, model_type):
        m = _make_matrix()
        result = train_cv(m, NUMERIC_COLS, CAT_COLS, model_type=model_type, cfg=FAST_CFG, n_splits=3)
        n_feats = len(NUMERIC_COLS) + len(CAT_COLS)
        assert len(result.feature_importance) == n_feats
        assert "feature" in result.feature_importance.columns
        assert "importance" in result.feature_importance.columns

    def test_mean_metrics_keys(self, model_type):
        m = _make_matrix()
        result = train_cv(m, NUMERIC_COLS, CAT_COLS, model_type=model_type, cfg=FAST_CFG, n_splits=3)
        for key in ("mae", "rmse", "r2", "spearman"):
            assert key in result.mean_metrics

    def test_std_metrics_keys(self, model_type):
        m = _make_matrix()
        result = train_cv(m, NUMERIC_COLS, CAT_COLS, model_type=model_type, cfg=FAST_CFG, n_splits=3)
        for key in ("mae", "rmse", "r2", "spearman"):
            assert key in result.std_metrics

    def test_nan_features_handled(self, model_type):
        """NaN in numeric features should not crash any model."""
        m = _make_matrix_with_nan()
        result = train_cv(m, NUMERIC_COLS, CAT_COLS, model_type=model_type, cfg=FAST_CFG, n_splits=3)
        assert not np.isnan(result.oof_predictions).any()


class TestTrainCVEdgeCases:

    def test_raises_invalid_model_type(self):
        m = _make_matrix()
        with pytest.raises(ValueError, match="model_type"):
            train_cv(m, NUMERIC_COLS, CAT_COLS, model_type="bad_model", cfg=FAST_CFG)  # type: ignore

    def test_raises_missing_target(self):
        m = _make_matrix().drop(columns=["AdjustedScore"])
        with pytest.raises(ValueError, match="AdjustedScore"):
            train_cv(m, NUMERIC_COLS, CAT_COLS, model_type="random_forest", cfg=FAST_CFG)

    def test_raises_missing_group_col(self):
        m = _make_matrix().drop(columns=["LandLordID"])
        with pytest.raises(ValueError, match="LandLordID"):
            train_cv(m, NUMERIC_COLS, CAT_COLS, model_type="random_forest", cfg=FAST_CFG)

    def test_n_splits_clamped_when_too_large(self):
        """With 5 landlords and n_splits=10, should clamp without crashing."""
        m = _make_matrix(n=5)
        result = train_cv(m, NUMERIC_COLS, CAT_COLS,
                          model_type="random_forest", cfg=FAST_CFG, n_splits=10)
        assert len(result.fold_results) <= 5

    def test_no_categorical_cols(self):
        """Works with an empty categorical list (numeric-only matrix)."""
        m = _make_matrix()
        result = train_cv(m, NUMERIC_COLS, [], model_type="random_forest",
                          cfg=FAST_CFG, n_splits=3)
        assert len(result.oof_predictions) == len(m)

    def test_no_numeric_cols(self):
        """Works with only categorical features."""
        m = _make_matrix()
        result = train_cv(m, [], CAT_COLS, model_type="random_forest",
                          cfg=FAST_CFG, n_splits=3)
        assert len(result.oof_predictions) == len(m)


# ---------------------------------------------------------------------------
# train_all
# ---------------------------------------------------------------------------

class TestTrainAll:

    def test_returns_dict_with_all_models(self):
        m = _make_matrix()
        results = train_all(m, NUMERIC_COLS, CAT_COLS, cfg=FAST_CFG, n_splits=3)
        assert set(results.keys()) == set(ALL_MODEL_TYPES)

    def test_all_values_are_cvresult(self):
        m = _make_matrix()
        results = train_all(m, NUMERIC_COLS, CAT_COLS, cfg=FAST_CFG, n_splits=3)
        for v in results.values():
            assert isinstance(v, CVResult)

    def test_subset_of_models(self):
        m = _make_matrix()
        results = train_all(m, NUMERIC_COLS, CAT_COLS,
                            cfg=FAST_CFG, n_splits=3,
                            model_types=["random_forest", "xgboost"])
        assert set(results.keys()) == {"random_forest", "xgboost"}

    def test_oof_length_consistent_across_models(self):
        m = _make_matrix()
        results = train_all(m, NUMERIC_COLS, CAT_COLS, cfg=FAST_CFG, n_splits=3)
        for result in results.values():
            assert len(result.oof_predictions) == len(m)
