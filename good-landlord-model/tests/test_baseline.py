"""
Tests for src/models/company_baseline.py

Uses only in-memory DataFrames — no real data files required.
sklearn and numpy are the only non-stdlib dependencies.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.company_baseline import (
    compute_oof_predictions,
    aggregate_landlord_scores,
    build_landlord_adjusted_scores,
    _auto_feature_cols,
    _resolve_cfg,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def rng():
    return np.random.default_rng(42)


@pytest.fixture
def model_base(rng):
    """50-row model_base: 5 landlords × 10 companies each, balanced classes."""
    n = 50
    y = np.tile([0, 1], n // 2)[:n].astype(float)
    return pd.DataFrame({
        "LandLordID": [f"L{i % 5 + 1}" for i in range(n)],
        "CompanyID": [f"C{i:03d}" for i in range(n)],
        "CompanyIsActive": y,
        "MonthlyBudget": rng.uniform(500, 5000, n),
        "SalesOfMainProduct": rng.uniform(1000, 20000, n),
        "TotalActiveClients": rng.integers(10, 200, n).astype(float),
        "ReturningClient": rng.integers(5, 100, n).astype(float),
    })


@pytest.fixture
def bridge(model_base):
    return model_base[["LandLordID", "CompanyID"]].copy()


@pytest.fixture
def model_base_with_oof(model_base):
    """Pre-computed OOF result to avoid re-training in every test."""
    return compute_oof_predictions(model_base)


# ---------------------------------------------------------------------------
# _resolve_cfg
# ---------------------------------------------------------------------------


class TestResolveCfg:

    def test_none_returns_empty_dict(self):
        assert _resolve_cfg(None) == {}

    def test_bare_dict_returned_as_is(self):
        d = {"seed": 7, "n_splits": 3}
        assert _resolve_cfg(d) == d

    def test_full_cfg_with_baseline_key(self):
        full = {"baseline": {"seed": 99, "n_splits": 3}, "other": "x"}
        assert _resolve_cfg(full) == {"seed": 99, "n_splits": 3}

    def test_full_cfg_with_model_key(self):
        full = {"model": {"seed": 7}, "paths": {}}
        assert _resolve_cfg(full) == {"seed": 7}


# ---------------------------------------------------------------------------
# _auto_feature_cols
# ---------------------------------------------------------------------------


class TestAutoFeatureCols:

    def test_returns_numeric_cols(self):
        df = pd.DataFrame({"a": [1.0], "b": ["x"], "c": [2.0]})
        assert set(_auto_feature_cols(df)) == {"a", "c"}

    def test_excludes_target_cols(self):
        df = pd.DataFrame({
            "CompanyIsActive": [1.0],
            "SuccessScore": [0.5],
            "MonthlyBudget": [1000.0],
        })
        cols = _auto_feature_cols(df)
        assert "CompanyIsActive" not in cols
        assert "SuccessScore" not in cols
        assert "MonthlyBudget" in cols

    def test_empty_df_returns_empty(self):
        df = pd.DataFrame({"CompanyID": ["C1"]})
        assert _auto_feature_cols(df) == []


# ---------------------------------------------------------------------------
# compute_oof_predictions
# ---------------------------------------------------------------------------


class TestComputeOofPredictions:

    def test_output_has_oof_columns(self, model_base):
        out = compute_oof_predictions(model_base)
        for col in ("OOFPredictedProb", "OOFResidual", "OOFFold"):
            assert col in out.columns, f"Missing column: {col}"

    def test_output_same_length_as_input(self, model_base):
        out = compute_oof_predictions(model_base)
        assert len(out) == len(model_base)

    def test_predicted_prob_in_zero_one(self, model_base):
        out = compute_oof_predictions(model_base)
        probs = out["OOFPredictedProb"].dropna()
        assert (probs >= 0.0).all() and (probs <= 1.0).all()

    def test_residual_equals_actual_minus_predicted(self, model_base):
        out = compute_oof_predictions(model_base)
        valid = out["OOFResidual"].notna()
        expected = out.loc[valid, "CompanyIsActive"] - out.loc[valid, "OOFPredictedProb"]
        pd.testing.assert_series_equal(
            out.loc[valid, "OOFResidual"].reset_index(drop=True),
            expected.reset_index(drop=True),
            check_names=False,
        )

    def test_fold_assigned_to_all_labeled_rows(self, model_base):
        out = compute_oof_predictions(model_base)
        labeled = out["CompanyIsActive"].notna()
        assert (out.loc[labeled, "OOFFold"] >= 0).all()

    def test_unlabeled_rows_get_nan(self, model_base):
        mb = model_base.copy()
        mb.loc[mb.index[:3], "CompanyIsActive"] = np.nan
        out = compute_oof_predictions(mb)
        assert out.loc[mb.index[:3], "OOFPredictedProb"].isna().all()
        assert out.loc[mb.index[:3], "OOFResidual"].isna().all()
        assert (out.loc[mb.index[:3], "OOFFold"] == -1).all()

    def test_does_not_mutate_input(self, model_base):
        cols_before = list(model_base.columns)
        compute_oof_predictions(model_base)
        assert list(model_base.columns) == cols_before

    def test_return_model_flag_returns_tuple(self, model_base):
        result = compute_oof_predictions(model_base, return_model=True)
        assert isinstance(result, tuple) and len(result) == 2
        df, pipeline = result
        assert isinstance(df, pd.DataFrame)
        assert hasattr(pipeline, "predict_proba")

    def test_explicit_feature_cols(self, model_base):
        out = compute_oof_predictions(model_base, feature_cols=["MonthlyBudget"])
        assert out["OOFPredictedProb"].notna().any()

    def test_raises_missing_target_column(self, model_base):
        with pytest.raises(ValueError, match="CompanyIsActive"):
            compute_oof_predictions(model_base.drop(columns=["CompanyIsActive"]))

    def test_raises_if_no_features(self, model_base):
        # Keep only non-numeric columns (plus target)
        df = model_base[["LandLordID", "CompanyID", "CompanyIsActive"]].copy()
        with pytest.raises(ValueError, match="feature"):
            compute_oof_predictions(df)

    def test_raises_if_feature_col_missing(self, model_base):
        with pytest.raises(ValueError, match="NonExistentCol"):
            compute_oof_predictions(model_base, feature_cols=["NonExistentCol"])

    def test_raises_single_class_target(self):
        df = pd.DataFrame({
            "LandLordID": ["L1"] * 10,
            "CompanyID": [f"C{i}" for i in range(10)],
            "CompanyIsActive": [1.0] * 10,
            "MonthlyBudget": np.linspace(500, 5000, 10),
        })
        with pytest.raises(ValueError, match="one class"):
            compute_oof_predictions(df)

    def test_custom_n_splits(self, model_base):
        out = compute_oof_predictions(model_base, cfg={"n_splits": 3})
        folds_used = out["OOFFold"][out["OOFFold"] >= 0].unique()
        assert len(folds_used) == 3

    def test_custom_seed_reproducible(self, model_base):
        out1 = compute_oof_predictions(model_base, cfg={"seed": 7})
        out2 = compute_oof_predictions(model_base, cfg={"seed": 7})
        pd.testing.assert_series_equal(
            out1["OOFPredictedProb"], out2["OOFPredictedProb"]
        )

    def test_nan_features_handled_gracefully(self):
        """NaN feature values are median-imputed; no error expected."""
        rng = np.random.default_rng(0)
        n = 40
        y = np.tile([0, 1], n // 2).astype(float)
        budgets = rng.uniform(500, 5000, n)
        budgets[:5] = np.nan   # inject NaNs
        df = pd.DataFrame({
            "LandLordID": [f"L{i % 4 + 1}" for i in range(n)],
            "CompanyID": [f"C{i}" for i in range(n)],
            "CompanyIsActive": y,
            "MonthlyBudget": budgets,
        })
        out = compute_oof_predictions(df)
        assert out["OOFPredictedProb"].notna().sum() == n


# ---------------------------------------------------------------------------
# aggregate_landlord_scores
# ---------------------------------------------------------------------------


class TestAggregateLandlordScores:

    def test_returns_dataframe(self, model_base_with_oof, bridge):
        result = aggregate_landlord_scores(model_base_with_oof, bridge)
        assert isinstance(result, pd.DataFrame)

    def test_required_columns_present(self, model_base_with_oof, bridge):
        result = aggregate_landlord_scores(model_base_with_oof, bridge)
        for col in ("LandLordID", "TenantCount", "RawAdjustedScore", "AdjustedScore"):
            assert col in result.columns

    def test_tenant_count_positive(self, model_base_with_oof, bridge):
        result = aggregate_landlord_scores(model_base_with_oof, bridge)
        assert (result["TenantCount"] > 0).all()

    def test_landlord_count_matches_model_base(self, model_base_with_oof, bridge):
        n_landlords = model_base_with_oof["LandLordID"].nunique()
        result = aggregate_landlord_scores(model_base_with_oof, bridge)
        assert len(result) == n_landlords

    def test_shrinkage_pulls_toward_global_mean(self, model_base_with_oof, bridge):
        """
        AdjustedScore should be between global_mean and RawAdjustedScore
        when m > 0.
        """
        result = aggregate_landlord_scores(model_base_with_oof, bridge)
        global_mean = result["RawAdjustedScore"].mean()
        # Check that AdjustedScore is closer to global than raw is
        for _, row in result.iterrows():
            raw_dist = abs(row["RawAdjustedScore"] - global_mean)
            adj_dist = abs(row["AdjustedScore"] - global_mean)
            # Strictly: adjusted should not be further from global than raw
            assert adj_dist <= raw_dist + 1e-9, (
                f"Shrinkage went wrong for {row['LandLordID']}: "
                f"raw_dist={raw_dist:.4f}, adj_dist={adj_dist:.4f}"
            )

    def test_zero_shrinkage_equals_raw(self, model_base_with_oof, bridge):
        """With m=0, AdjustedScore must equal RawAdjustedScore exactly."""
        result = aggregate_landlord_scores(
            model_base_with_oof, bridge, cfg={"shrinkage_m": 0}
        )
        pd.testing.assert_series_equal(
            result["AdjustedScore"],
            result["RawAdjustedScore"],
            check_names=False,
            rtol=1e-9,
        )

    def test_large_m_shrinks_to_global(self, model_base_with_oof, bridge):
        """With very large m, all AdjustedScores approach the global mean."""
        result = aggregate_landlord_scores(
            model_base_with_oof, bridge, cfg={"shrinkage_m": 1e9}
        )
        global_mean = float(
            model_base_with_oof["OOFResidual"].dropna().mean()
        )
        assert np.allclose(result["AdjustedScore"], global_mean, atol=1e-3)

    def test_sorted_descending(self, model_base_with_oof, bridge):
        result = aggregate_landlord_scores(model_base_with_oof, bridge)
        assert (result["AdjustedScore"].diff().dropna() <= 0).all()

    def test_raises_missing_residual_col(self, model_base, bridge):
        with pytest.raises(ValueError, match="OOFResidual"):
            aggregate_landlord_scores(model_base, bridge)

    def test_raises_missing_landlord_col(self, model_base_with_oof, bridge):
        df = model_base_with_oof.drop(columns=["LandLordID"])
        with pytest.raises(ValueError, match="LandLordID"):
            aggregate_landlord_scores(df, bridge)

    def test_custom_shrinkage_m_from_cfg(self, model_base_with_oof, bridge):
        r5 = aggregate_landlord_scores(
            model_base_with_oof, bridge, cfg={"shrinkage_m": 5}
        )
        r50 = aggregate_landlord_scores(
            model_base_with_oof, bridge, cfg={"shrinkage_m": 50}
        )
        # Higher m → scores should be more tightly clustered
        assert r50["AdjustedScore"].std() <= r5["AdjustedScore"].std()

    def test_all_nan_residuals_returns_empty(self, model_base_with_oof, bridge):
        df = model_base_with_oof.copy()
        df["OOFResidual"] = np.nan
        result = aggregate_landlord_scores(df, bridge)
        assert len(result) == 0


# ---------------------------------------------------------------------------
# build_landlord_adjusted_scores (end-to-end)
# ---------------------------------------------------------------------------


class TestBuildLandlordAdjustedScores:

    def test_returns_tuple_of_two_dataframes(self, model_base, bridge):
        result = build_landlord_adjusted_scores(model_base, bridge)
        assert isinstance(result, tuple) and len(result) == 2
        df1, df2 = result
        assert isinstance(df1, pd.DataFrame)
        assert isinstance(df2, pd.DataFrame)

    def test_model_base_with_oof_has_residuals(self, model_base, bridge):
        mb_oof, _ = build_landlord_adjusted_scores(model_base, bridge)
        assert "OOFResidual" in mb_oof.columns
        assert mb_oof["OOFResidual"].notna().any()

    def test_landlord_scores_has_adjusted_score(self, model_base, bridge):
        _, scores = build_landlord_adjusted_scores(model_base, bridge)
        assert "AdjustedScore" in scores.columns
        assert len(scores) > 0

    def test_tenant_count_sums_to_labeled_companies(self, model_base, bridge):
        _, scores = build_landlord_adjusted_scores(model_base, bridge)
        n_labeled = int(model_base["CompanyIsActive"].notna().sum())
        assert scores["TenantCount"].sum() == n_labeled

    def test_explicit_feature_cols_forwarded(self, model_base, bridge):
        mb_oof, _ = build_landlord_adjusted_scores(
            model_base, bridge, feature_cols=["MonthlyBudget", "SalesOfMainProduct"]
        )
        assert mb_oof["OOFPredictedProb"].notna().any()

    def test_cfg_forwarded_to_both_steps(self, model_base, bridge):
        cfg = {"seed": 99, "n_splits": 3, "shrinkage_m": 5}
        mb_oof, scores = build_landlord_adjusted_scores(model_base, bridge, cfg=cfg)
        # 3 folds used
        folds = mb_oof["OOFFold"][mb_oof["OOFFold"] >= 0].unique()
        assert len(folds) == 3
        # Scores exist
        assert len(scores) > 0
