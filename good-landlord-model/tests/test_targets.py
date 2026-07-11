"""
Tests for src/targets/construction.py

Uses only in-memory DataFrames — no real data files required.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.targets.construction import (
    binary_target,
    rank_normalize,
    composite_score,
    build_targets,
    sensitivity_analysis,
    _DEFAULT_WEIGHTS,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def companies():
    """Clean companies DataFrame with all performance columns."""
    return pd.DataFrame({
        "CompanyID": ["C1", "C2", "C3", "C4", "C5"],
        "CompanyStatus": ["Active", "Active", "Closed", "Active", "Suspended"],
        "MonthlyBudget": [1000, 2000, 500, 1500, 800],
        "SalesOfMainProduct": [5000, 8000, 1000, 6000, 3000],
        "TotalActiveClients": [50, 80, 10, 60, 30],
        "ReturningClient": [20, 40, 5, 25, 10],
        "ClientsInTheLast6Month": [30, 50, 8, 40, 20],
        "ClientsInTheLast12Month": [45, 75, 9, 55, 28],
        "TotalVisitorsInTheLast6Month": [100, 150, 30, 120, 60],
        "TotalVisitorsInTheLast12Month": [180, 280, 50, 200, 100],
    })


# ---------------------------------------------------------------------------
# binary_target
# ---------------------------------------------------------------------------


class TestBinaryTarget:

    def test_active_maps_to_1(self, companies):
        t = binary_target(companies)
        assert t.loc[companies["CompanyStatus"] == "Active"].eq(1).all()

    def test_closed_maps_to_0(self, companies):
        t = binary_target(companies)
        assert t.loc[companies["CompanyStatus"] == "Closed"].eq(0).all()

    def test_suspended_maps_to_0(self, companies):
        t = binary_target(companies)
        assert t.loc[companies["CompanyStatus"] == "Suspended"].eq(0).all()

    def test_case_insensitive(self):
        df = pd.DataFrame({"CompanyStatus": ["ACTIVE", "active", "Active", "closed"]})
        assert binary_target(df).tolist() == [1.0, 1.0, 1.0, 0.0]

    def test_missing_status_maps_to_0(self):
        df = pd.DataFrame({"CompanyStatus": ["Active", None, "Closed"]})
        assert binary_target(df).tolist() == [1.0, 0.0, 0.0]

    def test_name_is_company_is_active(self, companies):
        assert binary_target(companies).name == "CompanyIsActive"

    def test_dtype_is_float_nullable(self, companies):
        assert binary_target(companies).dtype == float

    def test_missing_column_raises(self, companies):
        with pytest.raises(ValueError, match="CompanyStatus"):
            binary_target(companies.drop(columns=["CompanyStatus"]))

    def test_custom_active_labels(self):
        df = pd.DataFrame({"CompanyStatus": ["Open", "Closed", "Open"]})
        assert binary_target(df, active_labels={"Open"}).tolist() == [1.0, 0.0, 1.0]

    def test_all_active(self):
        df = pd.DataFrame({"CompanyStatus": ["Active"] * 10})
        assert binary_target(df).sum() == 10

    def test_all_inactive(self):
        df = pd.DataFrame({"CompanyStatus": ["Closed"] * 5})
        assert binary_target(df).sum() == 0


# ---------------------------------------------------------------------------
# rank_normalize
# ---------------------------------------------------------------------------


class TestRankNormalize:

    def test_min_maps_to_zero(self):
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        assert rank_normalize(s).min() == pytest.approx(0.0)

    def test_max_maps_to_one(self):
        s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        assert rank_normalize(s).max() == pytest.approx(1.0)

    def test_monotone_increasing(self):
        s = pd.Series([10.0, 20.0, 30.0, 40.0])
        assert (rank_normalize(s).diff().dropna() > 0).all()

    def test_nan_stays_nan(self):
        s = pd.Series([1.0, np.nan, 3.0])
        assert np.isnan(rank_normalize(s).iloc[1])

    def test_all_same_value_no_crash(self):
        s = pd.Series([5.0, 5.0, 5.0])
        assert rank_normalize(s).between(0, 1).all()

    def test_single_value_returns_half(self):
        s = pd.Series([42.0])
        assert rank_normalize(s).iloc[0] == pytest.approx(0.5)

    def test_all_nan_returns_all_nan(self):
        assert rank_normalize(pd.Series([np.nan, np.nan])).isna().all()

    def test_output_in_zero_one_range(self):
        rng = np.random.default_rng(0)
        s = pd.Series(rng.standard_normal(100))
        assert rank_normalize(s).between(0, 1).all()

    def test_ties_get_average_rank(self):
        s = pd.Series([1.0, 2.0, 2.0, 3.0])
        out = rank_normalize(s)
        assert out.iloc[1] == pytest.approx(out.iloc[2])


# ---------------------------------------------------------------------------
# composite_score
# ---------------------------------------------------------------------------


class TestCompositeScore:

    def test_returns_series_same_length(self, companies):
        score = composite_score(companies)
        assert isinstance(score, pd.Series) and len(score) == len(companies)

    def test_values_in_zero_one(self, companies):
        valid = composite_score(companies).dropna()
        assert (valid >= 0.0).all() and (valid <= 1.0).all()

    def test_name_is_success_score(self, companies):
        assert composite_score(companies).name == "SuccessScore"

    def test_weights_sum_to_one_enforced(self, companies):
        w = {k: v * 2 for k, v in _DEFAULT_WEIGHTS.items()}
        valid = composite_score(companies, weights=w).dropna()
        assert (valid >= 0.0).all() and (valid <= 1.0).all()

    def test_missing_columns_produce_partial_score(self):
        df = pd.DataFrame({
            "SalesOfMainProduct": [5000, 1000],
            "MonthlyBudget": [1000, 500],
        })
        score = composite_score(df)
        assert len(score) == 2

    def test_return_components_flag(self, companies):
        score, comp_df = composite_score(companies, return_components=True)
        assert isinstance(score, pd.Series)
        assert isinstance(comp_df, pd.DataFrame)
        assert comp_df.shape[0] == len(companies)

    def test_all_nan_input_produces_nan_output(self):
        df = pd.DataFrame({col: [np.nan, np.nan] for col in [
            "SalesOfMainProduct", "MonthlyBudget", "TotalActiveClients",
            "ReturningClient", "ClientsInTheLast6Month",
            "TotalVisitorsInTheLast6Month", "ClientsInTheLast12Month",
            "TotalVisitorsInTheLast12Month",
        ]})
        assert composite_score(df).isna().all()

    def test_different_weights_give_different_scores(self, companies):
        w1 = {"sales_efficiency": 1.0, "retention": 0.0, "conversion": 0.0,
              "momentum": 0.0, "active_clients": 0.0}
        w2 = {"sales_efficiency": 0.0, "retention": 1.0, "conversion": 0.0,
              "momentum": 0.0, "active_clients": 0.0}
        assert not composite_score(companies, weights=w1).equals(
            composite_score(companies, weights=w2)
        )

    def test_single_company_no_crash(self):
        df = pd.DataFrame({
            "SalesOfMainProduct": [5000], "MonthlyBudget": [1000],
            "TotalActiveClients": [50], "ReturningClient": [20],
            "ClientsInTheLast6Month": [30], "TotalVisitorsInTheLast6Month": [100],
            "ClientsInTheLast12Month": [45], "TotalVisitorsInTheLast12Month": [180],
        })
        assert len(composite_score(df)) == 1


# ---------------------------------------------------------------------------
# build_targets
# ---------------------------------------------------------------------------


class TestBuildTargets:

    def test_returns_dataframe(self, companies):
        assert isinstance(build_targets(companies), pd.DataFrame)

    def test_adds_required_columns(self, companies):
        out = build_targets(companies)
        for col in ("CompanyIsActive", "SuccessScore", "SuccessScoreRank"):
            assert col in out.columns

    def test_rank_1_is_highest_score(self, companies):
        out = build_targets(companies)
        best_idx = out["SuccessScore"].idxmax()
        assert out.loc[best_idx, "SuccessScoreRank"] == 1

    def test_does_not_mutate_input(self, companies):
        cols_before = list(companies.columns)
        _ = build_targets(companies)
        assert list(companies.columns) == cols_before

    def test_cfg_weights_applied(self, companies):
        cfg = {"composite_weights": {
            "sales_efficiency": 1.0, "retention": 0.0, "conversion": 0.0,
            "momentum": 0.0, "active_clients": 0.0,
        }}
        out = build_targets(companies, cfg=cfg)
        assert "SuccessScore" in out.columns


# ---------------------------------------------------------------------------
# sensitivity_analysis
# ---------------------------------------------------------------------------


class TestSensitivityAnalysis:

    def test_returns_dataframe(self, companies):
        assert isinstance(sensitivity_analysis(companies), pd.DataFrame)

    def test_diagonal_is_one(self, companies):
        corr = sensitivity_analysis(companies)
        assert np.allclose(np.diag(corr.values), 1.0)

    def test_matrix_is_square(self, companies):
        corr = sensitivity_analysis(companies)
        assert corr.shape[0] == corr.shape[1]

    def test_values_in_minus_one_to_one(self, companies):
        corr = sensitivity_analysis(companies)
        assert (corr.values >= -1.0).all() and (corr.values <= 1.0).all()

    def test_custom_weight_configs(self, companies):
        configs = [
            {"sales_efficiency": 1.0, "retention": 0.0, "conversion": 0.0,
             "momentum": 0.0, "active_clients": 0.0},
            {"sales_efficiency": 0.0, "retention": 1.0, "conversion": 0.0,
             "momentum": 0.0, "active_clients": 0.0},
        ]
        corr = sensitivity_analysis(
            companies,
            weight_configs=configs,
            config_names=["sales_only", "retention_only"],
        )
        assert corr.shape == (2, 2)
        assert list(corr.columns) == ["sales_only", "retention_only"]
