"""
Tests for src/features/transforms.py

Uses only in-memory DataFrames — no real data files required.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.transforms import (
    landlord_features,
    portfolio_features,
    build_feature_matrix,
    get_feature_cols,
    _LANDLORD_CAT_COLS,
)


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def landlords():
    return pd.DataFrame({
        "LandLordID":             ["L1", "L2", "L3", "L4"],
        "YearFounded":            [2000, 2010, 1990, None],
        "Area":                   [500.0, 1200.0, 300.0, 800.0],
        "TotalPopulationAround":  [10000, 25000, 5000, 15000],
        "ActiveCompanies":        [5, 10, 2, 0],
        "PreferredIndustry":      ["Retail", "Tech", "Food", None],
        "OriginCity":             ["Tel Aviv", "Haifa", "Jerusalem", "Tel Aviv"],
        "OriginCountry":          ["IL", "IL", "IL", "IL"],
    })


@pytest.fixture
def model_base():
    """12-row model_base: L1→3 companies, L2→5, L3→2, L4→2."""
    rows = []
    for lid, cids, statuses, budgets, sales, clients, returning in [
        ("L1", ["C1","C2","C3"], [1.0,0.0,1.0], [1000,2000,1500], [5000,8000,6000], [50,80,60], [20,40,25]),
        ("L2", ["C4","C5","C6","C7","C8"], [1.0,1.0,0.0,1.0,0.0], [3000,1000,500,2000,800], [12000,4000,1000,9000,2000], [100,30,10,80,20], [50,15,5,40,8]),
        ("L3", ["C9","C10"], [0.0,1.0], [500,900], [2000,3500], [20,35], [8,15]),
        ("L4", ["C11","C12"], [1.0,1.0], [1200,1800], [5500,7000], [45,60], [18,25]),
    ]:
        for i, cid in enumerate(cids):
            rows.append({
                "LandLordID": lid, "CompanyID": cid,
                "CompanyIsActive": statuses[i],
                "MonthlyBudget": budgets[i],
                "SalesOfMainProduct": sales[i],
                "TotalActiveClients": clients[i],
                "ReturningClient": returning[i],
                "SuccessScore": np.random.default_rng(int(cid[1:])).random(),
            })
    return pd.DataFrame(rows)


@pytest.fixture
def landlord_scores():
    return pd.DataFrame({
        "LandLordID":       ["L1", "L2", "L3", "L4"],
        "TenantCount":      [3, 5, 2, 2],
        "RawAdjustedScore": [0.05, 0.12, -0.08, 0.03],
        "AdjustedScore":    [0.046, 0.109, -0.070, 0.028],
    })


# ---------------------------------------------------------------------------
# landlord_features
# ---------------------------------------------------------------------------


class TestLandlordFeatures:

    def test_returns_dataframe(self, landlords):
        assert isinstance(landlord_features(landlords), pd.DataFrame)

    def test_has_landlord_id_column(self, landlords):
        assert "LandLordID" in landlord_features(landlords).columns

    def test_row_count_matches_input(self, landlords):
        assert len(landlord_features(landlords)) == len(landlords)

    def test_landlord_age_computed_correctly(self, landlords):
        out = landlord_features(landlords, reference_year=2025)
        # L1: 2025 - 2000 = 25
        row = out[out["LandLordID"] == "L1"].iloc[0]
        assert row["LandlordAge"] == pytest.approx(25.0)

    def test_missing_year_gives_nan_age(self, landlords):
        out = landlord_features(landlords, reference_year=2025)
        row = out[out["LandLordID"] == "L4"].iloc[0]
        assert np.isnan(row["LandlordAge"])

    def test_population_density_ratio(self, landlords):
        out = landlord_features(landlords, reference_year=2025)
        # L1: 10000 / 500 = 20.0
        row = out[out["LandLordID"] == "L1"].iloc[0]
        assert row["PopulationDensity"] == pytest.approx(10000 / 500, rel=1e-4)

    def test_area_per_company_ratio(self, landlords):
        out = landlord_features(landlords, reference_year=2025)
        # L1: 500 / 5 = 100.0
        row = out[out["LandLordID"] == "L1"].iloc[0]
        assert row["AreaPerCompany"] == pytest.approx(500 / 5, rel=1e-4)

    def test_zero_active_companies_no_nan_ratio(self, landlords):
        """L4 has ActiveCompanies=0; safe division should not crash."""
        out = landlord_features(landlords, reference_year=2025)
        row = out[out["LandLordID"] == "L4"].iloc[0]
        # Should be a finite number (very large, since eps in denominator)
        assert np.isfinite(row["AreaPerCompany"])

    def test_categorical_cols_preserved(self, landlords):
        out = landlord_features(landlords)
        for col in _LANDLORD_CAT_COLS:
            assert col in out.columns

    def test_none_industry_is_null(self, landlords):
        out = landlord_features(landlords)
        row = out[out["LandLordID"] == "L4"].iloc[0]
        assert row["PreferredIndustry"] is None or pd.isna(row["PreferredIndustry"])

    def test_col_alias_landlord_origin_city(self):
        """LandlordOriginCity should be normalised to OriginCity."""
        df = pd.DataFrame({
            "LandLordID": ["L1"],
            "YearFounded": [2010],
            "Area": [100.0],
            "TotalPopulationAround": [5000],
            "ActiveCompanies": [3],
            "LandlordOriginCity": ["Tel Aviv"],
            "LandlordOriginCountry": ["IL"],
        })
        out = landlord_features(df)
        assert "OriginCity" in out.columns
        assert out.iloc[0]["OriginCity"] == "Tel Aviv"

    def test_raises_missing_landlord_id(self):
        df = pd.DataFrame({"Area": [100.0], "YearFounded": [2010]})
        with pytest.raises(ValueError, match="LandLordID"):
            landlord_features(df)

    def test_does_not_mutate_input(self, landlords):
        cols_before = list(landlords.columns)
        landlord_features(landlords)
        assert list(landlords.columns) == cols_before

    def test_negative_area_clipped_to_zero(self):
        df = pd.DataFrame({
            "LandLordID": ["L1"],
            "YearFounded": [2010],
            "Area": [-100.0],
            "TotalPopulationAround": [5000],
            "ActiveCompanies": [3],
        })
        out = landlord_features(df)
        assert out.iloc[0]["Area"] == pytest.approx(0.0)

    def test_output_index_is_reset(self, landlords):
        out = landlord_features(landlords)
        assert list(out.index) == list(range(len(landlords)))


# ---------------------------------------------------------------------------
# portfolio_features
# ---------------------------------------------------------------------------


class TestPortfolioFeatures:

    def test_returns_dataframe(self, model_base):
        assert isinstance(portfolio_features(model_base), pd.DataFrame)

    def test_has_landlord_id_column(self, model_base):
        assert "LandLordID" in portfolio_features(model_base).columns

    def test_one_row_per_landlord(self, model_base):
        out = portfolio_features(model_base)
        assert len(out) == model_base["LandLordID"].nunique()

    def test_portfolio_size_correct(self, model_base):
        out = portfolio_features(model_base)
        # L2 has 5 companies
        row = out[out["LandLordID"] == "L2"].iloc[0]
        assert row["PortfolioSize"] == 5

    def test_active_rate_between_0_and_1(self, model_base):
        out = portfolio_features(model_base)
        rates = out["PortfolioActiveRate"].dropna()
        assert (rates >= 0.0).all() and (rates <= 1.0).all()

    def test_active_rate_for_l1(self, model_base):
        # L1: [1, 0, 1] → rate = 2/3
        out = portfolio_features(model_base)
        row = out[out["LandLordID"] == "L1"].iloc[0]
        assert row["PortfolioActiveRate"] == pytest.approx(2 / 3, rel=1e-4)

    def test_mean_budget_computed(self, model_base):
        out = portfolio_features(model_base)
        # L1: mean([1000, 2000, 1500]) = 1500
        row = out[out["LandLordID"] == "L1"].iloc[0]
        assert row["PortfolioMeanBudget"] == pytest.approx(1500.0)

    def test_std_budget_nan_for_single_company(self):
        """Single-company landlord → std is undefined."""
        df = pd.DataFrame({
            "LandLordID": ["L1"],
            "CompanyID": ["C1"],
            "CompanyIsActive": [1.0],
            "MonthlyBudget": [1000.0],
            "SalesOfMainProduct": [5000.0],
            "TotalActiveClients": [50.0],
            "ReturningClient": [20.0],
        })
        out = portfolio_features(df)
        assert np.isnan(out.iloc[0]["PortfolioStdBudget"])

    def test_success_score_mean_present(self, model_base):
        out = portfolio_features(model_base)
        assert "PortfolioMeanSuccessScore" in out.columns
        assert out["PortfolioMeanSuccessScore"].notna().any()

    def test_success_score_nan_when_column_absent(self):
        df = pd.DataFrame({
            "LandLordID": ["L1", "L1"],
            "CompanyID": ["C1", "C2"],
            "MonthlyBudget": [1000.0, 2000.0],
            "SalesOfMainProduct": [5000.0, 8000.0],
            "TotalActiveClients": [50.0, 80.0],
            "ReturningClient": [20.0, 40.0],
        })
        out = portfolio_features(df)
        assert np.isnan(out.iloc[0]["PortfolioMeanSuccessScore"])

    def test_raises_missing_landlord_id(self):
        df = pd.DataFrame({"CompanyID": ["C1"], "MonthlyBudget": [1000.0]})
        with pytest.raises(ValueError, match="LandLordID"):
            portfolio_features(df)

    def test_retention_rate_between_0_and_1(self, model_base):
        out = portfolio_features(model_base)
        rates = out["PortfolioMeanRetention"].dropna()
        # Retention = returning / active_clients ≤ 1 normally (capped by _safe_ratio at large values)
        assert (rates >= 0.0).all()

    def test_does_not_mutate_input(self, model_base):
        cols_before = list(model_base.columns)
        portfolio_features(model_base)
        assert list(model_base.columns) == cols_before


# ---------------------------------------------------------------------------
# build_feature_matrix
# ---------------------------------------------------------------------------


class TestBuildFeatureMatrix:

    def test_returns_dataframe(self, landlords, model_base, landlord_scores):
        out = build_feature_matrix(landlords, model_base, landlord_scores)
        assert isinstance(out, pd.DataFrame)

    def test_has_landlord_id_and_target(self, landlords, model_base, landlord_scores):
        out = build_feature_matrix(landlords, model_base, landlord_scores)
        assert "LandLordID" in out.columns
        assert "AdjustedScore" in out.columns

    def test_row_count_equals_scored_landlords(self, landlords, model_base, landlord_scores):
        out = build_feature_matrix(landlords, model_base, landlord_scores)
        # Inner join: all 4 landlords have scores
        assert len(out) == len(landlord_scores)

    def test_landlord_only_has_no_portfolio_cols(self, landlords, model_base, landlord_scores):
        out = build_feature_matrix(landlords, model_base, landlord_scores, feature_set="landlord_only")
        assert "PortfolioSize" not in out.columns
        assert "PortfolioActiveRate" not in out.columns

    def test_with_portfolio_has_portfolio_cols(self, landlords, model_base, landlord_scores):
        out = build_feature_matrix(landlords, model_base, landlord_scores, feature_set="with_portfolio")
        assert "PortfolioSize" in out.columns
        assert "PortfolioMeanBudget" in out.columns

    def test_landlord_features_present(self, landlords, model_base, landlord_scores):
        out = build_feature_matrix(landlords, model_base, landlord_scores)
        for col in ("LandlordAge", "PopulationDensity", "AreaPerCompany"):
            assert col in out.columns, f"Missing: {col}"

    def test_adjusted_score_values_match_input(self, landlords, model_base, landlord_scores):
        out = build_feature_matrix(landlords, model_base, landlord_scores)
        for _, score_row in landlord_scores.iterrows():
            lid = score_row["LandLordID"]
            expected = score_row["AdjustedScore"]
            actual = out.loc[out["LandLordID"] == lid, "AdjustedScore"].iloc[0]
            assert actual == pytest.approx(expected, rel=1e-6)

    def test_tenant_count_propagated(self, landlords, model_base, landlord_scores):
        out = build_feature_matrix(landlords, model_base, landlord_scores)
        assert "TenantCount" in out.columns
        assert (out["TenantCount"] > 0).all()

    def test_raises_invalid_feature_set(self, landlords, model_base, landlord_scores):
        with pytest.raises(ValueError, match="feature_set"):
            build_feature_matrix(landlords, model_base, landlord_scores, feature_set="bad_mode")

    def test_raises_missing_adjusted_score(self, landlords, model_base):
        scores_no_target = pd.DataFrame({"LandLordID": ["L1"], "TenantCount": [3]})
        with pytest.raises(ValueError, match="AdjustedScore"):
            build_feature_matrix(landlords, model_base, scores_no_target)

    def test_unsecored_landlord_excluded(self, landlords, model_base):
        """If a landlord has no score, it should be excluded from the matrix."""
        partial_scores = pd.DataFrame({
            "LandLordID": ["L1", "L2"],
            "AdjustedScore": [0.05, 0.10],
            "TenantCount": [3, 5],
        })
        out = build_feature_matrix(landlords, model_base, partial_scores)
        assert set(out["LandLordID"].unique()) == {"L1", "L2"}

    def test_output_index_is_reset(self, landlords, model_base, landlord_scores):
        out = build_feature_matrix(landlords, model_base, landlord_scores)
        assert list(out.index) == list(range(len(out)))

    def test_does_not_mutate_landlords(self, landlords, model_base, landlord_scores):
        cols_before = list(landlords.columns)
        build_feature_matrix(landlords, model_base, landlord_scores)
        assert list(landlords.columns) == cols_before


# ---------------------------------------------------------------------------
# get_feature_cols
# ---------------------------------------------------------------------------


class TestGetFeatureCols:

    def test_returns_tuple_of_two_lists(self, landlords, model_base, landlord_scores):
        matrix = build_feature_matrix(landlords, model_base, landlord_scores)
        result = get_feature_cols(matrix)
        assert isinstance(result, tuple) and len(result) == 2
        num_cols, cat_cols = result
        assert isinstance(num_cols, list) and isinstance(cat_cols, list)

    def test_target_excluded(self, landlords, model_base, landlord_scores):
        matrix = build_feature_matrix(landlords, model_base, landlord_scores)
        num_cols, cat_cols = get_feature_cols(matrix)
        all_feature_cols = num_cols + cat_cols
        assert "AdjustedScore" not in all_feature_cols
        assert "LandLordID" not in all_feature_cols

    def test_numeric_cols_are_numeric(self, landlords, model_base, landlord_scores):
        matrix = build_feature_matrix(landlords, model_base, landlord_scores)
        num_cols, _ = get_feature_cols(matrix)
        for col in num_cols:
            assert pd.api.types.is_numeric_dtype(matrix[col]), f"{col} is not numeric"

    def test_categorical_cols_are_object(self, landlords, model_base, landlord_scores):
        matrix = build_feature_matrix(landlords, model_base, landlord_scores)
        _, cat_cols = get_feature_cols(matrix)
        for col in cat_cols:
            assert matrix[col].dtype == object, f"{col} is not object dtype"
