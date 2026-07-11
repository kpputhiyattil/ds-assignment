"""
Tests for src/features/inference.py — scoring-time feature engineering.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.features.inference import (
    build_scoring_feature_matrix,
    detect_input_kind,
    prepare_upload_for_scoring,
)


class TestDetectInputKind:
    def test_feature_matrix(self):
        df = pd.DataFrame({
            "LandLordID": ["L1"],
            "LandlordAge": [10],
            "PopulationDensity": [1.0],
            "PortfolioMeanClients": [5.0],
            "PortfolioActiveRate": [0.5],
            "PortfolioMeanBudget": [1000.0],
            "AreaPerCompany": [2.0],
        })
        assert detect_input_kind(df) == "feature_matrix"

    def test_raw_landlords(self):
        df = pd.DataFrame({
            "LandLordID": ["L1"],
            "AllCompanyID": ["['C1']"],
            "YearFounded": [2010],
            "Area": [100],
        })
        assert detect_input_kind(df) == "raw_landlords"

    def test_landlord_ids(self):
        df = pd.DataFrame({"LandLordID": ["L1", "L2"]})
        assert detect_input_kind(df) == "landlord_ids"


class TestBuildScoringFeatureMatrix:
    def test_builds_portfolio_features(self):
        landlords = pd.DataFrame({
            "LandLordID": ["L1"],
            "AllCompanyID": [["C1", "C2"]],
            "YearFounded": [2000],
            "Area": [100.0],
            "TotalPopulationAround": [1000.0],
            "ActiveCompanies": [2],
            "PreferredIndustry": ["Tech"],
            "OriginCity": ["X"],
            "OriginCountry": ["Y"],
        })
        companies = pd.DataFrame({
            "CompanyID": ["C1", "C2"],
            "MonthlyBudget": [1000.0, 2000.0],
            "SalesOfMainProduct": [1.0, 2.0],
            "TotalActiveClients": [10.0, 20.0],
            "ReturningClient": [5.0, 8.0],
            "CompanyIsActive": [1.0, 1.0],
            "SuccessScore": [0.4, 0.6],
        })
        matrix = build_scoring_feature_matrix(
            landlords,
            companies=companies,
            companies_scored=companies,
            feature_set="with_portfolio",
        )
        assert len(matrix) == 1
        assert "LandlordAge" in matrix.columns
        assert "PortfolioMeanClients" in matrix.columns
        assert matrix.iloc[0]["PortfolioSize"] == 2
        assert matrix.iloc[0]["PortfolioMeanBudget"] == pytest.approx(1500.0)
        assert matrix.iloc[0]["TenantCount"] == 2


class TestPrepareUploadForScoring:
    def test_raw_runs_pipeline(self):
        landlords = pd.DataFrame({
            "LandLordID": ["L1"],
            "AllCompanyID": [["C1", "C2"]],
            "YearFounded": [2000],
            "Area": [100.0],
            "TotalPopulationAround": [1000.0],
            "ActiveCompanies": [2],
            "PreferredIndustry": ["Tech"],
            "OriginCity": ["X"],
            "OriginCountry": ["Y"],
        })
        companies = pd.DataFrame({
            "CompanyID": ["C1", "C2"],
            "MonthlyBudget": [1000.0, 2000.0],
            "SalesOfMainProduct": [1.0, 2.0],
            "TotalActiveClients": [10.0, 20.0],
            "ReturningClient": [5.0, 8.0],
            "CompanyIsActive": [1.0, 1.0],
            "SuccessScore": [0.4, 0.6],
        })
        matrix, info = prepare_upload_for_scoring(
            landlords,
            companies=companies,
            companies_scored=companies,
            feature_set="with_portfolio",
        )
        assert info["pipeline_applied"] is True
        assert info["input_kind"] == "raw_landlords"
        assert "PortfolioMeanClients" in matrix.columns

    def test_feature_matrix_skips_pipeline(self):
        df = pd.DataFrame({
            "LandLordID": ["L1"],
            "LandlordAge": [10],
            "PopulationDensity": [1.0],
            "PortfolioMeanClients": [5.0],
            "PortfolioActiveRate": [0.5],
            "PortfolioMeanBudget": [1000.0],
            "AreaPerCompany": [2.0],
        })
        matrix, info = prepare_upload_for_scoring(df)
        assert info["pipeline_applied"] is False
        assert info["input_kind"] == "feature_matrix"
        assert len(matrix) == 1
