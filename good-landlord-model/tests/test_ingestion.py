"""
Tests for src/data/ingestion.py

Uses only in-memory DataFrames — no real data files required.
"""
from __future__ import annotations

import json

import pandas as pd
import pytest

from src.data.ingestion import (
    parse_all_company_id,
    build_landlord_company_bridge,
    build_model_base,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def landlords_list_format():
    """AllCompanyID stored as Python lists (pyarrow-loaded)."""
    return pd.DataFrame({
        "LandLordID": ["L1", "L2", "L3"],
        "AllCompanyID": [["C1", "C2"], ["C3"], ["C4", "C5"]],
    })


@pytest.fixture
def landlords_json_format():
    """AllCompanyID stored as JSON strings."""
    return pd.DataFrame({
        "LandLordID": ["L1", "L2", "L3"],
        "AllCompanyID": ['["C1","C2"]', '["C3"]', '["C4","C5"]'],
    })


@pytest.fixture
def landlords_csv_format():
    """AllCompanyID stored as comma-separated strings."""
    return pd.DataFrame({
        "LandLordID": ["L1", "L2", "L3"],
        "AllCompanyID": ["C1,C2", "C3", "C4,C5"],
    })


@pytest.fixture
def companies():
    return pd.DataFrame({
        "CompanyID": ["C1", "C2", "C3", "C4", "C5"],
        "CompanyStatus": ["Active", "Active", "Closed", "Active", "Active"],
        "MonthlyBudget": [1000, 2000, 500, 1500, 800],
    })


# ---------------------------------------------------------------------------
# parse_all_company_id
# ---------------------------------------------------------------------------


class TestParseAllCompanyId:
    """Unit tests for all supported input formats."""

    # -- null / empty --------------------------------------------------------

    def test_none_returns_empty_list(self):
        assert parse_all_company_id(None) == []

    def test_float_nan_returns_empty_list(self):
        import math
        assert parse_all_company_id(float("nan")) == []

    def test_empty_string_returns_empty_list(self):
        assert parse_all_company_id("") == []

    def test_string_nan_returns_empty_list(self):
        assert parse_all_company_id("nan") == []

    def test_string_null_returns_empty_list(self):
        assert parse_all_company_id("null") == []

    def test_empty_json_array_returns_empty_list(self):
        assert parse_all_company_id("[]") == []

    # -- Python list ---------------------------------------------------------

    def test_python_list_parsed_correctly(self):
        assert parse_all_company_id(["C1", "C2"]) == ["C1", "C2"]

    def test_python_list_strips_whitespace(self):
        assert parse_all_company_id([" C1 ", " C2"]) == ["C1", "C2"]

    def test_python_list_skips_none_elements(self):
        assert parse_all_company_id(["C1", None, "C2"]) == ["C1", "C2"]

    def test_python_tuple_works(self):
        assert parse_all_company_id(("C1", "C2")) == ["C1", "C2"]

    # -- JSON string ---------------------------------------------------------

    def test_json_array_string(self):
        assert parse_all_company_id('["C1","C2"]') == ["C1", "C2"]

    def test_json_array_single_element(self):
        assert parse_all_company_id('["C3"]') == ["C3"]

    def test_json_scalar_string(self):
        result = parse_all_company_id('"C1"')
        assert "C1" in result

    def test_json_fmt_explicit(self):
        assert parse_all_company_id('["C1","C2"]', fmt="json") == ["C1", "C2"]

    def test_json_fmt_invalid_falls_back_to_raw(self):
        result = parse_all_company_id("not-json", fmt="json")
        assert result == ["not-json"]

    # -- Comma-separated string ----------------------------------------------

    def test_csv_two_ids(self):
        assert parse_all_company_id("C1,C2") == ["C1", "C2"]

    def test_csv_three_ids(self):
        assert parse_all_company_id("C1,C2,C3") == ["C1", "C2", "C3"]

    def test_csv_fmt_explicit(self):
        assert parse_all_company_id("C1,C2", fmt="csv") == ["C1", "C2"]

    def test_csv_strips_spaces(self):
        assert parse_all_company_id("C1 , C2 , C3") == ["C1", "C2", "C3"]

    # -- Single scalar -------------------------------------------------------

    def test_single_string_id(self):
        assert parse_all_company_id("C1") == ["C1"]

    def test_integer_scalar(self):
        result = parse_all_company_id(123)
        assert result == ["123"]

    # -- auto format (default) -----------------------------------------------

    def test_auto_picks_json_for_bracket_prefix(self):
        assert parse_all_company_id('["C1","C2"]', fmt="auto") == ["C1", "C2"]

    def test_auto_picks_csv_for_comma(self):
        assert parse_all_company_id("C1,C2", fmt="auto") == ["C1", "C2"]

    def test_auto_single_value(self):
        assert parse_all_company_id("C99", fmt="auto") == ["C99"]

    def test_auto_python_list_repr_with_single_quotes(self):
        # Raw Landlords.parquet stores AllCompanyID as a Python list string
        assert parse_all_company_id("['COMPANY_1', 'COMPANY_2']", fmt="auto") == [
            "COMPANY_1",
            "COMPANY_2",
        ]

    def test_auto_python_list_repr_single_element(self):
        assert parse_all_company_id("['COMPANY_0335']", fmt="auto") == ["COMPANY_0335"]

    def test_fmt_list_python_repr(self):
        assert parse_all_company_id("['C1', 'C2']", fmt="list") == ["C1", "C2"]


# ---------------------------------------------------------------------------
# build_landlord_company_bridge
# ---------------------------------------------------------------------------


class TestBuildBridge:

    def test_list_format_produces_correct_rows(self, landlords_list_format):
        bridge = build_landlord_company_bridge(landlords_list_format)
        assert set(bridge.columns) == {"LandLordID", "CompanyID"}
        assert len(bridge) == 5  # C1,C2,C3,C4,C5

    def test_json_format_produces_correct_rows(self, landlords_json_format):
        bridge = build_landlord_company_bridge(landlords_json_format)
        assert len(bridge) == 5

    def test_csv_format_produces_correct_rows(self, landlords_csv_format):
        bridge = build_landlord_company_bridge(landlords_csv_format)
        assert len(bridge) == 5

    def test_deduplication(self):
        df = pd.DataFrame({
            "LandLordID": ["L1", "L1"],
            "AllCompanyID": [["C1", "C2"], ["C1", "C3"]],  # C1 appears twice
        })
        bridge = build_landlord_company_bridge(df)
        # L1-C1 should appear only once
        assert len(bridge[(bridge.LandLordID == "L1") & (bridge.CompanyID == "C1")]) == 1

    def test_missing_column_raises(self):
        df = pd.DataFrame({"LandLordID": ["L1"], "OtherCol": [1]})
        with pytest.raises(ValueError, match="AllCompanyID"):
            build_landlord_company_bridge(df)

    def test_landlord_id_preserved(self, landlords_list_format):
        bridge = build_landlord_company_bridge(landlords_list_format)
        assert set(bridge["LandLordID"].unique()) == {"L1", "L2", "L3"}

    def test_company_ids_correct(self, landlords_list_format):
        bridge = build_landlord_company_bridge(landlords_list_format)
        assert set(bridge["CompanyID"].unique()) == {"C1", "C2", "C3", "C4", "C5"}

    def test_empty_all_company_id_skipped(self):
        df = pd.DataFrame({
            "LandLordID": ["L1", "L2"],
            "AllCompanyID": [[], ["C3"]],
        })
        bridge = build_landlord_company_bridge(df)
        assert len(bridge) == 1
        assert bridge.iloc[0]["LandLordID"] == "L2"

    def test_null_all_company_id_skipped(self):
        df = pd.DataFrame({
            "LandLordID": ["L1", "L2"],
            "AllCompanyID": [None, ["C3"]],
        })
        bridge = build_landlord_company_bridge(df)
        assert len(bridge) == 1

    def test_fmt_argument_forwarded(self, landlords_json_format):
        bridge = build_landlord_company_bridge(landlords_json_format, fmt="json")
        assert len(bridge) == 5


# ---------------------------------------------------------------------------
# build_model_base
# ---------------------------------------------------------------------------


class TestBuildModelBase:

    def test_returns_tuple_of_two_dataframes(self, landlords_list_format, companies):
        result = build_model_base(landlords_list_format, companies)
        assert isinstance(result, tuple) and len(result) == 2
        bridge, model_base = result
        assert isinstance(bridge, pd.DataFrame)
        assert isinstance(model_base, pd.DataFrame)

    def test_model_base_has_all_company_columns(self, landlords_list_format, companies):
        _, model_base = build_model_base(landlords_list_format, companies)
        for col in companies.columns:
            assert col in model_base.columns, f"Missing column: {col}"

    def test_model_base_row_count_equals_bridge(self, landlords_list_format, companies):
        bridge, model_base = build_model_base(landlords_list_format, companies)
        assert len(model_base) == len(bridge)

    def test_full_coverage_no_nan_landlord_id(self, landlords_list_format, companies):
        _, model_base = build_model_base(landlords_list_format, companies)
        assert model_base["LandLordID"].notna().all()

    def test_unmatched_company_gives_nan_features(self, landlords_list_format):
        # Companies table is missing C5
        companies_partial = pd.DataFrame({
            "CompanyID": ["C1", "C2", "C3", "C4"],
            "CompanyStatus": ["Active", "Active", "Closed", "Active"],
        })
        _, model_base = build_model_base(landlords_list_format, companies_partial)
        unmatched_row = model_base[model_base["CompanyID"] == "C5"]
        assert len(unmatched_row) == 1
        assert unmatched_row["CompanyStatus"].isna().all()

    def test_bridge_is_deduplicated(self, landlords_list_format, companies):
        bridge, _ = build_model_base(landlords_list_format, companies)
        dupes = bridge.duplicated(subset=["LandLordID", "CompanyID"]).sum()
        assert dupes == 0
