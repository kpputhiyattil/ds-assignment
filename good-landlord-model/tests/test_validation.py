"""
Tests for src/data/validation.py

Uses synthetic DataFrames — no real data files required.
"""
from __future__ import annotations

import pandas as pd
import pytest

from src.data.validation import (
    QualityReport,
    SEVERITY_CRITICAL,
    SEVERITY_WARNING,
    SEVERITY_INFO,
    check_identifier_integrity,
    check_relationship_consistency,
    check_date_validity,
    check_ranges,
    check_time_window_consistency,
    check_category_quality,
    check_snapshot_structure,
    check_bridge_coverage,
    check_missing_rates,
    validate_all,
    NON_NEGATIVE_COLS_COMPANIES,
)


def _findings(report, check):
    return [f for f in report.findings if f.check == check]

def _severities(report, check):
    return {f.severity for f in _findings(report, check)}


@pytest.fixture
def clean_landlords():
    return pd.DataFrame({
        "LandLordID": ["L1", "L2", "L3"],
        "AllCompanyID": [["C1","C2"], ["C3"], ["C4","C5"]],
        "ActiveCompanies": [2, 1, 2],
        "Area": [100.0, 200.0, 150.0],
        "TotalPopulationAround": [5000, 3000, 4000],
        "PreferredIndustry": ["Retail", "Food", "Tech"],
        "OriginCity": ["Tel Aviv", "Haifa", "Jerusalem"],
        "OriginCountry": ["IL", "IL", "IL"],
        "LastUpdated": ["2023-01-15", "2022-06-30", "2023-03-01"],
        "YearFounded": [2010, 2015, 2018],
    })


@pytest.fixture
def clean_companies():
    return pd.DataFrame({
        "CompanyID": ["C1","C2","C3","C4","C5"],
        "CompanyStatus": ["Active","Active","Closed","Active","Active"],
        "PrimaryType": ["Retail","Retail","Food","Tech","Tech"],
        "MonthlyBudget": [1000, 2000, 500, 1500, 800],
        "SalesOfMainProduct": [5000, 8000, 1000, 6000, 3000],
        "SalesOfOtherProduct": [500, 200, 0, 300, 100],
        "TotalActiveClients": [50, 80, 10, 60, 30],
        "ReturningClient": [20, 40, 5, 25, 10],
        "ClientsInTheLast7Days": [5, 8, 1, 6, 3],
        "ClientsInTheLast6Month": [30, 50, 8, 40, 20],
        "ClientsInTheLast12Month": [45, 75, 9, 55, 28],
        "TotalVisitorsInTheLast7Days": [20, 30, 5, 25, 12],
        "TotalVisitorsInTheLast6Month": [100, 150, 30, 120, 60],
        "TotalVisitorsInTheLast12Month": [180, 280, 50, 200, 100],
        "Investments": [10000, 25000, 2000, 15000, 5000],
        "OriginCity": ["Tel Aviv","Haifa","Haifa","Jerusalem","Tel Aviv"],
        "OriginCountry": ["IL","IL","IL","IL","IL"],
        "YearFounded": [2015, 2018, 2012, 2019, 2020],
    })


@pytest.fixture
def clean_bridge():
    return pd.DataFrame({
        "LandLordID": ["L1","L1","L2","L3","L3"],
        "CompanyID":  ["C1","C2","C3","C4","C5"],
    })


class TestIdentifierIntegrity:
    def test_clean_data_gives_info(self, clean_landlords):
        r = QualityReport()
        check_identifier_integrity(clean_landlords, "LandLordID", "LandLords", r)
        assert SEVERITY_INFO in _severities(r, "identifier_integrity")
        assert SEVERITY_CRITICAL not in _severities(r, "identifier_integrity")

    def test_null_id_is_critical(self):
        df = pd.DataFrame({"LandLordID": ["L1", None, "L3"]})
        r = QualityReport()
        check_identifier_integrity(df, "LandLordID", "LandLords", r)
        assert SEVERITY_CRITICAL in _severities(r, "identifier_integrity")

    def test_duplicate_id_is_warning(self):
        df = pd.DataFrame({"LandLordID": ["L1","L1","L2"]})
        r = QualityReport()
        check_identifier_integrity(df, "LandLordID", "LandLords", r)
        assert SEVERITY_WARNING in _severities(r, "identifier_integrity")

    def test_blank_string_id_is_critical(self):
        df = pd.DataFrame({"LandLordID": ["L1","  ","L3"]})
        r = QualityReport()
        check_identifier_integrity(df, "LandLordID", "LandLords", r)
        assert SEVERITY_CRITICAL in _severities(r, "identifier_integrity")


class TestRelationshipConsistency:
    def test_consistent_counts(self, clean_landlords, clean_bridge):
        r = QualityReport()
        check_relationship_consistency(clean_landlords, clean_bridge, r)
        assert SEVERITY_CRITICAL not in _severities(r, "relationship_consistency")

    def test_missing_column_is_warning(self, clean_bridge):
        df = pd.DataFrame({"LandLordID": ["L1"], "AllCompanyID": [["C1"]]})
        r = QualityReport()
        check_relationship_consistency(df, clean_bridge, r)
        assert SEVERITY_WARNING in _severities(r, "relationship_consistency")

    def test_large_discrepancy_is_warning(self, clean_bridge):
        lls = pd.DataFrame({
            "LandLordID": ["L1","L2","L3"],
            "ActiveCompanies": [100, 1, 2],
        })
        r = QualityReport()
        check_relationship_consistency(lls, clean_bridge, r)
        assert SEVERITY_WARNING in _severities(r, "relationship_consistency")


class TestDateValidity:
    def test_clean_dates_pass(self, clean_landlords):
        r = QualityReport()
        check_date_validity(clean_landlords, ["LastUpdated","YearFounded"], "LandLords", r)
        assert SEVERITY_CRITICAL not in _severities(r, "date_validity")

    def test_future_date_is_warning(self):
        df = pd.DataFrame({"LastUpdated": ["2099-01-01","2022-06-01"]})
        r = QualityReport()
        check_date_validity(df, ["LastUpdated"], "Test", r)
        assert SEVERITY_WARNING in _severities(r, "date_validity")

    def test_unparseable_date_is_warning(self):
        df = pd.DataFrame({"LastUpdated": ["not-a-date","2022-01-01"]})
        r = QualityReport()
        check_date_validity(df, ["LastUpdated"], "Test", r)
        assert SEVERITY_WARNING in _severities(r, "date_validity")

    def test_missing_column_is_warning(self):
        df = pd.DataFrame({"SomeOtherCol": [1, 2]})
        r = QualityReport()
        check_date_validity(df, ["LastUpdated"], "Test", r)
        assert SEVERITY_WARNING in _severities(r, "date_validity")


class TestRanges:
    def test_clean_data_no_warnings(self, clean_companies):
        r = QualityReport()
        check_ranges(clean_companies, NON_NEGATIVE_COLS_COMPANIES, "Companies", r)
        assert SEVERITY_WARNING not in _severities(r, "range_check")

    def test_negative_sales_is_warning(self):
        df = pd.DataFrame({"SalesOfMainProduct": [-100, 500], "SalesOfOtherProduct": [200, 300], "MonthlyBudget": [1000, 2000]})
        r = QualityReport()
        check_ranges(df, ["SalesOfMainProduct"], "Companies", r)
        assert SEVERITY_WARNING in _severities(r, "range_check")

    def test_zero_denominator_is_info(self):
        df = pd.DataFrame({"Area": [0, 100], "ActiveCompanies": [5,3], "MonthlyBudget": [1000,2000]})
        r = QualityReport()
        check_ranges(df, ["Area"], "LandLords", r)
        assert SEVERITY_INFO in _severities(r, "denominator_guard")


class TestTimeWindowConsistency:
    def test_clean_windows_are_info(self, clean_companies):
        r = QualityReport()
        check_time_window_consistency(clean_companies, r)
        assert SEVERITY_CRITICAL not in _severities(r, "time_window_consistency")

    def test_7d_exceeding_6m_triggers_warning(self):
        df = pd.DataFrame({
            "ClientsInTheLast7Days": [100, 5],
            "ClientsInTheLast6Months": [10, 50],
            "ClientsInTheLast12Months": [15, 70],
            "TotalClientsInTheLast7Days": [50, 20],
            "TotalClientsInTheLast6Months": [200, 100],
            "TotalClientsInTheLast12Months": [300, 150],
        })
        r = QualityReport()
        check_time_window_consistency(df, r, violation_threshold_pct=0.0)
        assert SEVERITY_WARNING in _severities(r, "time_window_consistency")


class TestCategoryQuality:
    def test_mixed_casing_is_warning(self):
        df = pd.DataFrame({"CompanyStatus": ["Active","active","ACTIVE","Closed"]})
        r = QualityReport()
        check_category_quality(df, ["CompanyStatus"], "Companies", r)
        assert SEVERITY_WARNING in _severities(r, "category_quality")

    def test_blank_strings_are_warning(self):
        df = pd.DataFrame({"PreferredIndustry": ["Retail","  ","Food"]})
        r = QualityReport()
        check_category_quality(df, ["PreferredIndustry"], "LandLords", r)
        assert SEVERITY_WARNING in _severities(r, "category_quality")

    def test_clean_categories_no_critical(self, clean_companies):
        r = QualityReport()
        check_category_quality(clean_companies, ["CompanyStatus","PrimaryType"], "Companies", r)
        assert SEVERITY_CRITICAL not in _severities(r, "category_quality")


class TestSnapshotStructure:
    def test_unique_ids_gives_info(self, clean_landlords):
        r = QualityReport()
        check_snapshot_structure(clean_landlords, "LandLordID", "LandLords", r)
        assert SEVERITY_INFO in _severities(r, "snapshot_structure")

    def test_duplicate_ids_gives_warning(self):
        df = pd.DataFrame({"LandLordID": ["L1","L1","L2"]})
        r = QualityReport()
        check_snapshot_structure(df, "LandLordID", "LandLords", r)
        assert SEVERITY_WARNING in _severities(r, "snapshot_structure")


class TestBridgeCoverage:
    def test_full_coverage_gives_info(self, clean_bridge, clean_companies):
        r = QualityReport()
        check_bridge_coverage(clean_bridge, clean_companies, r)
        assert SEVERITY_CRITICAL not in _severities(r, "bridge_coverage")

    def test_missing_companies_critical_above_20pct(self, clean_companies):
        bridge = pd.DataFrame({"LandLordID": [f"L{i}" for i in range(10)], "CompanyID": [f"C{i}" for i in range(10)]})
        r = QualityReport()
        check_bridge_coverage(bridge, clean_companies, r)
        assert SEVERITY_CRITICAL in _severities(r, "bridge_coverage")


class TestMissingRates:
    def test_no_missing_is_info(self, clean_companies):
        r = QualityReport()
        check_missing_rates(clean_companies, "Companies", r)
        assert SEVERITY_CRITICAL not in _severities(r, "missing_rates")

    def test_high_missing_is_critical(self):
        df = pd.DataFrame({"A": [None]*100, "B": list(range(100))})
        r = QualityReport()
        check_missing_rates(df, "Test", r, critical_threshold=0.50)
        assert SEVERITY_CRITICAL in _severities(r, "missing_rates")


class TestValidateAll:
    def test_clean_data_no_criticals(self, clean_landlords, clean_companies, clean_bridge):
        report = validate_all(clean_landlords, clean_companies, clean_bridge)
        assert len(report.criticals) == 0, f"Unexpected CRITICALs:\n{report.summary()}"

    def test_strict_mode_raises_on_critical(self):
        lls = pd.DataFrame({"LandLordID": [None,"L2"], "AllCompanyID": [[],[]]})
        cos = pd.DataFrame({"CompanyID": ["C1"]})
        br = pd.DataFrame({"LandLordID": ["L2"], "CompanyID": ["C1"]})
        with pytest.raises(ValueError, match="CRITICAL"):
            validate_all(lls, cos, br, strict=True)

    def test_report_summary_string(self, clean_landlords, clean_companies, clean_bridge):
        report = validate_all(clean_landlords, clean_companies, clean_bridge)
        assert "Quality Report:" in report.summary()
