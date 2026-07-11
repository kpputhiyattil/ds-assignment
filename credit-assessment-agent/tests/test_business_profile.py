"""Unit tests for agent/tools/business_profile.py."""

import pytest

from agent.tools.business_profile import (
    BUSINESS_TYPE_RISK_MAP,
    STATUS_NORMALISATION,
    check_business_profile,
)


def invoke(kwargs: dict) -> dict:
    return check_business_profile.invoke(kwargs)


_BASE = {
    "company_status": "active",
    "primary_type": "Retail",
    "year_founded": 2010,
    "origin_country": "US",
    "rank": 4.2,
}


class TestStatusNormalisation:
    @pytest.mark.parametrize("raw,expected", [
        ("active", "active"),
        ("Active", "active"),
        ("ACTIVE", "active"),
        ("open", "active"),
        ("Actively Seeking New Employees", "active"),
        ("Will Consider New Projects", "active"),
        ("Acquired/Merged (Operating Subsidiary)", "active"),
        ("inactive", "inactive"),
        ("Inactive", "inactive"),
        ("dormant", "inactive"),
        ("Not Making New Products", "inactive"),
        ("Reducing Activity", "inactive"),
        ("Acquired/Merged", "inactive"),
        ("suspended", "suspended"),
        ("Suspended", "suspended"),
        ("closed", "closed"),
        ("Closed", "closed"),
        ("dissolved", "closed"),
        ("liquidated", "closed"),
        ("bankrupt", "closed"),
        ("Out of Business", "closed"),
    ])
    def test_known_statuses(self, raw, expected):
        r = invoke({**_BASE, "company_status": raw})
        assert r["status_flag"] == expected

    def test_unknown_status_returns_unknown(self):
        r = invoke({**_BASE, "company_status": "weird_value"})
        assert r["status_flag"] == "unknown"
        assert any("unknown_status_value" in i for i in r["data_quality_issues"])

    def test_null_status_returns_unknown(self):
        r = invoke({**_BASE, "company_status": None})
        assert r["status_flag"] == "unknown"
        assert any("missing: company_status" in i for i in r["data_quality_issues"])

    def test_empty_status_returns_unknown(self):
        r = invoke({**_BASE, "company_status": ""})
        assert r["status_flag"] == "unknown"
        assert any("missing: company_status" in i for i in r["data_quality_issues"])


class TestRiskTier:
    @pytest.mark.parametrize("primary_type,expected_tier", [
        ("Retail", "low"),
        ("Healthcare", "low"),
        ("Technology", "medium"),
        ("E-commerce", "medium"),
        ("Crypto", "high"),
        ("Gambling", "high"),
    ])
    def test_known_types(self, primary_type, expected_tier):
        r = invoke({**_BASE, "primary_type": primary_type})
        assert r["business_type_risk_tier"] == expected_tier

    def test_unknown_type_defaults_to_medium(self):
        r = invoke({**_BASE, "primary_type": "QuantumWidgets"})
        assert r["business_type_risk_tier"] == "medium"
        assert any("unknown_primary_type" in i for i in r["data_quality_issues"])

    def test_null_type_defaults_to_medium(self):
        r = invoke({**_BASE, "primary_type": None})
        assert r["business_type_risk_tier"] == "medium"
        assert any("missing: primary_type" in i for i in r["data_quality_issues"])


class TestCompanyAge:
    def test_age_computed(self):
        from datetime import datetime
        r = invoke({**_BASE, "year_founded": 2010})
        expected = datetime.now().year - 2010
        assert r["company_age_years"] == expected

    def test_null_year_founded(self):
        r = invoke({**_BASE, "year_founded": None})
        assert r["company_age_years"] is None
        assert any("missing: year_founded" in i for i in r["data_quality_issues"])

    def test_future_year_returns_none(self):
        from datetime import datetime
        future_year = datetime.now().year + 5
        r = invoke({**_BASE, "year_founded": future_year})
        assert r["company_age_years"] is None
        assert any("invalid: year_founded" in i for i in r["data_quality_issues"])


class TestRankBand:
    @pytest.mark.parametrize("rank,expected_band", [
        (4.5, "strong"),
        (4.01, "strong"),
        (4.0, "acceptable"),
        (3.5, "acceptable"),
        (3.0, "acceptable"),
        (2.9, "weak"),
        (1.0, "weak"),
    ])
    def test_rank_bands(self, rank, expected_band):
        r = invoke({**_BASE, "rank": rank})
        assert r["rank_band"] == expected_band

    def test_null_rank(self):
        r = invoke({**_BASE, "rank": None})
        assert r["rank_band"] == "unknown"


class TestCleanProfile:
    def test_no_issues_on_complete_known_data(self):
        r = invoke(_BASE)
        assert r["data_quality_issues"] == []
        assert r["country"] == "US"
        assert r["status_flag"] == "active"
        assert r["business_type_risk_tier"] == "low"
        assert r["rank_band"] == "strong"
