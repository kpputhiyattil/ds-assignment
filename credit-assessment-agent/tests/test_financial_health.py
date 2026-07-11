"""Unit tests for agent/tools/financial_health.py."""

import pytest

from agent.tools.financial_health import compute_financial_health


def invoke(kwargs: dict) -> dict:
    """Helper: invoke the LangChain tool with a dict of arguments."""
    return compute_financial_health.invoke(kwargs)


class TestRevenueEstimate:
    def test_both_sales_fields(self):
        r = invoke({"sales_main": 10000, "sales_other": 2000,
                    "monthly_budget": 8000, "investments": 50000})
        assert r["revenue_estimate"] == 12000.0

    def test_zero_sales_other(self):
        r = invoke({"sales_main": 5000, "sales_other": 0,
                    "monthly_budget": 4000, "investments": 20000})
        assert r["revenue_estimate"] == 5000.0

    def test_null_sales_other_uses_main(self):
        r = invoke({"sales_main": 7000, "sales_other": None,
                    "monthly_budget": 5000, "investments": 30000})
        assert r["revenue_estimate"] == 7000.0

    def test_both_null_returns_none(self):
        r = invoke({"sales_main": None, "sales_other": None,
                    "monthly_budget": 5000, "investments": 30000})
        assert r["revenue_estimate"] is None


class TestBudgetUtilisationRatio:
    def test_revenue_covers_budget(self):
        r = invoke({"sales_main": 10000, "sales_other": 0,
                    "monthly_budget": 8000, "investments": 1})
        assert r["budget_utilisation_ratio"] == pytest.approx(1.25, rel=1e-3)

    def test_revenue_below_budget(self):
        r = invoke({"sales_main": 1000, "sales_other": 0,
                    "monthly_budget": 50000, "investments": 1})
        assert r["budget_utilisation_ratio"] == pytest.approx(0.02, rel=1e-3)

    def test_zero_budget_returns_none(self):
        r = invoke({"sales_main": 5000, "sales_other": 0,
                    "monthly_budget": 0, "investments": 1})
        assert r["budget_utilisation_ratio"] is None
        assert any("zero: monthly_budget" in i for i in r["data_quality_issues"])

    def test_null_budget_returns_none(self):
        r = invoke({"sales_main": 5000, "sales_other": 0,
                    "monthly_budget": None, "investments": 1})
        assert r["budget_utilisation_ratio"] is None


class TestRevenuePerInvestment:
    def test_normal(self):
        r = invoke({"sales_main": 10000, "sales_other": 0,
                    "monthly_budget": 8000, "investments": 100000})
        assert r["revenue_per_investment"] == pytest.approx(0.1, rel=1e-3)

    def test_zero_investments_returns_none(self):
        r = invoke({"sales_main": 5000, "sales_other": 0,
                    "monthly_budget": 4000, "investments": 0})
        assert r["revenue_per_investment"] is None

    def test_null_investments_returns_none(self):
        r = invoke({"sales_main": 5000, "sales_other": 0,
                    "monthly_budget": 4000, "investments": None})
        assert r["revenue_per_investment"] is None


class TestSalesMixRatio:
    def test_pure_main_sales(self):
        r = invoke({"sales_main": 10000, "sales_other": 0,
                    "monthly_budget": 8000, "investments": 1})
        assert r["sales_mix_ratio"] == pytest.approx(1.0, rel=1e-3)

    def test_mixed_sales(self):
        r = invoke({"sales_main": 7000, "sales_other": 3000,
                    "monthly_budget": 8000, "investments": 1})
        assert r["sales_mix_ratio"] == pytest.approx(0.7, rel=1e-3)

    def test_null_main_returns_none(self):
        r = invoke({"sales_main": None, "sales_other": 3000,
                    "monthly_budget": 8000, "investments": 1})
        assert r["sales_mix_ratio"] is None


class TestDataQualityIssues:
    def test_no_issues_on_complete_data(self):
        r = invoke({"sales_main": 5000, "sales_other": 1000,
                    "monthly_budget": 4000, "investments": 20000})
        assert r["data_quality_issues"] == []

    def test_multiple_missing_fields_reported(self):
        r = invoke({"sales_main": None, "sales_other": None,
                    "monthly_budget": None, "investments": None})
        issues = r["data_quality_issues"]
        assert any("sales_main" in i for i in issues)
        assert any("sales_other" in i for i in issues)
        assert any("monthly_budget" in i for i in issues)
        assert any("investments" in i for i in issues)
