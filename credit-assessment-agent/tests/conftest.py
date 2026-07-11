"""
Shared pytest fixtures for the credit assessment agent test suite.

Fixture scopes
--------------
- session-scoped: expensive objects reused across all tests (none currently)
- function-scoped (default): reset per test for isolation

Environment
-----------
Tests run without a real LLM or Langfuse connection.
Settings are patched via monkeypatch + get_settings.cache_clear().
"""

from __future__ import annotations

import pytest

from agent.config import get_settings
from agent.models import AssessmentResult, Confidence, Recommendation


# ---------------------------------------------------------------------------
# Settings isolation
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clear_settings_cache():
    """
    Clear the lru_cache on get_settings before and after every test.

    This ensures that monkeypatched env vars take effect in Settings
    and don't bleed between tests.
    """
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


# ---------------------------------------------------------------------------
# Company profile fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def healthy_company_profile() -> dict:
    """A financially healthy, active company with strong engagement."""
    return {
        "CompanyID": "COMP_HEALTHY_001",
        "CompanyStatus": "active",
        "PrimaryType": "Retail",
        "YearFounded": 2015,
        "OriginCountry": "US",
        "Rank": 4.5,
        "SalesMain": 50000.0,
        "SalesOther": 10000.0,
        "MonthlyBudget": 40000.0,
        "Investments": 200000.0,
        "TodaysClients": 120,
        "ReturningClients": 90,
        "TotalActiveClients": 400,
        "Clients7D": 25,
        "Clients6M": 350,
        "Clients12M": 600,
        "Visitors7D": 800,
        "Visitors6M": 4000,
        "Visitors12M": 8500,
    }


@pytest.fixture
def closed_company_profile() -> dict:
    """A closed company — should always trigger the status guardrail."""
    return {
        "CompanyID": "COMP_CLOSED_001",
        "CompanyStatus": "closed",
        "PrimaryType": "Retail",
        "YearFounded": 2010,
        "OriginCountry": "US",
        "Rank": 4.0,
        "SalesMain": 30000.0,
        "SalesOther": 5000.0,
        "MonthlyBudget": 20000.0,
        "Investments": 100000.0,
        "TodaysClients": 50,
        "ReturningClients": 40,
        "TotalActiveClients": 150,
        "Clients7D": 10,
        "Clients6M": 120,
        "Clients12M": 200,
        "Visitors7D": 300,
        "Visitors6M": 1500,
        "Visitors12M": 3200,
    }


@pytest.fixture
def insolvent_company_profile() -> dict:
    """Revenue covers <10% of budget — should trigger budget burn guardrail."""
    return {
        "CompanyID": "COMP_INSOLVENT_001",
        "CompanyStatus": "active",
        "PrimaryType": "Technology",
        "YearFounded": 2020,
        "OriginCountry": "US",
        "Rank": 3.5,
        "SalesMain": 1000.0,    # very low revenue
        "SalesOther": 500.0,
        "MonthlyBudget": 50000.0,  # high budget → ratio = 0.03
        "Investments": 500000.0,
        "TodaysClients": 10,
        "ReturningClients": 8,
        "TotalActiveClients": 30,
        "Clients7D": 2,
        "Clients6M": 25,
        "Clients12M": 40,
        "Visitors7D": 100,
        "Visitors6M": 500,
        "Visitors12M": 1000,
    }


@pytest.fixture
def null_heavy_profile() -> dict:
    """Profile with many None values — tests null-safe computation paths."""
    return {
        "CompanyID": "COMP_NULLS_001",
        "CompanyStatus": "active",
        "PrimaryType": None,
        "YearFounded": None,
        "OriginCountry": None,
        "Rank": None,
        "SalesMain": None,
        "SalesOther": None,
        "MonthlyBudget": None,
        "Investments": None,
        "TodaysClients": None,
        "ReturningClients": None,
        "TotalActiveClients": None,
        "Clients7D": None,
        "Clients6M": None,
        "Clients12M": None,
        "Visitors7D": None,
        "Visitors6M": None,
        "Visitors12M": None,
    }


# ---------------------------------------------------------------------------
# AssessmentResult fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def continue_verdict() -> AssessmentResult:
    return AssessmentResult(
        recommendation=Recommendation.CONTINUE,
        confidence=Confidence.HIGH,
        rationale="Strong financials and engagement.",
    )


@pytest.fixture
def review_verdict() -> AssessmentResult:
    return AssessmentResult(
        recommendation=Recommendation.REVIEW,
        confidence=Confidence.MEDIUM,
        rationale="Mixed signals — recommend manual review.",
    )
