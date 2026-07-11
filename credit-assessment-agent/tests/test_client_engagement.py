"""Unit tests for agent/tools/client_engagement.py."""

import pytest

from agent.tools.client_engagement import assess_client_engagement

_FULL = {
    "todays_clients": 100,
    "returning_clients": 75,
    "total_active_clients": 300,
    "clients_7d": 20,
    "clients_6m": 180,
    "clients_12m": 320,
    "visitors_7d": 500,
    "visitors_6m": 2000,
    "visitors_12m": 4500,
}


def invoke(kwargs: dict) -> dict:
    return assess_client_engagement.invoke(kwargs)


class TestRetentionRate:
    def test_normal(self):
        r = invoke(_FULL)
        assert r["retention_rate"] == pytest.approx(0.75, rel=1e-3)

    def test_full_retention(self):
        d = {**_FULL, "returning_clients": 100}
        assert invoke(d)["retention_rate"] == pytest.approx(1.0, rel=1e-3)

    def test_zero_todays_clients(self):
        d = {**_FULL, "todays_clients": 0}
        assert invoke(d)["retention_rate"] is None

    def test_null_todays_clients(self):
        d = {**_FULL, "todays_clients": None}
        assert invoke(d)["retention_rate"] is None


class TestActiveClientRatio:
    def test_normal(self):
        r = invoke(_FULL)
        assert r["active_client_ratio"] == pytest.approx(3.0, rel=1e-3)

    def test_null_active_clients(self):
        d = {**_FULL, "total_active_clients": None}
        assert invoke(d)["active_client_ratio"] is None


class TestClientGrowthRates:
    def test_6m_growth_positive(self):
        # clients_12m=320, baseline=160; clients_6m=180 → growth = (180-160)/160 = 0.125
        r = invoke(_FULL)
        assert r["client_growth_rate_6m"] == pytest.approx(0.125, rel=1e-3)

    def test_12m_growth(self):
        # expected_annual = 180*2=360; clients_12m=320 → (320-360)/360 = -0.111
        r = invoke(_FULL)
        assert r["client_growth_rate_12m"] == pytest.approx(-0.1111, rel=1e-2)

    def test_null_clients_12m_returns_none(self):
        d = {**_FULL, "clients_12m": None}
        assert invoke(d)["client_growth_rate_6m"] is None


class TestConversionRate:
    def test_normal(self):
        r = invoke(_FULL)
        # 20 / 500 = 0.04
        assert r["conversion_rate_7d"] == pytest.approx(0.04, rel=1e-3)

    def test_zero_visitors(self):
        d = {**_FULL, "visitors_7d": 0}
        assert invoke(d)["conversion_rate_7d"] is None

    def test_null_visitors(self):
        d = {**_FULL, "visitors_7d": None}
        assert invoke(d)["conversion_rate_7d"] is None


class TestVisitorTrendRatio:
    def test_accelerating_traffic(self):
        # annualised_weekly = 500*52=26000; visitors_12m=4500 → ratio > 1
        r = invoke(_FULL)
        expected = round(500 * 52 / 4500, 4)
        assert r["visitor_trend_ratio"] == pytest.approx(expected, rel=1e-3)

    def test_null_visitors_12m(self):
        d = {**_FULL, "visitors_12m": None}
        assert invoke(d)["visitor_trend_ratio"] is None


class TestNullHeavyProfile:
    def test_all_nulls_returns_none_metrics(self):
        all_null = {k: None for k in _FULL}
        r = invoke(all_null)
        for key in [
            "retention_rate", "active_client_ratio",
            "client_growth_rate_6m", "client_growth_rate_12m",
            "conversion_rate_7d", "visitor_trend_ratio",
        ]:
            assert r[key] is None, f"Expected None for {key}"
        assert len(r["data_quality_issues"]) > 0
