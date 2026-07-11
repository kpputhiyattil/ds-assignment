"""Tests for Continue calibration (promote Review → Continue when eligible)."""

from agent.credit_agent import _promote_continue_if_eligible
from agent.models import AssessmentResult, Confidence, Recommendation


def _assessment(rec: Recommendation = Recommendation.REVIEW) -> AssessmentResult:
    return AssessmentResult(
        recommendation=rec,
        confidence=Confidence.MEDIUM,
        rationale="LLM said Review.",
    )


def test_promotes_strong_active_company():
    tools = {
        "business_profile": {"status_flag": "active"},
        "financial_health": {"budget_utilisation_ratio": 5.9},
        "client_engagement": {"retention_rate": 0.47},
    }
    out = _promote_continue_if_eligible(_assessment(), tools)
    assert out.recommendation == Recommendation.CONTINUE
    assert "CONTINUE CALIBRATION" in out.rationale


def test_does_not_promote_weak_util():
    tools = {
        "business_profile": {"status_flag": "active"},
        "financial_health": {"budget_utilisation_ratio": 0.05},
        "client_engagement": {"retention_rate": 0.5},
    }
    out = _promote_continue_if_eligible(_assessment(), tools)
    assert out.recommendation == Recommendation.REVIEW


def test_does_not_promote_closed():
    tools = {
        "business_profile": {"status_flag": "closed"},
        "financial_health": {"budget_utilisation_ratio": 2.0},
        "client_engagement": {"retention_rate": 0.5},
    }
    out = _promote_continue_if_eligible(_assessment(), tools)
    assert out.recommendation == Recommendation.REVIEW


def test_leaves_continue_unchanged():
    tools = {
        "business_profile": {"status_flag": "active"},
        "financial_health": {"budget_utilisation_ratio": 2.0},
        "client_engagement": {"retention_rate": 0.5},
    }
    out = _promote_continue_if_eligible(_assessment(Recommendation.CONTINUE), tools)
    assert out.recommendation == Recommendation.CONTINUE
    assert "CONTINUE CALIBRATION" not in out.rationale
