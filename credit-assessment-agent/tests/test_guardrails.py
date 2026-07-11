"""Unit tests for agent/guardrails.py."""

import pytest

from agent.guardrails import BUDGET_BURN_THRESHOLD, NON_ACTIVE_STATUSES, GuardrailEngine
from agent.models import AssessmentResult, Confidence, Recommendation


@pytest.fixture
def engine() -> GuardrailEngine:
    return GuardrailEngine()


@pytest.fixture
def continue_verdict() -> AssessmentResult:
    return AssessmentResult(
        recommendation=Recommendation.CONTINUE,
        confidence=Confidence.HIGH,
        rationale="Strong signals across all tools.",
    )


def _tool_outputs(status_flag="active", budget_ratio=1.5) -> dict:
    """Helper to build minimal tool_outputs dict."""
    return {
        "business_profile": {"status_flag": status_flag},
        "financial_health": {"budget_utilisation_ratio": budget_ratio},
        "client_engagement": {},
    }


class TestCompanyStatusRule:
    @pytest.mark.parametrize("status", ["inactive", "suspended", "closed"])
    def test_fires_for_non_active_statuses(self, engine, continue_verdict, status):
        outputs = _tool_outputs(status_flag=status)
        result = engine.run(outputs, continue_verdict)

        assert result.guardrail_fired is True
        assert result.recommendation == Recommendation.NO_GO
        assert result.confidence == Confidence.HIGH
        assert "company_status_hard_stop" in result.guardrail_reason
        assert status in result.guardrail_reason

    def test_does_not_fire_for_active(self, engine, continue_verdict):
        outputs = _tool_outputs(status_flag="active")
        result = engine.run(outputs, continue_verdict)
        assert result.guardrail_fired is False
        assert result.recommendation == Recommendation.CONTINUE

    def test_does_not_fire_for_unknown_status(self, engine, continue_verdict):
        """Unknown status is a warning, not a guardrail fire."""
        outputs = _tool_outputs(status_flag="unknown")
        result = engine.run(outputs, continue_verdict)
        assert result.guardrail_fired is False


class TestBudgetBurnRule:
    def test_fires_below_threshold(self, engine, continue_verdict):
        outputs = _tool_outputs(budget_ratio=BUDGET_BURN_THRESHOLD - 0.01)
        result = engine.run(outputs, continue_verdict)

        assert result.guardrail_fired is True
        assert result.recommendation == Recommendation.NO_GO
        assert "extreme_budget_burn" in result.guardrail_reason

    def test_fires_at_near_zero_ratio(self, engine, continue_verdict):
        outputs = _tool_outputs(budget_ratio=0.01)
        result = engine.run(outputs, continue_verdict)
        assert result.guardrail_fired is True

    def test_does_not_fire_at_threshold(self, engine, continue_verdict):
        outputs = _tool_outputs(budget_ratio=BUDGET_BURN_THRESHOLD)
        result = engine.run(outputs, continue_verdict)
        assert result.guardrail_fired is False

    def test_does_not_fire_above_threshold(self, engine, continue_verdict):
        outputs = _tool_outputs(budget_ratio=1.5)
        result = engine.run(outputs, continue_verdict)
        assert result.guardrail_fired is False

    def test_does_not_fire_when_ratio_is_none(self, engine, continue_verdict):
        """Missing data should not trigger the guardrail."""
        outputs = {
            "business_profile": {"status_flag": "active"},
            "financial_health": {"budget_utilisation_ratio": None},
        }
        result = engine.run(outputs, continue_verdict)
        assert result.guardrail_fired is False


class TestRulePriority:
    def test_status_rule_wins_over_budget_rule(self, engine, continue_verdict):
        """Both rules trigger — status rule fires first and wins."""
        outputs = _tool_outputs(status_flag="closed", budget_ratio=0.01)
        result = engine.run(outputs, continue_verdict)

        assert result.guardrail_fired is True
        assert "company_status_hard_stop" in result.guardrail_reason

    def test_budget_rule_fires_when_status_is_active(self, engine, continue_verdict):
        outputs = _tool_outputs(status_flag="active", budget_ratio=0.05)
        result = engine.run(outputs, continue_verdict)

        assert result.guardrail_fired is True
        assert "extreme_budget_burn" in result.guardrail_reason


class TestOverrideStructure:
    def test_override_preserves_original_rationale(self, engine, continue_verdict):
        outputs = _tool_outputs(status_flag="closed")
        result = engine.run(outputs, continue_verdict)

        assert "Strong signals across all tools." in result.rationale
        assert "[GUARDRAIL OVERRIDE" in result.rationale

    def test_override_sets_high_confidence(self, engine, continue_verdict):
        outputs = _tool_outputs(status_flag="closed")
        result = engine.run(outputs, continue_verdict)
        assert result.confidence == Confidence.HIGH

    def test_no_override_returns_original_verdict(self, engine, continue_verdict):
        outputs = _tool_outputs(status_flag="active", budget_ratio=2.0)
        result = engine.run(outputs, continue_verdict)

        assert result.recommendation == Recommendation.CONTINUE
        assert result.confidence == Confidence.HIGH
        assert result.guardrail_fired is False
        assert result.guardrail_reason == ""


class TestRuleRobustness:
    def test_empty_tool_outputs_does_not_crash(self, engine, continue_verdict):
        """Missing tool_output keys should not raise exceptions."""
        result = engine.run({}, continue_verdict)
        # status_flag missing → treated as active; ratio missing → no fire
        assert result.guardrail_fired is False

    def test_missing_financial_health_key(self, engine, continue_verdict):
        outputs = {"business_profile": {"status_flag": "active"}}
        result = engine.run(outputs, continue_verdict)
        assert result.guardrail_fired is False
