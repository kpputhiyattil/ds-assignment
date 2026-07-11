"""
Integration test for CreditAgent with a mocked LLM.

No real OpenAI calls are made — the LangGraph graph is patched at the
invoke level to return a deterministic JSON verdict payload.

Tests verify end-to-end wiring: agent invocation → tool output extraction
→ JSON parsing → guardrail application → AssessmentResult.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent.credit_agent import CreditAgent, _parse_verdict_json
from agent.models import AssessmentResult, Confidence, Recommendation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_tool_message(name: str, content: dict) -> ToolMessage:
    return ToolMessage(content=json.dumps(content), name=name, tool_call_id=f"id_{name}")


def _make_graph_result(recommendation: str, confidence: str, rationale: str) -> dict:
    """Build a fake graph result with three ToolMessages and a final AIMessage."""
    verdict_json = json.dumps({
        "recommendation": recommendation,
        "confidence": confidence,
        "rationale": rationale,
    })
    return {
        "messages": [
            HumanMessage(content="assess this company"),
            _make_tool_message("compute_financial_health", {
                "revenue_estimate": 60000.0,
                "budget_utilisation_ratio": 1.5,
                "revenue_per_investment": 0.3,
                "sales_mix_ratio": 0.83,
                "data_quality_issues": [],
            }),
            _make_tool_message("assess_client_engagement", {
                "retention_rate": 0.75,
                "active_client_ratio": 3.0,
                "client_growth_rate_6m": 0.12,
                "client_growth_rate_12m": -0.11,
                "conversion_rate_7d": 0.04,
                "visitor_trend_ratio": 5.78,
                "data_quality_issues": [],
            }),
            _make_tool_message("check_business_profile", {
                "status_flag": "active",
                "business_type_risk_tier": "low",
                "company_age_years": 14,
                "rank_band": "strong",
                "country": "US",
                "data_quality_issues": [],
            }),
            AIMessage(content=verdict_json),
        ]
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def agent(monkeypatch) -> CreditAgent:
    """
    CreditAgent with mocked LLM credentials so Settings validation passes,
    and the graph is NOT invoked in the constructor.
    """
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test-fake-key")

    with patch("agent.credit_agent.build_llm"), \
         patch("agent.credit_agent.create_react_agent") as mock_graph_factory, \
         patch("agent.credit_agent.get_langfuse_handler", return_value=None):

        mock_graph_factory.return_value = MagicMock()
        a = CreditAgent()

    return a


# ---------------------------------------------------------------------------
# Tests: happy path
# ---------------------------------------------------------------------------

class TestHappyPath:
    def test_continue_verdict_returned(self, agent, healthy_company_profile):
        graph_result = _make_graph_result("Continue", "high", "Strong metrics across all signals.")
        agent._graph.invoke = MagicMock(return_value=graph_result)

        result = agent.assess(healthy_company_profile)

        assert isinstance(result, AssessmentResult)
        assert result.recommendation == Recommendation.CONTINUE
        assert result.confidence == Confidence.HIGH
        assert result.guardrail_fired is False
        assert result.error is None

    def test_tool_outputs_populated(self, agent, healthy_company_profile):
        graph_result = _make_graph_result("Continue", "high", "Good company.")
        agent._graph.invoke = MagicMock(return_value=graph_result)

        result = agent.assess(healthy_company_profile)

        assert "financial_health" in result.tool_outputs
        assert "client_engagement" in result.tool_outputs
        assert "business_profile" in result.tool_outputs
        assert result.tool_outputs["financial_health"]["budget_utilisation_ratio"] == 1.5

    def test_steps_populated(self, agent, healthy_company_profile):
        graph_result = _make_graph_result("Continue", "high", "Good company.")
        agent._graph.invoke = MagicMock(return_value=graph_result)

        result = agent.assess(healthy_company_profile)
        assert len(result.steps) > 0


# ---------------------------------------------------------------------------
# Tests: guardrail integration via agent
# ---------------------------------------------------------------------------

class TestGuardrailIntegration:
    def test_llm_continue_overridden_for_closed_company(self, agent, closed_company_profile):
        """LLM says Continue but status_flag=closed → guardrail fires → No-Go."""
        # Patch the business_profile tool output to return closed status
        graph_result = _make_graph_result("Continue", "high", "Looks OK.")
        # Override the business_profile tool message
        graph_result["messages"][3] = _make_tool_message("check_business_profile", {
            "status_flag": "closed",
            "business_type_risk_tier": "low",
            "company_age_years": 14,
            "rank_band": "strong",
            "country": "US",
            "data_quality_issues": [],
        })
        agent._graph.invoke = MagicMock(return_value=graph_result)

        result = agent.assess(closed_company_profile)

        assert result.recommendation == Recommendation.NO_GO
        assert result.guardrail_fired is True
        assert "company_status_hard_stop" in result.guardrail_reason

    def test_budget_burn_guardrail_overrides_llm(self, agent, insolvent_company_profile):
        """LLM says Review but budget_utilisation_ratio=0.03 → guardrail fires → No-Go."""
        graph_result = _make_graph_result("Review", "medium", "Some concerns.")
        graph_result["messages"][1] = _make_tool_message("compute_financial_health", {
            "revenue_estimate": 1500.0,
            "budget_utilisation_ratio": 0.03,  # below threshold
            "revenue_per_investment": 0.003,
            "sales_mix_ratio": 0.67,
            "data_quality_issues": [],
        })
        agent._graph.invoke = MagicMock(return_value=graph_result)

        result = agent.assess(insolvent_company_profile)

        assert result.recommendation == Recommendation.NO_GO
        assert result.guardrail_fired is True
        assert "extreme_budget_burn" in result.guardrail_reason


# ---------------------------------------------------------------------------
# Tests: error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    def test_llm_exception_returns_error_result(self, agent, healthy_company_profile):
        agent._graph.invoke = MagicMock(side_effect=RuntimeError("API unavailable"))

        result = agent.assess(healthy_company_profile)

        assert result.is_error()
        assert result.recommendation == Recommendation.NO_GO
        assert result.confidence == Confidence.LOW
        assert "API unavailable" in result.error

    def test_malformed_json_returns_error_result(self, agent, healthy_company_profile):
        bad_result = {
            "messages": [
                HumanMessage(content="assess"),
                AIMessage(content="this is not json at all, sorry!"),
            ]
        }
        agent._graph.invoke = MagicMock(return_value=bad_result)

        result = agent.assess(healthy_company_profile)

        assert result.is_error()

    def test_empty_messages_returns_error_result(self, agent, healthy_company_profile):
        agent._graph.invoke = MagicMock(return_value={"messages": []})

        result = agent.assess(healthy_company_profile)

        assert result.is_error()


# ---------------------------------------------------------------------------
# Tests: JSON parser unit tests
# ---------------------------------------------------------------------------

class TestParseVerdictJson:
    def test_pure_json(self):
        d = _parse_verdict_json('{"recommendation": "Continue", "confidence": "high", "rationale": "Good."}')
        assert d["recommendation"] == "Continue"

    def test_json_with_preamble(self):
        content = 'Here is my verdict:\n{"recommendation": "No-Go", "confidence": "high", "rationale": "Bad."}\n'
        d = _parse_verdict_json(content)
        assert d["recommendation"] == "No-Go"

    def test_nested_json_extracts_outer(self):
        content = '{"recommendation": "Review", "confidence": "medium", "rationale": "Mixed."}'
        d = _parse_verdict_json(content)
        assert d["confidence"] == "medium"

    def test_unparseable_raises_value_error(self):
        with pytest.raises(ValueError):
            _parse_verdict_json("I cannot provide a verdict in JSON format.")
