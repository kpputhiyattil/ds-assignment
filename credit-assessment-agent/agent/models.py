"""
Data models for the credit assessment agent.

All inter-component contracts are typed here — the agent, guardrails,
tools, and UI all import from this module, never from each other.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class Recommendation(str, Enum):
    """Credit assessment recommendation."""

    CONTINUE = "Continue"
    REVIEW = "Review"
    NO_GO = "No-Go"


class Confidence(str, Enum):
    """Confidence level of the assessment."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class GuardrailResult:
    """Result of a single guardrail rule evaluation."""

    fired: bool
    rule_name: str
    reason: str


@dataclass
class AssessmentResult:
    """
    Structured output of a full credit assessment.

    Fields
    ------
    recommendation  : Final recommendation (may be overridden by a guardrail).
    confidence      : Confidence level in the recommendation.
    rationale       : Natural-language explanation (<200 words).
    tool_outputs    : Raw outputs from all three tools, keyed by tool name.
    guardrail_fired : True if any hard guardrail overrode the LLM verdict.
    guardrail_reason: Human-readable description of the guardrail that fired.
    steps           : LangGraph message trace (ToolMessage list) for UI display.
    error           : Set if the agent encountered a non-recoverable error.
    """

    recommendation: Recommendation
    confidence: Confidence
    rationale: str
    tool_outputs: dict[str, Any] = field(default_factory=dict)
    guardrail_fired: bool = False
    guardrail_reason: str = ""
    steps: list[Any] = field(default_factory=list)
    error: str | None = None

    def is_error(self) -> bool:
        return self.error is not None

    @classmethod
    def error_result(cls, message: str) -> "AssessmentResult":
        """Factory for a failed assessment (agent error, timeout, etc.)."""
        return cls(
            recommendation=Recommendation.NO_GO,
            confidence=Confidence.LOW,
            rationale=f"Assessment could not be completed: {message}",
            error=message,
        )
