"""
Guardrail Engine

Applies hard rule-based checks after tool outputs are collected.
Rules are deterministic, LLM-agnostic, and run in priority order —
the first rule that fires wins and overrides the LLM verdict.

Design principles
-----------------
- Rules read from tool_outputs, never from LLM rationale text
- A rule failure (exception) is caught and logged, never silently swallowed
- The threshold constants are module-level so they are easy to tune or
  override in tests without monkey-patching

Current rules (in priority order)
----------------------------------
1. Company Status Hard Stop  — inactive / suspended / closed → No-Go
2. Extreme Budget Burn       — budget_utilisation_ratio < BUDGET_BURN_THRESHOLD → No-Go

Threshold note (budget burn)
----------------------------
BUDGET_BURN_THRESHOLD = 0.10 (10%). Among rows with MonthlyBudget > 0, util is
heavily left-skewed (median near ~1%). Cuts at 30/40/50% would flag ~92–94% of
that subset and are too blunt. 10% is a catastrophic-coverage hard stop that
still fires often enough to be a meaningful financial guardrail for the
assignment, while missing ratios do not fire.
"""

from __future__ import annotations

import logging
from typing import Any

from agent.models import AssessmentResult, Confidence, GuardrailResult, Recommendation

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tunable thresholds
# ---------------------------------------------------------------------------

# Statuses that trigger an immediate No-Go regardless of financials
NON_ACTIVE_STATUSES: frozenset[str] = frozenset({"inactive", "suspended", "closed"})

# Hard financial guardrail (assignment requirement).
# Revenue must cover at least this fraction of monthly budget.
BUDGET_BURN_THRESHOLD: float = 0.10


class GuardrailEngine:
    """
    Evaluates hard rules against tool outputs and overwrites the LLM verdict
    if any rule fires.

    Usage
    -----
    engine = GuardrailEngine()
    final_result = engine.run(tool_outputs, llm_verdict)
    """

    def run(
        self,
        tool_outputs: dict[str, Any],
        verdict: AssessmentResult,
    ) -> AssessmentResult:
        """
        Run all guardrail rules in priority order.

        The first rule that fires overwrites the verdict and returns immediately
        (subsequent rules are not evaluated — one override is sufficient).

        Parameters
        ----------
        tool_outputs : Combined dict of all tool results, keyed by tool name:
                       "financial_health", "client_engagement", "business_profile"
        verdict      : The AssessmentResult produced by the LLM agent.

        Returns
        -------
        AssessmentResult — either the original verdict (no rule fired) or an
        overridden No-Go verdict with guardrail metadata set.
        """
        rules = [
            self._check_company_status,
            self._check_budget_burn,
        ]

        for rule in rules:
            try:
                result: GuardrailResult = rule(tool_outputs)
            except Exception as exc:
                # A broken rule must never silently pass — log and continue
                logger.error(
                    "Guardrail rule '%s' raised an unexpected error: %s",
                    rule.__name__,
                    exc,
                    exc_info=True,
                )
                continue

            if result.fired:
                logger.info(
                    "Guardrail fired — rule='%s' reason='%s' "
                    "original_recommendation='%s'",
                    result.rule_name,
                    result.reason,
                    verdict.recommendation.value,
                )
                return self._override(verdict, result)

        # No rule fired — return the LLM verdict unchanged
        return verdict

    # ------------------------------------------------------------------
    # Rule implementations
    # ------------------------------------------------------------------

    def _check_company_status(self, tool_outputs: dict[str, Any]) -> GuardrailResult:
        """
        Rule 1: Company Status Hard Stop

        A company that is not operationally active cannot service growth capital.
        No financial or engagement metric can override a fundamental inability to
        operate. This is a binary rule — 'inactive' is not a lesser form of 'active'.

        Trigger: status_flag in {inactive, suspended, closed}
        """
        profile = tool_outputs.get("business_profile", {})
        status_flag = profile.get("status_flag", "unknown")

        if status_flag in NON_ACTIVE_STATUSES:
            return GuardrailResult(
                fired=True,
                rule_name="company_status_hard_stop",
                reason=(
                    f"Company status is '{status_flag}'. "
                    "A non-active company cannot service or repay growth capital. "
                    "This rule fires regardless of financial or engagement metrics."
                ),
            )

        if status_flag == "unknown":
            # Unknown status is conservative — we cannot confirm the company is active
            logger.warning(
                "Company status is 'unknown' — could not confirm operational status. "
                "Consider treating this as a Review case."
            )

        return GuardrailResult(fired=False, rule_name="company_status_hard_stop", reason="")

    def _check_budget_burn(self, tool_outputs: dict[str, Any]) -> GuardrailResult:
        """
        Rule 2: Extreme Budget Burn

        If revenue covers less than BUDGET_BURN_THRESHOLD (10%) of the monthly
        budget, the company is structurally insolvent for lending purposes.
        Strong client engagement cannot overcome catastrophic unit economics.

        Trigger: budget_utilisation_ratio < 0.10
        Note: if the ratio is None (missing data), the rule does NOT fire —
        we do not penalise companies for missing fields.
        """
        financials = tool_outputs.get("financial_health", {})
        ratio = financials.get("budget_utilisation_ratio")

        if ratio is None:
            # Missing data — cannot evaluate; do not fire
            return GuardrailResult(fired=False, rule_name="extreme_budget_burn", reason="")

        if ratio < BUDGET_BURN_THRESHOLD:
            pct = round(ratio * 100, 1)
            return GuardrailResult(
                fired=True,
                rule_name="extreme_budget_burn",
                reason=(
                    f"Revenue covers only {pct}% of monthly budget "
                    f"(threshold: {int(BUDGET_BURN_THRESHOLD * 100)}%). "
                    "The company is structurally insolvent — it cannot repay growth capital."
                ),
            )

        return GuardrailResult(fired=False, rule_name="extreme_budget_burn", reason="")

    # ------------------------------------------------------------------
    # Override helper
    # ------------------------------------------------------------------

    @staticmethod
    def _override(
        verdict: AssessmentResult,
        guardrail: GuardrailResult,
    ) -> AssessmentResult:
        """
        Produce a new AssessmentResult with the guardrail override applied.

        The original rationale is preserved and the guardrail notice is prepended
        so the analyst can see both the LLM's reasoning and the override reason.
        """
        override_notice = (
            f"[GUARDRAIL OVERRIDE — {guardrail.rule_name}] {guardrail.reason}"
        )
        combined_rationale = f"{override_notice}\n\n{verdict.rationale}"

        return AssessmentResult(
            recommendation=Recommendation.NO_GO,
            confidence=Confidence.HIGH,
            rationale=combined_rationale,
            tool_outputs=verdict.tool_outputs,
            guardrail_fired=True,
            guardrail_reason=f"{guardrail.rule_name}: {guardrail.reason}",
            steps=verdict.steps,
            error=verdict.error,
        )
