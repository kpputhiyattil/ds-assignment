"""Continue / Review / No-Go decision layer (Step 8).

Turns a calibrated churn probability into a credit triage recommendation, with a
**data-quality guardrail**: when the inferred billing interval is low-confidence
(short or missing history), the case is routed to **Review** rather than being
auto-decided. Missing history is a data-quality problem, not evidence of credit
risk, so it must never trigger an automatic No-Go (nor an automatic Continue).
"""

from __future__ import annotations

from src.config import Config, get_config

CONTINUE = "Continue"
REVIEW = "Review"
NO_GO = "No-Go"


def decide(
    churn_prob: float,
    interval_confidence: float,
    cfg: Config | None = None,
) -> dict:
    """Map churn risk + interval confidence to a recommendation.

    Rules (in order):
    1. Low interval confidence  -> Review (data-quality guardrail).
    2. churn_prob <= continue_max_risk -> Continue.
    3. churn_prob >= nogo_min_risk     -> No-Go.
    4. otherwise                        -> Review.
    """
    cfg = cfg or get_config()
    d = cfg.decision

    if interval_confidence < d.min_interval_confidence_for_auto:
        return {
            "recommendation": REVIEW,
            "reason": "low_interval_confidence",
            "detail": (
                f"interval_confidence {interval_confidence:.2f} < "
                f"{d.min_interval_confidence_for_auto:.2f}; insufficient billing "
                "history to auto-decide (data-quality guardrail)."
            ),
            "auto_decided": False,
        }

    if churn_prob <= d.continue_max_risk:
        reco, reason = CONTINUE, "low_churn_risk"
    elif churn_prob >= d.nogo_min_risk:
        reco, reason = NO_GO, "high_churn_risk"
    else:
        reco, reason = REVIEW, "borderline_churn_risk"

    return {
        "recommendation": reco,
        "reason": reason,
        "detail": (
            f"churn_prob {churn_prob:.2f} vs Continue<= {d.continue_max_risk:.2f}, "
            f"No-Go>= {d.nogo_min_risk:.2f}."
        ),
        "auto_decided": reco != REVIEW,
    }
