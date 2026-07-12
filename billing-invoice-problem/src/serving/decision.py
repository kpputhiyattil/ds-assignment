"""Continue / Review / No-Go decision layer (Step 8).

Turns a calibrated churn probability into a credit triage recommendation, with a
**data-quality guardrail**: when the inferred billing interval is low-confidence
(short or missing history), the case is routed to **Review** rather than being
auto-decided. Missing history is a data-quality problem, not evidence of credit
risk, so it must never trigger an automatic No-Go (nor an automatic Continue).
"""

from __future__ import annotations

from typing import Iterable, Sequence

from src.config import Config, get_config

CONTINUE = "Continue"
REVIEW = "Review"
NO_GO = "No-Go"


def decide(
    churn_prob: float,
    interval_confidence: float,
    cfg: Config | None = None,
    *,
    continue_max_risk: float | None = None,
    nogo_min_risk: float | None = None,
    min_interval_confidence_for_auto: float | None = None,
) -> dict:
    """Map churn risk + interval confidence to a recommendation.

    Rules (in order):
    1. Low interval confidence  -> Review (data-quality guardrail).
    2. churn_prob <= continue_max_risk -> Continue.
    3. churn_prob >= nogo_min_risk     -> No-Go.
    4. otherwise                        -> Review.

    Optional keyword thresholds override ``cfg.decision`` (used for sensitivity).
    """
    cfg = cfg or get_config()
    d = cfg.decision
    cont = d.continue_max_risk if continue_max_risk is None else continue_max_risk
    nogo = d.nogo_min_risk if nogo_min_risk is None else nogo_min_risk
    min_conf = (
        d.min_interval_confidence_for_auto
        if min_interval_confidence_for_auto is None
        else min_interval_confidence_for_auto
    )

    if interval_confidence < min_conf:
        return {
            "recommendation": REVIEW,
            "reason": "low_interval_confidence",
            "detail": (
                f"interval_confidence {interval_confidence:.2f} < "
                f"{min_conf:.2f}; insufficient billing "
                "history to auto-decide (data-quality guardrail)."
            ),
            "auto_decided": False,
        }

    if churn_prob <= cont:
        reco, reason = CONTINUE, "low_churn_risk"
    elif churn_prob >= nogo:
        reco, reason = NO_GO, "high_churn_risk"
    else:
        reco, reason = REVIEW, "borderline_churn_risk"

    return {
        "recommendation": reco,
        "reason": reason,
        "detail": (
            f"churn_prob {churn_prob:.2f} vs Continue<= {cont:.2f}, "
            f"No-Go>= {nogo:.2f}."
        ),
        "auto_decided": reco != REVIEW,
    }


def recommendation_counts(
    probs: Sequence[float],
    confidences: Sequence[float],
    cfg: Config | None = None,
    *,
    continue_max_risk: float | None = None,
    nogo_min_risk: float | None = None,
    min_interval_confidence_for_auto: float | None = None,
) -> dict[str, int]:
    """Count Continue / Review / No-Go under a threshold setting."""
    counts = {CONTINUE: 0, REVIEW: 0, NO_GO: 0}
    for p, c in zip(probs, confidences, strict=True):
        reco = decide(
            float(p),
            float(c),
            cfg,
            continue_max_risk=continue_max_risk,
            nogo_min_risk=nogo_min_risk,
            min_interval_confidence_for_auto=min_interval_confidence_for_auto,
        )["recommendation"]
        counts[reco] += 1
    return counts


def threshold_sensitivity(
    probs: Sequence[float],
    confidences: Sequence[float],
    cfg: Config | None = None,
    *,
    continue_grid: Iterable[float] = (0.20, 0.30, 0.40, 0.50),
    nogo_grid: Iterable[float] = (0.60, 0.70, 0.80),
    min_conf_grid: Iterable[float] = (0.30, 0.50, 0.70),
) -> list[dict]:
    """Sweep decision thresholds; return one row per (continue, nogo, min_conf)."""
    cfg = cfg or get_config()
    n = len(probs)
    rows: list[dict] = []
    for min_conf in min_conf_grid:
        for cont in continue_grid:
            for nogo in nogo_grid:
                if cont >= nogo:
                    continue
                counts = recommendation_counts(
                    probs,
                    confidences,
                    cfg,
                    continue_max_risk=float(cont),
                    nogo_min_risk=float(nogo),
                    min_interval_confidence_for_auto=float(min_conf),
                )
                rows.append(
                    {
                        "continue_max_risk": float(cont),
                        "nogo_min_risk": float(nogo),
                        "min_interval_confidence": float(min_conf),
                        "n": n,
                        "Continue": counts[CONTINUE],
                        "Review": counts[REVIEW],
                        "No-Go": counts[NO_GO],
                        "continue_pct": round(100.0 * counts[CONTINUE] / max(n, 1), 2),
                        "review_pct": round(100.0 * counts[REVIEW] / max(n, 1), 2),
                        "nogo_pct": round(100.0 * counts[NO_GO] / max(n, 1), 2),
                        "is_config_default": (
                            abs(cont - cfg.decision.continue_max_risk) < 1e-9
                            and abs(nogo - cfg.decision.nogo_min_risk) < 1e-9
                            and abs(min_conf - cfg.decision.min_interval_confidence_for_auto)
                            < 1e-9
                        ),
                    }
                )
    return rows
