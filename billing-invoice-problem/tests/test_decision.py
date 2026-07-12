"""Tests for the Continue/Review/No-Go decision layer (Step 8)."""

from __future__ import annotations

from src.config import get_config
from src.serving.decision import CONTINUE, NO_GO, REVIEW, decide

CFG = get_config()
HIGH_CONF = 0.9  # above min_interval_confidence_for_auto


def test_continue_on_low_risk_high_confidence() -> None:
    r = decide(churn_prob=0.10, interval_confidence=HIGH_CONF, cfg=CFG)
    assert r["recommendation"] == CONTINUE
    assert r["auto_decided"]


def test_nogo_on_high_risk_high_confidence() -> None:
    r = decide(churn_prob=0.95, interval_confidence=HIGH_CONF, cfg=CFG)
    assert r["recommendation"] == NO_GO


def test_review_on_borderline_risk() -> None:
    mid = (CFG.decision.continue_max_risk + CFG.decision.nogo_min_risk) / 2
    r = decide(churn_prob=mid, interval_confidence=HIGH_CONF, cfg=CFG)
    assert r["recommendation"] == REVIEW


def test_guardrail_low_confidence_forces_review_even_at_high_risk() -> None:
    # Data-quality guardrail: never auto No-Go on thin history.
    low_conf = CFG.decision.min_interval_confidence_for_auto - 0.01
    r = decide(churn_prob=0.99, interval_confidence=low_conf, cfg=CFG)
    assert r["recommendation"] == REVIEW
    assert r["reason"] == "low_interval_confidence"
    assert not r["auto_decided"]


def test_guardrail_low_confidence_blocks_auto_continue() -> None:
    low_conf = CFG.decision.min_interval_confidence_for_auto - 0.01
    r = decide(churn_prob=0.01, interval_confidence=low_conf, cfg=CFG)
    assert r["recommendation"] == REVIEW


def test_threshold_sensitivity_changes_continue_share() -> None:
    from src.serving.decision import threshold_sensitivity

    probs = [0.25, 0.45, 0.85, 0.95]
    confs = [0.9, 0.9, 0.9, 0.9]
    rows = threshold_sensitivity(
        probs,
        confs,
        CFG,
        continue_grid=(0.20, 0.50),
        nogo_grid=(0.70,),
        min_conf_grid=(0.50,),
    )
    by_cont = {r["continue_max_risk"]: r for r in rows}
    assert by_cont[0.50]["Continue"] >= by_cont[0.20]["Continue"]
