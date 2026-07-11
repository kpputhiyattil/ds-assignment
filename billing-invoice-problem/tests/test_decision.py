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
