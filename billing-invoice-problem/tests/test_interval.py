"""Tests for the interval inference engine (Step 3).

No ground truth exists, so we assert the rule engine behaves as designed on
synthetic customers with known cadences, plus the historical-only guarantee.
"""

from __future__ import annotations

import datetime as dt

import numpy as np

from src.config import get_config
from src.features.interval import (
    INSUFFICIENT_HISTORY,
    IRREGULAR,
    MIXED,
    ONE_TIME_CANDIDATE,
    classify_gap_bases,
    infer_interval_single,
)

CFG = get_config()
SNAP = dt.date(2026, 1, 1)


def _seq(start: dt.date, step_days: int, n: int) -> list[dt.date]:
    return [start + dt.timedelta(days=step_days * i) for i in range(n)]


def test_classify_gap_bases_matches_multiples() -> None:
    # 30->monthly, 60->monthly (2x), 91->quarterly, 365->annual, 45->none
    gaps = np.array([30, 60, 91, 365, 45], dtype=float)
    matched = classify_gap_bases(gaps, CFG)
    assert matched[0] == "monthly"
    assert matched[1] == "monthly"      # skipped period
    assert matched[2] == "quarterly"
    assert matched[3] == "annual"
    assert matched[4] == ""             # 45 is not near any multiple within tol


def test_monthly_customer() -> None:
    dates = _seq(dt.date(2024, 1, 1), 30, 18)
    res = infer_interval_single(dates, [100.0] * 18, SNAP, CFG)
    assert res["billing_interval"] == "monthly"
    assert res["interval_confidence"] > 0.6


def test_quarterly_customer() -> None:
    dates = _seq(dt.date(2024, 1, 1), 91, 8)
    res = infer_interval_single(dates, [500.0] * 8, SNAP, CFG)
    assert res["billing_interval"] == "quarterly"


def test_annual_customer() -> None:
    dates = _seq(dt.date(2021, 1, 1), 365, 4)
    res = infer_interval_single(dates, [1200.0] * 4, SNAP, CFG)
    assert res["billing_interval"] == "annual"


def test_monthly_with_skipped_period() -> None:
    # Regular monthly but one missed cycle (60-day gap in the middle).
    dates = [dt.date(2024, 1, 1), dt.date(2024, 1, 31), dt.date(2024, 3, 1),
             dt.date(2024, 5, 1), dt.date(2024, 5, 31), dt.date(2024, 6, 30)]
    res = infer_interval_single(dates, [100.0] * len(dates), SNAP, CFG)
    assert res["billing_interval"] == "monthly"


def test_one_time_candidate() -> None:
    # Single invoice long before the snapshot (silence >= horizon).
    res = infer_interval_single([dt.date(2023, 1, 1)], [349.0], SNAP, CFG)
    assert res["billing_interval"] == ONE_TIME_CANDIDATE
    assert res["interval_confidence"] >= 0.5


def test_insufficient_history() -> None:
    # Single recent invoice (silence < horizon): too soon to tell.
    res = infer_interval_single([dt.date(2025, 12, 1)], [349.0], SNAP, CFG)
    assert res["billing_interval"] == INSUFFICIENT_HISTORY
    assert res["interval_confidence"] < 0.5


def test_irregular_customer() -> None:
    # Erratic gaps that match no base cadence.
    # Gaps of 45, 20, 50, 15 days -- none near a multiple of 30/91/182/365.
    dates = [dt.date(2024, 1, 1), dt.date(2024, 2, 15), dt.date(2024, 3, 6),
             dt.date(2024, 4, 25), dt.date(2024, 5, 10)]
    res = infer_interval_single(dates, [10.0, 500.0, 3.0, 900.0, 7.0], SNAP, CFG)
    assert res["billing_interval"] == IRREGULAR


def test_mixed_monthly_plus_annual() -> None:
    # Monthly subscription with an annual fee interleaved -> competing bases.
    monthly = _seq(dt.date(2024, 1, 15), 30, 12)
    annual = [dt.date(2024, 1, 1), dt.date(2025, 1, 1)]
    dates = sorted(monthly + annual)
    res = infer_interval_single(dates, [50.0] * len(dates), SNAP, CFG)
    assert res["billing_interval"] in {MIXED, "monthly"}
    assert res["n_distinct_bases"] >= 1


def test_historical_only_excludes_future_events() -> None:
    # Events after the snapshot must not influence the inference.
    past = _seq(dt.date(2024, 1, 1), 30, 6)
    future = _seq(dt.date(2026, 6, 1), 7, 10)
    res_all = infer_interval_single(past + future, [100.0] * 16, SNAP, CFG)
    res_past = infer_interval_single(past, [100.0] * 6, SNAP, CFG)
    assert res_all["billing_interval"] == res_past["billing_interval"] == "monthly"
    assert res_all["n_events"] == res_past["n_events"] == 6
