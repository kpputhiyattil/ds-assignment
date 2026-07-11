"""Tests for feature engineering + dollar-churn labels (Step 4).

The leakage test is the most important: no feature may change when
post-snapshot events are added.
"""

from __future__ import annotations

import datetime as dt

import polars as pl

from src.config import get_config
from src.features.build import (
    CORE_FEATURES,
    GAP_FEATURES,
    INTERVAL_FEATURES,
    build_feature_matrix,
    feature_columns,
)
from src.features.labels import Y_CHURN, Y_LOSS, compute_labels

CFG = get_config()
S = CFG.schema_
SNAP = dt.date(2025, 1, 1)


def _events(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            S.customer_id_col: [r["c"] for r in rows],
            S.date_col: [dt.datetime(*r["d"]) for r in rows],
            "event_amount": [float(r["a"]) for r in rows],
            "n_invoices": [r.get("n", 1) for r in rows],
        }
    )


def _monthly(cust: str, start: dt.date, months: int, amt: float) -> list[dict]:
    return [
        {"c": cust, "d": ((start + dt.timedelta(days=30 * i)).year,
                          (start + dt.timedelta(days=30 * i)).month,
                          (start + dt.timedelta(days=30 * i)).day), "a": amt}
        for i in range(months)
    ]


def test_labels_dollar_churn_math() -> None:
    # past-12mo revenue = 1200 (12x100), future revenue = 0 -> full churn.
    rows = _monthly("A", dt.date(2024, 1, 5), 12, 100.0)
    labels = compute_labels(_events(rows), SNAP, CFG)
    r = labels.filter(pl.col(S.customer_id_col) == "A").row(0, named=True)
    assert r["past_rev"] == 1200.0
    assert r["future_rev"] == 0.0
    assert r[Y_LOSS] == 1200.0
    assert r[Y_CHURN] == 1


def test_labels_no_churn_when_revenue_maintained() -> None:
    # Revenue continues at the same rate after the snapshot -> not churned.
    rows = _monthly("B", dt.date(2024, 1, 5), 24, 100.0)  # spans past & future
    labels = compute_labels(_events(rows), SNAP, CFG)
    r = labels.filter(pl.col(S.customer_id_col) == "B").row(0, named=True)
    assert r[Y_CHURN] == 0                       # revenue maintained -> not churned
    assert r[Y_LOSS] < 0.3 * r["past_rev"]       # at most minor window-edge loss


def test_cohort_excludes_zero_past_revenue() -> None:
    # Customer whose only activity is AFTER the snapshot has no baseline.
    rows = [{"c": "C", "d": (2025, 6, 1), "a": 500.0}]
    labels = compute_labels(_events(rows), SNAP, CFG)
    assert labels.filter(pl.col(S.customer_id_col) == "C").height == 0


def test_no_leakage_features_invariant_to_future_events() -> None:
    past = _monthly("D", dt.date(2024, 1, 5), 12, 100.0)
    future = [{"c": "D", "d": (2025, 3, 1), "a": 9999.0},
              {"c": "D", "d": (2025, 7, 1), "a": 8888.0}]

    mat_past = build_feature_matrix(_events(past), SNAP, CFG)
    mat_all = build_feature_matrix(_events(past + future), SNAP, CFG)

    feats = feature_columns()
    row_past = mat_past.filter(pl.col(S.customer_id_col) == "D").select(feats).row(0)
    row_all = mat_all.filter(pl.col(S.customer_id_col) == "D").select(feats).row(0)
    assert row_past == row_all, "features must not depend on post-snapshot events"


def test_labels_do_depend_on_future() -> None:
    # Sanity: labels SHOULD change with future events (unlike features).
    past = _monthly("E", dt.date(2024, 1, 5), 12, 100.0)
    future = _monthly("E", dt.date(2025, 1, 5), 12, 100.0)
    y_no_future = build_feature_matrix(_events(past), SNAP, CFG)[Y_CHURN][0]
    y_with_future = build_feature_matrix(_events(past + future), SNAP, CFG)[Y_CHURN][0]
    assert y_no_future == 1  # no future revenue -> churn
    assert y_with_future == 0  # revenue continues -> no churn


def test_feature_groups_present_and_disjoint() -> None:
    rows = _monthly("F", dt.date(2024, 1, 5), 12, 100.0)
    mat = build_feature_matrix(_events(rows), SNAP, CFG)
    for col in CORE_FEATURES + GAP_FEATURES + INTERVAL_FEATURES:
        assert col in mat.columns, f"missing feature column: {col}"
    groups = [set(CORE_FEATURES), set(GAP_FEATURES), set(INTERVAL_FEATURES)]
    assert set().union(*groups).__len__() == sum(len(g) for g in groups), "groups overlap"
