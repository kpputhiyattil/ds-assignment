"""End-to-end serving tests: packaged model + API (Step 8)."""

from __future__ import annotations

import datetime as dt

import numpy as np
import polars as pl
from lightgbm import LGBMClassifier, LGBMRegressor

from src.config import get_config
from src.features.build import build_scoring_features, feature_columns
from src.serving.model import ChurnDecisionModel

CFG = get_config()
S = CFG.schema_
SNAP = dt.date(2026, 1, 1)


def _events(cust: str, dates: list[dt.date], amt: float) -> pl.DataFrame:
    # Raw invoices (one row per invoice); the serving FE aggregates same-day.
    return pl.DataFrame({
        S.customer_id_col: [cust] * len(dates),
        S.date_col: [dt.datetime(d.year, d.month, d.day) for d in dates],
        S.amount_col: [amt] * len(dates),
    })


def test_scoring_features_have_no_labels() -> None:
    dates = [dt.date(2024, 1, 1) + dt.timedelta(days=30 * i) for i in range(12)]
    feats = build_scoring_features(_events("A", dates, 100.0), SNAP, CFG)
    for f in feature_columns():
        assert f in feats.columns
    for lbl in ("y_churn", "y_loss", "future_rev"):
        assert lbl not in feats.columns


def _tiny_bundle() -> ChurnDecisionModel:
    rng = np.random.default_rng(0)
    d = len(feature_columns())
    X = rng.normal(size=(300, d))
    y = (X[:, 0] > 0).astype(int)
    freq = LGBMClassifier(n_estimators=30, random_state=0, verbose=-1).fit(X, y)
    sev = LGBMRegressor(n_estimators=30, random_state=0, verbose=-1).fit(
        X[y == 1], rng.uniform(50, 500, int(y.sum()))
    )
    return ChurnDecisionModel(freq, sev, freq, feature_columns(), CFG, "test", "abc123")


def test_assess_returns_full_decision() -> None:
    bundle = _tiny_bundle()
    dates = [dt.date(2024, 1, 1) + dt.timedelta(days=30 * i) for i in range(12)]
    out = bundle.assess(_events("MONTHLY", dates, 100.0), SNAP, customer_id="MONTHLY")
    assert set(out) >= {"recommendation", "churn_probability", "expected_dollar_churn",
                        "inferred_interval", "interval_confidence", "briefing"}
    assert out["recommendation"] in {"Continue", "Review", "No-Go"}


def test_assess_guardrail_reviews_thin_history() -> None:
    # Single recent invoice -> insufficient_history -> low confidence -> Review.
    bundle = _tiny_bundle()
    out = bundle.assess(_events("NEW", [dt.date(2025, 12, 1)], 500.0), SNAP,
                        customer_id="NEW", explain=False)
    assert out["recommendation"] == "Review"
    assert out["reason"] == "low_interval_confidence"


def test_api_assess_endpoint() -> None:
    from fastapi.testclient import TestClient

    import src.serving.api as api
    api._MODEL = _tiny_bundle()  # inject tiny model, skip artifact load
    client = TestClient(api.app)

    payload = {
        "customer_id": "MONTHLY",
        "invoices": [
            {"date": f"2024-{m:02d}-05", "amount": 100.0} for m in range(1, 13)
        ],
        "snapshot": "2026-01-01",
    }
    r = client.post("/assess", json=payload)
    assert r.status_code == 200
    body = r.json()
    assert body["customer_id"] == "MONTHLY"
    assert body["recommendation"] in {"Continue", "Review", "No-Go"}
