"""End-to-end serving tests: packaged dual model + API (Step 8)."""

from __future__ import annotations

import datetime as dt
import io

import numpy as np
import polars as pl
from lightgbm import LGBMClassifier, LGBMRegressor

from src.config import get_config
from src.features.build import (
    CORE_FEATURES,
    GAP_FEATURES,
    build_scoring_features,
    feature_columns,
)
from src.serving.io import load_invoices_from_bytes, load_invoices_from_path, normalize_invoice_frame
from src.serving.model import ChurnDecisionModel, _ModelPair

CFG = get_config()
S = CFG.schema_
SNAP = dt.date(2026, 1, 1)


def _events(cust: str, dates: list[dt.date], amt: float) -> pl.DataFrame:
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


def _fit_pair(feats: list[str]) -> _ModelPair:
    rng = np.random.default_rng(0)
    d = len(feats)
    X = rng.normal(size=(300, d))
    y = (X[:, 0] > 0).astype(int)
    freq = LGBMClassifier(n_estimators=30, random_state=0, verbose=-1).fit(X, y)
    sev = LGBMRegressor(n_estimators=30, random_state=0, verbose=-1).fit(
        X[y == 1], rng.uniform(50, 500, int(y.sum()))
    )
    return _ModelPair(freq, sev, freq, feats)


def _tiny_bundle() -> ChurnDecisionModel:
    return ChurnDecisionModel(
        with_interval=_fit_pair(feature_columns()),
        without_interval=_fit_pair(CORE_FEATURES + GAP_FEATURES),
        cfg=CFG,
        version="test",
        git_commit="abc123",
        comparison_metrics={"narrative": "test holdout"},
    )


def test_assess_returns_dual_decision() -> None:
    bundle = _tiny_bundle()
    dates = [dt.date(2024, 1, 1) + dt.timedelta(days=30 * i) for i in range(12)]
    out = bundle.assess(_events("MONTHLY", dates, 100.0), SNAP, customer_id="MONTHLY")
    assert out["inferred_interval"]
    assert out["interval_reason"]
    assert out["decision_reason"]
    assert "with_interval" in out and "without_interval" in out
    assert "comparison" in out
    assert out["with_interval"]["decision_reason"]
    assert out["without_interval"]["decision_reason"]
    assert out["recommendation"] in {"Continue", "Review", "No-Go"}


def test_assess_guardrail_reviews_thin_history() -> None:
    bundle = _tiny_bundle()
    out = bundle.assess(
        _events("NEW", [dt.date(2025, 12, 1)], 500.0),
        SNAP,
        customer_id="NEW",
        explain=False,
    )
    assert out["recommendation"] == "Review"
    assert out["reason"] == "low_interval_confidence"
    assert "confidence" in (out.get("interval_reason") or "").lower() or "history" in (
        out.get("interval_reason") or ""
    ).lower()


def test_assess_many_batch_dual() -> None:
    bundle = _tiny_bundle()
    frames = []
    for cust, n in [("A", 12), ("B", 1)]:
        dates = [dt.date(2024, 1, 1) + dt.timedelta(days=30 * i) for i in range(n)]
        frames.append(_events(cust, dates, 80.0))
    events = pl.concat(frames)
    out = bundle.assess_many(events, SNAP, explain_top_n=1)
    assert out["n_customers"] == 2
    assert len(out["results"]) == 2
    assert "comparison_summary" in out
    assert out["results"][0]["with_interval"]["recommendation"]
    assert out["results"][0]["without_interval"]["recommendation"]
    assert out["results"][0]["interval_reason"]


def test_normalize_aliases() -> None:
    df = pl.DataFrame({
        "customer_id": ["x"],
        "invoice_date": [dt.date(2024, 1, 1)],
        "invoice_amount": [10.0],
    })
    out = normalize_invoice_frame(df, CFG)
    assert S.customer_id_col in out.columns
    assert out.height == 1


def test_api_assess_endpoint() -> None:
    from fastapi.testclient import TestClient

    import src.serving.api as api

    api._MODEL = _tiny_bundle()
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
    assert body["interval_reason"]
    assert body["with_interval"]["recommendation"]
    assert body["without_interval"]["recommendation"]


def test_api_assess_file_endpoint() -> None:
    from fastapi.testclient import TestClient

    import src.serving.api as api

    api._MODEL = _tiny_bundle()
    client = TestClient(api.app)

    csv = (
        "customerid,date,amount\n"
        "C1,2024-01-05,100\n"
        "C1,2024-02-05,100\n"
        "C2,2025-12-01,50\n"
    ).encode()
    r = client.post(
        "/assess/file",
        files={"file": ("invoices.csv", csv, "text/csv")},
        data={"snapshot": "2026-01-01", "explain_top_n": "1"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["n_customers"] == 2
    assert body["comparison_summary"]["n_customers"] == 2
    assert "interval_reason" in body["results"][0]
    assert "with_interval" in body["results"][0]
    assert "without_interval" in body["results"][0]


def test_load_invoices_from_bytes_parquet() -> None:
    df = pl.DataFrame({
        S.customer_id_col: ["Z"],
        S.date_col: [dt.datetime(2024, 6, 1)],
        S.amount_col: [12.5],
    })
    buf = io.BytesIO()
    df.write_parquet(buf)
    out = load_invoices_from_bytes(buf.getvalue(), "x.parquet", CFG)
    assert out.frame.height == 1
    assert out.n_customers == 1


def test_load_path_respects_max_customers(tmp_path) -> None:
    rows = []
    for i in range(20):
        rows.append({
            S.customer_id_col: f"C{i}",
            S.date_col: dt.datetime(2024, 1, 1),
            S.amount_col: 10.0,
        })
    path = tmp_path / "many.parquet"
    pl.DataFrame(rows).write_parquet(path)
    loaded = load_invoices_from_path(path, CFG, max_customers=5, allow_full=False)
    assert loaded.n_customers == 5
    assert loaded.truncated is True


def test_api_rejects_uncapped_by_default() -> None:
    from fastapi.testclient import TestClient

    import src.serving.api as api

    api._MODEL = _tiny_bundle()
    client = TestClient(api.app)
    csv = "customerid,date,amount\n"
    for i in range(3):
        csv += f"C{i},2024-01-05,100\n"
    r = client.post(
        "/assess/file",
        files={"file": ("invoices.csv", csv.encode(), "text/csv")},
        data={"snapshot": "2026-01-01", "explain_top_n": "0"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["max_customers_applied"] == CFG.serving.default_max_customers
