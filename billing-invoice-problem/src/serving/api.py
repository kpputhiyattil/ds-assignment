"""Optional FastAPI service exposing the churn-decision model (Step 8).

Loads the single packaged artifact once at startup (never per request) and
serves ``/assess``. Aggregates same-day invoices via the same feature-
engineering path used in training, so there is no train/serve skew.

Run:  uvicorn src.serving.api:app --reload
"""

from __future__ import annotations

import datetime as dt

import polars as pl
from fastapi import FastAPI, HTTPException

from src.config import get_config
from src.serving.package import load_package
from src.serving.schemas import AssessRequest, AssessResponse

app = FastAPI(title="Billing-Interval Churn Decision API", version="1.0")
_MODEL = None


def get_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = load_package()
    return _MODEL


@app.get("/health")
def health() -> dict:
    try:
        m = get_model()
        return {"status": "ok", "model_version": m.version}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"model not loaded: {exc}")


@app.post("/assess", response_model=AssessResponse)
def assess(req: AssessRequest) -> AssessResponse:
    cfg = get_config()
    s = cfg.schema_
    snapshot = req.snapshot or dt.date.today()
    events = pl.DataFrame(
        {
            s.customer_id_col: [req.customer_id] * len(req.invoices),
            s.date_col: [dt.datetime(i.date.year, i.date.month, i.date.day)
                         for i in req.invoices],
            s.amount_col: [i.amount for i in req.invoices],

        }
    )
    result = get_model().assess(events, snapshot, customer_id=req.customer_id)
    return AssessResponse(**{k: result.get(k) for k in AssessResponse.model_fields})
