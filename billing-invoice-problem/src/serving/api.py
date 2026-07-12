"""FastAPI service: single-customer and file-based invoice assessment.

Loads the packaged artifact once at startup. Endpoints:

* ``GET  /health``          -- liveness + model version
* ``GET  /model/info``      -- package checklist / version
* ``POST /assess``          -- JSON body (one customer, full SHAP briefing)
* ``POST /assess/file``     -- upload CSV/Parquet (capped load + top-N explain)

Interactive file loads are **capped** (default 500 customers, hard max from
config) so uploading the full 22M-row parquet cannot crash the service.

Run::

    make package
    uvicorn src.serving.api:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import datetime as dt
import logging
from typing import Annotated

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from src.config import get_config
from src.serving.io import (
    UploadTooLargeError,
    invoices_from_records,
    load_invoices_from_bytes,
)
from src.serving.package import load_package
from src.serving.schemas import (
    AssessRequest,
    AssessResponse,
    BatchAssessResponse,
    BatchSummary,
)

logger = logging.getLogger(__name__)

app = FastAPI(
    title="Billing-Interval Churn Decision API",
    version="1.0",
    description=(
        "Infer billing interval + dollar-churn risk from raw invoices. "
        "Returns Continue / Review / No-Go with SHAP briefings. "
        "File uploads are customer-capped to avoid loading the full dataset."
    ),
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_MODEL = None


def get_model():
    global _MODEL
    if _MODEL is None:
        _MODEL = load_package()
    # Dual package is required for with/without interval comparison in the UI.
    if not hasattr(_MODEL, "without_interval") or not hasattr(_MODEL, "with_interval"):
        raise HTTPException(
            status_code=503,
            detail=(
                "Serving artifact is a single-model package. "
                "Run `python -m src.serving.package` and restart the API."
            ),
        )
    return _MODEL


@app.post("/reload")
def reload_model() -> dict:
    """Drop the cached artifact so the next request loads the latest package."""
    global _MODEL
    _MODEL = None
    m = get_model()
    return {
        "status": "reloaded",
        "model_version": m.version,
        "dual_models": True,
        "n_features_with": len(m.with_interval.feature_cols),
        "n_features_without": len(m.without_interval.feature_cols),
    }


@app.get("/health")
def health() -> dict:
    try:
        m = get_model()
        return {"status": "ok", "model_version": m.version, "git_commit": m.git_commit}
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=503, detail=f"model not loaded: {exc}") from exc


@app.get("/model/info")
def model_info() -> dict:
    m = get_model()
    cfg = get_config()
    return {
        "model_version": m.version,
        "git_commit": m.git_commit,
        "n_features": len(m.feature_cols),
        "feature_cols": m.feature_cols,
        "checklist": {
            "feature_engineering": m.has_feature_engineering(),
            "models": m.has_models(),
            "explainer": m.has_explainer(),
            "decision_logic": True,
        },
        "load_limits": {
            "default_max_customers": cfg.serving.default_max_customers,
            "hard_max_customers": cfg.serving.hard_max_customers,
            "max_upload_mb": cfg.serving.max_upload_mb,
            "max_invoice_rows": cfg.serving.max_invoice_rows,
        },
        "dual_models": True,
        "holdout_comparison": getattr(m, "comparison_metrics", None),
    }


@app.post("/assess", response_model=AssessResponse)
def assess(req: AssessRequest) -> AssessResponse:
    """Score one customer from manually entered invoices (explainable)."""
    snapshot = req.snapshot or dt.date.today()
    try:
        events = invoices_from_records(
            req.customer_id,
            [{"date": i.date, "amount": i.amount} for i in req.invoices],
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    result = get_model().assess(
        events, snapshot, customer_id=req.customer_id, explain=req.explain
    )
    return AssessResponse(**{k: result.get(k) for k in AssessResponse.model_fields})


@app.post("/assess/file", response_model=BatchAssessResponse)
async def assess_file(
    file: Annotated[UploadFile, File(description="CSV or Parquet invoice file")],
    snapshot: Annotated[str | None, Form()] = None,
    explain_top_n: Annotated[int, Form()] = 10,
    max_customers: Annotated[int | None, Form()] = None,
    customer_id: Annotated[str | None, Form()] = None,
    recurring_only: Annotated[bool, Form()] = False,
) -> BatchAssessResponse:
    """Batch-score an uploaded file with a hard customer cap (never full dump)."""
    cfg = get_config()
    filename = file.filename or "upload.bin"

    max_bytes = int(cfg.serving.max_upload_mb * 1024 * 1024)
    cl = file.headers.get("content-length") if hasattr(file, "headers") else None
    if cl is not None:
        try:
            if int(cl) > max_bytes:
                raise HTTPException(
                    status_code=413,
                    detail=(
                        f"Upload Content-Length {int(cl) / (1024 * 1024):.1f} MB "
                        f"exceeds limit {cfg.serving.max_upload_mb:.0f} MB."
                    ),
                )
        except ValueError:
            pass

    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=422, detail="Empty upload")
    if len(raw) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=(
                f"Upload is {len(raw) / (1024 * 1024):.1f} MB; "
                f"limit is {cfg.serving.max_upload_mb:.0f} MB."
            ),
        )

    requested = max_customers if max_customers and max_customers > 0 else None
    if requested is None:
        requested = cfg.serving.default_max_customers
    capped = min(int(requested), cfg.serving.hard_max_customers)

    try:
        loaded = load_invoices_from_bytes(
            raw,
            filename,
            cfg,
            max_customers=capped,
            customer_id=customer_id.strip() if customer_id else None,
            allow_full=False,
            recurring_only=bool(recurring_only),
        )
    except UploadTooLargeError as exc:
        raise HTTPException(status_code=413, detail=str(exc)) from exc
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=422, detail=f"Failed to parse file: {exc}") from exc

    snap = dt.date.fromisoformat(snapshot) if snapshot else dt.date.today()

    try:
        out = get_model().assess_many(
            loaded.frame,
            snap,
            explain_top_n=max(0, int(explain_top_n)),
            explain_ids=[customer_id.strip()] if customer_id else None,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Batch assess failed")
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    warning = None
    if loaded.truncated:
        warning = (
            f"Input truncated to {loaded.max_customers_applied} customers "
            f"(hard max {cfg.serving.hard_max_customers}). "
            "For a full-file offline run use: "
            "python -m src.scripts.run_batch_assess --full"
        )
    if recurring_only:
        note = "Recurring-only filter applied (>=2 invoice days)."
        warning = f"{warning} {note}" if warning else note

    return BatchAssessResponse(
        snapshot=out["snapshot"],
        n_customers=out["n_customers"],
        summary=BatchSummary(**out["summary"]),
        results=out["results"],
        explanations=out["explanations"],
        comparison_summary=out.get("comparison_summary"),
        model_version=out["model_version"],
        source_filename=filename,
        truncated=loaded.truncated,
        max_customers_applied=loaded.max_customers_applied,
        n_invoice_rows_loaded=loaded.n_rows,
        warning=warning,
    )
