"""
FastAPI app — landlord quality scoring + SHAP explanations.

Run from project root:
    uvicorn service.app.main:app --reload --port 8080
"""
from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# Project root on path for src.* and service.*
_PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from service.app.config import DEFAULT_TOP_K, MAX_SCORE_ROWS, SCORER_PATH
from service.app.schemas import ScoreRequest
from service.app.scoring import (
    ScorerNotReady,
    dataframe_from_records,
    dataframe_from_upload,
    get_scorer,
    load_meta,
    score_dataframe,
)

logger = logging.getLogger(__name__)

SERVICE_DIR = Path(__file__).resolve().parents[1]
templates = Jinja2Templates(directory=str(SERVICE_DIR / "templates"))

app = FastAPI(
    title="Landlord Quality Scoring",
    description="Score landlords with the packed CatBoost model + human-readable SHAP drivers.",
    version="1.0.0",
)
app.mount("/static", StaticFiles(directory=str(SERVICE_DIR / "static")), name="static")


@app.on_event("startup")
def _startup() -> None:
    logging.basicConfig(level=logging.INFO)
    try:
        # Always load the on-disk artifact (picks up re-exports after restart)
        scorer = get_scorer(force_reload=True)
        manifest = scorer.package_manifest()
        logger.info("Scorer ready: %s", SCORER_PATH)
        logger.info(
            "Packaged components: %s",
            ", ".join(
                f"{k}={manifest[k]['included']}"
                for k in ("model", "preprocessing", "feature_engineering", "shap")
            ),
        )
    except ScorerNotReady as exc:
        logger.warning("Scorer not loaded at startup: %s", exc)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request) -> HTMLResponse:
    meta: dict[str, Any] = {}
    ready = True
    error = None
    try:
        meta = load_meta()
    except Exception as exc:
        ready = False
        error = str(exc)
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "ready": ready,
            "error": error,
            "meta": meta,
            "max_rows": MAX_SCORE_ROWS,
            "default_top_k": DEFAULT_TOP_K,
            "numeric_cols": meta.get("numeric_cols", []),
            "categorical_cols": meta.get("categorical_cols", []),
            "glossary": meta.get("feature_glossary", []),
            "package_components": meta.get("package_components", []),
            "scorer_path": str(SCORER_PATH),
        },
    )


@app.get("/api/health")
def health() -> dict[str, Any]:
    try:
        scorer = get_scorer()
        manifest = scorer.package_manifest()
        return {
            "status": "ok",
            "model_type": scorer.meta.model_type,
            "model_version": scorer.meta.version,
            "scorer_path": str(SCORER_PATH),
            "package_components": manifest.get("components", []),
            "package_manifest": {
                k: {"included": manifest[k]["included"]}
                for k in ("model", "preprocessing", "feature_engineering", "shap")
            },
            "pipeline_name": scorer.meta.pipeline_name,
            "pipeline_steps": list(scorer.meta.pipeline_steps),
        }
    except ScorerNotReady as exc:
        return {"status": "unavailable", "detail": str(exc)}


@app.get("/api/schema")
def schema() -> dict[str, Any]:
    try:
        meta = load_meta()
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {
        "numeric_cols": meta.get("numeric_cols", []),
        "categorical_cols": meta.get("categorical_cols", []),
        "feature_glossary": meta.get("feature_glossary", []),
        "model_type": meta.get("model_type"),
        "model_version": meta.get("version"),
        "package_components": meta.get("package_components", []),
        "pipeline_steps": meta.get("pipeline_steps", []),
        "max_rows": MAX_SCORE_ROWS,
        "scorer_path": str(SCORER_PATH),
    }


def _run_score(df, top_k: int) -> dict[str, Any]:
    try:
        return score_dataframe(df, top_k=top_k)
    except ScorerNotReady as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("Scoring failed")
        raise HTTPException(status_code=500, detail=f"Scoring failed: {exc}") from exc


@app.post("/api/score/json")
async def score_json(body: ScoreRequest) -> JSONResponse:
    records = [r.model_dump(exclude_none=False) for r in body.landlords]
    # Drop keys that are entirely None so optional blanks don't force columns wrongly
    cleaned = []
    for rec in records:
        cleaned.append({k: v for k, v in rec.items() if v is not None and v != ""})
    df = dataframe_from_records(cleaned)
    payload = _run_score(df, body.top_k)
    return JSONResponse(payload)


@app.post("/api/score/file")
async def score_file(
    file: UploadFile = File(...),
    top_k: int = Form(DEFAULT_TOP_K),
) -> JSONResponse:
    raw = await file.read()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty file upload.")
    try:
        df = dataframe_from_upload(file.filename or "upload.csv", raw)
    except Exception as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Could not parse file (use .csv or .parquet): {exc}",
        ) from exc
    if df.empty:
        raise HTTPException(status_code=400, detail="Uploaded table has no rows.")
    payload = _run_score(df, int(top_k))
    return JSONResponse(payload)


@app.exception_handler(ScorerNotReady)
async def scorer_not_ready_handler(request: Request, exc: ScorerNotReady):
    return JSONResponse(status_code=503, content={"detail": str(exc)})
