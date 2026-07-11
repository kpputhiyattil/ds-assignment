"""
Scoring helpers — thin wrapper around packed LandlordScorer (FE + model + SHAP).
"""
from __future__ import annotations

import io
import json
import logging
import sys
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from service.app.config import META_PATH, PROJECT_ROOT, SCORER_PATH

logger = logging.getLogger(__name__)

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


class ScorerNotReady(RuntimeError):
    """Raised when the artifact is missing or failed to load."""


@lru_cache(maxsize=1)
def _load_scorer():
    from src.models.scorer import LandlordScorer

    if not SCORER_PATH.exists():
        raise ScorerNotReady(
            f"Scorer artifact not found at {SCORER_PATH}. "
            "Run: python scripts/export_landlord_scorer.py"
        )
    logger.info("Loading scorer from %s", SCORER_PATH)
    return LandlordScorer.load(SCORER_PATH)


def get_scorer(*, force_reload: bool = False):
    """Load and cache the packed LandlordScorer artifact."""
    if force_reload:
        _load_scorer.cache_clear()
    return _load_scorer()


def load_meta() -> dict[str, Any]:
    if META_PATH.exists():
        with open(META_PATH, encoding="utf-8") as fp:
            return json.load(fp)
    scorer = get_scorer()
    return scorer.metadata_dict()


def required_feature_cols() -> tuple[list[str], list[str]]:
    meta = load_meta()
    return list(meta.get("numeric_cols", [])), list(meta.get("categorical_cols", []))


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Accept common aliases for the landlord id column."""
    out = df.copy()
    rename = {}
    lower_map = {c.lower().strip(): c for c in out.columns}
    for alias in ("landlordid", "landlord_id", "landlord id", "id"):
        if alias in lower_map and "LandLordID" not in out.columns:
            rename[lower_map[alias]] = "LandLordID"
            break
    for alias in ("tenantcount", "tenant_count", "n_tenants"):
        if alias in lower_map and "TenantCount" not in out.columns:
            rename[lower_map[alias]] = "TenantCount"
            break
    for alias in ("allcompanyid", "all_company_id", "company_ids"):
        if alias in lower_map and "AllCompanyID" not in out.columns:
            rename[lower_map[alias]] = "AllCompanyID"
            break
    if rename:
        out = out.rename(columns=rename)
    return out


def dataframe_from_upload(filename: str, raw: bytes) -> pd.DataFrame:
    name = (filename or "").lower()
    buf = io.BytesIO(raw)
    if name.endswith(".parquet") or name.endswith(".pq"):
        df = pd.read_parquet(buf)
    elif name.endswith(".csv"):
        df = pd.read_csv(buf)
    else:
        try:
            buf.seek(0)
            df = pd.read_parquet(buf)
        except Exception:
            buf.seek(0)
            df = pd.read_csv(buf)
    return _normalize_columns(df)


def dataframe_from_records(records: list[dict[str, Any]]) -> pd.DataFrame:
    if not records:
        raise ValueError("No landlord records provided.")
    return _normalize_columns(pd.DataFrame(records))


def sparse_feature_report(df: pd.DataFrame) -> dict[str, Any]:
    """Report which model features are entirely missing / all-null."""
    num, cat = required_feature_cols()
    empty_cols = [
        c for c in (num + cat)
        if c not in df.columns or int(df[c].notna().sum()) == 0
    ]
    portfolio_empty = [c for c in empty_cols if c.startswith("Portfolio")]
    warning = None
    if portfolio_empty:
        warning = (
            "Some portfolio features are still empty after the pipeline "
            "(often unmatched AllCompanyID → Companies). "
            "Check bridge coverage for these landlords."
        )
    elif empty_cols:
        warning = (
            f"{len(empty_cols)} feature column(s) have no values after processing."
        )
    return {
        "empty_feature_cols": empty_cols,
        "n_empty_features": len(empty_cols),
        "input_warning": warning,
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.floating, float)):
        v = float(value)
        if np.isnan(v) or np.isinf(v):
            return None
        return v
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if value is pd.NA:
        return None
    try:
        if value is not None and not isinstance(value, (str, bytes, dict, list)) and pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def score_dataframe(
    df: pd.DataFrame,
    *,
    top_k: int = 5,
    max_rows: int | None = None,
    force_pipeline: bool = False,
) -> dict[str, Any]:
    """
    Delegate to packed LandlordScorer: FE pipeline → predict → SHAP.
    """
    from service.app.config import MAX_SCORE_ROWS

    scorer = get_scorer(force_reload=False)
    limit = max_rows if max_rows is not None else MAX_SCORE_ROWS

    work = df
    truncated = False
    if len(work) > limit:
        work = work.head(limit).copy()
        truncated = True

    features, pipeline_info = scorer.transform(work, force_pipeline=force_pipeline)
    if "LandLordID" not in features.columns:
        features = features.copy()
        features["LandLordID"] = [f"ROW_{i+1}" for i in range(len(features))]

    sparsity = sparse_feature_report(features)
    results = scorer.score_with_explanation(features, top_k=top_k)
    clean = [_json_safe(row) for row in results]

    meta = scorer.metadata_dict()
    warning = sparsity.get("input_warning")
    note = pipeline_info.get("note")
    if note and warning:
        warning = f"{note} {warning}"
    elif pipeline_info.get("pipeline_applied") and not sparsity.get("n_empty_features"):
        warning = None

    return {
        "n_input": int(len(df)),
        "n_scored": int(len(clean)),
        "truncated": truncated,
        "max_rows": limit,
        "model_type": meta.get("model_type"),
        "model_version": meta.get("version"),
        "pipeline": pipeline_info,
        "pipeline_note": note,
        "input_warning": warning,
        "empty_feature_cols": sparsity.get("empty_feature_cols", []),
        "results": clean,
    }
