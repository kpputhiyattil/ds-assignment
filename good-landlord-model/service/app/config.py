"""
Service configuration — paths relative to the good-landlord-model project root.
"""
from __future__ import annotations

import os
from pathlib import Path

# service/ → project root
SERVICE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = SERVICE_DIR.parent

SCORER_PATH = Path(
    os.environ.get(
        "LANDLORD_SCORER_PATH",
        str(PROJECT_ROOT / "data" / "artifacts" / "landlord_scorer.joblib"),
    )
)
META_PATH = Path(
    os.environ.get(
        "LANDLORD_SCORER_META_PATH",
        str(PROJECT_ROOT / "data" / "artifacts" / "landlord_scorer_meta.json"),
    )
)

# Cap rows per request (SHAP is expensive)
MAX_SCORE_ROWS = int(os.environ.get("LANDLORD_SCORE_MAX_ROWS", "100"))
DEFAULT_TOP_K = int(os.environ.get("LANDLORD_SCORE_TOP_K", "5"))
