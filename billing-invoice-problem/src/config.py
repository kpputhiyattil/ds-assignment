"""Typed, validated configuration loader.

Single source of truth for paths, seeds, and hyperparameters. Loads
``configs/training_config.yaml`` into nested pydantic models so downstream code
gets attribute access with validation instead of raw dict lookups.

Usage
-----
>>> from src.config import get_config
>>> cfg = get_config()
>>> cfg.churn.outcome_window_days
365

Environment variables override YAML values via a ``__`` nested delimiter, e.g.
``CHURN__DROP_THRESHOLD=0.4`` overrides ``churn.drop_threshold`` and
``RANDOM_SEED=7`` overrides the top-level seed. Overrides are applied before
validation, so an out-of-range override still fails loudly.
"""

from __future__ import annotations

import copy
import datetime as dt
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field, field_validator

# Project root = two levels up from this file (src/config.py -> project/).
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "training_config.yaml"
ENV_NESTED_DELIMITER = "__"


class Paths(BaseModel):
    data_dir: str
    raw_invoices: str
    processed_dir: str
    artifacts_dir: str
    billing_events: str

    def resolve(self, value: str) -> Path:
        """Resolve a configured relative path against the project root."""
        p = Path(value)
        return p if p.is_absolute() else PROJECT_ROOT / p


class Schema(BaseModel):
    customer_id_col: str
    account_id_col: str
    amount_col: str
    date_col: str


class IntervalConfig(BaseModel):
    base_periods_days: dict[str, int]
    match_tolerance: float = Field(gt=0, lt=1)
    one_time_horizon_days: int = Field(gt=0)
    max_gap_cv: float = Field(gt=0)
    mixed_dominant_share_max: float = Field(gt=0, le=1)


class ChurnConfig(BaseModel):
    outcome_window_days: int = Field(gt=0)
    grace_days: int = Field(ge=0)
    drop_threshold: float = Field(gt=0, lt=1)


class ModelConfig(BaseModel):
    frequency: dict
    severity: dict


class DecisionConfig(BaseModel):
    continue_max_risk: float = Field(ge=0, le=1)
    nogo_min_risk: float = Field(ge=0, le=1)
    min_interval_confidence_for_auto: float = Field(ge=0, le=1)


class ServingConfig(BaseModel):
    """Safety limits for interactive file loads (API / Streamlit)."""

    # Default customer cap when the caller does not pass max_customers.
    default_max_customers: int = Field(default=500, gt=0)
    # Absolute ceiling even if the client requests a larger cap.
    hard_max_customers: int = Field(default=5000, gt=0)
    # Reject uploads larger than this (MB) before parsing.
    max_upload_mb: float = Field(default=128.0, gt=0)
    # Hard row ceiling after customer filtering (crash guard).
    max_invoice_rows: int = Field(default=2_000_000, gt=0)


class Config(BaseModel):
    """Root configuration object (validated)."""

    paths: Paths
    random_seed: int
    schema_: Schema = Field(alias="schema")
    interval: IntervalConfig
    churn: ChurnConfig
    model: ModelConfig
    snapshots: list[dt.date]
    decision: DecisionConfig
    serving: ServingConfig = Field(default_factory=ServingConfig)

    model_config = {"populate_by_name": True}

    @field_validator("snapshots", mode="before")
    @classmethod
    def _parse_snapshots(cls, v: list) -> list:
        parsed = [dt.date.fromisoformat(s) if isinstance(s, str) else s for s in v]
        if parsed != sorted(parsed):
            raise ValueError("snapshots must be listed in chronological order")
        return parsed


def _coerce(value: str) -> Any:
    """Best-effort coercion of an env-var string to int/float/bool/str."""
    low = value.lower()
    if low in {"true", "false"}:
        return low == "true"
    for cast in (int, float):
        try:
            return cast(value)
        except ValueError:
            continue
    return value


def _apply_env_overrides(data: dict, environ: dict) -> dict:
    """Deep-merge environment overrides into a config dict.

    An env var ``A__B__C`` sets ``data['a']['b']['c']``. Only env vars whose
    top-level segment matches an existing top-level config key are considered,
    so unrelated environment variables are ignored.
    """
    merged = copy.deepcopy(data)
    top_keys = {k.upper(): k for k in merged}
    for env_key, env_val in environ.items():
        segments = env_key.split(ENV_NESTED_DELIMITER)
        if segments[0] not in top_keys:
            continue
        path = [top_keys[segments[0]]] + [s.lower() for s in segments[1:]]
        cursor: Any = merged
        for key in path[:-1]:
            if not isinstance(cursor, dict) or key not in cursor:
                cursor = None
                break
            cursor = cursor[key]
        if isinstance(cursor, dict) and path[-1] in cursor:
            cursor[path[-1]] = _coerce(env_val)
    return merged


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> Config:
    """Load, apply env overrides to, and validate configuration from YAML."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        raw = yaml.safe_load(fh)
    merged = _apply_env_overrides(raw, dict(os.environ))
    return Config(**merged)


@lru_cache(maxsize=1)
def get_config() -> Config:
    """Return the cached, validated default configuration."""
    return load_config()
