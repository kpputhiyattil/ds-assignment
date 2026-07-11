"""Smoke tests for the configuration loader (Step 1).

These assert the scaffold is wired correctly before any ML logic exists.
"""

from __future__ import annotations

import datetime as dt

from src.config import get_config, load_config


def test_config_loads() -> None:
    cfg = get_config()
    assert cfg.random_seed == 42
    assert cfg.schema_.customer_id_col == "customerid"


def test_paths_resolve_absolute() -> None:
    cfg = get_config()
    raw = cfg.paths.resolve(cfg.paths.raw_invoices)
    assert raw.is_absolute()
    assert raw.name == "Invoices_users.parquet"


def test_snapshots_parsed_as_dates() -> None:
    cfg = get_config()
    assert all(isinstance(s, dt.date) for s in cfg.snapshots)
    assert cfg.snapshots == sorted(cfg.snapshots), "snapshots must be chronological"


def test_churn_window_is_twelve_months() -> None:
    cfg = get_config()
    # >= 12-month outcome window so annual customers aren't falsely churned.
    assert cfg.churn.outcome_window_days >= 365


def test_interval_bases_present() -> None:
    cfg = get_config()
    bases = cfg.interval.base_periods_days
    assert {"monthly", "quarterly", "semi_annual", "annual"} <= set(bases)


def test_env_override(monkeypatch, tmp_path) -> None:
    # Nested env override: CHURN__DROP_THRESHOLD should win over YAML.
    monkeypatch.setenv("CHURN__DROP_THRESHOLD", "0.4")
    cfg = load_config()
    assert cfg.churn.drop_threshold == 0.4
