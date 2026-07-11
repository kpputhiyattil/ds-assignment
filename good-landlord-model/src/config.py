"""
config.py — single source of truth for loading training_config.yaml.

Usage anywhere in the codebase:
    from src.config import cfg
    raw_path = cfg["paths"]["raw_landlords"]
"""
from __future__ import annotations

import os
from pathlib import Path

import yaml

# Project root is two levels above this file (src/config.py → project root)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = PROJECT_ROOT / "configs" / "training_config.yaml"


def load_config(path: str | Path = CONFIG_PATH) -> dict:
    """Load YAML config and resolve relative paths against PROJECT_ROOT."""
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    # Resolve all path values to absolute paths
    for key, value in raw.get("paths", {}).items():
        raw["paths"][key] = str(PROJECT_ROOT / value)

    return raw


# Module-level singleton — import `cfg` for convenience
cfg: dict = load_config()
