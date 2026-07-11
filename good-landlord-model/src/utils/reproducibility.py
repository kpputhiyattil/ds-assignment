"""
reproducibility.py — seed control and path helpers for deterministic runs.
"""
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Any

import numpy as np

from src.config import PROJECT_ROOT


def set_seed(seed: int = 42) -> int:
    """
    Seed Python, NumPy, and hash-affecting env for reproducible pipelines.

    Also attempts to seed torch / tensorflow if installed (no-op if absent).
    Returns the seed used.
    """
    seed = int(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass

    try:
        import tensorflow as tf

        tf.random.set_seed(seed)
    except ImportError:
        pass

    return seed


def repo_relpath(path: Path | str, root: Path | None = None) -> str:
    """Return a forward-slash path relative to the project root when possible."""
    root = root or PROJECT_ROOT
    p = Path(path)
    try:
        return p.resolve().relative_to(root.resolve()).as_posix()
    except ValueError:
        return p.as_posix()


def json_safe(obj: Any) -> Any:
    """Convert Path / numpy scalars in nested structures for JSON dumps."""
    if isinstance(obj, Path):
        return repo_relpath(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, dict):
        return {k: json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    return obj
