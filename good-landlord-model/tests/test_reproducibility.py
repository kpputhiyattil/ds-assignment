"""
Tests for src/utils/reproducibility.py and config seed wiring.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from src.config import PROJECT_ROOT, cfg, load_config
from src.utils.reproducibility import json_safe, repo_relpath, set_seed


class TestSetSeed:
    def test_returns_seed(self):
        assert set_seed(123) == 123

    def test_sets_pythonhashseed_env(self):
        set_seed(7)
        assert os.environ.get("PYTHONHASHSEED") == "7"

    def test_numpy_reproducible(self):
        set_seed(42)
        a = np.random.rand(5)
        set_seed(42)
        b = np.random.rand(5)
        np.testing.assert_array_equal(a, b)

    def test_different_seeds_differ(self):
        set_seed(1)
        a = np.random.rand(5)
        set_seed(2)
        b = np.random.rand(5)
        assert not np.array_equal(a, b)


class TestRepoRelpath:
    def test_relative_under_project(self, tmp_path):
        # Use a real path under PROJECT_ROOT
        target = PROJECT_ROOT / "configs" / "training_config.yaml"
        rel = repo_relpath(target)
        assert rel == "configs/training_config.yaml"
        assert "\\" not in rel

    def test_outside_root_returns_posix(self, tmp_path):
        outside = tmp_path / "other.txt"
        outside.write_text("x", encoding="utf-8")
        out = repo_relpath(outside)
        assert out.replace("\\", "/").endswith("other.txt")


class TestJsonSafe:
    def test_path_converted(self):
        p = PROJECT_ROOT / "reports"
        assert json_safe(p) == "reports"

    def test_numpy_scalars(self):
        assert json_safe(np.int64(3)) == 3
        assert isinstance(json_safe(np.float64(1.5)), float)

    def test_nested(self):
        payload = {"path": PROJECT_ROOT / "configs", "n": np.int32(2)}
        out = json_safe(payload)
        assert out["path"] == "configs"
        assert out["n"] == 2


class TestConfigSeed:
    def test_cfg_has_seed(self):
        assert "seed" in cfg
        assert int(cfg["seed"]) == 42

    def test_load_config_resolves_paths(self):
        loaded = load_config()
        assert Path(loaded["paths"]["reports_dir"]).is_absolute() or "reports" in str(
            loaded["paths"]["reports_dir"]
        )
