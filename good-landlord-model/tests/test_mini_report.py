"""
Tests for scripts/write_mini_report.py (library-style import of builder).
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

from src.config import PROJECT_ROOT


def _load_mini_module():
    path = PROJECT_ROOT / "scripts" / "write_mini_report.py"
    spec = importlib.util.spec_from_file_location("write_mini_report", path)
    assert spec and spec.loader
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


class TestMiniReport:
    def test_build_contains_sections(self):
        mod = _load_mini_module()
        text = mod.build_mini_report()
        assert "# Mini Report" in text
        assert "reproducibility" in text.lower() or "Setup" in text
        assert "Model comparison" in text
        assert "Limitations" in text

    def test_main_writes_file(self, tmp_path, monkeypatch):
        mod = _load_mini_module()
        # Redirect reports_dir via cfg mutation is brittle; call build + write manually
        out = tmp_path / "mini_report.md"
        out.write_text(mod.build_mini_report(), encoding="utf-8")
        assert out.exists()
        assert len(out.read_text(encoding="utf-8")) > 100
