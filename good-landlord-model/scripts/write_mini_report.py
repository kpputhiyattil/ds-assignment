"""
write_mini_report.py — synthesize an executive mini-report from pipeline artifacts.

Reads existing JSON/CSV reports (does not retrain). Safe to run after:
  run_company_targets → run_landlord_models → compare_landlord_models → run_shap_explainability

Usage:
    python scripts/write_mini_report.py
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.config import cfg
from src.utils.reproducibility import repo_relpath, set_seed

logger = logging.getLogger(__name__)


def _load_json(path: Path) -> dict | list | None:
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as fp:
        return json.load(fp)


def _md_table(headers: list[str], rows: list[list[str]]) -> list[str]:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return lines


def build_mini_report() -> str:
    reports = Path(cfg["paths"]["reports_dir"])
    processed = Path(cfg["paths"]["processed_dir"])
    seed = int(cfg.get("seed", 42))

    targets = _load_json(processed / "company_targets_summary.json") or {}
    eval_summary = _load_json(reports / "landlord_model_eval_summary.json") or {}
    recommendation = _load_json(reports / "model_comparison_recommendation.json") or {}
    case_studies = _load_json(reports / "shap_case_studies.json") or []
    importance_path = reports / "shap_global_importance.csv"
    importance = (
        pd.read_csv(importance_path) if importance_path.exists() else pd.DataFrame()
    )

    best = recommendation.get("best_model") or "n/a"
    reason = recommendation.get("reason", "")
    n_landlords = eval_summary.get("n_landlords") or recommendation.get("best_row", {}).get(
        "n_landlords"
    )
    feature_set = eval_summary.get("feature_set", "with_portfolio")

    lines: list[str] = []
    lines.append("# Mini Report — Landlord Quality Scoring")
    lines.append("")
    lines.append(f"_Generated at: {pd.Timestamp.utcnow().isoformat()}_")
    lines.append("")
    lines.append(
        "Executive summary of the good-vs-bad landlord pipeline: "
        "company success targets → OOF company baseline residuals → "
        "EB-shrunk `AdjustedScore` → landlord models (GroupKFold) → SHAP."
    )
    lines.append("")

    lines.append("## Setup & reproducibility")
    lines.append("")
    lines.append(f"- Config seed: **`{seed}`** (`configs/training_config.yaml`)")
    lines.append(f"- Feature set: **`{feature_set}`**")
    lines.append(f"- Landlords scored: **{n_landlords if n_landlords else 'n/a'}**")
    lines.append("- CV: **GroupKFold** by `LandLordID` (leakage control)")
    lines.append(
        "- Reproduce: `pip install -e \".[dev]\"` then run scripts in README order; "
        "`pytest tests/ -v` for unit checks."
    )
    lines.append("")

    lines.append("## Data & targets")
    lines.append("")
    if targets:
        lines.append(
            f"- Companies: **{targets.get('n_companies', 'n/a')}** · "
            f"binary-labeled: **{targets.get('n_binary_labeled', 'n/a')}** "
            f"({targets.get('n_active', 'n/a')} active / "
            f"{targets.get('n_inactive', 'n/a')} inactive)"
        )
        ss = targets.get("success_score") or {}
        if ss:
            lines.append(
                f"- SuccessScore mean={ss.get('mean', float('nan')):.3f}, "
                f"median={ss.get('median', float('nan')):.3f}"
            )
        if "sensitivity_spearman_min" in targets:
            lines.append(
                f"- Target-weight sensitivity (min Spearman): "
                f"**{targets['sensitivity_spearman_min']:.3f}**"
            )
    else:
        lines.append("_Run `scripts/run_company_targets.py` to populate target summary._")
    lines.append("")
    lines.append(
        "EDA notes: ~70% bridge↔Companies match; remaining unmatched IDs are a data gap. "
        "See `reports/eda_report.md`."
    )
    lines.append("")

    lines.append("## Model comparison")
    lines.append("")
    lines.append(f"**Recommended model: `{best}`**")
    lines.append("")
    if reason:
        lines.append(reason)
        lines.append("")

    comparison = recommendation.get("comparison") or []
    if comparison:
        rows = []
        for r in comparison:
            rows.append([
                str(r.get("model_type")),
                f"{r.get('mae_mean', float('nan')):.5f}",
                f"{r.get('rmse_mean', float('nan')):.5f}",
                f"{r.get('r2_mean', float('nan')):.4f}",
                f"{r.get('spearman_mean', float('nan')):.4f}",
            ])
        lines.extend(_md_table(
            ["Model", "MAE", "RMSE", "R²", "Spearman"],
            rows,
        ))
        lines.append("")

    winners = recommendation.get("winners_by_metric") or {}
    if winners:
        lines.append("Metric winners: " + ", ".join(
            f"`{m}`→`{w}`" for m, w in winners.items()
        ))
        lines.append("")

    lines.append("## Explainability (SHAP)")
    lines.append("")
    if not importance.empty:
        top = importance.head(5)
        rows = []
        for r in top.itertuples(index=False):
            label = getattr(r, "label", None) or str(r.feature)
            rows.append([label, f"`{r.feature}`", f"{r.mean_abs_shap:.5f}"])
        lines.extend(_md_table(["Meaning", "Column", "mean |SHAP|"], rows))
        lines.append("")
        lines.append(
            "Portfolio aggregates dominate predictions; landlord demographics "
            "(origin, industry) contribute less on average."
        )
        lines.append("")
    else:
        lines.append("_Run `scripts/run_shap_explainability.py` for SHAP tables/plots._")
        lines.append("")

    if case_studies:
        lines.append("### Case studies")
        lines.append("")
        for cs in case_studies:
            band = str(cs.get("band", "?")).upper()
            lid = cs.get("LandLordID")
            score = cs.get("AdjustedScore")
            tenants = cs.get("TenantCount")
            score_txt = f"{score:.4f}" if isinstance(score, (int, float)) else str(score)
            lines.append(
                f"- **[{band}]** `{lid}` — AdjustedScore={score_txt}"
                + (f", tenants={tenants}" if tenants is not None else "")
            )
            pos = cs.get("top_positive_drivers") or []
            if pos:
                drivers = "; ".join(
                    (
                        f"{d.get('label', d['feature'])} was {d.get('value_display', '?')} "
                        f"(+{d['shap_value']:.3f})"
                    )
                    for d in pos[:2]
                )
                lines.append(f"  - Raised score: {drivers}")
        lines.append("")
        lines.append(
            "Full narratives and waterfalls: `reports/shap_explainability_report.md`."
        )
        lines.append("")

    lines.append("## Limitations")
    lines.append("")
    lines.append(
        "- Association, not causation — tenant selection and location confound landlord effects."
    )
    lines.append(
        "- Survivorship / incomplete `AllCompanyID` coverage (~30% bridge IDs absent from Companies)."
    )
    lines.append(
        "- Single snapshot; small-N landlords are heavily EB-shrunk — treat low-tenant scores cautiously."
    )
    lines.append(
        "- SHAP explains model predictions, not true causal drivers."
    )
    lines.append("")

    lines.append("## Artifact index")
    lines.append("")
    artifacts = [
        ("EDA", "reports/eda_report.md"),
        ("Targets", "data/processed/company_targets_summary.json"),
        ("Model comparison", "reports/model_comparison_report.md"),
        ("Recommendation JSON", "reports/model_comparison_recommendation.json"),
        ("SHAP report", "reports/shap_explainability_report.md"),
        ("Case studies JSON", "reports/shap_case_studies.json"),
        ("Serveable scorer", "data/artifacts/landlord_scorer.joblib"),
        ("Scorer meta", "data/artifacts/landlord_scorer_meta.json"),
        ("This mini report", "reports/mini_report.md"),
    ]
    for label, path in artifacts:
        exists = (_ROOT / path).exists()
        mark = "[x]" if exists else "[ ]"
        lines.append(f"- {mark} **{label}:** `{path}`")
    lines.append("")
    lines.append(f"_Paths relative to repo root; config seed={seed}._")
    lines.append("")

    return "\n".join(lines)


def main() -> Path:
    set_seed(int(cfg.get("seed", 42)))
    out_path = Path(cfg["paths"]["reports_dir"]) / "mini_report.md"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    text = build_mini_report()
    out_path.write_text(text, encoding="utf-8")
    logger.info("Wrote %s", out_path)
    print("\n" + "=" * 60)
    print("MINI REPORT WRITTEN")
    print("=" * 60)
    print(f"Path: {repo_relpath(out_path)}")
    print("=" * 60)
    return out_path


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    main()
