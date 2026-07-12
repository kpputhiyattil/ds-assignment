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

    land_cfg = cfg.get("landlord_score", {})
    top_pct = land_cfg.get("top_percentile", 70)
    bottom_pct = land_cfg.get("bottom_percentile", 30)
    conf_high = land_cfg.get("confidence_high_threshold", 10)
    conf_med = land_cfg.get("confidence_medium_threshold", 5)
    shrink_m = land_cfg.get("shrinkage_m", 10)

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

    lines.append("## Business question & how to use the score")
    lines.append("")
    lines.append(
        "**Question:** *Is it worth for a business or shop to rent space from a given "
        "landlord?* The pipeline answers this with a single quality signal per landlord "
        "plus the reasons behind it, so a prospective tenant can compare options."
    )
    lines.append("")
    lines.append(
        "- **Score** — `AdjustedScore` (known landlords) or `PredictedScore` (new / "
        "unseen landlords). Higher = tenants under this landlord tend to *outperform* "
        "what their own profile predicts, i.e. the landlord adds value beyond location "
        "and tenant mix."
    )
    lines.append(
        f"- **QualityBand** — `good` (≥{top_pct}th percentile), `bad` "
        f"(≤{bottom_pct}th percentile), else `neutral`. Use the band for a quick "
        "shortlist; use `PercentileRank` to rank finalists."
    )
    lines.append(
        f"- **ConfidenceLevel** — `high` (≥{conf_high} tenants), `medium` "
        f"(≥{conf_med}), else `low`. Scores for landlords with few tenants are pulled "
        f"toward the market average (Empirical-Bayes shrinkage, m={shrink_m}); a `low` "
        "confidence `neutral` often just means *not enough evidence yet*, not *average*."
    )
    lines.append(
        "- **Drivers** — `TopPositive/NegativeDrivers` (local SHAP) tell the tenant "
        "*why*: e.g. strong tenant client base and healthy budgets raise a score."
    )
    lines.append("")
    lines.append(
        f"**Suggested decision rule:** prefer `good` + `high`/`medium` confidence; treat "
        f"`bad` + `high` confidence as a real red flag; for `low` confidence, weight the "
        "SHAP drivers and do independent due diligence rather than trusting the band."
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

    lines.append("## What I'd improve given more time")
    lines.append("")
    lines.append(
        "- **Causal framing** — move beyond association with a temporal / "
        "difference-in-differences design (tenant outcomes before vs after moving "
        "under a landlord) or matching on tenant profile + location to reduce "
        "selection confounding."
    )
    lines.append(
        "- **Recover the missing ~30%** — investigate the unmatched `AllCompanyID` "
        "bridge rows; if they are failed/churned tenants, their absence biases scores "
        "upward (survivorship). Quantify and correct for it."
    )
    lines.append(
        "- **Uncertainty per landlord** — publish a confidence interval / posterior "
        "for each score (e.g. quantile or Bayesian models) instead of a point estimate "
        "plus a coarse confidence band."
    )
    lines.append(
        "- **Location disentanglement** — add explicit geographic controls so the "
        "score reflects the *landlord*, not just a good catchment area."
    )
    lines.append(
        "- **Temporal validation & monitoring** — backtest on a time-split, then track "
        "drift on the served model (the FastAPI scorer already exposes the hooks)."
    )
    lines.append(
        "- **Target robustness** — the success target is a weighted composite; "
        "co-design the weights with domain stakeholders and expand the "
        "sensitivity sweep beyond the current min-Spearman check."
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
