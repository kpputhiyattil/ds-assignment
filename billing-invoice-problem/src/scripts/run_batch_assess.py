"""Batch-score invoices and write a **timestamped** assessment report.

Uses the packaged artifact (feature engineering + models + SHAP). Scores
customers as of a snapshot and writes a versioned run directory so previous
reports are never overwritten.

Also embeds a short summary of the **interval ablation** (assignment part 3:
churn model with vs without the inferred interval) from
``data/artifacts/ablation_report.json`` when available — that test is *not*
re-run here; it comes from ``make ablation``.

Usage (from project root)::

    make package
    python -m src.scripts.run_batch_assess          # capped (safe)
    python -m src.scripts.run_batch_assess --full   # entire file
"""

from __future__ import annotations

import json
import logging
import shutil
import sys
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import polars as pl  # noqa: E402

from src.config import PROJECT_ROOT, get_config  # noqa: E402
from src.serving.io import load_invoices_from_path  # noqa: E402
from src.serving.package import build_package, load_package  # noqa: E402

logger = logging.getLogger(__name__)

REPORT_ROOT = PROJECT_ROOT / "reports" / "batch_assessment"


def _ensure_package() -> Path:
    cfg = get_config()
    path = cfg.paths.resolve(cfg.paths.artifacts_dir) / "serving" / "churn_decision_model.joblib"
    if not path.exists():
        logger.info("Packaged model missing — building now…")
        build_package(cfg)
    return path


def _load_ablation_summary(art: Path) -> dict[str, Any] | None:
    """Load assignment part-3 interval-value test results if present."""
    path = art / "ablation_report.json"
    if not path.exists():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    abs_ = raw.get("ablations") or {}
    out: dict[str, Any] = {
        "interval_good_enough": raw.get("interval_good_enough"),
        "decision": raw.get("decision"),
        "source": str(path),
        "ablations": {},
    }
    for name, ab in abs_.items():
        overall = ab.get("overall") or {}
        verdict = overall.get("verdict") or {}
        pr = overall.get("pr_auc") or {}
        dr = overall.get("dollar_recall_top10pct") or {}
        mae = overall.get("expected_loss_mae") or {}
        out["ablations"][name] = {
            "base": ab.get("base"),
            "treatment": ab.get("treatment"),
            "n": overall.get("n"),
            "pr_auc_base": pr.get("base"),
            "pr_auc_treatment": pr.get("treatment"),
            "pr_auc_delta": pr.get("delta"),
            "pr_auc_significant": pr.get("significant"),
            "dollar_recall_base": dr.get("base"),
            "dollar_recall_treatment": dr.get("treatment"),
            "dollar_recall_delta": dr.get("delta"),
            "mae_base": mae.get("base"),
            "mae_treatment": mae.get("treatment"),
            "mae_delta": mae.get("delta"),
            "good_enough": verdict.get("good_enough"),
        }
    return out


def _write_markdown(
    summary: dict,
    explanations: list[dict],
    ablation: dict[str, Any] | None,
    out: Path,
    sensitivity: list[dict] | None = None,
) -> None:
    reco = summary.get("recommendation_counts") or {}
    lines = [
        "# Batch Invoice Assessment Report",
        "",
        f"_Generated: {summary.get('generated_at')}_",
        f"_Run dir: `{summary.get('run_dir')}`_",
        "",
        "## 0. Assignment checklist — interval value test",
        "",
        "Part 3 of the brief (*compare churn model with vs without the inferred "
        "interval*) is answered by the **ablation harness** (`make ablation`), "
        "not by re-scoring customers. Batch assessment below is the *serving* "
        "pass (Continue / Review / No-Go on invoice history).",
        "",
    ]
    if ablation is None:
        lines += [
            "_Ablation report not found._ Run `make ablation` then re-run batch "
            "assess. Expected file: `data/artifacts/ablation_report.json`.",
            "",
        ]
    else:
        lines += [
            f"**Verdict:** `{ablation.get('decision')}`",
            f"- `interval_good_enough` = **{ablation.get('interval_good_enough')}**",
            f"- Source: `{ablation.get('source')}`",
            f"- Narrative: `reports/assessment/mini_report.md`",
            "",
            "| Ablation | Base → Treatment | PR-AUC Δ | Dollar-recall@10% Δ | "
            "MAE Δ | Segment good_enough |",
            "|---|---|---:|---:|---:|:---:|",
        ]
        labels = {
            "A1_core_value": "A1 core value",
            "A2_beyond_raw_gaps": "A2 beyond raw gaps",
        }
        for key, row in (ablation.get("ablations") or {}).items():
            lines.append(
                f"| {labels.get(key, key)} | "
                f"`{row.get('base')}` → `{row.get('treatment')}` | "
                f"{row.get('pr_auc_delta')} | "
                f"{row.get('dollar_recall_delta')} | "
                f"{row.get('mae_delta')} | "
                f"{row.get('good_enough')} |"
            )
        lines += [
            "",
            "- **A1**: core features vs core+interval — tests whether the "
            "inferred interval adds value when CRM interval is missing.",
            "- **A2**: core+gap vs core+gap+interval — tests whether the "
            "*derived* interval adds anything beyond raw gap stats.",
            "",
        ]

    lines += [
        "## 1. Serving batch summary",
        "",
        f"- Snapshot: **{summary.get('snapshot')}**",
        f"- Customers scored: **{summary.get('n_customers', 0):,}**",
        f"- Invoice rows loaded: **{summary.get('n_invoice_rows_loaded', '?')}**",
        f"- Recurring-only (>=2 billing days): **{summary.get('recurring_only')}**",
        f"- Truncated: {summary.get('truncated')} "
        f"(max_customers={summary.get('max_customers_applied')}, "
        f"allow_full={summary.get('allow_full')})",
        f"- Model version: `{summary.get('model_version')}`",
        f"- Mean P(churn): {summary.get('mean_churn_probability')}",
        f"- Total expected $ churn: {summary.get('total_expected_dollar_churn'):,}",
        f"- Mean interval confidence: {summary.get('mean_interval_confidence')}",
        "",
        "### Recommendation mix (config thresholds)",
        "",
        "| Recommendation | Count | Share |",
        "|---|---:|---:|",
    ]
    n = max(int(summary.get("n_customers") or 1), 1)
    for k in ("Continue", "Review", "No-Go"):
        c = int(reco.get(k, 0))
        lines.append(f"| {k} | {c:,} | {100 * c / n:.1f}% |")

    if sensitivity:
        lines += [
            "",
            "## 1b. Threshold sensitivity",
            "",
            "Same scored probabilities/confidences, re-bucketed under alternate "
            "decision thresholds. Rows marked `*` use the config defaults "
            f"(Continue≤{summary.get('cfg_continue_max')}, "
            f"No-Go≥{summary.get('cfg_nogo_min')}, "
            f"min_conf={summary.get('cfg_min_conf')}).",
            "",
            "| Cont≤ | No-Go≥ | Min conf | Continue % | Review % | No-Go % | * |",
            "|---:|---:|---:|---:|---:|---:|:---:|",
        ]
        # Keep the table readable: show default min_conf rows + a few others.
        default_conf = summary.get("cfg_min_conf")
        shown = [
            r for r in sensitivity
            if r.get("is_config_default")
            or abs(r["min_interval_confidence"] - float(default_conf or 0.5)) < 1e-9
            or abs(r["min_interval_confidence"] - 0.30) < 1e-9
        ]
        # Deduplicate while preserving order
        seen: set[tuple] = set()
        for r in shown:
            key = (
                r["continue_max_risk"],
                r["nogo_min_risk"],
                r["min_interval_confidence"],
            )
            if key in seen:
                continue
            seen.add(key)
            star = "*" if r.get("is_config_default") else ""
            lines.append(
                f"| {r['continue_max_risk']:.2f} | {r['nogo_min_risk']:.2f} | "
                f"{r['min_interval_confidence']:.2f} | "
                f"{r['continue_pct']:.1f} | {r['review_pct']:.1f} | "
                f"{r['nogo_pct']:.1f} | {star} |"
            )
        lines.append("")
        lines.append(
            "Full sweep (all min_conf × continue × nogo) is in "
            "`threshold_sensitivity.json` in this run folder."
        )
        lines.append("")

    lines += ["", "## 2. Explainable briefings (top risk)", ""]
    if not explanations:
        lines.append("_No explanations requested._")
    for ex in explanations:
        b = ex.get("briefing") or {}
        lines += [
            f"### {ex.get('customer_id')}",
            "",
            f"- Recommendation: **{ex.get('recommendation')}** ({ex.get('reason')})",
            f"- P(churn): {ex.get('churn_probability')} · "
            f"Expected $: {ex.get('expected_dollar_churn')} · "
            f"Interval: `{ex.get('inferred_interval')}` "
            f"(conf={ex.get('interval_confidence')})",
            f"- Tier: {b.get('tier')} · Confidence: {b.get('confidence')}",
            "",
        ]
        main = b.get("main_reason") or {}
        if main:
            lines.append(
                f"**Main reason:** {main.get('label')} "
                f"({main.get('direction')} risk)"
            )
            lines.append("")
        inc = b.get("risk_increasing") or []
        dec = b.get("risk_decreasing") or []
        if inc:
            lines.append("Risk ↑:")
            for it in inc:
                lines.append(
                    f"- {it.get('label')} ({it.get('value')}) "
                    f"[shap {it.get('shap'):+.3f}]"
                )
            lines.append("")
        if dec:
            lines.append("Risk ↓:")
            for it in dec:
                lines.append(
                    f"- {it.get('label')} ({it.get('value')}) "
                    f"[shap {it.get('shap'):+.3f}]"
                )
            lines.append("")

    out.write_text("\n".join(lines), encoding="utf-8")


def _new_run_dir() -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    run_dir = REPORT_ROOT / f"run_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _refresh_latest(run_dir: Path) -> None:
    """Point ``reports/batch_assessment/latest`` at this run (copy, Windows-safe)."""
    latest = REPORT_ROOT / "latest"
    if latest.exists() or latest.is_symlink():
        if latest.is_dir() and not latest.is_symlink():
            shutil.rmtree(latest)
        else:
            latest.unlink()
    # Prefer junction/symlink; fall back to copying key files into latest/.
    try:
        latest.symlink_to(run_dir, target_is_directory=True)
    except OSError:
        latest.mkdir(parents=True, exist_ok=True)
        for name in (
            "batch_assessment_report.md",
            "batch_summary.json",
            "batch_explanations.json",
            "batch_results.parquet",
            "batch_results.csv",
            "ablation_summary.json",
            "threshold_sensitivity.json",
            "threshold_sensitivity.csv",
        ):
            src = run_dir / name
            if src.exists():
                shutil.copy2(src, latest / name)


def run(
    *,
    invoice_path: Path | None = None,
    snapshot: date | None = None,
    explain_top_n: int = 15,
    max_customers: int | None = None,
    allow_full: bool = False,
    recurring_only: bool = False,
) -> Path:
    cfg = get_config()
    _ensure_package()
    model = load_package(cfg)
    art = cfg.paths.resolve(cfg.paths.artifacts_dir)
    ablation = _load_ablation_summary(art)

    path = invoice_path or cfg.paths.resolve(cfg.paths.raw_invoices)
    if allow_full:
        logger.warning(
            "Loading FULL invoice file (%s) — this can use substantial memory.",
            path,
        )
    else:
        logger.info(
            "Loading invoices from %s with customer cap "
            "(pass --full only for intentional full-file runs)",
            path,
        )

    loaded = load_invoices_from_path(
        path,
        cfg,
        max_customers=max_customers,
        allow_full=allow_full,
        recurring_only=recurring_only,
    )
    events = loaded.frame
    snap = snapshot or sorted(cfg.snapshots)[-1]

    logger.info(
        "Scoring %d customers / %d rows as of %s "
        "(truncated=%s, recurring_only=%s, explain_top_n=%d)…",
        loaded.n_customers,
        loaded.n_rows,
        snap,
        loaded.truncated,
        recurring_only,
        explain_top_n,
    )
    out = model.assess_many(events, snap, explain_top_n=explain_top_n)

    probs = [float(r["churn_probability"] or 0.0) for r in out["results"]]
    confs = [float(r["interval_confidence"] or 0.0) for r in out["results"]]
    from src.serving.decision import threshold_sensitivity

    sensitivity = threshold_sensitivity(probs, confs, cfg)

    run_dir = _new_run_dir()
    generated_at = datetime.now(UTC).isoformat()

    results_df = pl.DataFrame(out["results"])
    results_df.write_parquet(run_dir / "batch_results.parquet")
    # CSV cannot store nested dict columns from the dual-model payload.
    nested_names = {"with_interval", "without_interval", "comparison", "briefing"}
    csv_cols = [
        c
        for c in results_df.columns
        if c not in nested_names and results_df[c].dtype != pl.Object
    ]
    results_df.select(csv_cols).write_csv(run_dir / "batch_results.csv")

    sens_df = pl.DataFrame(sensitivity)
    sens_df.write_csv(run_dir / "threshold_sensitivity.csv")
    (run_dir / "threshold_sensitivity.json").write_text(
        json.dumps(sensitivity, indent=2), encoding="utf-8"
    )

    summary = {
        **out["summary"],
        "snapshot": out["snapshot"],
        "model_version": out["model_version"],
        "source": str(path),
        "generated_at": generated_at,
        "run_dir": str(run_dir),
        "truncated": loaded.truncated,
        "max_customers_applied": loaded.max_customers_applied,
        "n_invoice_rows_loaded": loaded.n_rows,
        "allow_full": allow_full,
        "recurring_only": recurring_only,
        "interval_value_test_included": ablation is not None,
        "cfg_continue_max": cfg.decision.continue_max_risk,
        "cfg_nogo_min": cfg.decision.nogo_min_risk,
        "cfg_min_conf": cfg.decision.min_interval_confidence_for_auto,
    }
    (run_dir / "batch_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    (run_dir / "batch_explanations.json").write_text(
        json.dumps(out["explanations"], indent=2), encoding="utf-8"
    )
    if ablation is not None:
        (run_dir / "ablation_summary.json").write_text(
            json.dumps(ablation, indent=2), encoding="utf-8"
        )

    md_path = run_dir / "batch_assessment_report.md"
    _write_markdown(summary, out["explanations"], ablation, md_path, sensitivity)

    art.mkdir(parents=True, exist_ok=True)
    (art / "batch_assessment_report.md").write_text(
        md_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    results_df.write_parquet(art / "batch_results.parquet")
    _refresh_latest(run_dir)

    logger.info(
        "Scored %d customers -> %s (Continue=%s Review=%s No-Go=%s)",
        out["n_customers"],
        md_path,
        summary["recommendation_counts"].get("Continue", 0),
        summary["recommendation_counts"].get("Review", 0),
        summary["recommendation_counts"].get("No-Go", 0),
    )
    if ablation is None:
        logger.warning(
            "Ablation report missing — interval value test not embedded. "
            "Run `make ablation` first."
        )
    else:
        logger.info(
            "Embedded interval-value verdict: %s",
            ablation.get("decision"),
        )
    return md_path


def main() -> None:
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    p = argparse.ArgumentParser(
        description=(
            "Batch-assess invoice file with packaged model. "
            "Writes a timestamped run folder; embeds ablation interval-value "
            "summary and a threshold-sensitivity table. "
            "Defaults to a safe customer cap; pass --full for the entire file."
        )
    )
    p.add_argument("--invoices", type=Path, default=None, help="CSV or Parquet path")
    p.add_argument("--snapshot", type=str, default=None, help="YYYY-MM-DD")
    p.add_argument("--explain-top-n", type=int, default=15)
    p.add_argument(
        "--max-customers",
        type=int,
        default=None,
        help=f"Customer cap (default {get_config().serving.default_max_customers})",
    )
    p.add_argument(
        "--full",
        action="store_true",
        help="Score the entire file (memory-intensive; opt-in only)",
    )
    p.add_argument(
        "--recurring-only",
        action="store_true",
        help="Only customers with >=2 distinct invoice days (more Continues possible)",
    )
    args = p.parse_args()
    snap = date.fromisoformat(args.snapshot) if args.snapshot else None
    path = run(
        invoice_path=args.invoices,
        snapshot=snap,
        explain_top_n=args.explain_top_n,
        max_customers=args.max_customers,
        allow_full=args.full,
        recurring_only=args.recurring_only,
    )
    print(f"\nBatch assessment report: {path}")
    print(f"Latest pointer:        {REPORT_ROOT / 'latest'}")


if __name__ == "__main__":
    main()
