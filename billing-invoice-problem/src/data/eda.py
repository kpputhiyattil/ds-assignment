"""Invoice-focused exploratory data analysis (DuckDB / Polars, out-of-core).

Profiles the raw invoice parquet and (when present) billing events to surface
the structural facts that drive modeling choices: missingness, same-day
multi-invoice collapse, amount skew, customer history depth, and gap shape.

Outputs JSON + Markdown under ``reports/eda/`` plus optional figures.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import duckdb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np
import polars as pl

from src.config import PROJECT_ROOT, Config, get_config
from src.data.ingestion import EVENT_AMOUNT, N_INVOICES

logger = logging.getLogger(__name__)

REPORTS_DIR = PROJECT_ROOT / "reports" / "eda"
FIGURES_DIR = REPORTS_DIR / "figures"


def _pct(x: float) -> str:
    return f"{100.0 * x:.1f}%"


def _safe_float(v: Any) -> float | None:
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def profile_raw_invoices(cfg: Config) -> dict[str, Any]:
    """Schema, missingness, amount/date stats, and customer history depth."""
    raw_path = cfg.paths.resolve(cfg.paths.raw_invoices)
    s = cfg.schema_
    con = duckdb.connect()
    source = f"read_parquet('{raw_path.as_posix()}')"

    columns = pl.scan_parquet(raw_path).collect_schema().names()
    has_account = s.account_id_col in columns

    acct_expr = f"COUNT(DISTINCT {s.account_id_col})" if has_account else "0"
    pair_expr = (
        f"COUNT(DISTINCT ({s.account_id_col} || '|' || {s.customer_id_col}))"
        if has_account
        else f"COUNT(DISTINCT {s.customer_id_col})"
    )

    row = con.execute(
        f"""
        SELECT
            COUNT(*) AS n_rows,
            COUNT(DISTINCT {s.customer_id_col}) AS n_customers,
            {acct_expr} AS n_accounts,
            {pair_expr} AS n_pairs,
            AVG(CASE WHEN {s.customer_id_col} IS NULL THEN 1.0 ELSE 0.0 END) AS null_customer,
            AVG(CASE WHEN {s.date_col} IS NULL THEN 1.0 ELSE 0.0 END) AS null_date,
            AVG(CASE WHEN {s.amount_col} IS NULL THEN 1.0 ELSE 0.0 END) AS null_amount,
            AVG(CASE WHEN {s.amount_col} <= 0 THEN 1.0 ELSE 0.0 END) AS non_pos_amount,
            MIN({s.date_col}) AS date_min,
            MAX({s.date_col}) AS date_max,
            MIN({s.amount_col}) AS amount_min,
            approx_quantile({s.amount_col}, 0.5) AS amount_p50,
            approx_quantile({s.amount_col}, 0.75) AS amount_p75,
            approx_quantile({s.amount_col}, 0.95) AS amount_p95,
            approx_quantile({s.amount_col}, 0.99) AS amount_p99,
            AVG({s.amount_col}) AS amount_mean,
            MAX({s.amount_col}) AS amount_max
        FROM {source}
        """
    ).fetchone()

    (
        n_rows, n_customers, n_accounts, n_pairs,
        null_c, null_d, null_a, non_pos,
        date_min, date_max,
        amt_min, amt_p50, amt_p75, amt_p95, amt_p99, amt_mean, amt_max,
    ) = row

    # Same-day multi-invoice rate (key modeling fact).
    same_day = con.execute(
        f"""
        WITH day_counts AS (
            SELECT {s.customer_id_col}, CAST({s.date_col} AS DATE) AS d, COUNT(*) AS n
            FROM {source}
            GROUP BY 1, 2
        )
        SELECT
            COUNT(*) AS n_customer_days,
            AVG(CASE WHEN n > 1 THEN 1.0 ELSE 0.0 END) AS multi_invoice_day_rate,
            SUM(CASE WHEN n > 1 THEN 1 ELSE 0 END) AS n_multi_days,
            MAX(n) AS max_invoices_on_one_day
        FROM day_counts
        """
    ).fetchone()
    n_customer_days, multi_rate, n_multi_days, max_on_day = same_day

    # Invoices per customer (drives insufficient_history / one_time split).
    inv_pc = con.execute(
        f"""
        WITH per_cust AS (
            SELECT {s.customer_id_col}, COUNT(*) AS n
            FROM {source}
            GROUP BY 1
        )
        SELECT
            AVG(CASE WHEN n = 1 THEN 1.0 ELSE 0.0 END) AS share_single,
            approx_quantile(n, 0.5) AS p50,
            approx_quantile(n, 0.75) AS p75,
            approx_quantile(n, 0.95) AS p95,
            MAX(n) AS max_n,
            AVG(n) AS mean_n
        FROM per_cust
        """
    ).fetchone()
    share_single, inv_p50, inv_p75, inv_p95, inv_max, inv_mean = inv_pc

    return {
        "source": str(raw_path),
        "columns": columns,
        "n_rows": int(n_rows),
        "n_customers": int(n_customers),
        "n_accounts": int(n_accounts),
        "customer_id_is_unique": int(n_pairs) == int(n_customers),
        "missingness": {
            "null_customer_rate": round(float(null_c or 0), 6),
            "null_date_rate": round(float(null_d or 0), 6),
            "null_amount_rate": round(float(null_a or 0), 6),
            "non_positive_amount_rate": round(float(non_pos or 0), 6),
        },
        "date_range": {"min": str(date_min), "max": str(date_max)},
        "amount": {
            "min": _safe_float(amt_min),
            "p50": _safe_float(amt_p50),
            "p75": _safe_float(amt_p75),
            "p95": _safe_float(amt_p95),
            "p99": _safe_float(amt_p99),
            "mean": _safe_float(amt_mean),
            "max": _safe_float(amt_max),
        },
        "same_day": {
            "n_customer_days": int(n_customer_days),
            "multi_invoice_day_rate": round(float(multi_rate or 0), 4),
            "n_multi_invoice_days": int(n_multi_days),
            "max_invoices_on_one_day": int(max_on_day),
        },
        "invoices_per_customer": {
            "share_single_invoice": round(float(share_single or 0), 4),
            "p50": _safe_float(inv_p50),
            "p75": _safe_float(inv_p75),
            "p95": _safe_float(inv_p95),
            "mean": _safe_float(inv_mean),
            "max": int(inv_max),
        },
    }


def profile_billing_events(cfg: Config) -> dict[str, Any] | None:
    """Event counts, amount stats, and inter-event gap distribution."""
    events_path = cfg.paths.resolve(cfg.paths.billing_events)
    if not events_path.exists():
        logger.warning("Billing events not found at %s; skip event/gap EDA", events_path)
        return None

    s = cfg.schema_
    cust = s.customer_id_col
    date_col = s.date_col
    con = duckdb.connect()
    source = f"read_parquet('{events_path.as_posix()}')"

    totals = con.execute(
        f"""
        SELECT
            COUNT(*) AS n_events,
            COUNT(DISTINCT {cust}) AS n_customers,
            AVG({EVENT_AMOUNT}) AS mean_event_amount,
            approx_quantile({EVENT_AMOUNT}, 0.5) AS p50_event_amount,
            MAX({EVENT_AMOUNT}) AS max_event_amount,
            AVG({N_INVOICES}) AS mean_invoices_per_event
        FROM {source}
        """
    ).fetchone()
    n_events, n_customers, mean_amt, p50_amt, max_amt, mean_inv = totals

    events_pc = con.execute(
        f"""
        WITH per_cust AS (
            SELECT {cust}, COUNT(*) AS n
            FROM {source}
            GROUP BY 1
        )
        SELECT
            AVG(CASE WHEN n = 1 THEN 1.0 ELSE 0.0 END) AS share_single_event,
            AVG(CASE WHEN n >= 2 THEN 1.0 ELSE 0.0 END) AS share_recurring,
            approx_quantile(n, 0.5) AS p50,
            approx_quantile(n, 0.75) AS p75,
            approx_quantile(n, 0.95) AS p95,
            MAX(n) AS max_n
        FROM per_cust
        """
    ).fetchone()
    share_1, share_rec, ep50, ep75, ep95, emax = events_pc

    # Inter-event gaps (days) for customers with >=2 events.
    gaps = con.execute(
        f"""
        WITH ordered AS (
            SELECT
                {cust},
                {date_col},
                LAG({date_col}) OVER (PARTITION BY {cust} ORDER BY {date_col}) AS prev_date
            FROM {source}
        ),
        gaps AS (
            SELECT date_diff('day', prev_date, {date_col}) AS gap_days
            FROM ordered
            WHERE prev_date IS NOT NULL
        )
        SELECT
            COUNT(*) AS n_gaps,
            approx_quantile(gap_days, 0.25) AS p25,
            approx_quantile(gap_days, 0.5) AS p50,
            approx_quantile(gap_days, 0.75) AS p75,
            approx_quantile(gap_days, 0.9) AS p90,
            AVG(gap_days) AS mean_gap
        FROM gaps
        """
    ).fetchone()
    n_gaps, g25, g50, g75, g90, gmean = gaps

    # Sample of gap values for histogram (capped).
    gap_sample = con.execute(
        f"""
        WITH ordered AS (
            SELECT
                {cust},
                {date_col},
                LAG({date_col}) OVER (PARTITION BY {cust} ORDER BY {date_col}) AS prev_date
            FROM {source}
        )
        SELECT date_diff('day', prev_date, {date_col}) AS gap_days
        FROM ordered
        WHERE prev_date IS NOT NULL
          AND date_diff('day', prev_date, {date_col}) BETWEEN 1 AND 800
        USING SAMPLE 50000
        """
    ).fetchnumpy()["gap_days"]

    return {
        "source": str(events_path),
        "n_events": int(n_events),
        "n_customers": int(n_customers),
        "event_amount": {
            "mean": _safe_float(mean_amt),
            "p50": _safe_float(p50_amt),
            "max": _safe_float(max_amt),
            "mean_invoices_per_event": _safe_float(mean_inv),
        },
        "events_per_customer": {
            "share_single_event": round(float(share_1 or 0), 4),
            "share_recurring_ge2": round(float(share_rec or 0), 4),
            "p50": _safe_float(ep50),
            "p75": _safe_float(ep75),
            "p95": _safe_float(ep95),
            "max": int(emax),
        },
        "gaps_days": {
            "n_gaps": int(n_gaps),
            "p25": _safe_float(g25),
            "p50": _safe_float(g50),
            "p75": _safe_float(g75),
            "p90": _safe_float(g90),
            "mean": _safe_float(gmean),
        },
        "_gap_sample": np.asarray(gap_sample, dtype=float),
    }


def _derive_findings(raw: dict, events: dict | None) -> list[dict[str, str]]:
    findings: list[dict[str, str]] = []
    miss = raw["missingness"]
    if max(miss.values()) < 1e-6:
        findings.append({
            "severity": "info",
            "area": "missingness",
            "message": (
                "No nulls or non-positive amounts in customerid/date/amount — "
                "imputation is unnecessary; the real missing field is billing interval."
            ),
        })
    else:
        findings.append({
            "severity": "warn",
            "area": "missingness",
            "message": (
                f"Null/non-positive rates: customer={miss['null_customer_rate']}, "
                f"date={miss['null_date_rate']}, amount={miss['null_amount_rate']}, "
                f"non_pos={miss['non_positive_amount_rate']}."
            ),
        })

    sd = raw["same_day"]
    findings.append({
        "severity": "critical" if sd["multi_invoice_day_rate"] > 0.05 else "info",
        "area": "same_day_aggregation",
        "message": (
            f"{_pct(sd['multi_invoice_day_rate'])} of customer-days have >1 invoice "
            f"(max {sd['max_invoices_on_one_day']} on one day) — same-day aggregation "
            "is required before gap/interval logic."
        ),
    })

    ipc = raw["invoices_per_customer"]
    findings.append({
        "severity": "critical" if ipc["share_single_invoice"] > 0.4 else "info",
        "area": "history_depth",
        "message": (
            f"{_pct(ipc['share_single_invoice'])} of customers have a single invoice "
            f"(p50={ipc['p50']}, p75={ipc['p75']}, max={ipc['max']}) — "
            "split insufficient_history vs one_time_candidate; do not blanket one_time."
        ),
    })

    amt = raw["amount"]
    if amt["mean"] and amt["p50"] and amt["mean"] > 3 * amt["p50"]:
        findings.append({
            "severity": "warn",
            "area": "amount_skew",
            "message": (
                f"Amounts are right-skewed (median={amt['p50']}, mean={amt['mean']:.2f}, "
                f"max={amt['max']}) — use log1p / robust summaries for monetary features."
            ),
        })

    if events:
        gaps = events["gaps_days"]
        findings.append({
            "severity": "info",
            "area": "gap_shape",
            "message": (
                f"Inter-event gaps: n={gaps['n_gaps']}, p50={gaps['p50']}d, "
                f"p75={gaps['p75']}d, p90={gaps['p90']}d — match to calendar multiples "
                "(30/91/182/365) allowing skipped periods."
            ),
        })
        epc = events["events_per_customer"]
        findings.append({
            "severity": "info",
            "area": "recurring_share",
            "message": (
                f"After aggregation, {_pct(epc['share_recurring_ge2'])} of customers "
                f"have >=2 billing events (interval inference has a gap signal); "
                f"{_pct(epc['share_single_event'])} remain single-event."
            ),
        })

    findings.append({
        "severity": "info",
        "area": "modeling_implication",
        "message": (
            "Validate inferred intervals indirectly via temporally held-out ablation "
            "on dollar-churn (no ground-truth interval column exists)."
        ),
    })
    return findings


def _save_figures(raw: dict, events: dict | None, out_dir: Path) -> list[str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    figures: list[str] = []

    # Amount distribution (log1p histogram via DuckDB sample would be ideal;
    # use reported quantiles as a bar sketch + optional event amounts).
    fig, ax = plt.subplots(figsize=(7, 4))
    amt = raw["amount"]
    labels = ["p50", "p75", "p95", "p99", "mean"]
    vals = [amt.get(k) or 0 for k in ("p50", "p75", "p95", "p99", "mean")]
    ax.bar(labels, vals, color="#2F5D50")
    ax.set_ylabel("Amount")
    ax.set_title("Invoice amount quantiles (raw)")
    ax.spines[["top", "right"]].set_visible(False)
    path = out_dir / "amount_quantiles.png"
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    figures.append(str(path.relative_to(PROJECT_ROOT)))

    # Invoices-per-customer summary bars.
    fig, ax = plt.subplots(figsize=(7, 4))
    ipc = raw["invoices_per_customer"]
    ax.bar(
        ["single share", "p50", "p75", "p95"],
        [ipc["share_single_invoice"], ipc["p50"] or 0, ipc["p75"] or 0, ipc["p95"] or 0],
        color="#C45C26",
    )
    ax.set_title("Invoices per customer")
    ax.spines[["top", "right"]].set_visible(False)
    path = out_dir / "invoices_per_customer.png"
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    figures.append(str(path.relative_to(PROJECT_ROOT)))

    if events and "_gap_sample" in events:
        sample = events["_gap_sample"]
        if len(sample) > 0:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.hist(sample, bins=40, color="#3A6B8C", edgecolor="white", linewidth=0.3)
            for day, name in [(30, "mo"), (91, "qtr"), (182, "sa"), (365, "yr")]:
                ax.axvline(day, color="#222", linestyle="--", linewidth=0.8, alpha=0.7)
                ax.text(day, ax.get_ylim()[1] * 0.9 if ax.get_ylim()[1] else 1, name,
                        rotation=90, va="top", ha="right", fontsize=8)
            ax.set_xlabel("Gap days (1–800)")
            ax.set_ylabel("Count (sample)")
            ax.set_title("Inter-event gap distribution")
            ax.spines[["top", "right"]].set_visible(False)
            path = out_dir / "gap_histogram.png"
            fig.tight_layout()
            fig.savefig(path, dpi=120)
            plt.close(fig)
            figures.append(str(path.relative_to(PROJECT_ROOT)))

    return figures


def _to_markdown(report: dict) -> str:
    raw = report["raw"]
    events = report.get("events")
    lines = [
        "# Invoice EDA Report",
        "",
        f"_Generated: {report['generated_at']}_",
        "",
        "## Headline findings",
        "",
    ]
    for f in report["findings"]:
        lines.append(f"- **[{f['severity']}] {f['area']}:** {f['message']}")

    lines += [
        "",
        "## Raw invoices",
        "",
        f"- Source: `{raw['source']}`",
        f"- Rows: **{raw['n_rows']:,}** · Customers: **{raw['n_customers']:,}**",
        f"- Date range: {raw['date_range']['min']} → {raw['date_range']['max']}",
        f"- Customer id unique: {raw['customer_id_is_unique']}",
        "",
        "### Missingness",
        "",
        "| Field | Rate |",
        "|---|---|",
    ]
    for k, v in raw["missingness"].items():
        lines.append(f"| {k} | {v} |")

    amt = raw["amount"]
    lines += [
        "",
        "### Amounts",
        "",
        f"min={amt['min']}, p50={amt['p50']}, p75={amt['p75']}, "
        f"p95={amt['p95']}, p99={amt['p99']}, mean={amt['mean']}, max={amt['max']}",
        "",
        "### Same-day multi-invoice",
        "",
        f"- Customer-days: {raw['same_day']['n_customer_days']:,}",
        f"- Multi-invoice day rate: {_pct(raw['same_day']['multi_invoice_day_rate'])}",
        f"- Max invoices on one day: {raw['same_day']['max_invoices_on_one_day']}",
        "",
        "### Invoices per customer",
        "",
        f"- Single-invoice share: {_pct(raw['invoices_per_customer']['share_single_invoice'])}",
        f"- p50 / p75 / p95 / max: "
        f"{raw['invoices_per_customer']['p50']} / "
        f"{raw['invoices_per_customer']['p75']} / "
        f"{raw['invoices_per_customer']['p95']} / "
        f"{raw['invoices_per_customer']['max']}",
    ]

    if events:
        # Strip private sample before display (already in report dict separately handled)
        lines += [
            "",
            "## Billing events (post same-day aggregation)",
            "",
            f"- Events: **{events['n_events']:,}** · Customers: **{events['n_customers']:,}**",
            f"- Collapse ratio (1 − events/raw_rows): "
            f"**{_pct(1 - events['n_events'] / raw['n_rows'])}**",
            f"- Single-event share: "
            f"{_pct(events['events_per_customer']['share_single_event'])}",
            f"- Recurring (>=2 events): "
            f"{_pct(events['events_per_customer']['share_recurring_ge2'])}",
            "",
            "### Gap days",
            "",
            f"n={events['gaps_days']['n_gaps']:,}, "
            f"p25={events['gaps_days']['p25']}, p50={events['gaps_days']['p50']}, "
            f"p75={events['gaps_days']['p75']}, p90={events['gaps_days']['p90']}, "
            f"mean={events['gaps_days']['mean']}",
        ]

    if report.get("figures"):
        lines += ["", "## Figures", ""]
        for fig in report["figures"]:
            lines.append(f"- `{fig}`")

    lines += [
        "",
        "## Implications for the model",
        "",
        "1. Aggregate same-day invoices before computing gaps.",
        "2. Treat short history explicitly (`insufficient_history` vs `one_time_candidate`).",
        "3. Prefer robust / log1p transforms for skewed amounts.",
        "4. Infer interval from pre-snapshot history only; validate via dollar-churn ablation.",
        "",
    ]
    return "\n".join(lines)


def run_eda(cfg: Config | None = None, *, save_plots: bool = True) -> dict[str, Any]:
    """Run invoice EDA and return the in-memory report dict."""
    cfg = cfg or get_config()
    logger.info("Profiling raw invoices…")
    raw = profile_raw_invoices(cfg)
    logger.info("Profiling billing events / gaps…")
    events = profile_billing_events(cfg)
    if events is not None:
        events["collapse_ratio"] = round(1.0 - events["n_events"] / raw["n_rows"], 4)

    findings = _derive_findings(raw, events)
    figures: list[str] = []
    if save_plots:
        figures = _save_figures(raw, events, FIGURES_DIR)

    # Drop numpy sample from serializable events copy.
    events_public = None
    if events is not None:
        events_public = {k: v for k, v in events.items() if not k.startswith("_")}

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        "raw": raw,
        "events": events_public,
        "findings": findings,
        "figures": figures,
    }


def write_eda_report(
    cfg: Config | None = None, *, save_plots: bool = True
) -> tuple[Path, Path]:
    """Write ``reports/eda/eda_report.{json,md}`` and return their paths."""
    cfg = cfg or get_config()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report = run_eda(cfg, save_plots=save_plots)

    json_path = REPORTS_DIR / "eda_report.json"
    md_path = REPORTS_DIR / "eda_report.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(_to_markdown(report), encoding="utf-8")

    # Convenience copies under artifacts for the assessment script.
    art = cfg.paths.resolve(cfg.paths.artifacts_dir)
    art.mkdir(parents=True, exist_ok=True)
    (art / "eda_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    logger.info("Wrote EDA report -> %s", md_path)
    return json_path, md_path


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    write_eda_report()


if __name__ == "__main__":
    main()
