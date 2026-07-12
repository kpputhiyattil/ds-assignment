"""
Exploratory data analysis for Companies.parquet.

Profiles missingness, status mixes, and the financial / engagement ratios
that drive guardrail thresholds. Run **before** Streamlit or batch assessment.

Outputs
-------
reports/eda/eda_report.json
reports/eda/eda_report.md
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from agent.config import get_settings
from agent.data_loader import DataLoader
from agent.tools.business_profile import STATUS_NORMALISATION

logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
REPORTS_DIR = ROOT / "reports" / "eda"

# Candidate hard-stop cuts evaluated on budget_utilisation_ratio
UTIL_CUTS = (0.10, 0.30, 0.40, 0.50)


def _pct(x: float) -> float:
    return round(100.0 * float(x), 2)


def _quantile_dict(series: pd.Series) -> dict[str, float | None]:
    s = series.dropna()
    if s.empty:
        return {"count": 0, "p25": None, "p50": None, "p75": None, "p90": None, "p95": None, "mean": None}
    qs = s.quantile([0.25, 0.5, 0.75, 0.9, 0.95])
    return {
        "count": int(len(s)),
        "p25": round(float(qs.loc[0.25]), 4),
        "p50": round(float(qs.loc[0.5]), 4),
        "p75": round(float(qs.loc[0.75]), 4),
        "p90": round(float(qs.loc[0.9]), 4),
        "p95": round(float(qs.loc[0.95]), 4),
        "mean": round(float(s.mean()), 4),
    }


def _value_counts(series: pd.Series, top: int = 25) -> dict[str, int]:
    vc = series.fillna("(null)").astype(str).value_counts().head(top)
    return {str(k): int(v) for k, v in vc.items()}


def _normalise_status_series(raw: pd.Series) -> pd.Series:
    def one(v: Any) -> str:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return "unknown"
        if isinstance(v, str) and not v.strip():
            return "unknown"
        return STATUS_NORMALISATION.get(str(v), "unknown")

    return raw.map(one)


def build_eda_report(data_path: str | None = None) -> dict[str, Any]:
    """Compute the full EDA payload (no I/O besides reading parquet)."""
    path = data_path or get_settings().data_path
    loader = DataLoader(path)
    df = loader._df.copy()
    n = len(df)

    # Missingness
    null_rates = {
        col: {
            "null_count": int(df[col].isna().sum()),
            "null_pct": _pct(df[col].isna().mean()),
            "n_unique": int(df[col].nunique(dropna=True)),
        }
        for col in df.columns
    }
    high_null = {
        col: meta
        for col, meta in null_rates.items()
        if meta["null_pct"] >= 40.0
    }

    # Status
    status_raw = _value_counts(df["CompanyStatus"]) if "CompanyStatus" in df.columns else {}
    status_flag = (
        _normalise_status_series(df["CompanyStatus"])
        if "CompanyStatus" in df.columns
        else pd.Series(["unknown"] * n)
    )
    status_norm = _value_counts(status_flag)
    non_active = int(status_flag.isin(["inactive", "suspended", "closed"]).sum())

    # Financial ratios (same definitions as tools)
    sales_main = pd.to_numeric(df.get("SalesMain"), errors="coerce")
    sales_other = pd.to_numeric(df.get("SalesOther"), errors="coerce")
    budget = pd.to_numeric(df.get("MonthlyBudget"), errors="coerce")
    revenue = sales_main.fillna(0) + sales_other.fillna(0)
    both_sales_missing = sales_main.isna() & sales_other.isna()
    revenue = revenue.mask(both_sales_missing)

    util = revenue / budget
    util = util.mask(budget.isna() | (budget.abs() < 1e-9))
    util_positive_budget = util[budget > 0].dropna()

    cut_impact: dict[str, Any] = {}
    denom = max(len(util_positive_budget), 1)
    for cut in UTIL_CUTS:
        n_below = int((util_positive_budget < cut).sum())
        cut_impact[f"below_{cut:.2f}"] = {
            "n": n_below,
            "pct_of_positive_budget": _pct(n_below / denom),
        }

    # Engagement
    returning = pd.to_numeric(df.get("ReturningClients"), errors="coerce")
    todays = pd.to_numeric(df.get("TodaysClients"), errors="coerce")
    retention = returning / todays
    retention = retention.mask(todays.isna() | (todays.abs() < 1e-9))

    primary = _value_counts(df["PrimaryType"]) if "PrimaryType" in df.columns else {}

    findings: list[dict[str, str]] = []

    if high_null:
        cols = ", ".join(sorted(high_null)[:8])
        more = "..." if len(high_null) > 8 else ""
        findings.append(
            {
                "severity": "WARNING",
                "area": "missingness",
                "message": (
                    f"{len(high_null)} columns have >=40% null "
                    f"({cols}{more}). Lower confidence and Review rates will be common."
                ),
            }
        )

    findings.append(
        {
            "severity": "INFO",
            "area": "company_status",
            "message": (
                f"Normalised status: {status_norm}. "
                f"Non-active hard-stop candidates: {non_active}/{n} "
                f"({_pct(non_active / max(n, 1))}%)."
            ),
        }
    )

    med = cut_impact.get("below_0.10", {})
    blunt = cut_impact.get("below_0.30", {})
    util_stats = _quantile_dict(util_positive_budget)
    findings.append(
        {
            "severity": "INFO",
            "area": "budget_utilisation",
            "message": (
                f"Among MonthlyBudget>0 (n={util_stats['count']}), util median≈"
                f"{util_stats['p50']}. Share below 10%={med.get('pct_of_positive_budget')}%; "
                f"below 30%={blunt.get('pct_of_positive_budget')}%. "
                "10% is preferred as a catastrophic hard stop; 30%+ is too blunt."
            ),
        }
    )

    ret_stats = _quantile_dict(retention)
    findings.append(
        {
            "severity": "INFO",
            "area": "retention",
            "message": (
                f"retention_rate measurable for n={ret_stats['count']}; "
                f"median≈{ret_stats['p50']}, p25≈{ret_stats['p25']}."
            ),
        }
    )

    report: dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "data_path": str(Path(path).resolve()),
        "n_rows": n,
        "n_columns": int(df.shape[1]),
        "columns": list(df.columns),
        "null_rates": null_rates,
        "high_null_columns": high_null,
        "company_status_raw": status_raw,
        "company_status_normalised": status_norm,
        "non_active_count": non_active,
        "non_active_pct": _pct(non_active / max(n, 1)),
        "primary_type_top": primary,
        "budget_utilisation": {
            "with_positive_budget": util_stats,
            "cut_impact": cut_impact,
            "missing_or_zero_budget_rows": int(budget.isna().sum() + (budget.abs() < 1e-9).sum()),
        },
        "retention_rate": ret_stats,
        "findings": findings,
        "guardrail_implications": {
            "status_hard_stop": "inactive/suspended/closed → No-Go",
            "budget_burn_threshold": 0.10,
            "budget_burn_rationale": (
                "Cuts at 0.30/0.40/0.50 flag the large majority of rows with "
                "MonthlyBudget>0; 0.10 targets catastrophic coverage only."
            ),
        },
    }
    return report


def _markdown_from_report(report: dict[str, Any]) -> str:
    lines: list[str] = [
        "# Exploratory Data Analysis Report",
        "",
        f"_Generated at: {report['generated_at']}_",
        "",
        f"Raw input: `{report['data_path']}`",
        "",
        f"Shape: **{report['n_rows']}** rows × **{report['n_columns']}** columns",
        "",
        "## Findings (review before agent / guardrail tuning)",
        "",
    ]
    for i, finding in enumerate(report["findings"], 1):
        lines.append(f"### {i}. [{finding['severity']}] {finding['area']}")
        lines.append("")
        lines.append(finding["message"])
        lines.append("")

    lines.extend(
        [
            "## Company status (raw)",
            "",
            "```json",
            json.dumps(report["company_status_raw"], indent=2),
            "```",
            "",
            "## Company status (normalised)",
            "",
            "```json",
            json.dumps(report["company_status_normalised"], indent=2),
            "```",
            "",
            "## Budget utilisation cut impact (MonthlyBudget > 0)",
            "",
            "```json",
            json.dumps(report["budget_utilisation"], indent=2),
            "```",
            "",
            "## Retention rate",
            "",
            "```json",
            json.dumps(report["retention_rate"], indent=2),
            "```",
            "",
            "## High-null columns (>=40%)",
            "",
            "```json",
            json.dumps(report["high_null_columns"], indent=2),
            "```",
            "",
            "## Guardrail implications",
            "",
            "```json",
            json.dumps(report["guardrail_implications"], indent=2),
            "```",
            "",
            "## Next steps",
            "",
            "1. Confirm `BUDGET_BURN_THRESHOLD=0.10` still matches lending appetite.",
            "2. Run unit tests: `pytest tests/ -v`.",
            "3. Launch UI / batch: `streamlit run app/streamlit_app.py` or "
            "`python scripts/batch_assess.py`.",
            "",
        ]
    )
    return "\n".join(lines)


def write_eda_report(data_path: str | None = None) -> tuple[Path, Path]:
    """Run EDA and write JSON + Markdown under ``reports/eda/``."""
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report = build_eda_report(data_path=data_path)

    json_path = REPORTS_DIR / "eda_report.json"
    md_path = REPORTS_DIR / "eda_report.md"
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(_markdown_from_report(report), encoding="utf-8")

    # Also drop a machine-readable copy under data/artifacts for pipeline parity
    artifacts = ROOT / "data" / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    (artifacts / "eda_report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    logger.info("EDA written → %s , %s", json_path, md_path)
    return json_path, md_path
