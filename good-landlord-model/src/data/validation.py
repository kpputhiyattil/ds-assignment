"""
validation.py — Data quality checks for LandLords and Companies datasets.

All checks return a structured QualityReport dict rather than raising immediately,
so the full picture of data quality issues is visible before the pipeline aborts.
A `strict=True` mode raises on any CRITICAL finding; default is to warn and continue.

Checks implemented
------------------
1.  Identifier integrity       — null, duplicate, malformed LandLordID / CompanyID
2.  Relationship consistency   — ActiveCompanies vs. actual bridge count
3.  Date validity              — parse LastUpdated / YearFounded; reject future dates
4.  Range / denominator guards — non-negative numeric fields; guard division-by-zero sources
5.  Nested time-window         — 7-day clients/visitors should not exceed 6/12-month totals
6.  Category quality           — standardise casing; flag rare categories; detect blank strings
7.  Snapshot structure         — detect duplicate IDs (multiple rows per entity)
8.  Bridge coverage            — companies in bridge but absent from Companies.parquet
9.  Missing-value rates        — per-column null percentages with configurable thresholds
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Report container
# ---------------------------------------------------------------------------

SEVERITY_CRITICAL = "CRITICAL"
SEVERITY_WARNING = "WARNING"
SEVERITY_INFO = "INFO"


@dataclass
class QualityFinding:
    check: str
    severity: str          # CRITICAL | WARNING | INFO
    message: str
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class QualityReport:
    findings: list[QualityFinding] = field(default_factory=list)

    def add(self, check: str, severity: str, message: str, **detail):
        self.findings.append(QualityFinding(check, severity, message, detail))

    @property
    def criticals(self) -> list[QualityFinding]:
        return [f for f in self.findings if f.severity == SEVERITY_CRITICAL]

    @property
    def warnings(self) -> list[QualityFinding]:
        return [f for f in self.findings if f.severity == SEVERITY_WARNING]

    def summary(self) -> str:
        lines = [
            f"Quality Report: {len(self.findings)} findings "
            f"({len(self.criticals)} CRITICAL, {len(self.warnings)} WARNING)"
        ]
        for f in self.findings:
            lines.append(f"  [{f.severity}] {f.check}: {f.message}")
        return "\n".join(lines)

    def raise_if_critical(self):
        if self.criticals:
            msgs = "\n".join(f.message for f in self.criticals)
            raise ValueError(f"Data quality CRITICAL failures:\n{msgs}")


# ---------------------------------------------------------------------------
# Check 1 — Identifier integrity
# ---------------------------------------------------------------------------

def check_identifier_integrity(
    df: pd.DataFrame,
    id_col: str,
    dataset_name: str,
    report: QualityReport,
) -> None:
    """Check for null, duplicate, and blank-string IDs."""
    null_count = df[id_col].isna().sum()
    if null_count:
        report.add(
            "identifier_integrity",
            SEVERITY_CRITICAL,
            f"{dataset_name}.{id_col}: {null_count} null values",
            null_count=int(null_count),
        )

    blank_count = (df[id_col].astype(str).str.strip() == "").sum()
    if blank_count:
        report.add(
            "identifier_integrity",
            SEVERITY_CRITICAL,
            f"{dataset_name}.{id_col}: {blank_count} blank/whitespace values",
            blank_count=int(blank_count),
        )

    dup_count = df[id_col].duplicated().sum()
    if dup_count:
        report.add(
            "identifier_integrity",
            SEVERITY_WARNING,
            f"{dataset_name}.{id_col}: {dup_count} duplicate values "
            f"({df[id_col].nunique()} unique out of {len(df)} rows)",
            duplicate_count=int(dup_count),
            unique_count=int(df[id_col].nunique()),
        )
    else:
        report.add(
            "identifier_integrity",
            SEVERITY_INFO,
            f"{dataset_name}.{id_col}: all {len(df)} values unique and non-null",
        )


# ---------------------------------------------------------------------------
# Check 2 — Relationship consistency
# ---------------------------------------------------------------------------

def check_relationship_consistency(
    landlords: pd.DataFrame,
    bridge: pd.DataFrame,
    report: QualityReport,
) -> None:
    """
    Compare ActiveCompanies (claimed count) vs. actual linked company count in bridge.
    Large discrepancies suggest AllCompanyID parsing issues or stale metadata.
    """
    if "ActiveCompanies" not in landlords.columns:
        report.add(
            "relationship_consistency",
            SEVERITY_WARNING,
            "ActiveCompanies column not found in LandLords — skipping consistency check",
        )
        return

    actual_counts = bridge.groupby("LandLordID")["CompanyID"].nunique().rename("ActualCount")
    merged = landlords[["LandLordID", "ActiveCompanies"]].merge(
        actual_counts, on="LandLordID", how="left"
    )
    merged["ActualCount"] = merged["ActualCount"].fillna(0).astype(int)
    merged["Claimed"] = pd.to_numeric(merged["ActiveCompanies"], errors="coerce").fillna(0).astype(int)
    merged["Discrepancy"] = (merged["Claimed"] - merged["ActualCount"]).abs()

    material_threshold = 5  # flag if off by more than 5 companies
    material = merged[merged["Discrepancy"] > material_threshold]
    if not material.empty:
        report.add(
            "relationship_consistency",
            SEVERITY_WARNING,
            f"{len(material)} landlords have ActiveCompanies count differing from "
            f"parsed AllCompanyID count by >{material_threshold}",
            sample_landlord_ids=material["LandLordID"].head(5).tolist(),
            max_discrepancy=int(merged["Discrepancy"].max()),
        )
    else:
        report.add(
            "relationship_consistency",
            SEVERITY_INFO,
            "ActiveCompanies counts are consistent with parsed AllCompanyID (within threshold)",
        )


# ---------------------------------------------------------------------------
# Check 3 — Date validity
# ---------------------------------------------------------------------------

def check_date_validity(
    df: pd.DataFrame,
    date_cols: list[str],
    dataset_name: str,
    report: QualityReport,
    reference_date: pd.Timestamp | None = None,
) -> None:
    """Check date columns for parse failures and future dates."""
    if reference_date is None:
        reference_date = pd.Timestamp.now().normalize()

    for col in date_cols:
        if col not in df.columns:
            report.add(
                "date_validity",
                SEVERITY_WARNING,
                f"{dataset_name}.{col}: column not found — skipping",
            )
            continue

        parsed = pd.to_datetime(df[col], errors="coerce")
        parse_failures = parsed.isna().sum() - df[col].isna().sum()  # new nulls from parse error
        if parse_failures > 0:
            report.add(
                "date_validity",
                SEVERITY_WARNING,
                f"{dataset_name}.{col}: {parse_failures} values could not be parsed as dates",
                parse_failures=int(parse_failures),
            )

        future_dates = (parsed > reference_date).sum()
        if future_dates > 0:
            report.add(
                "date_validity",
                SEVERITY_WARNING,
                f"{dataset_name}.{col}: {future_dates} future dates (after {reference_date.date()})",
                future_count=int(future_dates),
            )

        if col == "YearFounded":
            # Year column — values before 1800 or after current year are suspicious
            numeric = pd.to_numeric(df[col], errors="coerce")
            bad = numeric[(numeric < 1800) | (numeric > reference_date.year)].count()
            if bad > 0:
                report.add(
                    "date_validity",
                    SEVERITY_WARNING,
                    f"{dataset_name}.YearFounded: {bad} implausible year values (<1800 or >current year)",
                    bad_count=int(bad),
                )

        if parse_failures == 0 and future_dates == 0:
            report.add(
                "date_validity",
                SEVERITY_INFO,
                f"{dataset_name}.{col}: all non-null values parse correctly and are non-future",
            )


# ---------------------------------------------------------------------------
# Check 4 — Range and denominator guards
# ---------------------------------------------------------------------------

NON_NEGATIVE_COLS_LANDLORDS = [
    "ActiveCompanies", "Area", "TotalPopulationAround",
]

NON_NEGATIVE_COLS_COMPANIES = [
    "SalesOfMainProduct", "SalesOfOtherProduct", "MonthlyBudget",
    "TotalActiveClients", "TodaysClients", "ClientsInTheLast7Days",
    "ClientsInTheLast6Months", "ClientsInTheLast12Months",
    "TotalClientsInTheLast7Days", "TotalClientsInTheLast6Months",
    "TotalClientsInTheLast12Months",
    "Investments", "ReturningClient", "Rank",
]

DENOMINATOR_COLS = ["Area", "ActiveCompanies", "MonthlyBudget"]


def check_ranges(
    df: pd.DataFrame,
    non_negative_cols: list[str],
    dataset_name: str,
    report: QualityReport,
) -> None:
    """Flag negative values in columns that must be non-negative."""
    for col in non_negative_cols:
        if col not in df.columns:
            continue
        numeric = pd.to_numeric(df[col], errors="coerce")
        neg_count = (numeric < 0).sum()
        if neg_count > 0:
            report.add(
                "range_check",
                SEVERITY_WARNING,
                f"{dataset_name}.{col}: {neg_count} negative values",
                negative_count=int(neg_count),
                min_value=float(numeric.min()),
            )

    # Denominator safety — flag zero/null values in columns used as denominators
    for col in DENOMINATOR_COLS:
        if col not in df.columns:
            continue
        numeric = pd.to_numeric(df[col], errors="coerce")
        zero_or_null = ((numeric == 0) | numeric.isna()).sum()
        if zero_or_null > 0:
            report.add(
                "denominator_guard",
                SEVERITY_INFO,
                f"{dataset_name}.{col}: {zero_or_null} zero/null values — "
                "safe-division (epsilon) applied during feature engineering",
                zero_null_count=int(zero_or_null),
            )


# ---------------------------------------------------------------------------
# Check 5 — Nested time-window consistency
# ---------------------------------------------------------------------------

TIME_WINDOW_PAIRS = [
    # (short_col, long_col, description) — names match Companies.parquet
    ("ClientsInTheLast7Days", "ClientsInTheLast6Months", "clients 7d vs 6m"),
    ("ClientsInTheLast7Days", "ClientsInTheLast12Months", "clients 7d vs 12m"),
    ("ClientsInTheLast6Months", "ClientsInTheLast12Months", "clients 6m vs 12m"),
    ("TotalClientsInTheLast7Days", "TotalClientsInTheLast6Months", "total clients 7d vs 6m"),
    ("TotalClientsInTheLast7Days", "TotalClientsInTheLast12Months", "total clients 7d vs 12m"),
    ("TotalClientsInTheLast6Months", "TotalClientsInTheLast12Months", "total clients 6m vs 12m"),
]


def check_time_window_consistency(
    companies: pd.DataFrame,
    report: QualityReport,
    violation_threshold_pct: float = 0.05,
) -> None:
    """
    Flag rows where shorter-window count exceeds longer-window count.
    A small number of violations may be acceptable (different measurement definitions),
    but a large fraction suggests data alignment problems.
    """
    for short_col, long_col, desc in TIME_WINDOW_PAIRS:
        if short_col not in companies.columns or long_col not in companies.columns:
            continue

        short = pd.to_numeric(companies[short_col], errors="coerce")
        long_ = pd.to_numeric(companies[long_col], errors="coerce")
        both_valid = short.notna() & long_.notna()
        violations = (short[both_valid] > long_[both_valid]).sum()
        total_valid = both_valid.sum()
        pct = violations / total_valid if total_valid > 0 else 0.0

        severity = SEVERITY_WARNING if pct > violation_threshold_pct else SEVERITY_INFO
        report.add(
            "time_window_consistency",
            severity,
            f"{desc}: {violations}/{total_valid} rows ({pct:.1%}) have "
            f"{short_col} > {long_col}",
            violations=int(violations),
            total_valid=int(total_valid),
            violation_pct=round(pct, 4),
        )


# ---------------------------------------------------------------------------
# Check 6 — Category quality
# ---------------------------------------------------------------------------

CATEGORICAL_COLS_LANDLORDS = [
    "PreferredIndustry", "OriginCity", "OriginCountry",
]

CATEGORICAL_COLS_COMPANIES = [
    "CompanyStatus", "PrimaryType", "OriginCity", "OriginCountry",
]


def check_category_quality(
    df: pd.DataFrame,
    cat_cols: list[str],
    dataset_name: str,
    report: QualityReport,
    rare_threshold: float = 0.01,
) -> None:
    """
    Check for: blank strings (not null), high cardinality, rare categories,
    and mixed casing that would create spurious duplicates.
    """
    for col in cat_cols:
        if col not in df.columns:
            continue

        series = df[col].astype(str).where(df[col].notna(), other=None)

        # Blank strings (not pd.NA but empty after strip)
        blank = (series.str.strip() == "").sum()
        if blank > 0:
            report.add(
                "category_quality",
                SEVERITY_WARNING,
                f"{dataset_name}.{col}: {blank} blank/whitespace strings "
                "(should be NaN, not empty string)",
                blank_count=int(blank),
            )

        # Mixed casing — if lowercase version has fewer unique values, casing is inconsistent
        non_null = series.dropna()
        if len(non_null) == 0:
            continue
        raw_unique = non_null.nunique()
        lower_unique = non_null.str.lower().str.strip().nunique()
        if lower_unique < raw_unique:
            report.add(
                "category_quality",
                SEVERITY_WARNING,
                f"{dataset_name}.{col}: {raw_unique - lower_unique} extra unique values "
                "due to casing/whitespace (e.g. 'Active' vs 'active')",
                raw_unique=int(raw_unique),
                normalised_unique=int(lower_unique),
            )

        # Rare categories
        value_counts = non_null.str.strip().value_counts(normalize=True)
        rare = value_counts[value_counts < rare_threshold]
        if not rare.empty:
            report.add(
                "category_quality",
                SEVERITY_INFO,
                f"{dataset_name}.{col}: {len(rare)} rare categories "
                f"(each <{rare_threshold:.0%} of non-null values) — consider grouping as 'Other'",
                rare_category_count=len(rare),
                examples=rare.head(5).index.tolist(),
            )


# ---------------------------------------------------------------------------
# Check 7 — Snapshot structure (duplicate entity IDs)
# ---------------------------------------------------------------------------

def check_snapshot_structure(
    df: pd.DataFrame,
    id_col: str,
    dataset_name: str,
    report: QualityReport,
) -> None:
    """
    Detect whether the dataset has multiple rows per entity (historical snapshots).
    This matters for split strategy and leakage controls.
    """
    if id_col not in df.columns:
        return

    dup_ids = df[df[id_col].duplicated(keep=False)][id_col].nunique()
    total_ids = df[id_col].nunique()

    if dup_ids == 0:
        report.add(
            "snapshot_structure",
            SEVERITY_INFO,
            f"{dataset_name}: one row per {id_col} — cross-sectional dataset (no time series splits needed)",
            total_entities=int(total_ids),
        )
    else:
        report.add(
            "snapshot_structure",
            SEVERITY_WARNING,
            f"{dataset_name}: {dup_ids}/{total_ids} {id_col} values appear in multiple rows — "
            "dataset may have historical snapshots; use time-based splitting",
            duplicate_id_count=int(dup_ids),
            total_id_count=int(total_ids),
        )


# ---------------------------------------------------------------------------
# Check 8 — Bridge coverage
# ---------------------------------------------------------------------------

def check_bridge_coverage(
    bridge: pd.DataFrame,
    companies: pd.DataFrame,
    report: QualityReport,
) -> None:
    """
    Find CompanyIDs in bridge that have no matching record in Companies.parquet.
    These rows will have NaN for all company features.
    """
    bridge_ids = set(bridge["CompanyID"].astype(str))
    company_ids = set(companies["CompanyID"].astype(str))
    missing = bridge_ids - company_ids
    pct = len(missing) / len(bridge_ids) if bridge_ids else 0.0

    severity = SEVERITY_CRITICAL if pct > 0.20 else (
        SEVERITY_WARNING if pct > 0.05 else SEVERITY_INFO
    )
    report.add(
        "bridge_coverage",
        severity,
        f"{len(missing)}/{len(bridge_ids)} unique CompanyIDs in bridge not found "
        f"in Companies ({pct:.1%} unmatched)",
        missing_count=int(len(missing)),
        total_bridge_ids=int(len(bridge_ids)),
        missing_pct=round(pct, 4),
        sample_missing=sorted(missing)[:5],
    )


# ---------------------------------------------------------------------------
# Check 9 — Missing-value rates
# ---------------------------------------------------------------------------

def check_missing_rates(
    df: pd.DataFrame,
    dataset_name: str,
    report: QualityReport,
    critical_threshold: float = 0.50,
    warning_threshold: float = 0.20,
) -> None:
    """Report per-column null rates; flag columns above thresholds."""
    null_rates = df.isna().mean().sort_values(ascending=False)
    high = null_rates[null_rates >= critical_threshold]
    medium = null_rates[(null_rates >= warning_threshold) & (null_rates < critical_threshold)]

    if not high.empty:
        report.add(
            "missing_rates",
            SEVERITY_CRITICAL,
            f"{dataset_name}: {len(high)} columns with >={critical_threshold:.0%} missing values",
            columns={col: f"{rate:.1%}" for col, rate in high.items()},
        )
    if not medium.empty:
        report.add(
            "missing_rates",
            SEVERITY_WARNING,
            f"{dataset_name}: {len(medium)} columns with "
            f"{warning_threshold:.0%}–{critical_threshold:.0%} missing values",
            columns={col: f"{rate:.1%}" for col, rate in medium.items()},
        )
    if high.empty and medium.empty:
        report.add(
            "missing_rates",
            SEVERITY_INFO,
            f"{dataset_name}: all columns below {warning_threshold:.0%} missing threshold",
        )


# ---------------------------------------------------------------------------
# Top-level orchestrator
# ---------------------------------------------------------------------------

def validate_all(
    landlords: pd.DataFrame,
    companies: pd.DataFrame,
    bridge: pd.DataFrame,
    strict: bool = False,
) -> QualityReport:
    """
    Run all quality checks and return a consolidated QualityReport.

    Parameters
    ----------
    landlords : raw LandLords DataFrame
    companies : raw Companies DataFrame
    bridge    : exploded (LandLordID, CompanyID) bridge
    strict    : if True, raise ValueError on any CRITICAL finding

    Returns
    -------
    QualityReport with all findings.
    """
    report = QualityReport()

    logger.info("Running data quality checks...")

    # 1. Identifier integrity
    check_identifier_integrity(landlords, "LandLordID", "LandLords", report)
    check_identifier_integrity(companies, "CompanyID", "Companies", report)

    # 2. Relationship consistency
    check_relationship_consistency(landlords, bridge, report)

    # 3. Date validity
    check_date_validity(
        landlords,
        ["LastUpdated", "YearFounded"],
        "LandLords",
        report,
    )
    check_date_validity(
        companies,
        ["YearFounded", "LastUpdated"],
        "Companies",
        report,
    )

    # 4. Range / denominator guards
    check_ranges(landlords, NON_NEGATIVE_COLS_LANDLORDS, "LandLords", report)
    check_ranges(companies, NON_NEGATIVE_COLS_COMPANIES, "Companies", report)

    # 5. Time-window consistency
    check_time_window_consistency(companies, report)

    # 6. Category quality
    check_category_quality(landlords, CATEGORICAL_COLS_LANDLORDS, "LandLords", report)
    check_category_quality(companies, CATEGORICAL_COLS_COMPANIES, "Companies", report)

    # 7. Snapshot structure
    check_snapshot_structure(landlords, "LandLordID", "LandLords", report)
    check_snapshot_structure(companies, "CompanyID", "Companies", report)

    # 8. Bridge coverage
    check_bridge_coverage(bridge, companies, report)

    # 9. Missing-value rates
    check_missing_rates(landlords, "LandLords", report)
    check_missing_rates(companies, "Companies", report)

    logger.info(report.summary())

    if strict:
        report.raise_if_critical()

    return report


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json
    import sys
    from pathlib import Path

    # Allow `python src/data/validation.py` as well as `python -m src.data.validation`
    _root = Path(__file__).resolve().parents[2]
    if str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    from src.data.ingestion import build_model_base, load_companies, load_landlords

    landlords = load_landlords()
    companies = load_companies()
    bridge, _ = build_model_base(landlords, companies)

    report = validate_all(landlords, companies, bridge, strict=False)

    # Save report as JSON for inspection
    from src.config import cfg

    processed_dir = Path(cfg["paths"]["processed_dir"])
    processed_dir.mkdir(parents=True, exist_ok=True)
    report_path = processed_dir / "data_quality_report.json"

    report_dict = [
        {
            "check": f.check,
            "severity": f.severity,
            "message": f.message,
            "detail": f.detail,
        }
        for f in report.findings
    ]
    with open(report_path, "w") as fp:
        json.dump(report_dict, fp, indent=2, default=str)

    print(report.summary())
    print(f"\nFull report saved to {report_path}")

    # Exit non-zero if any CRITICAL findings
    sys.exit(1 if report.criticals else 0)
