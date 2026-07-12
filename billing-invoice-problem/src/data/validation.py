"""Raw-data validation and profiling for the invoice ingestion layer.

Validation gates run *before* any aggregation so schema drift or a broken
identifier assumption fails loudly instead of silently corrupting downstream
features. All functions accept a Polars ``LazyFrame`` so checks stream over the
22M-row file without materializing it.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import polars as pl

from src.config import Config


class SchemaError(ValueError):
    """Raised when the raw invoice file is missing required columns."""


class IdentifierError(ValueError):
    """Raised when the assumed customer identifier is not unique."""


def required_columns(cfg: Config) -> list[str]:
    """Columns the raw invoice file must contain."""
    s = cfg.schema_
    return [s.customer_id_col, s.amount_col, s.date_col]


def validate_schema(columns: list[str], cfg: Config) -> None:
    """Assert every required column is present; raise ``SchemaError`` if not."""
    missing = [c for c in required_columns(cfg) if c not in columns]
    if missing:
        raise SchemaError(
            f"Raw invoices missing required column(s): {missing}. Present: {columns}"
        )


@dataclass
class IdentifierReport:
    n_customers: int
    n_accounts: int
    n_pairs: int
    customer_id_is_unique: bool
    key_columns: list[str] = field(default_factory=list)


def identifier_report(lf: pl.LazyFrame, cfg: Config) -> IdentifierReport:
    """Determine whether ``customerid`` alone is a safe grain key.

    If a single ``customerid`` ever spans multiple ``accountid`` values, the
    composite ``(accountid, customerid)`` must be used as the key instead.
    """
    s = cfg.schema_
    has_account = s.account_id_col in lf.collect_schema().names()

    n_customers = lf.select(pl.col(s.customer_id_col).n_unique()).collect().item()

    if has_account:
        n_accounts = lf.select(pl.col(s.account_id_col).n_unique()).collect().item()
        n_pairs = (
            lf.select([s.account_id_col, s.customer_id_col]).unique().collect().height
        )
    else:
        n_accounts = 0
        n_pairs = n_customers

    is_unique = n_pairs == n_customers
    key_columns = (
        [s.customer_id_col]
        if is_unique
        else [s.account_id_col, s.customer_id_col]
    )
    return IdentifierReport(
        n_customers=n_customers,
        n_accounts=n_accounts,
        n_pairs=n_pairs,
        customer_id_is_unique=is_unique,
        key_columns=key_columns,
    )


@dataclass
class QualityReport:
    n_rows: int
    null_customer_rate: float
    null_date_rate: float
    null_amount_rate: float
    non_positive_amount_rate: float
    date_min: str
    date_max: str


def quality_report(lf: pl.LazyFrame, cfg: Config) -> QualityReport:
    """Compute null rates, non-positive amounts, and the observed date range."""
    s = cfg.schema_
    agg = lf.select(
        pl.len().alias("n_rows"),
        pl.col(s.customer_id_col).is_null().mean().alias("null_customer"),
        pl.col(s.date_col).is_null().mean().alias("null_date"),
        pl.col(s.amount_col).is_null().mean().alias("null_amount"),
        (pl.col(s.amount_col) <= 0).mean().alias("non_pos_amount"),
        pl.col(s.date_col).min().alias("date_min"),
        pl.col(s.date_col).max().alias("date_max"),
    ).collect()
    row = agg.row(0, named=True)
    return QualityReport(
        n_rows=int(row["n_rows"]),
        null_customer_rate=float(row["null_customer"] or 0.0),
        null_date_rate=float(row["null_date"] or 0.0),
        null_amount_rate=float(row["null_amount"] or 0.0),
        non_positive_amount_rate=float(row["non_pos_amount"] or 0.0),
        date_min=str(row["date_min"]),
        date_max=str(row["date_max"]),
    )


def profile_raw(lf: pl.LazyFrame, cfg: Config) -> tuple[IdentifierReport, QualityReport]:
    """Compute identifier + quality metrics in a SINGLE pass over the file.

    Equivalent to calling ``identifier_report`` and ``quality_report``
    separately, but folds every aggregation into one ``collect`` so the 22M-row
    file is scanned once instead of five times.
    """
    s = cfg.schema_
    has_account = s.account_id_col in lf.collect_schema().names()

    exprs = [
        pl.len().alias("n_rows"),
        pl.col(s.customer_id_col).n_unique().alias("n_customers"),
        pl.col(s.customer_id_col).is_null().mean().alias("null_customer"),
        pl.col(s.date_col).is_null().mean().alias("null_date"),
        pl.col(s.amount_col).is_null().mean().alias("null_amount"),
        (pl.col(s.amount_col) <= 0).mean().alias("non_pos_amount"),
        pl.col(s.date_col).min().alias("date_min"),
        pl.col(s.date_col).max().alias("date_max"),
    ]
    if has_account:
        exprs.append(pl.col(s.account_id_col).n_unique().alias("n_accounts"))
        exprs.append(
            pl.struct([s.account_id_col, s.customer_id_col]).n_unique().alias("n_pairs")
        )

    row = lf.select(exprs).collect().row(0, named=True)

    n_customers = int(row["n_customers"])
    n_accounts = int(row.get("n_accounts", 0) or 0)
    n_pairs = int(row.get("n_pairs", n_customers) or n_customers)
    is_unique = n_pairs == n_customers
    id_rep = IdentifierReport(
        n_customers=n_customers,
        n_accounts=n_accounts,
        n_pairs=n_pairs,
        customer_id_is_unique=is_unique,
        key_columns=[s.customer_id_col]
        if is_unique
        else [s.account_id_col, s.customer_id_col],
    )
    q_rep = QualityReport(
        n_rows=int(row["n_rows"]),
        null_customer_rate=float(row["null_customer"] or 0.0),
        null_date_rate=float(row["null_date"] or 0.0),
        null_amount_rate=float(row["null_amount"] or 0.0),
        non_positive_amount_rate=float(row["non_pos_amount"] or 0.0),
        date_min=str(row["date_min"]),
        date_max=str(row["date_max"]),
    )
    return id_rep, q_rep
