"""Invoice ingestion: raw invoices -> validated per-customer billing events.

The single most important transform here is **same-day aggregation**: ~17% of
``(customerid, date)`` pairs contain more than one invoice (invoice lines /
related charges). Collapsing them into one billing event per customer-day is a
prerequisite for correct inter-invoice gap calculations downstream -- without
it, spurious 0-day gaps corrupt every inferred interval.

Two engines, by design:

* ``aggregate_same_day`` -- a pure **Polars** transform that defines the
  canonical aggregation semantics. It is what serving uses on a single
  customer's invoices at integration time, and what the unit tests exercise.
* ``build_billing_events`` -- the **DuckDB** batch job for the full 22M-row
  file (out-of-core distinct counts + grouped aggregation are far faster in
  DuckDB). ``test_duckdb_matches_polars`` guarantees the DuckDB SQL and the
  Polars transform produce identical output, so there is no train/serve skew.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict
from pathlib import Path

import duckdb
import polars as pl

from src.config import Config, get_config
from src.data.validation import (
    IdentifierError,
    IdentifierReport,
    QualityReport,
    validate_schema,
)

logger = logging.getLogger(__name__)

# Output schema of a billing event (one row per customer per billing day).
EVENT_AMOUNT = "event_amount"
N_INVOICES = "n_invoices"


def aggregate_same_day(frame: pl.LazyFrame | pl.DataFrame, cfg: Config) -> pl.LazyFrame:
    """Collapse invoices to one billing event per (customer, date).

    Canonical aggregation semantics, shared with serving. Sums the amount and
    counts the invoice lines per event, then sorts by customer and date so gap
    logic can diff consecutive rows directly.
    """
    lf = frame.lazy() if isinstance(frame, pl.DataFrame) else frame
    s = cfg.schema_
    return (
        lf.group_by([s.customer_id_col, s.date_col])
        .agg(
            pl.col(s.amount_col).sum().alias(EVENT_AMOUNT),
            pl.len().alias(N_INVOICES),
        )
        .sort([s.customer_id_col, s.date_col])
    )


def _events_sql(cfg: Config, source: str) -> str:
    """DuckDB SQL mirroring ``aggregate_same_day`` (same columns, same order)."""
    s = cfg.schema_
    return f"""
        SELECT
            {s.customer_id_col},
            {s.date_col},
            SUM({s.amount_col}) AS {EVENT_AMOUNT},
            COUNT(*)            AS {N_INVOICES}
        FROM {source}
        GROUP BY {s.customer_id_col}, {s.date_col}
        ORDER BY {s.customer_id_col}, {s.date_col}
    """


def _profile_duckdb(con: duckdb.DuckDBPyConnection, cfg: Config, source: str,
                    has_account: bool) -> tuple[IdentifierReport, QualityReport]:
    """Single-query identifier + quality profile over the raw file (fast)."""
    s = cfg.schema_
    pair_expr = (
        f"COUNT(DISTINCT ({s.account_id_col} || '|' || {s.customer_id_col}))"
        if has_account
        else f"COUNT(DISTINCT {s.customer_id_col})"
    )
    acct_expr = f"COUNT(DISTINCT {s.account_id_col})" if has_account else "0"
    row = con.execute(
        f"""
        SELECT
            COUNT(*)                                   AS n_rows,
            COUNT(DISTINCT {s.customer_id_col})        AS n_customers,
            {acct_expr}                                AS n_accounts,
            {pair_expr}                                AS n_pairs,
            AVG(CASE WHEN {s.customer_id_col} IS NULL THEN 1 ELSE 0 END) AS null_customer,
            AVG(CASE WHEN {s.date_col} IS NULL THEN 1 ELSE 0 END)        AS null_date,
            AVG(CASE WHEN {s.amount_col} IS NULL THEN 1 ELSE 0 END)      AS null_amount,
            AVG(CASE WHEN {s.amount_col} <= 0 THEN 1 ELSE 0 END)         AS non_pos_amount,
            MIN({s.date_col})                          AS date_min,
            MAX({s.date_col})                          AS date_max
        FROM {source}
        """
    ).fetchone()

    (n_rows, n_customers, n_accounts, n_pairs, null_c, null_d, null_a,
     non_pos, date_min, date_max) = row
    is_unique = int(n_pairs) == int(n_customers)
    id_rep = IdentifierReport(
        n_customers=int(n_customers),
        n_accounts=int(n_accounts),
        n_pairs=int(n_pairs),
        customer_id_is_unique=is_unique,
        key_columns=[s.customer_id_col]
        if is_unique
        else [s.account_id_col, s.customer_id_col],
    )
    q_rep = QualityReport(
        n_rows=int(n_rows),
        null_customer_rate=float(null_c or 0.0),
        null_date_rate=float(null_d or 0.0),
        null_amount_rate=float(null_a or 0.0),
        non_positive_amount_rate=float(non_pos or 0.0),
        date_min=str(date_min),
        date_max=str(date_max),
    )
    return id_rep, q_rep


def build_billing_events(cfg: Config | None = None, *, strict: bool = True) -> Path:
    """Run the full ingestion pipeline and write ``billing_events.parquet``.

    Steps: validate schema -> profile (identifier + quality) -> same-day
    aggregation -> write parquet + JSON summary. Uses DuckDB for scale.
    """
    cfg = cfg or get_config()
    s = cfg.schema_
    raw_path = cfg.paths.resolve(cfg.paths.raw_invoices)
    out_path = cfg.paths.resolve(cfg.paths.billing_events)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Scanning raw invoices: %s", raw_path)
    columns = pl.scan_parquet(raw_path).collect_schema().names()
    validate_schema(columns, cfg)
    has_account = s.account_id_col in columns

    con = duckdb.connect()
    source = f"read_parquet('{raw_path.as_posix()}')"

    id_rep, q_rep = _profile_duckdb(con, cfg, source, has_account)
    logger.info(
        "Identifier: %d customers, %d accounts, unique=%s, key=%s",
        id_rep.n_customers, id_rep.n_accounts,
        id_rep.customer_id_is_unique, id_rep.key_columns,
    )
    if strict and not id_rep.customer_id_is_unique:
        raise IdentifierError(
            f"'{s.customer_id_col}' is not unique across accounts; "
            f"use composite key {id_rep.key_columns}."
        )
    logger.info(
        "Quality: %d rows, null_cust=%.4f, non_pos_amount=%.4f, dates %s..%s",
        q_rep.n_rows, q_rep.null_customer_rate,
        q_rep.non_positive_amount_rate, q_rep.date_min, q_rep.date_max,
    )

    logger.info("Aggregating same-day invoices -> %s", out_path)
    con.execute(
        f"COPY ({_events_sql(cfg, source)}) "
        f"TO '{out_path.as_posix()}' (FORMAT PARQUET)"
    )

    ev = pl.scan_parquet(out_path)
    n_events = ev.select(pl.len()).collect().item()

    summary = {
        "raw_rows": q_rep.n_rows,
        "billing_events": int(n_events),
        "customers": id_rep.n_customers,
        "same_day_collapse_ratio": round(1 - n_events / q_rep.n_rows, 4),
        "identifier": asdict(id_rep),
        "quality": asdict(q_rep),
        "output": str(out_path),
    }
    summary_path = cfg.paths.resolve(cfg.paths.artifacts_dir) / "ingestion_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info(
        "Wrote %d billing events for %d customers (collapsed %.1f%% of rows)",
        n_events, id_rep.n_customers, summary["same_day_collapse_ratio"] * 100,
    )
    return out_path


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    build_billing_events()


if __name__ == "__main__":
    main()
