"""Invoice I/O helpers for serving: CSV / Parquet / in-memory rows.

Interactive paths (API / Streamlit) **never** materialize the full 22M-row file.
Loaders apply customer caps *before* collecting rows (lazy parquet/CSV scan when
possible) and reject oversized uploads.
"""

from __future__ import annotations

import datetime as dt
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from src.config import Config, get_config

logger = logging.getLogger(__name__)

# Common aliases accepted from uploaded files / manual forms.
_CUSTOMER_ALIASES = {"customerid", "customer_id", "customer", "cust_id", "id_customer"}
_DATE_ALIASES = {"date", "invoice_date", "billing_date", "invoice_dt"}
_AMOUNT_ALIASES = {"amount", "invoice_amount", "amt", "value", "total"}

# Safe defaults when config.serving is unavailable.
DEFAULT_MAX_CUSTOMERS = 500
HARD_MAX_CUSTOMERS = 5_000
DEFAULT_MAX_UPLOAD_MB = 128.0
DEFAULT_MAX_INVOICE_ROWS = 2_000_000


@dataclass
class LoadResult:
    """Normalized invoices plus truncation metadata."""

    frame: pl.DataFrame
    n_rows: int
    n_customers: int
    max_customers_applied: int | None
    truncated: bool
    source: str


class UploadTooLargeError(ValueError):
    """Raised when an upload exceeds the configured byte limit."""


def _resolve_col(columns: list[str], aliases: set[str], canonical: str) -> str:
    lower = {c.lower(): c for c in columns}
    for alias in aliases:
        if alias in lower:
            return lower[alias]
    raise ValueError(
        f"Could not find column for '{canonical}'. "
        f"Looked for {sorted(aliases)}; got {columns}."
    )


def _serving_limits(cfg: Config) -> tuple[int, int, float, int]:
    serving = getattr(cfg, "serving", None)
    if serving is None:
        return (
            DEFAULT_MAX_CUSTOMERS,
            HARD_MAX_CUSTOMERS,
            DEFAULT_MAX_UPLOAD_MB,
            DEFAULT_MAX_INVOICE_ROWS,
        )
    return (
        int(serving.default_max_customers),
        int(serving.hard_max_customers),
        float(serving.max_upload_mb),
        int(serving.max_invoice_rows),
    )


def _clamp_max_customers(
    max_customers: int | None,
    cfg: Config,
    *,
    allow_full: bool = False,
) -> int | None:
    """Resolve the customer cap. Full file only when ``allow_full=True``."""
    default_cap, hard_cap, _, _ = _serving_limits(cfg)
    if allow_full and (max_customers is None or max_customers <= 0):
        return None
    if max_customers is None or max_customers <= 0:
        return default_cap
    return min(int(max_customers), hard_cap)


def normalize_invoice_frame(df: pl.DataFrame, cfg: Config | None = None) -> pl.DataFrame:
    """Map arbitrary invoice columns onto the config schema."""
    cfg = cfg or get_config()
    s = cfg.schema_
    cols = df.columns
    cust = _resolve_col(cols, _CUSTOMER_ALIASES | {s.customer_id_col.lower()}, s.customer_id_col)
    date_c = _resolve_col(cols, _DATE_ALIASES | {s.date_col.lower()}, s.date_col)
    amt = _resolve_col(cols, _AMOUNT_ALIASES | {s.amount_col.lower()}, s.amount_col)

    out = df.select(
        pl.col(cust).cast(pl.Utf8).alias(s.customer_id_col),
        pl.col(date_c).alias(s.date_col),
        pl.col(amt).cast(pl.Float64).alias(s.amount_col),
    )
    if out[s.date_col].dtype == pl.Utf8:
        out = out.with_columns(pl.col(s.date_col).str.to_datetime(strict=False))
    elif out[s.date_col].dtype == pl.Date:
        out = out.with_columns(pl.col(s.date_col).cast(pl.Datetime("us")))
    elif out[s.date_col].dtype not in (pl.Datetime,):
        out = out.with_columns(pl.col(s.date_col).cast(pl.Datetime("us"), strict=False))

    out = out.filter(
        pl.col(s.customer_id_col).is_not_null()
        & pl.col(s.date_col).is_not_null()
        & pl.col(s.amount_col).is_not_null()
        & (pl.col(s.amount_col) > 0)
    )
    if out.height == 0:
        raise ValueError("No valid invoice rows after normalization (check columns/amounts).")
    return out


def _column_map(columns: list[str], cfg: Config) -> tuple[str, str, str]:
    s = cfg.schema_
    cust = _resolve_col(columns, _CUSTOMER_ALIASES | {s.customer_id_col.lower()}, s.customer_id_col)
    date_c = _resolve_col(columns, _DATE_ALIASES | {s.date_col.lower()}, s.date_col)
    amt = _resolve_col(columns, _AMOUNT_ALIASES | {s.amount_col.lower()}, s.amount_col)
    return cust, date_c, amt


def _limit_lazy_by_customers(
    lf: pl.LazyFrame,
    cust_col: str,
    date_col: str,
    max_customers: int | None,
    customer_id: str | None,
    *,
    recurring_only: bool = False,
    min_events: int = 2,
) -> tuple[pl.LazyFrame, int | None, bool]:
    """Filter a lazy frame to one customer or the first N customer ids."""
    if customer_id:
        return lf.filter(pl.col(cust_col).cast(pl.Utf8) == str(customer_id)), 1, True

    if max_customers is None and not recurring_only:
        return lf, None, False

    # Prefer customers with enough billing days when recurring_only is set.
    id_lf = lf.select(
        pl.col(cust_col).cast(pl.Utf8).alias(cust_col),
        pl.col(date_col).cast(pl.Date).alias("_d"),
    )
    if recurring_only:
        id_lf = (
            id_lf.group_by(cust_col)
            .agg(pl.col("_d").n_unique().alias("n_days"))
            .filter(pl.col("n_days") >= min_events)
            .select(cust_col)
        )
    else:
        id_lf = id_lf.select(cust_col).unique(maintain_order=True)

    if max_customers is not None:
        id_lf = id_lf.head(max_customers)

    ids = id_lf.collect().get_column(cust_col).to_list()
    truncated = max_customers is not None and len(ids) >= max_customers
    return (
        lf.filter(pl.col(cust_col).cast(pl.Utf8).is_in(ids)),
        max_customers,
        truncated,
    )


def _collect_capped(lf: pl.LazyFrame, max_rows: int) -> pl.DataFrame:
    df = lf.collect()
    if df.height > max_rows:
        logger.warning(
            "Loaded %d invoice rows exceeds max_invoice_rows=%d; truncating rows "
            "(prefer a smaller max_customers).",
            df.height,
            max_rows,
        )
        df = df.head(max_rows)
    return df


def load_invoices_from_path(
    path: str | Path,
    cfg: Config | None = None,
    *,
    max_customers: int | None = None,
    customer_id: str | None = None,
    allow_full: bool = False,
    recurring_only: bool = False,
    min_events: int = 2,
) -> LoadResult:
    """Load CSV/Parquet with an early customer cap (lazy scan when possible).

    Set ``allow_full=True`` only for offline batch jobs that intentionally score
    the entire file. ``recurring_only`` keeps customers with >= ``min_events``
    distinct invoice days (after column normalize, before collect).
    """
    cfg = cfg or get_config()
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    _, _, _, max_rows = _serving_limits(cfg)
    cap = _clamp_max_customers(max_customers, cfg, allow_full=allow_full)
    suffix = path.suffix.lower()

    if suffix == ".parquet":
        lf = pl.scan_parquet(path)
    elif suffix in {".csv", ".tsv"}:
        sep = "\t" if suffix == ".tsv" else ","
        lf = pl.scan_csv(path, try_parse_dates=True, separator=sep)
    else:
        raise ValueError(f"Unsupported file type '{suffix}'. Use .csv or .parquet.")

    schema_names = lf.collect_schema().names()
    cust_src, date_src, amt_src = _column_map(schema_names, cfg)
    s = cfg.schema_

    lf = lf.select(
        pl.col(cust_src).cast(pl.Utf8).alias(s.customer_id_col),
        pl.col(date_src).alias(s.date_col),
        pl.col(amt_src).cast(pl.Float64).alias(s.amount_col),
    )
    lf, applied, truncated = _limit_lazy_by_customers(
        lf,
        s.customer_id_col,
        s.date_col,
        cap,
        customer_id,
        recurring_only=recurring_only,
        min_events=min_events,
    )
    df = _collect_capped(lf, max_rows)
    df = normalize_invoice_frame(df, cfg)

    n_cust = df[s.customer_id_col].n_unique()
    logger.info(
        "Loaded %d rows / %d customers from %s "
        "(max_customers=%s, truncated=%s, recurring_only=%s)",
        df.height,
        n_cust,
        path.name,
        applied,
        truncated,
        recurring_only,
    )
    return LoadResult(
        frame=df,
        n_rows=df.height,
        n_customers=n_cust,
        max_customers_applied=applied,
        truncated=truncated,
        source=str(path),
    )


def load_invoices_from_bytes(
    data: bytes,
    filename: str,
    cfg: Config | None = None,
    *,
    max_customers: int | None = None,
    customer_id: str | None = None,
    allow_full: bool = False,
    recurring_only: bool = False,
    min_events: int = 2,
) -> LoadResult:
    """Load an upload safely: size-check, temp file, then capped lazy scan."""
    cfg = cfg or get_config()
    _, _, max_upload_mb, _ = _serving_limits(cfg)
    max_bytes = int(max_upload_mb * 1024 * 1024)
    if len(data) > max_bytes:
        raise UploadTooLargeError(
            f"Upload is {len(data) / (1024 * 1024):.1f} MB; "
            f"limit is {max_upload_mb:.0f} MB. "
            "Score a customer subset or use the offline batch job with --full."
        )

    name = filename.lower()
    if name.endswith(".parquet"):
        suffix = ".parquet"
    elif name.endswith(".csv"):
        suffix = ".csv"
    elif name.endswith(".tsv"):
        suffix = ".tsv"
    else:
        raise ValueError("Upload must be .csv or .parquet")

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(data)
        tmp_path = Path(tmp.name)

    try:
        result = load_invoices_from_path(
            tmp_path,
            cfg,
            max_customers=max_customers,
            customer_id=customer_id,
            allow_full=allow_full,
            recurring_only=recurring_only,
            min_events=min_events,
        )
        result.source = filename
        return result
    finally:
        tmp_path.unlink(missing_ok=True)


def invoices_from_records(
    customer_id: str,
    invoices: list[dict],
    cfg: Config | None = None,
) -> pl.DataFrame:
    """Build a normalized frame from manual {date, amount} records."""
    cfg = cfg or get_config()
    s = cfg.schema_
    if not invoices:
        raise ValueError("At least one invoice is required.")
    dates, amounts = [], []
    for inv in invoices:
        d = inv["date"]
        if isinstance(d, str):
            d = dt.date.fromisoformat(d)
        if isinstance(d, dt.datetime):
            d = d.date()
        dates.append(dt.datetime(d.year, d.month, d.day))
        amounts.append(float(inv["amount"]))
    return pl.DataFrame(
        {
            s.customer_id_col: [str(customer_id)] * len(dates),
            s.date_col: dates,
            s.amount_col: amounts,
        }
    )


# Back-compat: older call sites expecting a bare DataFrame.
def load_invoices_from_path_df(path: str | Path, cfg: Config | None = None) -> pl.DataFrame:
    return load_invoices_from_path(path, cfg).frame


def load_invoices_from_bytes_df(
    data: bytes, filename: str, cfg: Config | None = None
) -> pl.DataFrame:
    return load_invoices_from_bytes(data, filename, cfg).frame
