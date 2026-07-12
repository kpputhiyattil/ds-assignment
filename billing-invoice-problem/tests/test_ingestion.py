"""Tests for the ingestion/validation layer (Step 2)."""

from __future__ import annotations

import datetime as dt

import polars as pl
import pytest

from src.config import get_config
from src.data.ingestion import EVENT_AMOUNT, N_INVOICES, aggregate_same_day
from src.data.validation import (
    SchemaError,
    identifier_report,
    quality_report,
    validate_schema,
)

CFG = get_config()
S = CFG.schema_


def _raw(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(rows).with_columns(pl.col(S.date_col).cast(pl.Date))


def test_aggregate_collapses_same_day() -> None:
    df = _raw(
        [
            # customer A: two invoices on the same day -> one event, summed
            {S.customer_id_col: "A", S.date_col: dt.date(2024, 1, 1), S.amount_col: 100.0},
            {S.customer_id_col: "A", S.date_col: dt.date(2024, 1, 1), S.amount_col: 1.0},
            {S.customer_id_col: "A", S.date_col: dt.date(2024, 2, 1), S.amount_col: 100.0},
            # customer B: single invoice
            {S.customer_id_col: "B", S.date_col: dt.date(2024, 1, 5), S.amount_col: 50.0},
        ]
    )
    out = aggregate_same_day(df, CFG).collect()
    assert out.height == 3  # A has 2 events, B has 1

    a_jan = out.filter(
        (pl.col(S.customer_id_col) == "A") & (pl.col(S.date_col) == dt.date(2024, 1, 1))
    ).row(0, named=True)
    assert a_jan[EVENT_AMOUNT] == 101.0
    assert a_jan[N_INVOICES] == 2


def test_aggregate_sorted_by_customer_then_date() -> None:
    df = _raw(
        [
            {S.customer_id_col: "A", S.date_col: dt.date(2024, 3, 1), S.amount_col: 1.0},
            {S.customer_id_col: "A", S.date_col: dt.date(2024, 1, 1), S.amount_col: 1.0},
        ]
    )
    out = aggregate_same_day(df, CFG).collect()
    dates = out[S.date_col].to_list()
    assert dates == sorted(dates)


def test_validate_schema_raises_on_missing() -> None:
    with pytest.raises(SchemaError):
        validate_schema(["customerid", "amount"], CFG)  # missing 'date'


def test_validate_schema_passes() -> None:
    validate_schema([S.customer_id_col, S.amount_col, S.date_col], CFG)


def test_identifier_report_detects_unique() -> None:
    lf = _raw(
        [
            {S.customer_id_col: "A", S.account_id_col: "acct1",
             S.date_col: dt.date(2024, 1, 1), S.amount_col: 1.0},
            {S.customer_id_col: "B", S.account_id_col: "acct1",
             S.date_col: dt.date(2024, 1, 1), S.amount_col: 1.0},
        ]
    ).lazy()
    rep = identifier_report(lf, CFG)
    assert rep.customer_id_is_unique
    assert rep.key_columns == [S.customer_id_col]


def test_identifier_report_detects_collision() -> None:
    lf = _raw(
        [
            {S.customer_id_col: "A", S.account_id_col: "acct1",
             S.date_col: dt.date(2024, 1, 1), S.amount_col: 1.0},
            {S.customer_id_col: "A", S.account_id_col: "acct2",
             S.date_col: dt.date(2024, 1, 1), S.amount_col: 1.0},
        ]
    ).lazy()
    rep = identifier_report(lf, CFG)
    assert not rep.customer_id_is_unique
    assert rep.key_columns == [S.account_id_col, S.customer_id_col]


def test_quality_report_flags_non_positive() -> None:
    lf = _raw(
        [
            {S.customer_id_col: "A", S.date_col: dt.date(2024, 1, 1), S.amount_col: 10.0},
            {S.customer_id_col: "A", S.date_col: dt.date(2024, 1, 2), S.amount_col: -5.0},
        ]
    ).lazy()
    rep = quality_report(lf, CFG)
    assert rep.n_rows == 2
    assert rep.non_positive_amount_rate == 0.5
    assert rep.null_customer_rate == 0.0


def test_duckdb_matches_polars_aggregation(tmp_path) -> None:
    """DuckDB batch SQL must produce identical events to the Polars transform."""
    import duckdb

    from src.data.ingestion import _events_sql

    df = _raw(
        [
            {S.customer_id_col: "A", S.date_col: dt.date(2024, 1, 1), S.amount_col: 100.0},
            {S.customer_id_col: "A", S.date_col: dt.date(2024, 1, 1), S.amount_col: 1.0},
            {S.customer_id_col: "A", S.date_col: dt.date(2024, 3, 2), S.amount_col: 100.0},
            {S.customer_id_col: "B", S.date_col: dt.date(2024, 1, 5), S.amount_col: 50.0},
            {S.customer_id_col: "B", S.date_col: dt.date(2024, 1, 5), S.amount_col: 50.0},
        ]
    )

    polars_out = aggregate_same_day(df, CFG).collect().sort(
        [S.customer_id_col, S.date_col]
    )

    con = duckdb.connect()
    con.register("t", df.to_arrow())
    duck_out = pl.from_arrow(con.execute(_events_sql(CFG, "t")).arrow()).sort(
        [S.customer_id_col, S.date_col]
    )

    assert polars_out[S.customer_id_col].to_list() == duck_out[S.customer_id_col].to_list()
    assert polars_out[EVENT_AMOUNT].to_list() == duck_out[EVENT_AMOUNT].to_list()
    assert polars_out[N_INVOICES].to_list() == duck_out[N_INVOICES].to_list()
