"""Dollar-churn label construction (Step 4).

Churn is defined on **realized revenue**, deliberately independent of the
inferred billing interval, so the Step-6 ablation ("does the interval help?")
is not circular.

For a customer at ``snapshot``:

* ``past_rev``   = revenue in ``[snapshot - W, snapshot)``            (12 months back)
* ``future_rev`` = revenue in ``[snapshot, snapshot + W + grace)``    (12 months forward)
* ``dollar_churn`` (``y_loss``) = ``max(0, past_rev - future_rev)``
* ``y_churn`` = 1 if ``future_rev < drop_threshold * past_rev`` else 0

Only customers with ``past_rev > 0`` are in the modeling cohort -- you cannot
churn revenue you never had. ``past_rev`` doubles as ``dollar_at_risk`` (the
exposure the credit decision cares about).

The ``grace`` buffer on the forward window means a late-arriving annual invoice
still counts, so annual customers are not spuriously flagged as churned.
"""

from __future__ import annotations

import datetime as dt

import polars as pl

from src.config import Config, get_config

Y_CHURN = "y_churn"
Y_LOSS = "y_loss"
PAST_REV = "past_rev"
FUTURE_REV = "future_rev"
DOLLAR_AT_RISK = "dollar_at_risk"


def compute_labels(
    events: pl.LazyFrame | pl.DataFrame,
    snapshot: dt.date,
    cfg: Config | None = None,
) -> pl.DataFrame:
    """Build dollar-churn labels for the cohort with positive past revenue."""
    cfg = cfg or get_config()
    s = cfg.schema_
    cust, amount = s.customer_id_col, "event_amount"
    w = cfg.churn.outcome_window_days
    grace = cfg.churn.grace_days
    thr = cfg.churn.drop_threshold

    lf = events.lazy() if isinstance(events, pl.DataFrame) else events
    d = pl.col(s.date_col).cast(pl.Date)

    past_start = pl.lit(snapshot - dt.timedelta(days=w)).cast(pl.Date)
    snap = pl.lit(snapshot).cast(pl.Date)
    future_end = pl.lit(snapshot + dt.timedelta(days=w + grace)).cast(pl.Date)

    agg = (
        lf.with_columns(d.alias("_d"))
        .group_by(cust)
        .agg(
            pl.col(amount)
            .filter((pl.col("_d") >= past_start) & (pl.col("_d") < snap))
            .sum()
            .alias(PAST_REV),
            pl.col(amount)
            .filter((pl.col("_d") >= snap) & (pl.col("_d") < future_end))
            .sum()
            .alias(FUTURE_REV),
        )
        .collect()
    )

    agg = agg.with_columns(
        pl.col(PAST_REV).fill_null(0.0),
        pl.col(FUTURE_REV).fill_null(0.0),
    ).filter(pl.col(PAST_REV) > 0.0)  # cohort: must have a revenue baseline

    labels = agg.with_columns(
        (pl.col(PAST_REV) - pl.col(FUTURE_REV)).clip(lower_bound=0.0).alias(Y_LOSS),
        (pl.col(FUTURE_REV) < thr * pl.col(PAST_REV)).cast(pl.Int8).alias(Y_CHURN),
        pl.col(PAST_REV).alias(DOLLAR_AT_RISK),
    )
    return labels.select([cust, Y_CHURN, Y_LOSS, DOLLAR_AT_RISK, PAST_REV, FUTURE_REV])
