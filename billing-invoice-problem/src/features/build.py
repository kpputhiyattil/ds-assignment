"""Leakage-safe feature engineering (Step 4).

Every feature is computed from events strictly **before** the snapshot, so a
feature can never see the outcome window. Features are organized into three
groups that stay separable for the Step-6 ablations:

* ``CORE``     -- RFM, monetary, tenure, revenue-trend (customer health).
* ``GAP``      -- raw inter-event spacing statistics.
* ``INTERVAL`` -- the inferred billing interval (one-hots) + its confidence
                  and competing-pattern diagnostics.

``build_feature_matrix`` is the single code path used by the batch job, the
tests, and serving. It joins core monetary features to the Step-3 interval
diagnostics and the Step-4 labels, keeping only the modeling cohort (customers
with positive past revenue).
"""

from __future__ import annotations

import datetime as dt
import json
import logging

import polars as pl

from src.config import Config, get_config
from src.data.ingestion import aggregate_same_day
from src.features.interval import (
    CONFIDENCE,
    INTERVAL,
    infer_intervals,
)
from src.features.labels import Y_CHURN, Y_LOSS, compute_labels

logger = logging.getLogger(__name__)

# Canonical interval classes -> one-hot columns (order fixed for determinism).
INTERVAL_CLASSES = [
    "insufficient_history", "one_time_candidate", "monthly", "quarterly",
    "semi_annual", "annual", "mixed", "irregular",
]

# Feature-group registry (consumed by Steps 5-6). Interval one-hots appended below.
CORE_FEATURES = [
    "recency_days", "tenure_days", "n_events", "n_events_past12",
    "total_amount", "past12_amount", "prior12_amount", "mean_amount",
    "last_amount", "max_amount", "n_invoices_total", "revenue_trend", "amount_cv",
]
GAP_FEATURES = ["median_gap_days", "gap_cv", "n_gaps"]
INTERVAL_FEATURES = [CONFIDENCE, "dominant_share", "n_distinct_bases"] + [
    f"is_{c}" for c in INTERVAL_CLASSES
]

LABEL_COLUMNS = [Y_CHURN, Y_LOSS, "dollar_at_risk", "past_rev", "future_rev"]


def _core_features(
    events: pl.LazyFrame, snapshot: dt.date, cfg: Config
) -> pl.DataFrame:
    """Monetary / frequency / trend features from pre-snapshot events only."""
    s = cfg.schema_
    cust, amount = s.customer_id_col, "event_amount"
    w = cfg.churn.outcome_window_days

    d = pl.col(s.date_col).cast(pl.Date)
    snap = pl.lit(snapshot).cast(pl.Date)
    past_start = pl.lit(snapshot - dt.timedelta(days=w)).cast(pl.Date)
    prior_start = pl.lit(snapshot - dt.timedelta(days=2 * w)).cast(pl.Date)

    hist = events.with_columns(d.alias("_d")).filter(pl.col("_d") < snap)

    core = hist.group_by(cust).agg(
        pl.col(amount).sum().alias("total_amount"),
        pl.col(amount).mean().alias("mean_amount"),
        pl.col(amount).max().alias("max_amount"),
        pl.col(amount).sort_by("_d").last().alias("last_amount"),
        pl.col("n_invoices").sum().alias("n_invoices_total"),
        pl.col(amount)
        .filter((pl.col("_d") >= past_start) & (pl.col("_d") < snap))
        .sum()
        .alias("past12_amount"),
        pl.col(amount)
        .filter((pl.col("_d") >= past_start) & (pl.col("_d") < snap))
        .count()
        .alias("n_events_past12"),
        pl.col(amount)
        .filter((pl.col("_d") >= prior_start) & (pl.col("_d") < past_start))
        .sum()
        .alias("prior12_amount"),
    )
    core = core.with_columns(
        # Revenue trend: last-12mo vs prior-12mo. Undefined (no prior revenue)
        # -> null, which LightGBM handles natively.
        pl.when(pl.col("prior12_amount") > 0)
        .then(pl.col("past12_amount") / pl.col("prior12_amount"))
        .otherwise(None)
        .alias("revenue_trend")
    )
    return core.collect()


def build_feature_matrix(
    events: pl.LazyFrame | pl.DataFrame,
    snapshot: dt.date,
    cfg: Config | None = None,
) -> pl.DataFrame:
    """Assemble the full feature+label matrix for one snapshot (cohort only)."""
    cfg = cfg or get_config()
    s = cfg.schema_
    cust = s.customer_id_col
    lf = events.lazy() if isinstance(events, pl.DataFrame) else events

    core = _core_features(lf, snapshot, cfg)

    # Interval diagnostics (Step 3) supply recency, tenure, gap and interval cols.
    iv = infer_intervals(lf, snapshot, cfg).rename({"days_since_last": "recency_days"})
    iv = iv.with_columns(
        [
            (pl.col(INTERVAL) == c).cast(pl.Int8).alias(f"is_{c}")
            for c in INTERVAL_CLASSES
        ]
    )

    labels = compute_labels(lf, snapshot, cfg)

    # Inner joins => cohort = has history AND positive past revenue.
    mat = (
        labels.join(core, on=cust, how="inner")
        .join(iv, on=cust, how="inner")
        .with_columns(pl.lit(snapshot).cast(pl.Date).alias("snapshot_date"))
    )
    return mat


def build_scoring_features(
    events: pl.LazyFrame | pl.DataFrame,
    snapshot: dt.date,
    cfg: Config | None = None,
) -> pl.DataFrame:
    """Serving-time features (no labels, no cohort filter).

    Identical feature engineering to :func:`build_feature_matrix` but without the
    outcome window -- used to score customers at integration time, when the
    future is unknown. Returns one row per customer that has any pre-snapshot
    history, with the inferred interval label attached for the guardrail.
    """
    cfg = cfg or get_config()
    s = cfg.schema_
    cust = s.customer_id_col
    # Serving receives RAW invoices: apply the same same-day aggregation used to
    # build training billing events, so there is no train/serve skew.
    lf = aggregate_same_day(events, cfg)

    core = _core_features(lf, snapshot, cfg)
    iv = infer_intervals(lf, snapshot, cfg).rename({"days_since_last": "recency_days"})
    iv = iv.with_columns(
        [(pl.col(INTERVAL) == c).cast(pl.Int8).alias(f"is_{c}") for c in INTERVAL_CLASSES]
    )
    mat = iv.join(core, on=cust, how="left")
    diag = [
        "n_events", "n_gaps", "median_gap_days", "gap_cv", "amount_cv",
        "dominant_base", "dominant_share", "n_distinct_bases",
        "recency_days", "tenure_days",
    ]
    keep = [cust, INTERVAL, CONFIDENCE] + feature_columns() + diag
    seen: set[str] = set()
    keep = [c for c in keep if c in mat.columns and not (c in seen or seen.add(c))]
    return mat.select(keep)


def feature_columns() -> list[str]:
    """All model feature columns (order stable)."""
    return CORE_FEATURES + GAP_FEATURES + INTERVAL_FEATURES


def run(cfg: Config | None = None) -> pl.DataFrame:
    """Build the feature matrix for every snapshot and persist it."""
    cfg = cfg or get_config()
    events = pl.read_parquet(cfg.paths.resolve(cfg.paths.billing_events))

    frames, summary = [], {}
    for snap in cfg.snapshots:
        mat = build_feature_matrix(events, snap, cfg)
        frames.append(mat)
        summary[str(snap)] = {
            "cohort": mat.height,
            "churn_rate": round(float(mat[Y_CHURN].mean()), 4),
            "mean_dollar_loss": round(float(mat[Y_LOSS].mean()), 2),
            "total_dollars_at_risk": round(float(mat["dollar_at_risk"].sum()), 2),
        }
        logger.info("Snapshot %s: cohort=%d churn_rate=%.3f",
                    snap, mat.height, summary[str(snap)]["churn_rate"])

    combined = pl.concat(frames, how="vertical_relaxed")
    out_path = cfg.paths.resolve(cfg.paths.processed_dir) / "features.parquet"
    combined.write_parquet(out_path)

    groups = {
        "core": CORE_FEATURES,
        "gap": GAP_FEATURES,
        "interval": INTERVAL_FEATURES,
        "labels": LABEL_COLUMNS,
    }
    (cfg.paths.resolve(cfg.paths.artifacts_dir) / "feature_groups.json").write_text(
        json.dumps(groups, indent=2), encoding="utf-8"
    )
    (cfg.paths.resolve(cfg.paths.artifacts_dir) / "feature_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    logger.info("Wrote %d feature rows -> %s", combined.height, out_path)
    return combined


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    run()


if __name__ == "__main__":
    main()
