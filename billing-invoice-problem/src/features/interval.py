"""Billing-interval inference from invoice history (Step 3).

There is no ground-truth interval, so inference is a deterministic, auditable
rule engine over the spacing of billing events. Every label reduces to a
traceable statement like "median gap 31d matches monthly (1x30) with 3 observed
cycles => monthly, confidence 0.78".

Design principles
-----------------
* **Historical-only**: only events strictly before the snapshot date are used,
  so the inferred interval obeys the same leakage rules as every other feature.
* **Skipped periods**: a gap is matched to the nearest *multiple* of a base
  period, so a 60-day gap still supports monthly (2x30) with one missed cycle.
* **Eight classes**: ``insufficient_history``, ``one_time_candidate``,
  ``monthly``, ``quarterly``, ``semi_annual``, ``annual``, ``mixed``,
  ``irregular``.
* **One code path**: ``infer_interval_single`` (serving / tests) wraps the same
  batch ``infer_intervals`` used at scale, so there is no train/serve skew.

Confidence combines five factors: number of observed billing cycles, gap
consistency, amount consistency, history length, and whether competing patterns
are present.
"""

from __future__ import annotations

import datetime as dt
import json
import logging

import numpy as np
import polars as pl

from src.config import Config, get_config

logger = logging.getLogger(__name__)

# --- Interval class labels ---
INSUFFICIENT_HISTORY = "insufficient_history"
ONE_TIME_CANDIDATE = "one_time_candidate"
MIXED = "mixed"
IRREGULAR = "irregular"
# Recurring base labels come from cfg.interval.base_periods_days keys, e.g.
# "monthly", "quarterly", "semi_annual", "annual".

GAP = "gap_days"
INTERVAL = "billing_interval"
CONFIDENCE = "interval_confidence"


def _bases(cfg: Config) -> tuple[list[str], np.ndarray]:
    """Return (base names, base day-lengths) sorted by length for determinism."""
    items = sorted(cfg.interval.base_periods_days.items(), key=lambda kv: kv[1])
    names = [k for k, _ in items]
    days = np.array([v for _, v in items], dtype=float)
    return names, days


def classify_gap_bases(gaps: np.ndarray, cfg: Config) -> np.ndarray:
    """Match each gap to the nearest multiple of a base period (or "").

    For each gap and base, the closest integer number of cycles ``k>=1`` is
    taken and the relative error ``|gap - k*base| / (k*base)`` computed. The
    base with the smallest relative error wins if that error is within
    ``match_tolerance``; otherwise the gap is unmatched ("").
    """
    names, base_days = _bases(cfg)
    if gaps.size == 0:
        return np.array([], dtype=object)
    g = gaps.astype(float)[:, None]              # (G, 1)
    b = base_days[None, :]                        # (1, B)
    k = np.clip(np.round(g / b), 1, None)         # nearest cycle count >= 1
    pred = k * b                                  # predicted gap
    rel_err = np.abs(g - pred) / pred             # (G, B)
    best = np.argmin(rel_err, axis=1)
    best_err = rel_err[np.arange(g.shape[0]), best]
    matched = np.where(best_err <= cfg.interval.match_tolerance,
                       np.array(names, dtype=object)[best], "")
    return matched


def _assign_labels_and_confidence(df: pl.DataFrame, cfg: Config) -> pl.DataFrame:
    """Vectorized label + confidence assignment on the per-customer table."""
    horizon = cfg.interval.one_time_horizon_days
    max_cv = cfg.interval.max_gap_cv
    mixed_max = cfg.interval.mixed_dominant_share_max

    n_events = df["n_events"].to_numpy()
    n_gaps = np.nan_to_num(df["n_gaps"].to_numpy().astype(float), nan=0.0)
    gap_cv = np.nan_to_num(df["gap_cv"].to_numpy().astype(float), nan=0.0)
    amount_cv = np.nan_to_num(df["amount_cv"].to_numpy().astype(float), nan=0.0)
    tenure = np.nan_to_num(df["tenure_days"].to_numpy().astype(float), nan=0.0)
    days_since_last = df["days_since_last"].to_numpy().astype(float)
    dominant_base = df["dominant_base"].to_list()
    dominant_share = np.nan_to_num(df["dominant_share"].to_numpy().astype(float), nan=0.0)
    n_distinct = np.nan_to_num(df["n_distinct_bases"].to_numpy().astype(float), nan=0.0)

    n = len(df)
    labels = np.empty(n, dtype=object)

    single = n_events <= 1
    has_base = np.array([b is not None and b != "" for b in dominant_base])
    recurring = ~single
    is_mixed = recurring & has_base & (n_distinct >= 2) & (dominant_share < mixed_max)
    is_base = recurring & has_base & ~is_mixed
    is_irregular = recurring & ~has_base

    # Single-event customers: silence decides one-time vs too-recent.
    labels[single & (days_since_last >= horizon)] = ONE_TIME_CANDIDATE
    labels[single & (days_since_last < horizon)] = INSUFFICIENT_HISTORY
    labels[is_mixed] = MIXED
    labels[is_irregular] = IRREGULAR
    for i in np.where(is_base)[0]:
        labels[i] = dominant_base[i]

    # --- Confidence ---
    conf = np.zeros(n, dtype=float)

    cycles_conf = 1.0 - 1.0 / (1.0 + n_gaps)                 # more cycles -> higher
    reg_conf = np.clip(1.0 - gap_cv / max_cv, 0.0, 1.0)      # low CV -> higher
    amt_conf = np.clip(1.0 - amount_cv, 0.0, 1.0)
    hist_conf = np.clip(tenure / 365.0, 0.0, 1.0)
    blend = 0.40 * cycles_conf + 0.30 * reg_conf + 0.15 * amt_conf + 0.15 * hist_conf

    base_conf = np.clip(blend * dominant_share, 0.0, 1.0)
    conf[is_base] = base_conf[is_base]
    conf[is_mixed] = (0.7 * base_conf)[is_mixed]             # competing-pattern penalty

    conf[is_irregular] = np.clip(
        0.3 * (1.0 - np.clip(gap_cv / max_cv, 0, 1)), 0.05, 0.4
    )[is_irregular]

    ot = single & (days_since_last >= horizon)
    conf[ot] = np.clip(days_since_last / (2.0 * horizon), 0.5, 0.95)[ot]

    ih = single & (days_since_last < horizon)
    conf[ih] = np.clip(0.5 * days_since_last / horizon, 0.0, 0.5)[ih]

    return df.with_columns(
        pl.Series(INTERVAL, labels, dtype=pl.String),
        pl.Series(CONFIDENCE, np.round(conf, 4)),
    )


def infer_intervals(
    events: pl.LazyFrame | pl.DataFrame,
    snapshot: dt.date,
    cfg: Config | None = None,
) -> pl.DataFrame:
    """Infer a billing interval + confidence per customer as of ``snapshot``.

    Only events strictly before ``snapshot`` are used. Returns one row per
    customer with the label, confidence, and the diagnostic statistics that
    produced them.
    """
    cfg = cfg or get_config()
    s = cfg.schema_
    cust, date_col, amount = s.customer_id_col, s.date_col, "event_amount"
    lf = events.lazy() if isinstance(events, pl.DataFrame) else events

    snap = pl.lit(snapshot).cast(pl.Date)
    hist = (
        lf.with_columns(pl.col(date_col).cast(pl.Date).alias("_d"))
        .filter(pl.col("_d") < snap)
        .sort([cust, "_d"])
    )

    per_cust = hist.group_by(cust).agg(
        pl.len().alias("n_events"),
        pl.col("_d").min().alias("first_date"),
        pl.col("_d").max().alias("last_date"),
        pl.col(amount).mean().alias("amount_mean"),
        pl.col(amount).std().alias("amount_std"),
    )

    gaps = (
        hist.with_columns(
            (pl.col("_d").diff().over(cust).dt.total_days()).alias(GAP)
        )
        .drop_nulls(GAP)
        .select([cust, GAP])
    ).collect()

    per_cust = per_cust.collect()

    if gaps.height:
        gap_stats = gaps.group_by(cust).agg(
            pl.col(GAP).median().alias("median_gap_days"),
            pl.col(GAP).mean().alias("gap_mean"),
            pl.col(GAP).std().alias("gap_std"),
            pl.len().alias("n_gaps"),
        )
        matched = classify_gap_bases(gaps[GAP].to_numpy(), cfg)
        gclass = gaps.with_columns(pl.Series("matched_base", matched, dtype=pl.String))
        counts = (
            gclass.filter(pl.col("matched_base") != "")
            .group_by([cust, "matched_base"])
            .agg(pl.len().alias("c"))
        )
        base_dist = counts.group_by(cust).agg(
            pl.col("matched_base").sort_by("c", descending=True).first().alias("dominant_base"),
            pl.col("c").max().alias("dominant_count"),
            pl.col("matched_base").n_unique().alias("n_distinct_bases"),
            pl.col("c").sum().alias("total_matched"),
        ).with_columns(
            (pl.col("dominant_count") / pl.col("total_matched")).alias("dominant_share")
        )
    else:
        gap_stats = per_cust.select(cust).clear()
        base_dist = per_cust.select(cust).clear()

    df = per_cust.join(gap_stats, on=cust, how="left").join(base_dist, on=cust, how="left")

    # Ensure every expected column exists even when no customer had a gap.
    expected: dict[str, pl.DataType] = {
        "median_gap_days": pl.Float64, "gap_mean": pl.Float64, "gap_std": pl.Float64,
        "n_gaps": pl.Int64, "dominant_base": pl.String, "dominant_count": pl.Int64,
        "n_distinct_bases": pl.Int64, "total_matched": pl.Int64,
        "dominant_share": pl.Float64,
    }
    for col, tp in expected.items():
        if col not in df.columns:
            df = df.with_columns(pl.lit(None).cast(tp).alias(col))

    df = df.with_columns(
        (pl.lit(snapshot).cast(pl.Date) - pl.col("last_date")).dt.total_days().alias("days_since_last"),
        (pl.col("last_date") - pl.col("first_date")).dt.total_days().alias("tenure_days"),
        (pl.col("amount_std") / pl.col("amount_mean")).fill_nan(0.0).fill_null(0.0).alias("amount_cv"),
        (pl.col("gap_std") / pl.col("gap_mean")).alias("gap_cv"),
    )
    df = df.with_columns(
        pl.col("n_gaps").fill_null(0),
        pl.col("dominant_share").fill_null(0.0),
        pl.col("n_distinct_bases").fill_null(0),
    )

    result = _assign_labels_and_confidence(df, cfg)
    keep = [cust, INTERVAL, CONFIDENCE, "n_events", "n_gaps", "median_gap_days",
            "gap_cv", "amount_cv", "days_since_last", "tenure_days",
            "dominant_base", "dominant_share", "n_distinct_bases"]
    keep = [c for c in keep if c in result.columns]
    return result.select(keep)


def infer_interval_single(
    dates: list[dt.date],
    amounts: list[float],
    snapshot: dt.date,
    cfg: Config | None = None,
) -> dict:
    """Infer interval for one customer (serving / tests) via the batch path."""
    cfg = cfg or get_config()
    s = cfg.schema_
    if not dates:
        return {INTERVAL: INSUFFICIENT_HISTORY, CONFIDENCE: 0.0, "n_events": 0}
    df = pl.DataFrame(
        {
            s.customer_id_col: ["_single"] * len(dates),
            s.date_col: [dt.datetime(d.year, d.month, d.day) for d in dates],
            "event_amount": list(amounts),
            "n_invoices": [1] * len(dates),
        }
    )
    out = infer_intervals(df, snapshot, cfg)
    if out.height == 0:
        return {INTERVAL: INSUFFICIENT_HISTORY, CONFIDENCE: 0.0, "n_events": 0}
    return out.row(0, named=True)


def run(cfg: Config | None = None) -> pl.DataFrame:
    """Infer intervals for every configured snapshot and persist the result.

    Writes ``data/processed/intervals.parquet`` (one row per customer per
    snapshot) and a per-snapshot class-distribution summary to artifacts.
    """
    cfg = cfg or get_config()
    events = pl.read_parquet(cfg.paths.resolve(cfg.paths.billing_events))

    frames, summary = [], {}
    for snap in cfg.snapshots:
        out = infer_intervals(events, snap, cfg).with_columns(
            pl.lit(snap).cast(pl.Date).alias("snapshot_date")
        )
        frames.append(out)
        dist = (
            out.group_by(INTERVAL).len().sort("len", descending=True)
            .to_dict(as_series=False)
        )
        summary[str(snap)] = {
            "customers": out.height,
            "class_counts": dict(zip(dist[INTERVAL], dist["len"])),
            "mean_confidence": round(float(out[CONFIDENCE].mean()), 4),
        }
        logger.info("Snapshot %s: %d customers, classes=%s",
                    snap, out.height, summary[str(snap)]["class_counts"])

    combined = pl.concat(frames)
    out_path = cfg.paths.resolve(cfg.paths.processed_dir) / "intervals.parquet"
    combined.write_parquet(out_path)

    summary_path = cfg.paths.resolve(cfg.paths.artifacts_dir) / "interval_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    logger.info("Wrote %d interval rows -> %s", combined.height, out_path)
    return combined


def main() -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )
    run()


if __name__ == "__main__":
    main()
