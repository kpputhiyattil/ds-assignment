# Invoice EDA Report

_Generated: 2026-07-12T14:09:42.432680+00:00_

## Headline findings

- **[info] missingness:** No nulls or non-positive amounts in customerid/date/amount — imputation is unnecessary; the real missing field is billing interval.
- **[critical] same_day_aggregation:** 17.2% of customer-days have >1 invoice (max 9387 on one day) — same-day aggregation is required before gap/interval logic.
- **[critical] history_depth:** 62.0% of customers have a single invoice (p50=1.0, p75=2.0, max=9387) — split insufficient_history vs one_time_candidate; do not blanket one_time.
- **[warn] amount_skew:** Amounts are right-skewed (median=1.0, mean=10.77, max=11880.0) — use log1p / robust summaries for monetary features.
- **[info] gap_shape:** Inter-event gaps: n=282613, p50=8.0d, p75=31.0d, p90=365.0d — match to calendar multiples (30/91/182/365) allowing skipped periods.
- **[info] recurring_share:** After aggregation, 23.9% of customers have >=2 billing events (interval inference has a gap signal); 76.0% remain single-event.
- **[info] modeling_implication:** Validate inferred intervals indirectly via temporally held-out ablation on dollar-churn (no ground-truth interval column exists).

## Raw invoices

- Source: `D:\Personal\DS\ds-assignment\billing-invoice-problem\ds-assignment\billing-invoice-problem\data\raw\Invoices_users.parquet`
- Rows: **22,083,629** · Customers: **562,851**
- Date range: 2023-01-09 00:00:00 → 2026-04-05 00:00:00
- Customer id unique: True

### Missingness

| Field | Rate |
|---|---|
| null_customer_rate | 0.0 |
| null_date_rate | 0.0 |
| null_amount_rate | 0.0 |
| non_positive_amount_rate | 0.0 |

### Amounts

min=0.95, p50=1.0, p75=1.0, p95=1.0, p99=348.0, mean=10.773526551274768, max=11880.0

### Same-day multi-invoice

- Customer-days: 845,464
- Multi-invoice day rate: 17.2%
- Max invoices on one day: 9387

### Invoices per customer

- Single-invoice share: 62.0%
- p50 / p75 / p95 / max: 1.0 / 2.0 / 123.0 / 9387

## Billing events (post same-day aggregation)

- Events: **845,464** · Customers: **562,851**
- Collapse ratio (1 − events/raw_rows): **96.2%**
- Single-event share: 76.0%
- Recurring (≥2 events): 23.9%

### Gap days

n=282,613, p25=2.0, p50=8.0, p75=31.0, p90=365.0, mean=69.82910198752357

## Figures

- `reports\eda\figures\amount_quantiles.png`
- `reports\eda\figures\invoices_per_customer.png`
- `reports\eda\figures\gap_histogram.png`

## Implications for the model

1. Aggregate same-day invoices before computing gaps.
2. Treat short history explicitly (`insufficient_history` vs `one_time_candidate`).
3. Prefer robust / log1p transforms for skewed amounts.
4. Infer interval from pre-snapshot history only; validate via dollar-churn ablation.
