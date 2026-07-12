# Exploratory Data Analysis Report

_Generated at: 2026-07-12T19:24:32.550049+00:00_

Raw input: `D:\Personal\DS\ds-assignment\credit-assessment-agent\ds-assignment\credit-assessment-agent\data\raw\Companies.parquet`

Shape: **26981** rows × **20** columns

## Findings (review before agent / guardrail tuning)

### 1. [WARNING] missingness

10 columns have >=40% null (Clients12M, Clients6M, Clients7D, CompanyStatus, MonthlyBudget, Rank, Visitors12M, Visitors6M...). Lower confidence and Review rates will be common.

### 2. [INFO] company_status

Normalised status: {'active': 13527, 'unknown': 12962, 'closed': 364, 'inactive': 128}. Non-active hard-stop candidates: 492/26981 (1.82%).

### 3. [INFO] budget_utilisation

Among MonthlyBudget>0 (n=5342), util median≈0.0092. Share below 10%=81.75%; below 30%=92.18%. 10% is preferred as a catastrophic hard stop; 30%+ is too blunt.

### 4. [INFO] retention

retention_rate measurable for n=26739; median≈0.3333, p25≈0.1385.

## Company status (raw)

```json
{
  "Actively Seeking New Employees": 13315,
  "(null)": 12938,
  "Out of Business": 364,
  "Acquired/Merged (Operating Subsidiary)": 194,
  "Acquired/Merged": 73,
  "Not Making New Products": 47,
  "Other": 24,
  "Will Consider New Projects": 18,
  "Reducing Activity": 8
}
```

## Company status (normalised)

```json
{
  "active": 13527,
  "unknown": 12962,
  "closed": 364,
  "inactive": 128
}
```

## Budget utilisation cut impact (MonthlyBudget > 0)

```json
{
  "with_positive_budget": {
    "count": 5342,
    "p25": 0.0001,
    "p50": 0.0092,
    "p75": 0.06,
    "p90": 0.2262,
    "p95": 0.553,
    "mean": 20.8588
  },
  "cut_impact": {
    "below_0.10": {
      "n": 4367,
      "pct_of_positive_budget": 81.75
    },
    "below_0.30": {
      "n": 4924,
      "pct_of_positive_budget": 92.18
    },
    "below_0.40": {
      "n": 4998,
      "pct_of_positive_budget": 93.56
    },
    "below_0.50": {
      "n": 5043,
      "pct_of_positive_budget": 94.4
    }
  },
  "missing_or_zero_budget_rows": 21639
}
```

## Retention rate

```json
{
  "count": 26739,
  "p25": 0.1385,
  "p50": 0.3333,
  "p75": 1.0,
  "p90": 1.0,
  "p95": 1.0,
  "mean": 0.4574
}
```

## High-null columns (>=40%)

```json
{
  "MonthlyBudget": {
    "null_count": 21631,
    "null_pct": 80.17,
    "n_unique": 3103
  },
  "Rank": {
    "null_count": 23217,
    "null_pct": 86.05,
    "n_unique": 3137
  },
  "YearFounded": {
    "null_count": 14940,
    "null_pct": 55.37,
    "n_unique": 209
  },
  "CompanyStatus": {
    "null_count": 12938,
    "null_pct": 47.95,
    "n_unique": 8
  },
  "Clients7D": {
    "null_count": 26481,
    "null_pct": 98.15,
    "n_unique": 6
  },
  "Clients6M": {
    "null_count": 19805,
    "null_pct": 73.4,
    "n_unique": 43
  },
  "Clients12M": {
    "null_count": 15781,
    "null_pct": 58.49,
    "n_unique": 63
  },
  "Visitors7D": {
    "null_count": 26067,
    "null_pct": 96.61,
    "n_unique": 7
  },
  "Visitors6M": {
    "null_count": 16739,
    "null_pct": 62.04,
    "n_unique": 78
  },
  "Visitors12M": {
    "null_count": 12639,
    "null_pct": 46.84,
    "n_unique": 113
  }
}
```

## Guardrail implications

```json
{
  "status_hard_stop": "inactive/suspended/closed \u2192 No-Go",
  "budget_burn_threshold": 0.1,
  "budget_burn_rationale": "Cuts at 0.30/0.40/0.50 flag the large majority of rows with MonthlyBudget>0; 0.10 targets catastrophic coverage only."
}
```

## Next steps

1. Confirm `BUDGET_BURN_THRESHOLD=0.10` still matches lending appetite.
2. Run unit tests: `pytest tests/ -v`.
3. Launch UI / batch: `streamlit run app/streamlit_app.py` or `python scripts/batch_assess.py`.
