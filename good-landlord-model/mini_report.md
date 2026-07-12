# Mini Report — Landlord Quality Scoring

_Generated at: 2026-07-12T20:30:33.200635+00:00_

Executive summary of the good-vs-bad landlord pipeline: company success targets → OOF company baseline residuals → EB-shrunk `AdjustedScore` → landlord models (GroupKFold) → SHAP.

## Business question & how to use the score

**Question:** *Is it worth for a business or shop to rent space from a given landlord?* The pipeline answers this with a single quality signal per landlord plus the reasons behind it, so a prospective tenant can compare options.

- **Score** — `AdjustedScore` (known landlords) or `PredictedScore` (new / unseen landlords). Higher = tenants under this landlord tend to *outperform* what their own profile predicts, i.e. the landlord adds value beyond location and tenant mix.
- **QualityBand** — `good` (≥70th percentile), `bad` (≤30th percentile), else `neutral`. Use the band for a quick shortlist; use `PercentileRank` to rank finalists.
- **ConfidenceLevel** — `high` (≥10 tenants), `medium` (≥5), else `low`. Scores for landlords with few tenants are pulled toward the market average (Empirical-Bayes shrinkage, m=10); a `low` confidence `neutral` often just means *not enough evidence yet*, not *average*.
- **Drivers** — `TopPositive/NegativeDrivers` (local SHAP) tell the tenant *why*: e.g. strong tenant client base and healthy budgets raise a score.

**Suggested decision rule:** prefer `good` + `high`/`medium` confidence; treat `bad` + `high` confidence as a real red flag; for `low` confidence, weight the SHAP drivers and do independent due diligence rather than trusting the band.

## Setup & reproducibility

- Config seed: **`42`** (`configs/training_config.yaml`)
- Feature set: **`with_portfolio`**
- Landlords scored: **14838**
- CV: **GroupKFold** by `LandLordID` (leakage control)
- Reproduce: `pip install -e ".[dev]"` then run scripts in README order; `pytest tests/ -v` for unit checks.

## Data & targets

- Companies: **26981** · binary-labeled: **13752** (13333 active / 419 inactive)
- SuccessScore mean=0.494, median=0.517
- Target-weight sensitivity (min Spearman): **0.541**

EDA notes: ~70% bridge↔Companies match; remaining unmatched IDs are a data gap. See `reports/eda_report.md`.

## Model comparison

**Recommended model: `catboost`**

catboost is recommended: best mae_mean=0.01192 (±0.00022), spearman_mean=0.87069. Near-tie with xgboost (mae_mean gap 0.34%).

| Model | MAE | RMSE | R² | Spearman |
| --- | --- | --- | --- | --- |
| catboost | 0.01192 | 0.01613 | 0.7804 | 0.8707 |
| xgboost | 0.01196 | 0.01608 | 0.7818 | 0.8700 |
| lightgbm | 0.01207 | 0.01622 | 0.7781 | 0.8680 |
| random_forest | 0.01338 | 0.01788 | 0.7301 | 0.8415 |

Metric winners: `mae_mean`→`catboost`, `rmse_mean`→`xgboost`, `r2_mean`→`xgboost`, `spearman_mean`→`catboost`

## Explainability (SHAP)

| Meaning | Column | mean |SHAP| |
| --- | --- | --- |
| Average tenant client base | `PortfolioMeanClients` | 0.01320 |
| Average tenant monthly budget | `PortfolioMeanBudget` | 0.00803 |
| Share of active tenants | `PortfolioActiveRate` | 0.00745 |
| Average tenant sales intensity | `PortfolioMeanSales` | 0.00621 |
| Average tenant client retention | `PortfolioMeanRetention` | 0.00606 |

Portfolio aggregates dominate predictions; landlord demographics (origin, industry) contribute less on average.

### Case studies

- **[GOOD]** `LANDLORD_5976` — AdjustedScore=0.5467, tenants=23
  - Raised score: Average tenant client base was 10.9 (+0.028); Average tenant sales intensity was 0.3793 (+0.013)
- **[GOOD]** `LANDLORD_8695` — AdjustedScore=0.5369, tenants=25
  - Raised score: Average tenant client base was 16.3 (+0.025); Average tenant sales intensity was 0.2564 (+0.012)
- **[BAD]** `LANDLORD_5634` — AdjustedScore=0.2484, tenants=25
  - Raised score: Nearby population was 841,693.2 (+0.000); Landlord age (years) was 8 (+0.000)
- **[BAD]** `LANDLORD_10894` — AdjustedScore=0.2224, tenants=6
  - Raised score: Average tenant client base was 24.8 (+0.011); Average tenant client retention was 22.8% (+0.002)
- **[NEUTRAL]** `LANDLORD_3491` — AdjustedScore=0.4382, tenants=5
  - Raised score: Average tenant sales intensity was 1.833 (+0.005); Share of active tenants was 100.0% (+0.004)

Full narratives and waterfalls: `reports/shap_explainability_report.md`.

## Limitations

- Association, not causation — tenant selection and location confound landlord effects.
- Survivorship / incomplete `AllCompanyID` coverage (~30% bridge IDs absent from Companies).
- Single snapshot; small-N landlords are heavily EB-shrunk — treat low-tenant scores cautiously.
- SHAP explains model predictions, not true causal drivers.

## What I'd improve given more time

- **Causal framing** — move beyond association with a temporal / difference-in-differences design (tenant outcomes before vs after moving under a landlord) or matching on tenant profile + location to reduce selection confounding.
- **Recover the missing ~30%** — investigate the unmatched `AllCompanyID` bridge rows; if they are failed/churned tenants, their absence biases scores upward (survivorship). Quantify and correct for it.
- **Uncertainty per landlord** — publish a confidence interval / posterior for each score (e.g. quantile or Bayesian models) instead of a point estimate plus a coarse confidence band.
- **Location disentanglement** — add explicit geographic controls so the score reflects the *landlord*, not just a good catchment area.
- **Temporal validation & monitoring** — backtest on a time-split, then track drift on the served model (the FastAPI scorer already exposes the hooks).
- **Target robustness** — the success target is a weighted composite; co-design the weights with domain stakeholders and expand the sensitivity sweep beyond the current min-Spearman check.

## Artifact index

- [x] **EDA:** `reports/eda_report.md`
- [x] **Targets:** `data/processed/company_targets_summary.json`
- [x] **Model comparison:** `reports/model_comparison_report.md`
- [x] **Recommendation JSON:** `reports/model_comparison_recommendation.json`
- [x] **SHAP report:** `reports/shap_explainability_report.md`
- [x] **Case studies JSON:** `reports/shap_case_studies.json`
- [x] **Serveable scorer:** `data/artifacts/landlord_scorer.joblib`
- [x] **Scorer meta:** `data/artifacts/landlord_scorer_meta.json`
- [x] **This mini report:** `reports/mini_report.md`

_Paths relative to repo root; config seed=42._
