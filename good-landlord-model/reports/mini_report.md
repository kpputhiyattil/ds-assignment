# Mini Report — Landlord Quality Scoring

_Generated at: 2026-07-11T12:52:59.866002+00:00_

Executive summary of the good-vs-bad landlord pipeline: company success targets → OOF company baseline residuals → EB-shrunk `AdjustedScore` → landlord models (GroupKFold) → SHAP.

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
