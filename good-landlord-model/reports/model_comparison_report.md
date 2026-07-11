# Landlord Model Comparison Report

_Generated at: 2026-07-11T12:14:10.394230+00:00_

## Recommendation

**Best model: `catboost`**

catboost is recommended: best mae_mean=0.01192 (±0.00022), spearman_mean=0.87069. Near-tie with xgboost (mae_mean gap 0.34%).

### Metric winners

| Metric | Direction | Winner |
| --- | --- | --- |
| MAE | lower better | `catboost` |
| RMSE | lower better | `xgboost` |
| R² | higher better | `xgboost` |
| Spearman | higher better | `catboost` |

## Setup

- Landlords scored: **14838**
- CV: **GroupKFold**, 5 folds (grouped by `LandLordID`)
- Feature set: **with_portfolio**
- Target: `AdjustedScore` (EB-shrunk mean company residual)
- Models: CatBoost, XGBoost, LightGBM, Random Forest

## Comparison table (sorted by MAE)

| Model | MAE | MAE std | RMSE | RMSE std | R² | R² std | Spearman | Spearman std | Folds | N |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| catboost | 0.01192 | 0.00022 | 0.01613 | 0.00039 | 0.78040 | 0.00340 | 0.87069 | 0.00480 | 5 | 14838 |
| xgboost | 0.01196 | 0.00024 | 0.01608 | 0.00040 | 0.78181 | 0.00374 | 0.87005 | 0.00402 | 5 | 14838 |
| lightgbm | 0.01207 | 0.00028 | 0.01622 | 0.00045 | 0.77806 | 0.00484 | 0.86797 | 0.00388 | 5 | 14838 |
| random_forest | 0.01338 | 0.00019 | 0.01788 | 0.00035 | 0.73012 | 0.00661 | 0.84154 | 0.00477 | 5 | 14838 |

## How the best model was chosen

1. **Primary:** lowest mean Absolute Error (MAE) across GroupKFold folds — interpretable average scoring error.
2. **Tie-break:** highest Spearman rank correlation — preserves landlord ordering for recommendations.
3. RMSE and R² are reported for diagnostics; they do not override MAE unless MAE values are effectively tied (<2% relative gap).

## Notes

- Comparison CSV: `D:/Personal/DS/ds-assignment/good-vs-bad-landlord/ds-assignment/good-landlord-model/reports/landlord_model_comparison.csv`
- Primary selection rule: lowest GroupKFold MAE, then highest Spearman.
- CatBoost uses native categoricals; other models use ordinal-encoded categories.

---
_This report estimates predictive association with adjusted tenant outcomes; it does not prove causal landlord impact._
