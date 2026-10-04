# Fraud model evaluation (sub-phase 5.4)

Data: corrected PaySim dataset, DVC data hash `87c929358f0950a86aac62ce382795e8`. Models trained on the full train split (3,817,572 rows) with tuned hyperparameters, scored once on the test split (1,272,524 rows, 6,362 fraud).

## Test-set results (sorted by PR-AUC)

| Model | PR-AUC | ROC-AUC | Train seconds |
|---|---|---|---|
| xgboost | 0.9947 | 0.9999 | 7.4 |
| random_forest | 0.9922 | 0.9999 | 399.6 |
| catboost | 0.9912 | 0.9998 | 21.7 |

**Winner: xgboost** - selection rule: highest test PR-AUC (an exact tie would be broken by ROC-AUC).

## Per-segment metrics for xgboost

| Segment | Rows | Fraud | PR-AUC | ROC-AUC | Note |
|---|---|---|---|---|---|
| Overall | 1,272,524 | 6,362 | 0.9947 | 0.9999 |  |
| Type: CASH_OUT | 447,772 | 4,703 | 0.9911 | 0.9997 |  |
| Type: TRANSFER | 106,344 | 1,659 | 1.0000 | 1.0000 |  |
| Type: PAYMENT | 429,397 | 0 | undefined | undefined | no fraud present - metric undefined |
| Type: CASH_IN | 280,726 | 0 | undefined | undefined | no fraud present - metric undefined |
| Type: DEBIT | 8,285 | 0 | undefined | undefined | no fraud present - metric undefined |
| Amount: bottom quartile | 318,131 | 1,572 | 0.9987 | 1.0000 |  |
| Amount: middle two quartiles | 636,262 | 3,203 | 0.9924 | 0.9997 |  |
| Amount: top quartile | 318,131 | 1,587 | 0.9931 | 0.9999 |  |
| Pattern: account_takeover vs non-fraud | 1,268,589 | 2,427 | 0.9985 | 1.0000 |  |
| Pattern: mule_account vs non-fraud | 1,268,486 | 2,324 | 0.9988 | 1.0000 |  |
| PaySim-native fraud (is_adversarial=False) | 1,267,773 | 1,611 | 0.9444 | 0.9994 |  |

Amount tiers use the test split's own quartile cut points: 13,347.52 and 208,195.07.

## Confusion matrix at threshold 0.5 (provisional operating point, not tuned)

| | Predicted non-fraud | Predicted fraud |
|---|---|---|
| Actual non-fraud | 1,263,657 | 2,505 |
| Actual fraud | 14 | 6,348 |

## Plots

- Feature importance: `reports/model/fraud_feature_importance_top20.png`
- SHAP summary (50,000-row stratified test sample): `reports/model/fraud_shap_summary.png`

Note: PaySim-native fraud is a small segment of the test set (about a quarter of the fraud rows), so the overall score is dominated by the injected account_takeover and mule_account patterns; read the native segment line to see how the model does on native fraud alone.
