| Model | ROC-AUC | PR-AUC | Train time (s) |
|---|---|---|---|
| xgboost | 0.9994 | 0.9377 | 5.1 |
| random_forest | 0.9920 | 0.9365 | 90.6 |
| catboost | 0.9993 | 0.9092 | 13.5 |
| logreg | 0.9903 | 0.5627 | 8.4 |
| lightgbm | 0.8514 | 0.0186 | 12.7 |

Baseline, default-ish params, no tuning yet. Winner selection happens after hyperparameter tuning.
