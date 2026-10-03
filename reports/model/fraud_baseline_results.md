| Model | PR-AUC | ROC-AUC | Train seconds |
|---|---|---|---|
| xgboost | 0.9933 | 0.9998 | 6.7 |
| catboost | 0.9871 | 0.9998 | 13.6 |
| random_forest | 0.9832 | 0.9985 | 107.4 |
| logreg | 0.6051 | 0.9935 | 12.2 |
| lightgbm | 0.1973 | 0.9793 | 25.2 |

These results supersede the pre-correction baseline results (they were produced on the corrected dataset, DVC data hash `87c929358f0950a86aac62ce382795e8`). Baseline settings, no tuning; winner selection comes later.
