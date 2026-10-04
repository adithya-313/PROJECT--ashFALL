# Calibration summary (sub-phase 5.5)

Model: tuned XGBoost retrained on the full train split and saved at `models/xgboost/model.ubj`. Isotonic calibrator fitted on the calibration split only (1,272,524 rows). DVC data hash `87c929358f0950a86aac62ce382795e8`.

## Reliability on the calibration split (10 equal-width bins; the acceptance check)

| Bin | Rows | Mean predicted | Actual fraud rate | Deviation | Within 10pp |
|---|---|---|---|---|---|
| 0.0-0.1 | 1,265,178 | 0.0001 | 0.0001 | 0.0000 | PASS |
| 0.1-0.2 | 570 | 0.1404 | 0.1404 | 0.0000 | PASS |
| 0.2-0.3 | 184 | 0.2174 | 0.2174 | 0.0000 | PASS |
| 0.3-0.4 | 430 | 0.3465 | 0.3465 | 0.0000 | PASS |
| 0.4-0.5 | 86 | 0.4186 | 0.4186 | 0.0000 | PASS |
| 0.5-0.6 | 96 | 0.5833 | 0.5833 | 0.0000 | PASS |
| 0.6-0.7 | 3 | 0.6667 | 0.6667 | 0.0000 | PASS |
| 0.7-0.8 | 113 | 0.7522 | 0.7522 | 0.0000 | PASS |
| 0.8-0.9 | 75 | 0.8667 | 0.8667 | 0.0000 | PASS |
| 0.9-1.0 | 5,789 | 0.9978 | 0.9978 | 0.0000 | PASS |

All non-empty bins within 10pp: **PASS**. This split is the one the calibrator was fitted on, so it is an in-sample check and is expected to look good.

## Reliability on the test split (unseen; confirmation only)

| Bin | Rows | Mean predicted | Actual fraud rate | Deviation | Within 10pp |
|---|---|---|---|---|---|
| 0.0-0.1 | 1,265,167 | 0.0001 | 0.0000 | 0.0000 | PASS |
| 0.1-0.2 | 547 | 0.1408 | 0.1353 | 0.0055 | PASS |
| 0.2-0.3 | 190 | 0.2166 | 0.2105 | 0.0061 | PASS |
| 0.3-0.4 | 423 | 0.3463 | 0.3286 | 0.0177 | PASS |
| 0.4-0.5 | 81 | 0.4198 | 0.4198 | 0.0001 | PASS |
| 0.5-0.6 | 99 | 0.5843 | 0.5657 | 0.0186 | PASS |
| 0.6-0.7 | 6 | 0.6531 | 0.3333 | 0.3198 | FAIL |
| 0.7-0.8 | 103 | 0.7491 | 0.6990 | 0.0501 | PASS |
| 0.8-0.9 | 77 | 0.8661 | 0.8442 | 0.0220 | PASS |
| 0.9-1.0 | 5,831 | 0.9978 | 0.9985 | 0.0006 | PASS |

All non-empty bins within 10pp on test: **FAIL**. Bins with few rows swing easily, so read each deviation together with its row count.

## Probability checks

- Calibrated probabilities inside [0, 1]: **True** (min 0.000001, max 1.000000)
- Monotonic in the raw score: **True**

## Threshold analysis (thresholds chosen on the calibration split; test is confirmation only)

| Operating point | Threshold | Calibration precision / recall / F1 | Calibration FP / FN | Test precision / recall / F1 | Test FP / FN |
|---|---|---|---|---|---|
| (a) F1-optimal | 0.42 | 0.9849 / 0.9404 / 0.9621 | 92 / 379 | 0.9837 / 0.9459 / 0.9644 | 100 / 344 |
| (b) High-recall (>= 0.995) | 0.03 | 0.7757 / 0.9958 / 0.8721 | 1,832 / 27 | 0.7773 / 0.9962 / 0.8732 | 1,816 / 24 |

Neither point is recommended here. The final operating point is a business decision (missed fraud versus false alarms) and is not finalised in Level 5.

Plots: `reports/calibration/reliability_diagram.png`, `reports/calibration/threshold_analysis.png`.
