# Fairness table: false-positive rate by segment (winner: XGBoost, test split)

False-positive rate (FPR) = false positives / actual non-fraud rows in the segment. A false positive is a legitimate transaction the model flags as fraud. FPR depends on the decision threshold, so it is reported at three thresholds. **They are on two different scales**: `raw` is the model's own score (the scale the confusion matrix in `evaluation.md` used), `calibrated` is the score after isotonic calibration. A raw 0.5 is not the same decision as a calibrated 0.5. No threshold is recommended here; the operating point is a business decision.

Each cell shows FPR (number of false positives). The interval column is the 95% Wilson confidence interval of the FPR at `raw score >= 0.5 (as in the confusion matrix)`; wide intervals mean a small sample.

| Group | Segment | Rows | Non-fraud | FPR, raw score >= 0.5 (as in the confusion matrix) | FPR, calibrated >= 0.42 (F1-optimal) | FPR, calibrated >= 0.03 (high-recall) | 95% interval (first threshold) |
|---|---|---|---|---|---|---|---|
| Transaction type | CASH_OUT | 447,772 | 443,069 | 0.547% (2,424) | 0.022% (98) | 0.399% (1,766) | 0.53% to 0.57% |
| Transaction type | TRANSFER | 106,344 | 104,685 | 0.077% (81) | 0.002% (2) | 0.048% (50) | 0.06% to 0.10% |
| Transaction type | PAYMENT | 429,397 | 429,397 | 0.000% (0) | 0.000% (0) | 0.000% (0) | 0.00% to 0.00% |
| Transaction type | CASH_IN | 280,726 | 280,726 | 0.000% (0) | 0.000% (0) | 0.000% (0) | 0.00% to 0.00% |
| Transaction type | DEBIT | 8,285 | 8,285 | 0.000% (0) | 0.000% (0) | 0.000% (0) | 0.00% to 0.05% |
| Amount tier | bottom quartile | 318,131 | 316,559 | 0.078% (246) | 0.001% (3) | 0.057% (179) | 0.07% to 0.09% |
| Amount tier | middle two quartiles | 636,262 | 633,059 | 0.276% (1,746) | 0.003% (21) | 0.196% (1,238) | 0.26% to 0.29% |
| Amount tier | top quartile | 318,131 | 316,544 | 0.162% (513) | 0.024% (76) | 0.126% (399) | 0.15% to 0.18% |
| Merchant risk tier (confounded) | high | 421 | 421 | 0.000% (0) | 0.000% (0) | 0.000% (0) | 0.00% to 0.90% |
| Merchant risk tier (confounded) | medium | 843,127 | 836,765 | 0.299% (2,505) | 0.012% (100) | 0.217% (1,816) | 0.29% to 0.31% |
| Merchant risk tier (confounded) | low | 428,976 | 428,976 | 0.000% (0) | 0.000% (0) | 0.000% (0) | 0.00% to 0.00% |
| All rows | all | 1,272,524 | 1,266,162 | 0.198% (2,505) | 0.008% (100) | 0.143% (1,816) | 0.19% to 0.21% |

Amount tiers use the test split's own quartile cut points: 13,347.52 and 208,195.07.

## Recall (share of actual fraud caught) at the same thresholds

FPR alone can hide missed fraud, so recall is shown for segments that contain fraud (undefined otherwise).

| Group | Segment | Fraud rows | Recall, raw score >= 0.5 (as in the confusion matrix) | Recall, calibrated >= 0.42 (F1-optimal) | Recall, calibrated >= 0.03 (high-recall) |
|---|---|---|---|---|---|
| Transaction type | CASH_OUT | 4,703 | 99.70% | 92.81% | 99.49% |
| Transaction type | TRANSFER | 1,659 | 100.00% | 99.64% | 100.00% |
| Amount tier | bottom quartile | 1,572 | 99.75% | 97.65% | 99.62% |
| Amount tier | middle two quartiles | 3,203 | 99.78% | 92.88% | 99.56% |
| Amount tier | top quartile | 1,587 | 99.81% | 95.02% | 99.75% |
| Merchant risk tier (confounded) | medium | 6,362 | 99.78% | 94.59% | 99.62% |
| All rows | all | 6,362 | 99.78% | 94.59% | 99.62% |

## Note on the merchant risk tier rows (read before using them)

Merchant risk tier is confounded with transaction type. The `high` and `low` tiers exist only on PAYMENT rows (payments to merchant accounts), and PAYMENT rows contain no fraud at all. Every other transaction type is assigned `medium`, and all 6,362 test fraud rows are in `medium`. So the 0.000% false-positive rates for `high` and `low` are NOT evidence of fair treatment: they only show that the model correctly treats PAYMENT rows as non-fraud. The real driver is transaction type, which is why that is the fairness axis to read here.

The `high` tier has only 421 rows (0 fraud, 421 non-fraud). Its false-positive count at the first threshold is 0 of 421, with a 95% interval of 0.00% to 0.90%. That supports no conclusion about the `high` tier.

