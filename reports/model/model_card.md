# Model card: fraud scoring model (XGBoost)

## 1. What the model does and what it was trained on

The model scores a single payment transaction with a probability that it is fraud (`isFraud`). It is an XGBoost classifier that takes **15 numeric features** in a fixed order: `amount`, the four account balances (`oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, `newbalanceDest`), five engineered signals (`account_avg_amount`, `many_small_inbound_count`, `large_outbound_amount`, `new_device_flag`, `geo_velocity_spike`) and a one-hot encoding of the transaction type (CASH_IN, CASH_OUT, DEBIT, PAYMENT, TRANSFER). Identifiers, labels and `step` are never inputs.

**Data.** The corrected PaySim dataset (DVC hash `87c929358f0950a86aac62ce382795e8`), split 60/20/20 into train / calibration / test with a fixed seed (42), stratified on the fraud label, so each split has about 0.5% fraud. The model trained on the train split (3,817,572 rows). Fraud in the dataset is 31,813: 8,213 PaySim fraud rows plus 23,600 synthetic rows (11,800 account-takeover and 11,800 mule-account), made by `scripts/inject_adversarial.py`.

**Model.** XGBoost, chosen as the winner because it had the highest test PR-AUC of three tuned models. Hyperparameters (tuned with Optuna on a 25% train sample): `max_depth` = 5; `learning_rate` = 0.1160184349121161; `n_estimators` = 350; `min_child_weight` = 1; `subsample` = 0.915514604777331; `reg_lambda` = 0.01193765046114114. It was trained with a heavy fraud weighting (`scale_pos_weight`, about 199), so its raw output is **not a probability**. A separate isotonic calibrator turns it into one, and the serving client always applies it (`serving/fraud_client.py`). Native feature importance is led by `type_PAYMENT` (55%), `newbalanceOrig` (17%), `oldbalanceOrg` (11%), `amount` (4%), `type_CASH_IN` (4%).

## 2. Performance (test split)

Overall: PR-AUC 0.9947, ROC-AUC 0.9999 on 1,272,524 rows with 6,362 fraud. The per-segment table below is copied from `reports/model/evaluation.md`.

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

Confusion matrix at the provisional raw threshold of 0.5 (not a recommended operating point):

| | Predicted non-fraud | Predicted fraud |
|---|---|---|
| Actual non-fraud | 1,263,657 | 2,505 |
| Actual fraud | 14 | 6,348 |

## 3. Fairness: false-positive rate by segment

Produced by `scripts/fraud_evaluation.py --fairness` (same script as the performance section) and copied from `reports/model/fairness_table.md`. The real axes are transaction type and amount tier. Read the merchant-tier note below the tables before using those rows.


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

### Recall (share of actual fraud caught) at the same thresholds

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

### Note on the merchant risk tier rows (read before using them)

Merchant risk tier is confounded with transaction type. The `high` and `low` tiers exist only on PAYMENT rows (payments to merchant accounts), and PAYMENT rows contain no fraud at all. Every other transaction type is assigned `medium`, and all 6,362 test fraud rows are in `medium`. So the 0.000% false-positive rates for `high` and `low` are NOT evidence of fair treatment: they only show that the model correctly treats PAYMENT rows as non-fraud. The real driver is transaction type, which is why that is the fairness axis to read here.

The `high` tier has only 421 rows (0 fraud, 421 non-fraud). Its false-positive count at the first threshold is 0 of 421, with a 95% interval of 0.00% to 0.90%. That supports no conclusion about the `high` tier.


## 4. Comparison with the runner-up models

Overall test results for all three tuned models (copied from `evaluation.md`):

| Model | PR-AUC | ROC-AUC | Train seconds |
|---|---|---|---|
| xgboost | 0.9947 | 0.9999 | 7.4 |
| random_forest | 0.9922 | 0.9999 | 399.6 |
| catboost | 0.9912 | 0.9998 | 21.7 |

The runners-up are random_forest, catboost, within 0.0035 PR-AUC of the winner at most. Per-segment metrics and the fairness table were computed for the winner only. Producing them for the runners-up would mean retraining those models (the random forest alone takes several minutes), so that comparison was not done.

## 5. Known limitations

- **The data is synthetic.** PaySim is a simulator, not real transactions. Scores on it say how well the model learned this simulator, not how it would do on real fraud.
- **The signal is strongly balance- and type-driven.** The leading features are `type_PAYMENT` (55%), `newbalanceOrig` (17%), `oldbalanceOrg` (11%), `amount` (4%), `type_CASH_IN` (4%). That reflects how the simulator makes fraud (accounts being emptied) and is optimistic compared with real-world fraud, which is rarely this clean. No no-balance version was measured on the final dataset, so this is stated qualitatively, without a number.
- **The adversarial patterns are injected.** The account-takeover and mule-account rows are synthetic, built from documented real-world behaviour (`reports/data/fraud_pattern_research.md`) but still artificial. They cannot carry network, session or time-window signals that the tabular format does not hold.
- **Not validated on real fraud data.** No real-world evaluation has been done.
- **Native versus injected fraud.** PaySim's own fraud scores PR-AUC 0.9444 (1,611 test fraud rows), against 0.9985 (account takeover) and 0.9988 (mule account) for the injected patterns. The overall 0.9947 is dominated by the injected rows, which make up about three quarters of the test fraud, so the native figure is the more conservative one.
- **Calibration is thin in the middle of the range.** On the test split, the eight middle probability bins (0.1 to 0.9) hold only 6 to 547 rows each, against 1,265,167 in the lowest bin. The thinnest middle bin is 0.6-0.7 (6 rows). It is also the worst-calibrated bin on test (deviation 0.3198). Calibration in the middle of the range is therefore not well established.
- **Merchant risk tier is not a valid fairness axis.** It is confounded with transaction type (see the note in section 3), so the model has not been shown to be fair across merchant risk levels.
- **No operating point is chosen.** The thresholds shown are provisional. The final threshold is a business decision about missed fraud versus false alarms.
- **Not connected to live requests.** The model needs an already-prepared row of 15 features. Building those features from a raw payment request (account history, balances) is not done yet, and the API's `/score` endpoint is still a stub that does not call the model.

## 6. Reproducibility

- Code: git commit `ac1b98fffe63638384d3d92f483760b88a48ea23` (this card is added in the commit that follows it).
- Data: DVC content hash `87c929358f0950a86aac62ce382795e8`.
- MLflow: evaluation run `780831d8c7be4feabff20d894a1abd62` in experiment `level5_fraud_evaluation` (store `mlflow.db`). This card and the fairness table are also logged there as artifacts of the run named `model_card_xgboost`.
- Seed: 42. Reload the model from `models/xgboost/model.ubj` (not committed; DVC-tracked later), calibrator from `models/calibration/isotonic_calibrator.pkl`, ONNX export in `models/xgboost/model.onnx`.
