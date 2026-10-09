# Audit of model v1

## 1. What was audited and how

Model v1 (XGBoost, PR-AUC 0.99) was left untouched; its files are hashed before and after the audit and are unchanged. Copies of its training recipe were retrained in the MLflow experiment `level9_v1_audit` (seed 42, CPU): with all 15 features (control), without the balance columns, without the type flags, without the five generated columns, on an out-of-time split, and with each feature alone.

## 2. Results

| Variant | Group | Split | PR-AUC | MLflow run id |
|---|---|---|---|---|
| original_v1 | original | level5_random | 0.99466 | 780831d8c7be4feabff20d894a1abd62 |
| control_all15 | control | level5_random | 0.99436 | d4ec335c6eef4f09a795759e15629f83 |
| ablate_no_balance | ablation | level5_random | 0.81339 | 15a4d16c1c164e44bd72a7b21f1a243e |
| ablate_no_type | ablation | level5_random | 0.98783 | b74185b5c9484005af89581838a4e7ec |
| ablate_no_generated | ablation | level5_random | 0.96016 | 9236f35d567e468c82d78ecc2b5c5c1b |
| oot_all15 | out_of_time | out_of_time | 0.99226 | ba13a947db9741309459284b59cf806e |
| oot_no_balance | out_of_time | out_of_time | 0.66447 | 6c32a22bb9224ed9bcb5b123ad308c87 |
| single_amount | single_feature | level5_random | 0.01448 | 62ebde03c655496fa8c9091746740e33 |
| single_oldbalanceOrg | single_feature | level5_random | 0.00937 | c7b05723ba8049e6a329c0b0e41bf2a5 |
| single_newbalanceOrig | single_feature | level5_random | 0.19097 | d26ecd1a14ec409eb70a871b4158c652 |
| single_oldbalanceDest | single_feature | level5_random | 0.00648 | 720f0f8c39074a1f8892677d386cddab |
| single_newbalanceDest | single_feature | level5_random | 0.00780 | 5e502eadaef84ed1af61174b0ccd01f0 |
| single_type_CASH_IN | single_feature | level5_random | 0.00641 | 5550712fb5dd4db3aa1175dba0ccdc52 |
| single_type_CASH_OUT | single_feature | level5_random | 0.00907 | 04c437a1451b4676a33c8462f0df463d |
| single_type_DEBIT | single_feature | level5_random | 0.00503 | 59eb42d7bce84038b4918377430baebd |
| single_type_PAYMENT | single_feature | level5_random | 0.00755 | cc3b55316227457489870a9af07b5adf |
| single_type_TRANSFER | single_feature | level5_random | 0.00776 | af40ed87b5fd4a4988e7e25a033e0a40 |
| single_new_device_flag | single_feature | level5_random | 0.00994 | e561085ecd1248c1b369ad63c589a4d3 |
| single_geo_velocity_spike | single_feature | level5_random | 0.00619 | a3405b8428274c3b9c2e01733215ebcc |
| single_account_avg_amount | single_feature | level5_random | 0.01231 | 92b5f98d2b39459e86d3853e0cee3b5e |
| single_many_small_inbound_count | single_feature | level5_random | 0.00579 | 929288294933471e8b1cd5307349d5d0 |
| single_large_outbound_amount | single_feature | level5_random | 0.02874 | 52b805c29a044dc3bbd62f1976bcb28c |

Base rate of the test split: 0.0050.

## 3. What the model learned

The three strongest single features:

- newbalanceOrig alone: PR-AUC 0.191
- large_outbound_amount alone: PR-AUC 0.029
- amount alone: PR-AUC 0.014

How easy the generated columns are to separate:

| Generated column | Non-zero in fraud rows | Non-zero in other rows |
|---|---|---|
| account_avg_amount | 99.9% | 100.0% |
| geo_velocity_spike | 32.4% | 20.0% |
| large_outbound_amount | 99.9% | 100.0% |
| many_small_inbound_count | 95.6% | 95.0% |
| new_device_flag | 46.1% | 15.0% |

## 4. Why the score misled (or did not)

Model v1's original PR-AUC was 0.995, while a model with no skill would score about the base rate of 0.005. A copy retrained on the CPU with all 15 features scored 0.994; the gap of 0.0003 to the original is an ESTIMATE of CPU versus GPU training (it was not tested). Without the balance columns, PR-AUC fell from 0.994 to 0.813 (a large drop of 0.181). Without the type flags, PR-AUC fell from 0.994 to 0.988 (a small drop of 0.007). Without the five generated columns, PR-AUC fell from 0.994 to 0.960 (a small drop of 0.034). Removing the balance columns dropped PR-AUC from 0.994 to 0.813, and it still stayed well above the base rate of 0.005. So the score DEPENDS on the balance group, which is recorded after the transaction (a leak). The injected patterns may also be easy to separate on their own. The separability diagnostic (the table in section 3) shows how easy the injected patterns are to tell apart: new_device_flag is non-zero in 46% of fraud rows and 15% of other rows. On an out-of-time split (train on steps up to 372, test on later steps) all 15 features scored 0.992, a drop of 0.002 from the control; without the balance columns the out-of-time score was 0.664. The strongest single feature was newbalanceOrig with PR-AUC 0.191, 19% of the original score. No single feature explains the score; its strength comes from combining columns, so this is not evidence that the score was fine. type_PAYMENT alone scored 0.008, which is 0.8% of the original score and 1.5 times the base rate of 0.005.

## 5. PaySim's own guidance

**NOT VERIFIED — source sentence still to be added**

Source to check: https://www.kaggle.com/datasets/ealaxi/paysim1. The page text could not be read automatically, so no sentence is quoted here. Nothing in this report depends on it.

## 6. Design rules that follow

- Never use balances recorded after the transaction (new balances) as model inputs.
- Evaluate with an out-of-time split, not only a random split.
- Ablate groups of features and test features alone before trusting a score.
- Keep frozen artifacts behind a hash check, and never write to them from analysis code.

## 7. Limits of this audit

- The audit models were trained on the CPU; model v1 was trained on the GPU, so scores differ slightly.
- One seed (42) and one out-of-time cutoff (step 372) were used.
- The data is synthetic (PaySim plus injected patterns): scores say how well a model learned this simulator.
- Only PR-AUC was compared; thresholds, calibration and fairness were not re-audited.
