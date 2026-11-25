# Level 18b handoff: calibration, ONNX export, Triton, reason codes, model cards and the quality gate

Date: 2026-11-25. Rules: `docs/research/amendments.md`, A-055 to A-064. Every number below is SIMULATED and
copied from the committed console files under `reports/level_18/console/` (A-063) or the generated model
cards under `reports/level_18/model_cards/` (A-064).

## What this part adds

- **Calibration (A-059).** Isotonic calibrators for both models, fitted on validation rows only. The test
  summaries and reliability bins are in `calibration.json`; the calibrator files and their sha256 are recorded
  in the same file and in both model cards.
- **ONNX export (A-061, A-061b).** `mule_account_xgb.onnx` and `mule_payment_xgb.onnx`, with the account
  merge proof and the 1e-5 parity tolerance on validation and test. `onnx_parity.json` carries both.
- **Triton folders (Phase 9, A-062a).** `serving/model_repository/mule_account_xgb/1/model.onnx` and
  `mule_payment_xgb/1/model.onnx`, each with a `config.pbtxt` following model v1's folder conventions; v1's
  folder is untouched. The live validation-row load is recorded in `docs/handoff/level_18b_partial.md` and
  `runs/full/level18b/triton/triton_load.json`.
- **Reason codes (A-058).** TreeSHAP reason codes for both models over 20,000 sampled rows each, in
  `reports/level_18/console/reason_codes.json`.
- **Frozen-results console (A-063).** `scripts/level18_console.py` writes the nine JSON files under
  `reports/level_18/console/` from frozen result files only; it fits nothing and scores nothing.
- **Model cards and the quality gate (A-064).** `scripts/level18_model_cards.py` writes
  `reports/level_18/model_cards/{mule_account_xgb,mule_payment_xgb}.{json,md}` from seven console files only.
  `scripts/mule_model_quality_gate.py` checks the committed cards against
  `config/mule_model_quality_gate.yaml` and runs as the CI job `Mule model quality gate`; model v1's gate is a
  separate job and is untouched.

## Key numbers (test split, across worlds)

Account model, median (p10, p90) over 120 worlds:

| Metric | Value |
| --- | --- |
| Recall at capacity | 0.642857 (0.477045, 0.833333) |
| Genuine restriction rate | 0.005882 (0.000000, 0.008958) |
| Median time to detection | 11.675 hours (1.1675, 33.780833) |
| Yield measured | 0.923077 (0.800000, 1.000000) |
| Yield re-weighted (E1) | 0.272955 (0.086673, 1.000000), 118 worlds |
| Recall at hidden depth 5 | 1.000000 (0.595556, 1.000000), 30 worlds |

Payment model:

| Metric | Test | Stress |
| --- | --- | --- |
| Truth AP | 0.632033 (0.385806, 0.788270) | 0.070938 (0.015870, 0.353742) |
| Precision | 0.994179 (0.872121, 1.000000) | 0.093817 (0.009401, 0.412842) |
| Recall | 0.247117 (0.091315, 0.487675) | 0.174830 (0.054337, 0.436448) |
| Hold rate | 0.005003 (0.001374, 0.011879) | 0.028382 (0.007690, 0.064041) |
| Same-bank truth AP | 0.876438 (0.560906, 0.969466) | 0.103087 (0.015904, 0.557882) |
| Other-bank truth AP | 0.467391 (0.195569, 0.749861) | 0.080064 (0.009906, 0.396097) |

Payee-level precision is 0.911046 over 1,023 distinct payees and 11,196 held payments, not the per-world
median 0.994179. The payee-risk feature alone reaches truth AP 0.922653 on test and 0.543550 under stress.

Calibration, test split:

| Model | Raw ECE | Calibrated ECE |
| --- | --- | --- |
| Account | 0.127137 | 0.001843 |
| Payment | 0.012284 | 0.012821 |

Leakage: the account checks pass except check 6b (average precision over base rate 10.415126 against the 2.0
limit, A-057); the payment items pass except `id_leak`, and the overall payment status is `fail`. Both cards
state this plainly.

ONNX maximum absolute difference against the native model: account 1.788418e-07 (validation) and
1.678902e-07 (test); payment 4.470348e-07 and 5.364418e-07. The account merge proof records Spearman
0.9999999998845, 0 decision flips and a maximum margin difference of 1.310349e-05.

## Model cards and the gate

Each card states its intended use and the four "not intended for" uses, the SIMULATED training label, the
metrics above with spreads, calibration before and after, every leakage outcome, ONNX parity and the known
limitations. The five limitations are per-payee precision (0.911046, not 0.994179), stress collapse (truth AP
0.070938 against the payee feature's 0.543550), missing payee scores at other banks (NaN payee-risk share
0.617159), simulator new-account cleanness (`world/bank_connectors.py:344-369`) and capacity rounding to one
per bank-day (`evaluation/level17_scoring.py:527`). The account card also records check 6b as failed.

The gate applies 14 thresholds from `config/mule_model_quality_gate.yaml`, each carrying `derived_from`; the
committed cards pass all 14. There is no flag, environment variable or switch that skips a threshold.

Regenerate and check:

```
PYTHONPATH=. py -3.12 -B scripts/level18_model_cards.py
PYTHONPATH=. py -3.12 -B scripts/mule_model_quality_gate.py
```

## Facts

The block below is the verbatim output of:

```
PYTHONPATH=. py -3.12 -B scripts/handoff_facts.py --part 18b
```

## Register Changes

---

## Public Interfaces

Extracted directly via `inspect.signature`:

### `evaluation/calibration.py`

```python
account_split_rows(datasets: 'Sequence[Path]', split: 'str', model: 'PUBaggedAccountModel', truth_root: 'Path') -> 'tuple[np.ndarray, np.ndarray, dict[str, Any]]'
```

```python
assert_ranking_preserved(probabilities: 'Sequence[float] | np.ndarray', calibrated: 'Sequence[float] | np.ndarray', label: 'str') -> 'dict[str, Any]'
```

```python
calibration_metrics(probabilities: 'Sequence[float] | np.ndarray', truth: 'Sequence[int] | np.ndarray', bins: 'int' = 10) -> 'dict[str, Any]'
```

```python
isotonic_apply(calibrator: 'Calibrator', probabilities: 'Sequence[float] | np.ndarray') -> 'np.ndarray'
```

```python
isotonic_fit(probabilities: 'Sequence[float] | np.ndarray', truth: 'Sequence[int] | np.ndarray') -> 'Calibrator'
```

```python
load_calibrator(path: 'Path') -> 'Calibrator'
```

```python
main(argv: 'Sequence[str] | None' = None) -> 'int'
```

```python
payment_test_rows(datasets: 'Sequence[Path]', features: 'Sequence[str]', encodings: 'Mapping[str, Sequence[str]]', model: 'xgboost.XGBClassifier', truth_root: 'Path', account_model: 'PUBaggedAccountModel', account_config: 'Path', reference_path: 'Path', expected_rows: 'int', expected_positives: 'int') -> 'tuple[np.ndarray, np.ndarray, dict[str, Any]]'
```

```python
payment_validation_rows(datasets: 'Sequence[Path]', features: 'Sequence[str]', encodings: 'Mapping[str, Sequence[str]]', model: 'xgboost.XGBClassifier', oof_root: 'Path') -> 'tuple[np.ndarray, np.ndarray, dict[str, Any]]'
```

```python
run(validation_dirs: 'Sequence[Path]', test_dirs: 'Sequence[Path]', out_root: 'Path' = WindowsPath('C:/Users/sinne/Downloads/PROJECT-ashFALL/runs/full/level18b/calibration')) -> 'dict[str, Any]'
```

```python
truth_label(role: 'str') -> 'int'
```

```python
verify_task4_reproduction(records: 'Sequence[WorldRecord]', reference: 'Mapping[str, Any]', expected_rows: 'int', expected_positives: 'int') -> 'dict[str, Any]'
```

```python
write_calibrator(path: 'Path', calibrator: 'Calibrator') -> 'None'
```

```python
Calibrator(x_thresholds: 'tuple[float, ...]', y_thresholds: 'tuple[float, ...]', y_min: 'float' = 0.0, y_max: 'float' = 1.0) -> None
```
- dataclass

```python
WorldRecord(world_id: 'str', rows: 'int', positives: 'int', truth_ap: 'float | None') -> None
```
- dataclass

### `evaluation/onnx_export.py`

```python
account_sample(datasets: 'Sequence[Path]', split: 'str', feature_names: 'Sequence[str]') -> 'tuple[np.ndarray, list[tuple[str, ...]]]'
```

```python
decision_flips(native: 'Sequence[float] | np.ndarray', merged: 'Sequence[float] | np.ndarray', cut: 'float') -> 'int'
```

```python
export_model(booster: 'xgboost.Booster', n_features: 'int') -> 'onnx.ModelProto'
```

```python
load_account_cut(path: 'Path' = WindowsPath('C:/Users/sinne/Downloads/PROJECT-ashFALL/runs/full/run_a320_stage5c/extra_reporting.json')) -> 'float'
```

```python
main(argv: 'Sequence[str] | None' = None) -> 'int'
```

```python
merge_gate(native_probability: 'Sequence[float] | np.ndarray', merged_probability: 'Sequence[float] | np.ndarray', native_margin: 'Sequence[float] | np.ndarray', merged_margin: 'Sequence[float] | np.ndarray', cut: 'float') -> 'dict[str, Any]'
```

```python
merge_member_models(raws: 'Sequence[Mapping[str, Any]]', n_bags: 'int', feature_count: 'int') -> 'dict[str, Any]'
```

```python
merge_proof(merged_booster: 'xgboost.Booster', account_model: 'PUBaggedAccountModel', matrix: 'np.ndarray', cut: 'float') -> 'dict[str, Any]'
```

```python
midranks(values: 'Sequence[float] | np.ndarray') -> 'np.ndarray'
```

```python
onnx_probabilities(session: 'onnxruntime.InferenceSession', matrix: 'np.ndarray') -> 'np.ndarray'
```

```python
open_session(path: 'Path') -> 'onnxruntime.InferenceSession'
```

```python
parity_summary(native: 'Sequence[float] | np.ndarray', converted: 'Sequence[float] | np.ndarray', nan_rows: 'int') -> 'dict[str, Any]'
```

```python
run(validation_dirs: 'Sequence[Path]', test_dirs: 'Sequence[Path]', out_root: 'Path' = WindowsPath('C:/Users/sinne/Downloads/PROJECT-ashFALL/runs/full/level18b/onnx')) -> 'dict[str, Any]'
```

```python
save_onnx(model: 'onnx.ModelProto', path: 'Path') -> 'None'
```

```python
spearman_correlation(left: 'Sequence[float] | np.ndarray', right: 'Sequence[float] | np.ndarray') -> 'float'
```

```python
top_fraction_indices(scores: 'Sequence[float] | np.ndarray', fraction: 'float') -> 'set[int]'
```

```python
write_margin_booster(merged: 'Mapping[str, Any]', path: 'Path', expected_rounds: 'int', feature_count: 'int') -> 'xgboost.Booster'
```

### `evaluation/reason_codes.py`

```python
account_shap_values(model: 'PUBaggedAccountModel', matrix: 'list[list[float]]', feature_count: 'int') -> 'np.ndarray'
```

```python
build_reason_code_map(schema_path: 'Path' = WindowsPath('C:/Users/sinne/Downloads/PROJECT-ashFALL/data/schemas/feature_set_m1.yaml'), payment_feature_order_path: 'Path' = WindowsPath('C:/Users/sinne/Downloads/PROJECT-ashFALL/config/mule_payment_feature_order.yaml')) -> 'ReasonCodeMap'
```

```python
frequency_table(rows: 'Sequence[Mapping[str, Any]]', codes: 'Sequence[ReasonCode]') -> 'dict[str, Any]'
```

```python
load_reason_code_map(path: 'Path' = WindowsPath('C:/Users/sinne/Downloads/PROJECT-ashFALL/config/feature_reason_codes.yaml'), schema_path: 'Path' = WindowsPath('C:/Users/sinne/Downloads/PROJECT-ashFALL/data/schemas/feature_set_m1.yaml'), payment_feature_order_path: 'Path' = WindowsPath('C:/Users/sinne/Downloads/PROJECT-ashFALL/config/mule_payment_feature_order.yaml')) -> 'ReasonCodeMap'
```

```python
main(argv: 'Sequence[str] | None' = None) -> 'int'
```

```python
payment_shap_values(model: 'xgboost.XGBClassifier', matrix: 'np.ndarray', feature_count: 'int') -> 'np.ndarray'
```

```python
reason_codes_for_features(features: 'Sequence[str]', feature_codes: 'Mapping[str, str]') -> 'list[str]'
```

```python
row_reasons(shap_row: 'Sequence[float]', feature_names: 'Sequence[str]', feature_codes: 'Mapping[str, str]') -> 'tuple[list[RowReason], list[str]]'
```

```python
run(validation_dirs: 'Sequence[Path]', out_root: 'Path' = WindowsPath('C:/Users/sinne/Downloads/PROJECT-ashFALL/runs/full/level18b/reason_codes')) -> 'dict[str, Any]'
```

```python
select_sample(rows: 'Iterator[Mapping[str, Any]] | Sequence[Mapping[str, Any]]', key_fields: 'Sequence[str]', seed: 'int' = 1717, sample_rows: 'int' = 20000) -> 'list[dict[str, Any]]'
```

```python
top_reason_features(shap_row: 'Sequence[float]', feature_names: 'Sequence[str]', top_n: 'int' = 3) -> 'list[ReasonFeature]'
```

```python
verify_account_checkpoint(checkpoint_root: 'Path', extra_reporting_path: 'Path' = WindowsPath('C:/Users/sinne/Downloads/PROJECT-ashFALL/runs/full/run_a320_stage5c/extra_reporting.json')) -> 'dict[str, str]'
```

```python
write_reason_code_map(path: 'Path' = WindowsPath('C:/Users/sinne/Downloads/PROJECT-ashFALL/config/feature_reason_codes.yaml'), schema_path: 'Path' = WindowsPath('C:/Users/sinne/Downloads/PROJECT-ashFALL/data/schemas/feature_set_m1.yaml'), payment_feature_order_path: 'Path' = WindowsPath('C:/Users/sinne/Downloads/PROJECT-ashFALL/config/mule_payment_feature_order.yaml')) -> 'ReasonCodeMap'
```

```python
ReasonCode(group: 'str', code: 'str', plain_reason: 'str') -> None
```
- dataclass

```python
ReasonCodeMap(schema_path: 'str', schema_sha256: 'str', payment_feature_order_path: 'str', payment_feature_order_sha256: 'str', codes: 'tuple[ReasonCode, ...]', features: 'dict[str, str]') -> None
```
- dataclass

```python
ReasonFeature(feature: 'str', shap: 'float') -> None
```
- dataclass

```python
RowReason(feature: 'str', shap: 'float', code: 'str') -> None
```
- dataclass

---

## Records and Fields

Extracted directly via `dataclasses.fields`:

- `Calibrator`:
  - Fields: `x_thresholds: tuple[float, ...]`, `y_thresholds: tuple[float, ...]`, `y_min: float`, `y_max: float`
  - Path: `evaluation/calibration.py`

- `WorldRecord`:
  - Fields: `world_id: str`, `rows: int`, `positives: int`, `truth_ap: float | None`
  - Path: `evaluation/calibration.py`

- `ReasonCode`:
  - Fields: `group: str`, `code: str`, `plain_reason: str`
  - Path: `evaluation/reason_codes.py`

- `ReasonCodeMap`:
  - Fields: `schema_path: str`, `schema_sha256: str`, `payment_feature_order_path: str`, `payment_feature_order_sha256: str`, `codes: tuple[ReasonCode, ...]`, `features: dict[str, str]`
  - Path: `evaluation/reason_codes.py`

- `ReasonFeature`:
  - Fields: `feature: str`, `shap: float`
  - Path: `evaluation/reason_codes.py`

- `RowReason`:
  - Fields: `feature: str`, `shap: float`, `code: str`
  - Path: `evaluation/reason_codes.py`

---

## Configuration Keys

### `config/feature_reason_codes.yaml`
- **version**
- **schema**
- **schema_sha256**
- **payment_feature_order**
- **payment_feature_order_sha256**
- **reason_codes**
- **features**
  - `distinct_payers_1h`
  - `distinct_payers_24h`
  - `distinct_payers_7d`
  - `first_time_payer_share_24h`
  - `small_credit_count_24h`
  - `small_credit_count_7d`
  - `largest_credit_paise_24h`
  - `below_threshold_credit_count_7d`
  - `pass_through_ratio_24h`
  - `pass_through_ratio_7d`
  - `median_credit_to_debit_seconds_7d`
  - `top_receiver_share_7d`
  - `other_bank_outflow_share_7d`
  - `outflow_to_person_paise_7d`
  - `outflow_to_merchant_paise_7d`
  - `cash_atm_paise_7d`
  - `cash_branch_paise_7d`
  - `current_balance_paise`
  - `min_balance_paise_7d`
  - `near_zero_cycles_7d`
  - `salary_like_payer_count`
  - `recurring_merchant_payee_count`
  - `regular_transfer_payee_count`
  - `routine_life_present`
  - `history_days_available`
  - `account_age_days`
  - `account_type`
  - `dormancy_days_before_recent_activity`
  - `volume_vs_own_history_ratio`
  - `days_since_device_change`
  - `accounts_on_same_device`
  - `device_hardware_changed`
  - `sim_swapped`
  - `app_reinstalled_same_phone`
  - `new_device_r5`
  - `distinct_fingerprints_lifetime`
  - `machine_regular_timing_score`
  - `distinct_same_bank_payer_regions_7d`
  - `unknown_region_payer_count_7d`
  - `hops_to_confirmed_label`
  - `shared_counterparties_with_flagged_7d`
  - `local_cluster_density`
  - `hops_to_complained_account`
  - `registry_match_count`
  - `registry_match_age_days`
  - `payee_registry_match`
  - `first_payment_to_payee`
  - `amount_vs_payer_median_ratio`
  - `amount_vs_payer_max_ratio`
  - `payee_same_bank_risk`
  - `hour_ist`
  - `payee_type`
  - `customer_failed_attempts_24h`
  - `payee_risk`

### `config/mule_model_quality_gate.yaml`
- **version**
- **threshold_count**
- **thresholds**

### `config/mule_payment_feature_order.yaml`
- **version**
- **target**
- **payee_risk_source**
- **feature_sha256**
- **features**
- **groups**
  - `payee_risk`
  - `payment_level`
  - `payer_account`
- **categorical_encodings**
  - `account_type`
  - `payee_type`

---

## Random Stream Children and Tie-Break Order

- `random_stream_children`:
  - `4`: `fraud_payment_failures`
  - `5`: `resumed_payment_failures`
  - `6`: `gang_devices`
  - `7`: `gang_planning`
  - `8`: `genuine_account_openings`
  - `9`: `gang_account_openings`
  - `10`: `opened_account_routine`
  - `11`: `opened_routine_failures`
  - `12`: `rented_owner_activity`
  - `13`: `cash_withdrawal_failures`
  - `14`: `gang_observation_feed`
  - `15`: `gang_reactions`
  - `16`: `account_market`
  - `17`: `label_world_settings`
  - `18`: `true_complaints`
  - `19`: `false_complaints`
  - `20`: `registry_reports`
  - `21`: `bulk_confirmations`
  - `22`: `gang_mix`
  - `23`: `account_opening_dates`
  - `24`: `registry_seed`
  - `25`: `registry_screen_delay`
  - `26`: `hold_request`
  - `27`: `gang_take_control_flags`

- `tie_break_order`:
  - `[outside_events, planned_payments, fraud_payments, resumed_payments, gang_device_events, opened_routine_payments, cash_withdrawals, gang_observations]`
