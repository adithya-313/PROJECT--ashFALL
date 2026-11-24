# Level 18b Handoff Note (partial) — reason codes, calibration, ONNX export, Triton folders

All figures below are SIMULATED and come from the frozen full datasets named in each run manifest.

## Where things stand
Level 18b adds the serving-side parts of the mule models on top of the frozen Run A account model and the
frozen payment model: plain-English reason codes, isotonic calibration fitted on validation, a merged
account booster exported to ONNX next to the payment export, and the Triton model folders for both.

| Part | Amendment | Code commits | Pull request |
| --- | --- | --- | --- |
| Reason codes (task 5) | A-059 | `31b61ed`, `ae0d558`, `b5ee071`, `c097f0b` | #174, merged as `1c41243` |
| Calibration (task 7) | A-060, A-060a | `27c4899`, `ef6959d`, `fb70523`, `22008d8` | #175, merged as `b989085` |
| ONNX export (task 8) | A-061, A-061a, A-061b | `4b8738a`, `2d9c6d9`, `9deaea6`, `97ae69c`, `6f2c4fe`, `34cdbb5` | #176, merged as `caf6617` |
| Triton folders (task 9) | A-062 | this branch, `feature/level-18b-triton`, based on `caf6617` | #177 |

Task 9 adds only the two model folders, their `config.pbtxt` files and this note. The `1/model.onnx`
files inside the folders are copies of the Phase 8 outputs; `*.onnx` is gitignored, so they are not
committed, the same as `serving/model_repository/fraud_xgboost/1/model.onnx`. A fresh checkout therefore
holds only a tracked `1/.gitkeep` in each version folder: copy `runs/full/level18b/onnx/mule_account_xgb.onnx`
and `runs/full/level18b/onnx/mule_payment_xgb.onnx` into the two `1/model.onnx` paths before starting the
`serving` profile.

## What each component does
- `evaluation/reason_codes.py`: builds the committed feature-to-code map from the M1 schema and the payment
  feature order, takes the top TreeSHAP features of each sampled row and turns them into at most three
  plain-English reasons (`ROUTINE_LIFE`, `OUTFLOW`, `PAYEE_RISK`, …), and writes one JSON per frozen model
  with the sampled rows, the per-code frequency and the map it used.
- `evaluation/calibration.py`: fits one isotonic calibrator per frozen model on validation probabilities
  against validation truth, writes the `x`/`y` thresholds to JSON, applies the calibrator unchanged to
  test, proves the calibration never reorders rows, and reports test Brier, expected calibration error and
  equal-count reliability points before and after calibration. The payment test rows are rebuilt with the
  task 4 production path and must reproduce task 4's totals and truth AP exactly.
- `evaluation/onnx_export.py`: merges the 50 frozen account members into one margin-space booster (every
  leaf weight divided by 50), reloads the file and proves its round and feature counts, gates the merge on
  the probability error, the decisions at the frozen account cut and the ranking, exports the merged
  booster and the frozen payment booster with `onnxmltools`, and checks each ONNX file against the native
  production probability on 20,000 validation and 20,000 test rows that keep their NaNs.
- `serving/model_repository/mule_account_xgb/` and `mule_payment_xgb/`: Triton model folders in the same
  layout as `fraud_xgboost` (a `config.pbtxt` plus a version folder holding `model.onnx`). The configs fix
  the input name `input`, 42 and 54 float32 features, the outputs `label` and `probabilities`, the
  `max_batch_size: 128` used by the rest of the repository, and the CPU instance group.

## Rules
A-059 pre-registers the reason-code map, the 20,000-row sampling rule and the top-3 TreeSHAP ranking.
A-060 and A-060a fix the calibration truth definitions, the validation-only fit, the monotone rule, the
ten equal-count bins and the payment test check under the owner ruling. A-061 and A-061a fix the merge
rule, the converter contract, the parity tolerance and the output shape. A-061b replaces A-061's 1e-6
margin gate with the probability, decision and ranking gates and states explicitly that it was written
after the margin gate failed at 1.310e-05. A-062 fixes the Triton folder layout, the input and output
contract, the batch limit and the smoke-load rule, and states explicitly that it was recorded after the
load ran.

## Reason codes (SIMULATED, validation, 20,000 rows per model)
Run: exit 0, git `c097f0b`, wall 138.67 s (wrapper 146.06 s), peak committed 18.19 GiB. Both models:
20,000 sampled rows, 0 rows without a reason. Shares overlap because one row can carry several codes.

| Code | Account rows | Account share | Payment rows | Payment share |
| --- | --- | --- | --- | --- |
| ROUTINE_LIFE | 12,035 | 60.2% | 6,007 | 30.0% |
| REGION | 10,180 | 50.9% | 20 | 0.1% |
| OUTFLOW | 8,280 | 41.4% | 7,449 | 37.2% |
| CONTROL | 6,040 | 30.2% | 1,628 | 8.1% |
| LIFECYCLE | 4,636 | 23.2% | 1,649 | 8.2% |
| NETWORK | 4,045 | 20.3% | 6,945 | 34.7% |
| INFLOW | 3,709 | 18.5% | 344 | 1.7% |
| BALANCE_PATTERN | 3,038 | 15.2% | 3,994 | 20.0% |
| PAYMENT_LEVEL | 0 | 0.0% | 16,504 | 82.5% |
| PAYEE_RISK | 0 | 0.0% | 7,585 | 37.9% |
| INTELLIGENCE | 0 | 0.0% | 0 | 0.0% |

## Calibration (SIMULATED)
Run: exit 0, git `22008d8`, wall 1,047.62 s (wrapper 1,055.55 s), peak committed 18.94 GiB.
The fit uses validation only: account 1,441,229 rows / 37,619 positives, payment 1,834,135 rows /
25,750 positives. Test is a check only, never a choice.

| Model | Test rows | Positives | Brier before → after | ECE before → after |
| --- | --- | --- | --- | --- |
| Account | 1,395,478 | 40,751 | 0.058347 → 0.009745 | 0.127137 → 0.001843 |
| Payment | 1,846,587 | 43,239 | 0.017319 → 0.017060 | 0.012284 → 0.012821 |

The payment ECE rose slightly while its Brier fell; both numbers are reported as measured. The monotone
check passes on validation and test for both models. The task 4 reproduction matches exactly: 120 worlds,
1,846,587 rows, 43,239 positives, truth AP median 0.632033, against `runs/full/payment_model_v1/test_score/result.json`
(sha256 `e157175f60c0926dec4d38035634c96e3e23291dc1aa377e42dbb97378739ab6`).
Calibrators: `account_isotonic.json` sha256 `0d287917a2c68e089f2375d6d85c2600db69b5b3b01aea57d3fa11f648afa33d`,
`payment_isotonic.json` sha256 `00ce58b8d49604174bc83de889ded17f0af7cbe0584556321fab0fbf62276818`.

## ONNX export (SIMULATED)
Run: exit 0, git `34cdbb5`, wall 533.25 s (wrapper 552.51 s), peak committed 23.49 GiB. Merged booster:
5,900 trees, leaf scale 0.02, sha256 `a21ba6033ad957dc75f413eca92348abf11e2835af996f3f7651b15e17f4a954`.

Merge gate on the 20,000-row account validation sample (all 20,000 rows hold at least one NaN feature):

- probability max |merged − bagged| 7.059898836092593e-07, mean 8.631150705059898e-08 (limit 1e-5)
- margin difference, reported only: max 1.3103485107635038e-05, mean 2.957019532760149e-06
- decisions at the frozen cut 0.9631869778454679 (`manifest.xgboost_cut` in
  `runs/full/run_a320_stage5c/extra_reporting.json`): 0 flips
- ranking: Spearman 0.9999999998845 (minimum 0.999999), top-1% 200 rows identical

Parity, ONNX versus native production probability, 20,000 rows per sample:

| Sample | Max abs diff | Mean abs diff | Rows over 1e-5 |
| --- | --- | --- | --- |
| Account validation | 1.7884182645921953e-07 | 3.294305767656536e-08 | 0 |
| Account test | 1.6789018467133765e-07 | 3.299944884183842e-08 | 0 |
| Payment validation | 4.470348358154297e-07 | 3.724137277458794e-08 | 0 |
| Payment test | 5.364418029785156e-07 | 3.7679821252822874e-08 | 0 |

Converter: `onnxmltools.convert_xgboost` from onnxmltools 1.16.0 (onnx 1.20.1, onnxruntime 1.24.2,
xgboost 3.2.0), opset `ai.onnx.ml:1` for both graphs, one `TreeEnsembleClassifier`, input `input`,
outputs `label` and `probabilities`. Files: `mule_account_xgb.onnx` sha256
`e1823d5d35b930e16344ca2fe792f04ae726a10207459c6015ee95f0db2bb4ef`, `mule_payment_xgb.onnx` sha256
`7eb86c0e582c49a008fd89858cc220f4936ad8b90ac8566b720ae7aad20371d0`.

## Triton folders (task 9, A-062)
Both folders load in the local Docker Triton image and serve 100-row requests. A-062 was recorded after
this load ran; it states that no setting or check was taken from the result.

Triton load: PASS (Docker engine 29.0.1 running, `nvcr.io/nvidia/tritonserver:24.01-py3`, gRPC on
127.0.0.1:8001, 100 rows per model, seed 20261124, one NaN feature in each batch). Server log:

```
I1124 20:35:35.955126 1 model_lifecycle.cc:827] successfully loaded 'mule_account_xgb'
I1124 20:35:35.976062 1 model_lifecycle.cc:827] successfully loaded 'mule_payment_xgb'
```

| Model | Response | Max abs diff vs the same file on local onnxruntime | Labels |
| --- | --- | --- | --- |
| mule_account_xgb | probabilities (100, 2), label int64 (100, 1) | 1.1920928955078125e-07 | match |
| mule_payment_xgb | probabilities (100, 2), label int64 (100, 1) | 2.980232238769531e-07 | match |

The container was stopped after the check. `probabilities` column 1 is the RAW model probability; the
isotonic calibrators above are applied by the client, not by Triton.

## Generated facts
The block below is the verbatim output of:

```
PYTHONPATH=. py -3.12 -B scripts/handoff_facts.py --part 18b
```

No register entry is touched by Level 18b, so the Register Changes section is empty.

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

## Open items
- No mule production client exists yet: building the 42/54-feature rows and applying the isotonic
  calibrator after Triton are not part of Level 18b.
- The dynamic-batching settings in the two new configs are copied from `fraud_xgboost`; they were not
  tuned for these models.
- `tests/integration/test_triton_io_contract.py` freezes only `fraud_xgboost`; no mule Triton test was
  added.
- The Triton load ran against the local compose service; CI has no Triton job.

## How to verify
```
py -3.12 -m pytest -p no:cacheprovider tests/unit/evaluation/test_reason_codes.py tests/unit/evaluation/test_calibration.py tests/unit/evaluation/test_onnx_export.py
py -3.12 -m ruff check evaluation/reason_codes.py evaluation/calibration.py evaluation/onnx_export.py scripts/handoff_facts.py
py -3.12 -m ruff format --check evaluation/reason_codes.py evaluation/calibration.py evaluation/onnx_export.py scripts/handoff_facts.py
py -3.12 -m mypy --follow-imports=skip evaluation/reason_codes.py evaluation/calibration.py evaluation/onnx_export.py scripts/handoff_facts.py
py -3.12 -m core.imports_check
py -3.12 -m core.truth_import_check
py -3.12 -B scripts/verify_world.py
```

The three runs can be repeated with the scripts that produced them:
`runs/full/level18b/reason_codes/run.ps1` (`py -3.12 -u -m evaluation.reason_codes run`),
`runs/full/level18b/calibration/run.ps1` (`py -3.12 -u -m evaluation.calibration run`) and
`runs/full/level18b/onnx/run.ps1` (`py -3.12 -u -m evaluation.onnx_export run --validation … --test … --out-root …`).
