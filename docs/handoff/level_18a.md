# Level 18a handoff: the frozen-results console

Date: 2026-11-25. Rules: `docs/research/amendments.md`, A-063. All numbers below are SIMULATED and
copied from `reports/level_18/console/`, which the generator built from frozen result files only.

## What this part adds

`scripts/level18_console.py` reads the already-written Level 18 result files and the 120 world cards
and writes nine JSON files under `reports/level_18/console/`. It fits no model, scores no row, runs
no SHAP or Triton work and reads no dataset. Each file carries `"simulated": true` and a `manifest`
with the sha256 of every input it read and the generating commit hash.

| File | What it carries |
| --- | --- |
| `comparison.json` | XGBoost, rules, anomaly and the combined baseline on the 120 Run A test worlds: genuine restriction rate, time to detection, recall at capacity, measured and E1-re-weighted yield, each with median and p10-p90; the frozen cut; the account-model stress status |
| `hidden_depth.json` | Hidden-mule recall by depth 1-5 with accounts and worlds used |
| `held_out_families.json` | Held-out families `device_rotation` and `slow_drip`, recall at capacity and at cut |
| `calibration.json` | Every test reliability bin before and after isotonic calibration, with Brier and ECE, and the calibrator sha256 |
| `reason_codes.json` | Reason-code frequencies for both models over 20,000 sampled rows each |
| `payment_results.json` | Payment test and stress results, same-bank vs other-bank truth AP, NaN payee-risk share, payee precision, payee_risk feature AP, frozen model identity and the payment leakage statuses |
| `leakage.json` | The seven published leakage checks and the trivial-world sanity check, with check 4 recorded as not run |
| `onnx_parity.json` | ONNX file identity, validation and test parity, the account merge proof, tolerances and converter versions |
| `yield_e1.json` | Per-world E1 inflation factors with measured and re-weighted yield, and the E1 aggregate |

Sources: `runs/full/run_a320_stage5c/final/45c496799170/result.json`,
`runs/full/run_a320_stage5c/extra_reporting.json`,
`runs/full/run_a320_stage5c/final/45c496799170/leakage_checks.json`,
`runs/full/run_a320_stage5c/leakage_checks_5_6.json`,
`runs/full/payment_model_v1/test_score/result.json`,
`runs/full/payment_model_v1/diagnosis/diagnosis.json`,
`runs/full/level18b/calibration/calibration_data.json`,
`runs/full/level18b/reason_codes/account_model.json`,
`runs/full/level18b/reason_codes/payment_model.json`,
`runs/full/level18b/onnx/onnx_export.json`, and the 120
`runs/full/world_cards/<run_id>/base_rate.json` cards.

Regenerate with:

```
PYTHONPATH=. py -3.12 -B scripts/level18_console.py
```

## Key numbers

Comparison on the 120 Run A test worlds, median (p10, p90):

| Scorer | Recall at capacity | E1-re-weighted yield | Genuine restriction rate | Time to detection (hours) |
| --- | --- | --- | --- | --- |
| xgboost | 0.642857 (0.477045, 0.833333) | 0.272955 (0.086673, 1.0) | 0.005882 (0.0, 0.008958) | 11.675 (1.1675, 33.780833), 120 worlds |
| rules | 0.368112 (0.181818, 0.567029) | 0.026074 (0.012054, 0.067862) | 0.023529 (0.014706, 0.032964) | 17.354167 (4.66, 55.645833), 120 worlds |
| anomaly | 0.0 (0.0, 0.090909) | 0.0 (0.0, 0.078996) | 0.002985 (0.0, 0.011765) | 10.45 (0.0, 453.043333), 37 worlds |
| rules_and_anomaly | 0.375 (0.193424, 0.567029) | 0.022743 (0.010462, 0.061900) | 0.026786 (0.017694, 0.041176) | 17.570833 (4.446667, 55.128333), 120 worlds |

Hidden-mule recall by depth, median (p10, p90), worlds used:

| Depth | Accounts | Median accounts per world | Recall |
| --- | --- | --- | --- |
| 1 | 704 | 6.0 | 1.0 (0.83, 1.0), 120 |
| 2 | 687 | 6.0 | 0.944444 (0.666667, 1.0), 120 |
| 3 | 526 | 6.0 | 1.0 (0.6, 1.0), 90 |
| 4 | 353 | 6.0 | 0.857143 (0.597143, 1.0), 60 |
| 5 | 171 | 5.0 | 1.0 (0.595556, 1.0), 30 |

Held-out families:

| Family | Accounts | Recall at capacity | Recall at cut | Worlds used / absent |
| --- | --- | --- | --- | --- |
| device_rotation | 448 | 0.666667 (0.398462, 1.0) | 0.930288 (0.666667, 1.0) | 60 / 60 |
| slow_drip | 446 | 0.8 (0.5, 1.0) | 0.909091 (0.660606, 1.0) | 59 / 61 |

Calibration on test (isotonic fitted on validation only):

| Model | Rows | Positives | Brier raw -> calibrated | ECE raw -> calibrated |
| --- | --- | --- | --- | --- |
| account | 1,395,478 | 40,751 | 0.058347 -> 0.009745 | 0.127137 -> 0.001843 |
| payment | 1,846,587 | 43,239 | 0.017319 -> 0.017060 | 0.012284 -> 0.012820 |

Reason codes over 20,000 sampled rows each (11 codes per model), top codes by rows:

| Model | Top codes (rows, share) |
| --- | --- |
| account | ROUTINE_LIFE (12,035, 0.60175), REGION (10,180, 0.509), OUTFLOW (8,280, 0.414), CONTROL (6,040, 0.302) |
| payment | PAYMENT_LEVEL (16,504, 0.8252), PAYEE_RISK (7,585, 0.37925), OUTFLOW (7,449, 0.37245), NETWORK (6,945, 0.34725) |

Payment model, median (p10, p90) where published:

| Split | Truth AP | Same-bank AP | Other-bank AP | Precision | Recall | Hold rate | NaN payee risk |
| --- | --- | --- | --- | --- | --- | --- | --- |
| test | 0.632033 (0.385806, 0.788270) | 0.876438 | 0.467391 | 0.994179 | 0.247117 | 0.005003 | 0.617159 |
| stress | 0.070938 (0.015870, 0.353742) | 0.103087 | 0.080064 | 0.093817 | 0.174830 | 0.028382 | 0.610330 |

Payee precision on test: 0.911046 per payee (932 positive of 1,023 distinct payees over 11,196 held
payments), 0.942390 per payment, against the reported per-world precision median 0.994179. The
`payee_risk` feature alone scores truth AP 0.922653 on test and 0.543550 on stress.

Leakage: `truth_leak` pass, `split_leak` pass, `check_5a` pass, `check_5b` pass, `check_6a` pass,
`check_6b` fail (validation AP 0.637781 over base rate 0.061236, ratio 10.415126, limit 2.0),
`section_23_check_7` pass. The trivial-world sanity check passes (best feature `account_age_days`).
Check 4 is recorded as not run: no repository file evaluates it. The payment diagnosis reports
`truth_and_simulator_only_fields` pass, `temporal_cutoff` pass, `split_isolation` pass, `id_leak` fail
(the same Run A 6b result), `label_dependent_inputs_by_arrival` pass.

ONNX parity, maximum absolute probability difference (20,000 rows per split, NaN injected):

| Model | File | sha256 (first 12) | Validation | Test | Rows over 1e-5 |
| --- | --- | --- | --- | --- | --- |
| account | mule_account_xgb.onnx | e1823d5d35b9 | 1.788418e-07 | 1.678902e-07 | 0 |
| payment | mule_payment_xgb.onnx | 7eb86c0e582c | 4.470348e-07 | 5.364418e-07 | 0 |

Account merge proof: probability max difference 7.059899e-07, decision flips 0 at cut 0.9631869778,
Spearman 0.9999999998845, top-1% set identical; margin max difference 1.310349e-05 is reported only.

E1: the per-world inflation factor has median 39.316691 over the 118 worlds with a positive factor and
0.0 in 2 worlds, where the re-weighted yield is therefore undefined. Aggregate re-weighted yield:
xgboost 0.272955, rules 0.026074, anomaly 0.0, rules_and_anomaly 0.022743 (median; 118 worlds defined).

## Handoff facts

Verbatim output of `PYTHONPATH=. py -3.12 -B scripts/handoff_facts.py --part 18a`:

## Register Changes

---

## Public Interfaces

Extracted directly via `inspect.signature`:

### `scripts/level18_console.py`

```python
build(root: 'Path' = WindowsPath('C:/Users/sinne/Downloads/PROJECT-ashFALL'), output_dir: 'Path | None' = None) -> 'list[Path]'
```

```python
collect_calibration(document: 'dict[str, Any]') -> 'dict[str, Any]'
```

```python
collect_comparison(run_a: 'dict[str, Any]', extra: 'dict[str, Any]') -> 'dict[str, Any]'
```

```python
collect_held_out_families(extra: 'dict[str, Any]') -> 'dict[str, Any]'
```

```python
collect_hidden_depth(extra: 'dict[str, Any]') -> 'dict[str, Any]'
```

```python
collect_leakage(leakage: 'dict[str, Any]', leakage_5_6: 'dict[str, Any]') -> 'dict[str, Any]'
```

```python
collect_onnx_parity(document: 'dict[str, Any]') -> 'dict[str, Any]'
```

```python
collect_payment_results(result: 'dict[str, Any]', diagnosis: 'dict[str, Any]') -> 'dict[str, Any]'
```

```python
collect_reason_codes(account: 'dict[str, Any]', payment: 'dict[str, Any]') -> 'dict[str, Any]'
```

```python
collect_yield_e1(run_a: 'dict[str, Any]', cards: 'dict[str, dict[str, Any]]') -> 'dict[str, Any]'
```

```python
main() -> 'int'
```

---

## Records and Fields

Extracted directly via `dataclasses.fields`:

---

## Configuration Keys

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
