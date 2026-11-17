# Level 18a4 Handoff Note — exact label-time cut, sanity report, label delay

All figures below are from tiny sampler worlds (SIMULATED) unless stated otherwise.

## Where things stand
Level 18a4 prepares the account model for honest training rows: a row is kept only if the bank could
have seen it before the label, the inputs and the capacity ceiling are measured before a model is
judged, and the label delay is measured so a maturity window can be chosen later.
Branch: `feature/level-18a4-label-cut-sanity`, based on `d61a65e` of
`feature/level-18-mule-models`.
Commits:
- `4bb46f3` `feat(decisions): cut account-model training rows at the label arrival time`
- `baa3086` `feat(evaluation): add a validation sanity report for feature fill and the oracle ceiling`
- `c44d802` `feat(evaluation): measure label delay and add an off-by-default maturity window`

## Generated facts
The key list below is the verbatim output of:

```
PYTHONPATH=. py -3.12 -B -c "
import yaml
with open('config/mule_model.yaml','r',encoding='utf-8') as f:
    raw=yaml.safe_load(f)
print('\n'.join(sorted(raw['parameters'])))
"
label_cut
label_maturity_days
```

## What each component does
- `config/mule_model.yaml`: two labelled `policy` blocks.
  - `label_cut: arrival_time` — the bank learns about the account at the label's exact arrival time;
    `label_day` stays as the legacy day-granularity cut used only to reproduce the first pilot
    dataset.
  - `label_maturity_days: null` with `not_found: true` — off until the delay has been measured on the
    full training worlds; the reason says so.
- `decisions/mule_account_model.py`
  - `MuleModelConfig` and `load_mule_model_config`: strict loader (unknown root keys, unknown
    parameter keys, missing blocks, missing `label`/`derived_from`, invalid values all raise
    `ValueError`; a missing file raises `FileNotFoundError`). A null `label_maturity_days` is
    accepted only with `not_found: true`.
  - `build_training_rows(..., label_cut=None, label_maturity_days=None)`: omitted switches are read
    from `config/mule_model.yaml`.
    - `arrival_time` keeps a positive account's snapshots with `snapshot_time <=` its earliest label
      arrival (the boundary instant is kept); `label_day` keeps the legacy whole-IST-day cut.
    - `label_maturity_days = N` drops each world's final N IST days, measured against that world's
      own last snapshot day; null keeps every day.
- `evaluation/level18_sanity.py`
  - `feature_fill_report`: per-feature share of rows carrying a value; a feature below
    `FEATURE_FILL_FLAG_SHARE = 0.999` is flagged.
  - `oracle_report`: scores account-days 1.0 for mule roles and 0.0 otherwise, then runs the pilot's
    own `daily_escalations`, `ground_sets`, `recall_at_capacity` and `yield_rate` with the capacity
    policy from `config/evaluation.yaml`.
  - CLI: `--validation` (repeatable), `--truth-root`, `--out`.
- `evaluation/measure_label_delay.py`
  - `measure_label_delay`: uses the evaluation's `first_illicit_credit` clock; label delay is the
    first label arrival minus that credit in days; percentiles use `percentile_linear`; mules with
    no bank-visible label are reported as censored.
  - `build_label_delay_report`: one block per world plus pooled statistics.
  - CLI: `--training` (repeatable), `--truth-root`, `--out`.

## Label cut on the pilot training split
- `label_day` (the previous behaviour): 69865 rows, 272 positives.
- `arrival_time` (the new config default): 69865 rows, 272 positives. No pilot training row sat
  after its label's exact arrival time, so the switch changes nothing on this dataset while
  correcting the contract.

## Sanity report on pilot validation (`validation-20261031T1830-efb891155a7e`)
- 35874 snapshot rows, 39 account-model features, flag threshold 0.999; 18 features flagged:
  `first_time_payer_share_24h` 0.4003, `pass_through_ratio_24h` 0.4003, `pass_through_ratio_7d`
  0.5546, `median_credit_to_debit_seconds_7d` 0.3783, `top_receiver_share_7d` 0.8373,
  `other_bank_outflow_share_7d` 0.8373, `current_balance_paise` 0.998, `min_balance_paise_7d`
  0.998, `near_zero_cycles_7d` 0.998, `salary_like_payer_count` 0.424,
  `recurring_merchant_payee_count` 0.424, `regular_transfer_payee_count` 0.424,
  `routine_life_present` 0.424, `dormancy_days_before_recent_activity` 0.8951,
  `volume_vs_own_history_ratio` 0.9667, `days_since_device_change` 0.0575,
  `machine_regular_timing_score` 0.8672, `local_cluster_density` 0.9409.
- Oracle ceiling (perfect ranking, same capacity policy): world `validation-0000` 12909 account-days,
  14 mule accounts, 130 escalations, recall at capacity 0.7143, measured yield 1.0; `validation-0001`
  10824 / 28 / 125 / 0.6786 / 1.0; `validation-0002` 12141 / 23 / 88 / 0.6957 / 1.0. Even a perfect
  ranker misses about three mules in ten under this capacity, which sets the real ceiling.

## Label delay on pilot training (`training-20261031T1830-687fcb9c6d16`)
- Pooled: 142 accounts with a first illicit credit, 31 labelled, 111 censored; p50 3.9 days, p90
  45.89 days, p95 46.90 days; min -4.91 days (a label arriving before the measured credit), max
  47.96 days.
- Per world (credit / labelled / censored / p50 / p90 / p95 days): `training-0000` 27 / 5 / 22 /
  1.13 / 7.95 / 9.19; `training-0001` 27 / 8 / 19 / 29.65 / 47.90 / 47.93; `training-0002` 22 / 3 /
  19 / 2.41 / 3.08 / 3.16; `training-0003` 29 / 1 / 28 / 7.02 / 7.02 / 7.02; `training-0004` 18 / 6 /
  12 / 0.16 / 5.08 / 5.84; `training-0005` 19 / 8 / 11 / 6.83 / 20.05 / 21.60.
- Censoring dominates: most mules never receive a bank-visible label before their world ends, so a
  maturity window measured only on labelled mules would understate the immature tail. The config
  stays null until a value is chosen with this in mind.

## Verification
- Full unit suite: `2329 passed, 1 skipped` in 465.04s (`env -u NO_PROXY PYTHONPATH=. py -3.12 -m
  pytest tests/unit -q -p no:cacheprovider`).
- `tests/baselines`: 11 passed.
- `scripts/verify_world.py`: `ALL GOLDEN CHECKS PASSED`.
- Fixture oracle test: recall at capacity 0.7692, measured yield 1.0; the all-1.0 sabotage scored
  0.6154 and 0.0526, so the test goes red without the role-based oracle score.

## Rulings applied
- C1 option (a): `tests/unit/decisions/test_mule_account_model.py` got `label_cut="label_day"` in its
  four `build_training_rows` calls and one new `arrival_time` test; no assertion changed.
- C3: `label_maturity_days: null` with `not_found: true` approved, matching
  `core/config_validator.py:187` and `config/legal_actions.yaml`.

## Open items
- Choose a `label_maturity_days` value after measuring on the full training worlds; today it is off.
- `evaluation/run_level_18_pilot.py` was not edited and now runs with the `arrival_time` default by
  design.
