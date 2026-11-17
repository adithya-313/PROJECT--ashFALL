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
- `9de39f5` `fix(decisions): drop a positive snapshot taken at the label's arrival instant`
- `88b7103` `fix(evaluation): fill report flags constant and empty features from config`
- `1e8f55f` `feat(evaluation): explain mules the oracle misses (report only, no metric change)`

## Generated facts
The key list below is the verbatim output of:

```
PYTHONPATH=. py -3.12 -B -c "
import yaml
with open('config/mule_model.yaml','r',encoding='utf-8') as f:
    raw=yaml.safe_load(f)
print('\n'.join(sorted(raw['parameters'])))
"
feature_fill_flag_share
label_cut
label_maturity_days
```

## What each component does
- `config/mule_model.yaml`: three labelled `policy` blocks.
  - `label_cut: arrival_time` — the bank learns about the account at the label's exact arrival time;
    `label_day` stays as the legacy day-granularity cut used only to reproduce the first pilot
    dataset.
  - `label_maturity_days: null` with `not_found: true` — off until the delay has been measured on the
    full training worlds; the reason says so.
  - `feature_fill_flag_share: 0.999` — a feature whose most common value covers this share of rows
    or more cannot separate anything and is flagged by the sanity report.
- `decisions/mule_account_model.py`
  - `MuleModelConfig` and `load_mule_model_config`: strict loader (unknown root keys, unknown
    parameter keys, missing blocks, missing `label`/`derived_from`, invalid values all raise
    `ValueError`; a missing file raises `FileNotFoundError`). A null `label_maturity_days` is
    accepted only with `not_found: true`; `feature_fill_flag_share` must be in (0.0, 1.0].
  - `build_training_rows(..., label_cut=None, label_maturity_days=None)`: omitted switches are read
    from `config/mule_model.yaml`.
    - `arrival_time` keeps a positive account's snapshots only strictly before its earliest label
      arrival (`snapshot_time < arrival`; a snapshot at the arrival instant is dropped);
      `label_day` keeps the legacy whole-IST-day cut.
    - `label_maturity_days = N` drops each world's final N IST days, measured against that world's
      own last snapshot day; null keeps every day.
- `evaluation/level18_sanity.py`
  - `feature_fill_report(snapshot_rows, config_path=None)`: per feature it reports rows,
    `fill_share`, `missing_share`, `most_common_value`, `most_common_share` (a missing value counts
    as one value) and `distinct_count`. A feature is flagged when it is missing from every row or
    its `most_common_share` reaches `feature_fill_flag_share` (read from config). Features missing
    from more than half the rows are also listed in `mostly_missing_features`, as information.
  - `oracle_report(snapshot_rows, payment_rows)`: scores account-days 1.0 for mule roles and 0.0
    otherwise, then runs the pilot's own `daily_escalations`, `ground_sets`, `recall_at_capacity`
    and `yield_rate` with the capacity policy from `config/evaluation.yaml`. `misses` explains every
    mule the oracle did not escalate:
    - `ground_set_mules`, `found_mules`, `truth_mule_accounts`;
    - `missed_without_mule_day`: mule accounts named by payment truth that have no snapshot day with
      a mule role;
    - `missed_with_mule_days`: ground-set mules never escalated;
    - `blocked_slots_on_missed_mule_days` / `missed_mule_day_slots` / `blocked_slot_share`: on each
      missed mule's mule days, the escalation slots already spent on accounts escalated on a
      strictly earlier day;
    - `total_escalations`, `repeat_escalations`, `repeat_escalation_share`: a repeat is an
      escalation of an account escalated on a strictly earlier day.
    The oracle score, the escalation function and every metric are unchanged.
  - CLI: `--validation` (repeatable), `--truth-root`, `--out`.
- `evaluation/measure_label_delay.py`
  - `measure_label_delay`: uses the evaluation's `first_illicit_credit` clock; label delay is the
    first label arrival minus that credit in days; percentiles use `percentile_linear`; mules with
    no bank-visible label are reported as censored. Every label row must carry a `source` in
    complaint / police_bulk / registry, and a negative delay is listed with that source in
    `negative_delays` (plus `negative_delay_count`).
  - `build_label_delay_report`: one block per world plus pooled statistics, including
    `negative_delay_count`.
  - CLI: `--training` (repeatable), `--truth-root`, `--out`.

## Label cut on the pilot training split
- `label_day` (the previous behaviour): 69865 rows, 272 positives.
- `arrival_time` (the config default): 69865 rows, 272 positives. No pilot training row sat at or
  after its label's exact arrival time, so the strict boundary changes nothing on this dataset
  while correcting the contract.

## Sanity report on pilot validation (`validation-20261031T1830-efb891155a7e`) (SIMULATED)
- Feature fill: 35874 snapshot rows, 39 account-model features, flag share 0.999; 4 features
  flagged for concentration: `cash_atm_paise_7d` most common 0 (0.99980),
  `cash_branch_paise_7d` 0 (0.99997), `sim_swapped` 0 (0.99964), `app_reinstalled_same_phone` 0
  (1.0); no feature is missing from every row. 8 features are listed as mostly missing (missing
  share): `first_time_payer_share_24h` 0.5997, `pass_through_ratio_24h` 0.5997,
  `median_credit_to_debit_seconds_7d` 0.6217, `salary_like_payer_count` 0.5760,
  `recurring_merchant_payee_count` 0.5760, `regular_transfer_payee_count` 0.5760,
  `routine_life_present` 0.5760, `days_since_device_change` 0.9425.
- Oracle ceiling (perfect ranking, same capacity policy): world `validation-0000` 12909 account-days,
  14 mule accounts, 130 escalations, recall at capacity 0.7143, measured yield 1.0; `validation-0001`
  10824 / 28 / 125 / 0.6786 / 1.0; `validation-0002` 12141 / 23 / 88 / 0.6957 / 1.0. Even a perfect
  ranker misses about three mules in ten under this capacity, which sets the real ceiling.
- Oracle misses (a–f, per world and total):

| measure | 0000 | 0001 | 0002 | total |
|---|---|---|---|---|
| a ground-set mules | 14 | 28 | 23 | 65 |
| b found by the oracle | 10 | 19 | 16 | 45 |
| c missed, no mule-day snapshot | 0 | 0 | 0 | 0 |
| d missed, had mule days | 4 | 9 | 7 | 20 |
| e blocked slots / mule-day slots | 6 / 17 | 215 / 258 | 45 / 81 | 266 / 356 |
| e blocked-slot share | 0.3529 | 0.8333 | 0.5556 | 0.7472 |
| f escalations / repeats | 130 / 120 | 125 / 106 | 88 / 72 | 343 / 298 |
| f repeat share | 0.9231 | 0.8480 | 0.8182 | 0.8688 |

  The oracle misses come from review slots spent on accounts already escalated, not from mules
  absent from the snapshots; on this split 20 of 65 ground-set mules never got a slot.

## Label delay on pilot training (`training-20261031T1830-687fcb9c6d16`) (SIMULATED)
- Pooled: 142 accounts with a first illicit credit, 31 labelled, 111 censored; p50 3.9 days, p90
  45.89 days, p95 46.90 days; min -4.91 days (a label arriving before the measured credit), max
  47.96 days.
- Per world (credit / labelled / censored / p50 / p90 / p95 days): `training-0000` 27 / 5 / 22 /
  1.13 / 7.95 / 9.19; `training-0001` 27 / 8 / 19 / 29.65 / 47.90 / 47.93; `training-0002` 22 / 3 /
  19 / 2.41 / 3.08 / 3.16; `training-0003` 29 / 1 / 28 / 7.02 / 7.02 / 7.02; `training-0004` 18 / 6 /
  12 / 0.16 / 5.08 / 5.84; `training-0005` 19 / 8 / 11 / 6.83 / 20.05 / 21.60.
- Negative delays (g): 1 pooled, `acc_person_000088` in `training-0000` at -4.913 days, label
  source `registry`; no other world has one.
- Censoring dominates: most mules never receive a bank-visible label before their world ends, so a
  maturity window measured only on labelled mules would understate the immature tail. The config
  stays null until a value is chosen with this in mind.

## Verification
- Full unit suite (round 1): `2329 passed, 1 skipped` in 465.04s (`env -u NO_PROXY PYTHONPATH=. py
  -3.12 -m pytest tests/unit -q -p no:cacheprovider`).
- Fix round test runs: `test_mule_model_labels.py` 8 passed; `test_mule_account_model.py` +
  `test_run_level_18_pilot.py` 11 passed in one 19-test run; `test_level18_sanity.py` 9 passed;
  `test_measure_label_delay.py` 6 passed; `test_rules_metrics.py` included in the 40-test run.
- `tests/baselines`: 11 passed.
- `scripts/verify_world.py`: `ALL GOLDEN CHECKS PASSED`.
- `python -m core.config_validator config/mule_model.yaml`: exit 0.
- Fixture oracle test: recall at capacity 0.7692, measured yield 1.0; the all-1.0 sabotage scored
  0.6154 and 0.0526, so the test goes red without the role-based oracle score.
- Fix-round sabotage proofs: reverting the cut to `>` fails
  `test_snapshot_exactly_at_arrival_time_is_dropped`; dropping the most-common-value condition fails
  the constant-feature and config-share tests; counting same-day repeats fails the capacity-1
  oracle-miss test. Each was restored and re-run green.

## Rulings applied
- C1 option (a): `tests/unit/decisions/test_mule_account_model.py` got `label_cut="label_day"` in its
  four `build_training_rows` calls and one new `arrival_time` test; no assertion changed.
- C3: `label_maturity_days: null` with `not_found: true` approved, matching
  `core/config_validator.py:187` and `config/legal_actions.yaml`.

## Open items
- Choose a `label_maturity_days` value after measuring on the full training worlds; today it is off.
- `evaluation/run_level_18_pilot.py` was not edited and now runs with the `arrival_time` default by
  design.
