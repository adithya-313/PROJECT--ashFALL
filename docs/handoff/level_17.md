# Level 17 Handoff Note — rules baseline and cold start

The section "Generated facts" is the verbatim output of `py -3.12 scripts/handoff_facts.py --part 17`
(signatures from `inspect.signature`, fields from `dataclasses.fields`, keys read from the config
files). Regenerate it instead of editing it. Everything else is written by hand.

## Where things stand
Level 17 builds the bar ashFALL's models must beat: a rules engine from the typologies, an
isolation-forest anomaly detector for cold start, a cold-start policy, and an evaluator with
Section 7's metrics. Integration branch: `feature/level-17-rules-baseline`.
- 17a (PR #127): `largest_credit_paise_24h` (53 features; 45 account features); `decisions/`
  rules table `rule_1`..`rule_5` with reason codes RB1..RB5; pure `evaluate_rules`.
- 17b (PR #128): scikit-learn 1.8.0 in app requirements; isolation forest on early unlabelled
  traffic; cold-start policy and review ranking.
- 17c-1 (PR #129): optional `personas.jsonl` truth file (off by default); amendment A-039
  (persona weights); `evaluation/rules_metrics.py` checked against hand-computed numbers.
- 17c-2a (PR #130): per-account-day scoring; joint threshold tuning on validation (972 points).
- 17c-2b (PR #131): dataset script `--write-evaluation-inputs`; `python -m evaluation.run_level_17`
  writing `result.json` with a run manifest; untuned-threshold guard.
- 17d-1 (PRs #132, #133): `--sampler-seed` / `--find-sampler-seed`; committed showcase fixtures
  (4 real replayed tiny worlds, sampler seed 2); tuning harm measured per genuine account.
- 17d-2 (PR #134): `make show LEVEL=17`, 7 checks.
- 17d-3 (PR #135): issues 17-1..17-5, R5 evidence, this note.

## What each component does
- Rules (`decisions/rules_engine.py`, `config/rules_baseline.yaml`): five rules read only the 53
  bank-visible features. A rule fires only when all its conditions hold; a missing (None) feature
  never passes and makes the rule `not_evaluable`. Score = number of rules fired (no weights).
  rule_5 = `new_device_r5 == 1` AND money passing through — an OS update alone never fires it (R5).
- Anomaly detector (`decisions/anomaly_detector.py`): IsolationForest (200 trees, seed 1717) on
  39 features — the 45 account features minus 5 label-derived ones (hops_to_confirmed_label,
  shared_counterparties_with_flagged_7d, hops_to_complained_account, registry_match_count,
  registry_match_age_days) and account_type. Trains only on the first 14 days of training
  worlds; refuses validation/test/stress. Missing values = training median plus a `__missing`
  0/1 column; features missing in every row are listed in `all_missing_features`. Saved with
  joblib plus a manifest; load refuses a changed model file (SHA-256) or a different
  scikit-learn version. No labels parameter exists.
- Cold start (`decisions/cold_start.py`): rules + anomaly until a bank has >= 200 confirmed fraud
  labels (arrived by then, own bank only) AND an approved model exists. Review order: rules
  score, then anomaly score, then account_id.
- Evaluation (`evaluation/level17_scoring.py`, `rules_tuning.py`, `level17_evaluation.py`,
  `run_level_17.py`): one row per account per IST day; restriction = rules (any rule fires),
  anomaly (score >= tuned cut) or both; capacity = 0.25% of a bank's active accounts per day
  (min 1); harm counts only days an account was genuine (genuine or victim); collectors and
  cash-outs are mules but only role `mule` carries a depth; first flag = snapshot time of the
  first escalated day; test and stress summarised separately; summaries carry units (tuning
  rates are percent, summaries are fractions).

## Decisions and why
- Q1: add `largest_credit_paise_24h` — R1/R2 need a credit amount and no feature had one.
- Q2: rule_5 needs money passing through; report its near-zero mule recall honestly (17-1).
- Q3: emulator / remote-access flags do not exist (removed in Level 15); not used.
- Q4: no registry rule in Level 17; I4C not wired (17-2).
- Q5: evaluator built and tested now; >= 20-world medians in 18a (17-3).
- Q6: scikit-learn 1.8.0 in the app requirements (Level 19 fallback needs it).
- Q7: `personas.jsonl` behind a switch; A-039 weights retiree/student/gig_worker 2, others 1.
- Q8: starting thresholds `untuned_start`; joint tuning on validation to the 0.5% budget;
  test results refused while untuned.
- Q9: rule ids rule_1..rule_5, reason codes RB1..RB5, unweighted score.
- Q10: rule_3 fires on a busy tea stall — kept, tested and counted, not hidden.
- Q11: parts 17a..17d into one integration branch.
- numpy is not pinned in the app: a pin changed CI's numpy and broke a frozen bit-for-bit geo
  test; the evaluator uses a standard-library percentile instead (same as numpy "linear").
- Harm in tuning and evaluation uses one definition: genuine accounts restricted on >= 1
  genuine day / genuine active accounts.
- Real multi-world evaluation deferred: tiny worlds could not hold the gang mix (sampler seed 1)
  and are trivial for metrics; small worlds take hours with today's replay; 18a makes replay
  fast and re-runs this evaluator.

## Results — SIMULATED, tiny fixture worlds, 1 test world (proves the pipeline, not a result)
Validation tuning: rules rate 2.78% (1 of 36 genuine accounts), recall 0.538, meets_budget
false; anomaly cut 0.5805, rate 1.39%, recall 0.0, meets_budget false. A 0.5% budget is
unreachable with ~36 genuine accounts. Test world (spreads: not enough worlds):
rules recall 0.444, yield 0.80 measured / 0.023 reweighted, genuine restriction 0.0286,
median time to detection 10.2 h; anomaly recall 0.0; rules+anomaly recall 0.444, yield 0.40.

## Known gaps and deferrals
- 17-1 device rule mule recall near zero (gang device_acquired unmapped; SIM swaps 16-6).
- 17-2 I4C feed unused by bank features.
- 17-3 real >= 20-world evaluation not run (needs fast replay, 16-4 / 18a).
- 17-4 CI level-showcase times out on the Level 16 showcase (two ~45 min replays); Level 17's
  own showcase uses committed fixtures and is fast. Accept a red level-showcase only when its
  log has no check FAIL line.
- 17-5 tuned thresholds are not committed; Level 19 must commit full-scale validation values.
- Genuine-harm parts not measurable offline: downstream shops, lien release wait, warnings to
  genuine payers — reported "not run".
- `account_type` is not used by the anomaly detector (text category; encoding deferred).
- `evaluate_rules` reloads the YAML when no config is passed; the live decision path must load
  the config once and pass it in.
- A world's start is its earliest snapshot time.

## Coding-agent watch list (Level 17 incidents)
- Reported `meets_budget: true` when result.json said false — print values from files with a
  command, never retype.
- Replaced the planned tuning grid, claiming the values were unrecoverable — compare config
  values to the prompt.
- Anchored first flag at midnight (detection looked earlier) and mixed test with stress — check
  definitions against the spec.
- Added error raises on empty input that would crash honest "nothing happened" cases.
- Pinned counts were missed by the architect: test_account_snapshots (44), test_m1_balance (19),
  showcase level_15_support_c CANONICAL_FEATURE_COUNT (52) — the last was hidden by the CI
  timeout. Search tests AND showcase for pinned counts; read the showcase log for FAIL lines.

## How to verify
- `py -3.12 -m showcase --level 17` and `--quick` -> 7 checks, Overall: PASSED.
- `py -3.12 -m pytest -q -p no:cacheprovider tests/unit/decisions tests/unit/evaluation`
- `py -3.12 -B scripts/verify_world.py` -> ALL GOLDEN CHECKS PASSED.
- Fixture evaluation: `py -3.12 -m evaluation.run_level_17 --training showcase/fixtures/level_17/datasets/training-20261031T1830-7a356ef53e9e --validation showcase/fixtures/level_17/datasets/validation-20261031T1830-9d983967b617 --test showcase/fixtures/level_17/datasets/test-20261031T1830-9a2856ce53b2 --truth-root showcase/fixtures/level_17/truth --world-cards-root showcase/fixtures/level_17/world_cards`

## Generated facts

## Register Changes
- 17-1: Device rule catches almost no mules: gang phone acquisition is not in the bank feed and SIM swaps during a run are not modelled (status: planned)
- 17-2: The I4C feed is not wired into bank features; the non-DPIP bank has no registry signal (status: planned)
- 17-3: Level 17 real multi-world evaluation not run: replay is too slow (16-4) (status: planned)
- 17-4: CI level-showcase times out: the Level 16 showcase replays twice (~2.5 h) (status: planned)
- 17-5: Rule thresholds are tuned only per evaluation run; the decision path needs thresholds tuned on full-scale validation worlds (status: planned)
- R5: OS updates do not trigger new-device rules (status: built)

---

## Public Interfaces

Extracted directly via `inspect.signature`:

### `decisions/rules_config.py`

```python
all_thresholds_tuned(config: decisions.rules_config.RulesConfig | None = None) -> bool
```

```python
load_rules_config(path: pathlib.Path | str | None = None) -> decisions.rules_config.RulesConfig
```

```python
with_thresholds(config: decisions.rules_config.RulesConfig, new_values: dict[str, float | int], source: str) -> decisions.rules_config.RulesConfig
```

```python
Rule(id: str, reason_code: str, plain_english: str, conditions: tuple[decisions.rules_config.RuleCondition, ...]) -> None
```
- dataclass

```python
RuleCondition(feature: str, op: str, threshold: str | int | float, threshold_key: str | None, threshold_value: int | float) -> None
```
- dataclass

```python
RuleThreshold(key: str, value: int | float, label: str, reason: str, derived_from: str, status: str) -> None
```
- dataclass

```python
RulesConfig(thresholds: dict[str, decisions.rules_config.RuleThreshold], rules: tuple[decisions.rules_config.Rule, ...]) -> None
```
- dataclass

### `decisions/rules_engine.py`

```python
evaluate_rules(features: collections.abc.Mapping[str, float | int | None], config: decisions.rules_config.RulesConfig | None = None) -> decisions.rules_engine.RulesDecision
```

```python
RulesDecision(fired: tuple[tuple[str, str], ...], not_evaluable: tuple[str, ...], score: int) -> None
```
- dataclass

### `decisions/anomaly_config.py`

```python
input_feature_names(config: decisions.anomaly_config.AnomalyDetectorConfig | None = None) -> tuple[str, ...]
```

```python
load_anomaly_detector_config(path: pathlib.Path | str | None = None) -> decisions.anomaly_config.AnomalyDetectorConfig
```

```python
load_cold_start_config(path: pathlib.Path | str | None = None) -> decisions.anomaly_config.ColdStartConfig
```

```python
AnomalyDetectorConfig(excluded_features: tuple[str, ...], excluded_features_reasons: dict[str, str], early_window_days: int, n_estimators: int, max_samples: str | int | float, contamination: str | float, random_state: int, allowed_training_splits: tuple[str, ...]) -> None
```
- dataclass

```python
ColdStartConfig(min_positive_labels_per_bank: int, counted_verdict: str) -> None
```
- dataclass

### `decisions/anomaly_detector.py`

```python
build_feature_matrix(rows: collections.abc.Sequence[collections.abc.Mapping[str, typing.Any]], feature_names: collections.abc.Sequence[str], medians: collections.abc.Mapping[str, float], missing_features: collections.abc.Sequence[str] | None = None) -> tuple[numpy.ndarray, list[str]]
```

```python
load_detector(directory: pathlib.Path | str) -> decisions.anomaly_detector.TrainedAnomalyDetector
```

```python
save_detector(detector: decisions.anomaly_detector.TrainedAnomalyDetector, directory: pathlib.Path | str) -> None
```

```python
score_rows(detector: decisions.anomaly_detector.TrainedAnomalyDetector, rows: collections.abc.Sequence[collections.abc.Mapping[str, typing.Any]]) -> list[float]
```

```python
train_anomaly_detector(rows: collections.abc.Sequence[collections.abc.Mapping[str, typing.Any]], split: str, world_start_by_world: collections.abc.Mapping[str, datetime.datetime], config: decisions.anomaly_config.AnomalyDetectorConfig | None = None) -> decisions.anomaly_detector.TrainedAnomalyDetector
```

```python
TrainedAnomalyDetector(model: sklearn.ensemble._iforest.IsolationForest, feature_names: tuple[str, ...], column_names: tuple[str, ...], missing_features: tuple[str, ...], all_missing_features: tuple[str, ...], medians: dict[str, float], training_row_count: int, training_rows_sha256: str, model_sha256: str, sklearn_version: str) -> None
```
- dataclass

### `decisions/cold_start.py`

```python
choose_scorer(positive_label_count: int, approved_model_available: bool, config: decisions.anomaly_config.ColdStartConfig | None = None) -> str
```

```python
count_positive_labels(labels: collections.abc.Sequence[collections.abc.Mapping[str, typing.Any]], bank_id: str, as_of: datetime.datetime, config: decisions.anomaly_config.ColdStartConfig | None = None) -> int
```

```python
rank_for_review(candidates: collections.abc.Sequence[collections.abc.Mapping[str, typing.Any] | typing.Any], capacity: int) -> list[str]
```

```python
ReviewCandidate(account_id: str, rules_score: int | float, anomaly_score: float) -> None
```
- dataclass

### `evaluation/rules_metrics.py`

```python
analyst_load(daily_escalations: dict[datetime.date, list[str]], capacity: int) -> dict[str, float | int]
```

```python
genuine_harm(restricted_days: dict[str, set[datetime.date]], genuine_active: set[str], persona_of: collections.abc.Callable[[str], str], weights: dict[str, int], default_weight: int) -> dict[str, float | int | dict[str, str]]
```

```python
hidden_mule_recall_by_depth(flagged: set[str], depth_of: dict[str, int], reported: set[str]) -> dict[int, float]
```

```python
percentile_linear(sorted_values: list[float], fraction: float) -> float
```

```python
recall_at_capacity(daily_escalations: dict[datetime.date, list[str]], mule_accounts: set[str]) -> float
```

```python
summarise_across_worlds(values: list[float], min_worlds: int) -> dict[str, float | int | str]
```

```python
time_to_detection(first_illicit_credit: dict[str, datetime.datetime], first_flag: dict[str, datetime.datetime]) -> dict[str, float | int | None]
```

```python
yield_rate(escalations: list[tuple[str, datetime.date]], is_mule: collections.abc.Callable[[str, datetime.date], bool], inflation_factor: float) -> dict[str, float | None]
```

### `evaluation/level17_scoring.py`

```python
anomaly_restricted_days(scored: list[evaluation.level17_scoring.ScoredDay], threshold: float) -> dict[str, set[datetime.date]]
```

```python
daily_escalations(scored: list[evaluation.level17_scoring.ScoredDay], mode: str, anomaly_threshold: float | None, capacity_pct: float) -> dict[datetime.date, list[str]]
```

```python
feature_values_only(row: dict[str, typing.Any]) -> dict[str, typing.Any]
```

```python
genuine_restricted_days(restricted: dict[str, set[datetime.date]], scored: list[evaluation.level17_scoring.ScoredDay]) -> dict[str, set[datetime.date]]
```

```python
ground_sets(scored: list[evaluation.level17_scoring.ScoredDay]) -> tuple[set[str], set[str], dict[str, int], collections.abc.Callable[[str, datetime.date], bool]]
```

```python
one_row_per_account_day(rows: list[dict[str, typing.Any]]) -> list[dict[str, typing.Any]]
```

```python
rules_restricted_days(scored: list[evaluation.level17_scoring.ScoredDay]) -> dict[str, set[datetime.date]]
```

```python
score_account_days(day_rows: list[dict[str, typing.Any]], rules_config: decisions.rules_config.RulesConfig, detector: decisions.anomaly_detector.TrainedAnomalyDetector | None) -> list[evaluation.level17_scoring.ScoredDay]
```

```python
ScoredDay(bank_id: str, account_id: str, day: datetime.date, rules_score: int, rules_fired: tuple[str, ...], anomaly_score: float | None, role: str, depth: int | None) -> None
```
- dataclass

### `evaluation/rules_tuning.py`

```python
load_tuning_config(path: pathlib.Path | str | None = None) -> evaluation.rules_tuning.TuningConfig
```

```python
tune_anomaly_threshold(scored: list[evaluation.level17_scoring.ScoredDay], tuning: evaluation.rules_tuning.TuningConfig, split: str, budget_pct: float) -> evaluation.rules_tuning.AnomalyTuningResult
```

```python
tune_rules(rows: list[dict[str, typing.Any]], config: decisions.rules_config.RulesConfig, tuning: evaluation.rules_tuning.TuningConfig, split: str, budget_pct: float) -> evaluation.rules_tuning.RulesTuningResult
```

```python
AnomalyTuningResult(threshold: float, validation_rate_pct: float, validation_recall: float, meets_budget: bool) -> None
```
- dataclass

```python
RuleStandalone(reason_code: str, validation_rate_pct: float, validation_recall: float) -> None
```
- dataclass

```python
RulesTuningResult(thresholds: dict[str, int | float], validation_rate_pct: float, validation_recall: float, meets_budget: bool, combinations_evaluated: int, per_rule_standalone: tuple[evaluation.rules_tuning.RuleStandalone, ...]) -> None
```
- dataclass

```python
TuningConfig(candidates: dict[str, tuple[int | float, ...]], allowed_tuning_splits: tuple[str, ...], label: str, reason: str, derived_from: str) -> None
```
- dataclass

### `evaluation/level17_evaluation.py`

```python
evaluate_world(scored: list[evaluation.level17_scoring.ScoredDay], scorer: str, anomaly_threshold: float | None, capacity_pct: float, inflation_factor: float, persona_of: collections.abc.Callable[[str], str], weights: dict[str, int], default_weight: int, snapshot_time_by_account_day: collections.abc.Mapping[tuple[str, datetime.date], datetime.datetime], first_credit: collections.abc.Mapping[str, datetime.datetime], reported: set[str]) -> dict[str, typing.Any]
```

```python
first_flag_times(daily_escalations: collections.abc.Mapping[datetime.date, collections.abc.Sequence[str]], scored: collections.abc.Sequence[evaluation.level17_scoring.ScoredDay], snapshot_time_by_account_day: collections.abc.Mapping[tuple[str, datetime.date], datetime.datetime]) -> dict[str, datetime.datetime]
```

```python
first_illicit_credit(payment_rows: collections.abc.Sequence[collections.abc.Mapping[str, typing.Any]], story_of: collections.abc.Callable[[str], collections.abc.Mapping[str, typing.Any]]) -> dict[str, datetime.datetime]
```

```python
require_tuned(rules_config: decisions.rules_config.RulesConfig) -> None
```

```python
world_start_by_world(snapshot_rows: collections.abc.Sequence[collections.abc.Mapping[str, typing.Any]]) -> dict[str, datetime.datetime]
```

### `evaluation/run_level_17.py`

```python
build_result_manifest(dataset_entries: collections.abc.Sequence[collections.abc.Mapping[str, typing.Any]], truth_manifests: collections.abc.Mapping[str, str], detector: decisions.anomaly_detector.TrainedAnomalyDetector, anomaly_random_state: int, tuned: evaluation.run_level_17.TunedThresholds, settings: evaluation.run_level_17.EvaluationSettings) -> dict[str, typing.Any]
```

```python
evaluate_worlds(window: str, datasets: collections.abc.Sequence[tuple[pathlib.Path, replay.dataset_builder.Dataset]], truth_root: pathlib.Path, world_cards_root: pathlib.Path, tuned: evaluation.run_level_17.TunedThresholds, detector: decisions.anomaly_detector.TrainedAnomalyDetector, settings: evaluation.run_level_17.EvaluationSettings) -> dict[str, dict[str, typing.Any]]
```

```python
json_safe_result(per_world_result: collections.abc.Mapping[str, typing.Any]) -> dict[str, typing.Any]
```

```python
load_evaluation_settings(path: pathlib.Path | str | None = None) -> evaluation.run_level_17.EvaluationSettings
```

```python
load_split_datasets(split: str, dataset_dirs: collections.abc.Sequence[pathlib.Path]) -> list[replay.dataset_builder.Dataset]
```

```python
main(argv: collections.abc.Sequence[str] | None = None) -> int
```

```python
read_world_base_rate(world_cards_root: pathlib.Path, run_id: str) -> dict[str, typing.Any]
```

```python
reported_accounts(labels_rows: collections.abc.Sequence[collections.abc.Mapping[str, typing.Any]], run_id: str) -> set[str]
```

```python
run_evaluation(training_dir: pathlib.Path, validation_dir: pathlib.Path, test_dirs: collections.abc.Sequence[pathlib.Path], stress_dirs: collections.abc.Sequence[pathlib.Path], truth_root: pathlib.Path, world_cards_root: pathlib.Path, out_root: pathlib.Path = WindowsPath('C:/Users/sinne/Downloads/PROJECT-ashFALL/reports/level_17_evaluation'), evaluation_config_path: pathlib.Path | None = None) -> pathlib.Path
```

```python
run_key(dataset_ids: collections.abc.Sequence[str]) -> str
```

```python
snapshot_time_by_account_day(day_rows: collections.abc.Sequence[collections.abc.Mapping[str, typing.Any]]) -> dict[tuple[str, datetime.date], datetime.datetime]
```

```python
summarise_worlds(per_world: collections.abc.Mapping[str, collections.abc.Mapping[str, typing.Any]], min_worlds: int) -> dict[str, dict[str, dict[str, typing.Any]]]
```

```python
tune_on_validation(validation_dir: pathlib.Path, truth_root: pathlib.Path, base_rules_config: decisions.rules_config.RulesConfig, tuning: evaluation.rules_tuning.TuningConfig, detector: decisions.anomaly_detector.TrainedAnomalyDetector, budget_pct: float) -> evaluation.run_level_17.TunedThresholds
```

```python
write_result(out_root: pathlib.Path, key: str, manifest: collections.abc.Mapping[str, typing.Any], per_world: collections.abc.Mapping[str, typing.Any], across_worlds: collections.abc.Mapping[str, typing.Any]) -> pathlib.Path
```

```python
EvaluationSettings(genuine_restriction_budget_pct: float, capacity_pct_active_accounts_per_day: float, min_test_worlds: int, persona_weights: dict[str, int], default_persona_weight: int) -> None
```
- dataclass

```python
TunedThresholds(rules_config: decisions.rules_config.RulesConfig, rules_result: evaluation.rules_tuning.RulesTuningResult, anomaly_result: evaluation.rules_tuning.AnomalyTuningResult) -> None
```
- dataclass

---

## Records and Fields

Extracted directly via `dataclasses.fields`:

- `Rule`:
  - Fields: `id: str`, `reason_code: str`, `plain_english: str`, `conditions: tuple[decisions.rules_config.RuleCondition, ...]`
  - Path: `decisions/rules_config.py`

- `RuleCondition`:
  - Fields: `feature: str`, `op: str`, `threshold: str | int | float`, `threshold_key: str | None`, `threshold_value: int | float`
  - Path: `decisions/rules_config.py`

- `RuleThreshold`:
  - Fields: `key: str`, `value: int | float`, `label: str`, `reason: str`, `derived_from: str`, `status: str`
  - Path: `decisions/rules_config.py`

- `RulesConfig`:
  - Fields: `thresholds: dict[str, decisions.rules_config.RuleThreshold]`, `rules: tuple[decisions.rules_config.Rule, ...]`
  - Path: `decisions/rules_config.py`

- `RulesDecision`:
  - Fields: `fired: tuple[tuple[str, str], ...]`, `not_evaluable: tuple[str, ...]`, `score: int`
  - Path: `decisions/rules_engine.py`

- `AnomalyDetectorConfig`:
  - Fields: `excluded_features: tuple[str, ...]`, `excluded_features_reasons: dict[str, str]`, `early_window_days: int`, `n_estimators: int`, `max_samples: str | int | float`, `contamination: str | float`, `random_state: int`, `allowed_training_splits: tuple[str, ...]`
  - Path: `decisions/anomaly_config.py`

- `ColdStartConfig`:
  - Fields: `min_positive_labels_per_bank: int`, `counted_verdict: str`
  - Path: `decisions/anomaly_config.py`

- `TrainedAnomalyDetector`:
  - Fields: `model: IsolationForest`, `feature_names: tuple[str, ...]`, `column_names: tuple[str, ...]`, `missing_features: tuple[str, ...]`, `all_missing_features: tuple[str, ...]`, `medians: dict[str, float]`, `training_row_count: int`, `training_rows_sha256: str`, `model_sha256: str`, `sklearn_version: str`
  - Path: `decisions/anomaly_detector.py`

- `ReviewCandidate`:
  - Fields: `account_id: str`, `rules_score: int | float`, `anomaly_score: float`
  - Path: `decisions/cold_start.py`

- `ScoredDay`:
  - Fields: `bank_id: str`, `account_id: str`, `day: date`, `rules_score: int`, `rules_fired: tuple[str, ...]`, `anomaly_score: float | None`, `role: str`, `depth: int | None`
  - Path: `evaluation/level17_scoring.py`

- `AnomalyTuningResult`:
  - Fields: `threshold: float`, `validation_rate_pct: float`, `validation_recall: float`, `meets_budget: bool`
  - Path: `evaluation/rules_tuning.py`

- `RuleStandalone`:
  - Fields: `reason_code: str`, `validation_rate_pct: float`, `validation_recall: float`
  - Path: `evaluation/rules_tuning.py`

- `RulesTuningResult`:
  - Fields: `thresholds: dict[str, int | float]`, `validation_rate_pct: float`, `validation_recall: float`, `meets_budget: bool`, `combinations_evaluated: int`, `per_rule_standalone: tuple[evaluation.rules_tuning.RuleStandalone, ...]`
  - Path: `evaluation/rules_tuning.py`

- `TuningConfig`:
  - Fields: `candidates: dict[str, tuple[int | float, ...]]`, `allowed_tuning_splits: tuple[str, ...]`, `label: str`, `reason: str`, `derived_from: str`
  - Path: `evaluation/rules_tuning.py`

- `EvaluationSettings`:
  - Fields: `genuine_restriction_budget_pct: float`, `capacity_pct_active_accounts_per_day: float`, `min_test_worlds: int`, `persona_weights: dict[str, int]`, `default_persona_weight: int`
  - Path: `evaluation/run_level_17.py`

- `TunedThresholds`:
  - Fields: `rules_config: RulesConfig`, `rules_result: RulesTuningResult`, `anomaly_result: AnomalyTuningResult`
  - Path: `evaluation/run_level_17.py`

---

## Configuration Keys

### `config/rules_baseline.yaml`
- **thresholds**
  - `large_credit_amount_paise`
  - `outflow_share`
  - `time_window_seconds`
  - `new_account_age_days`
  - `payer_count`
  - `near_zero_balance_paise`
- **rules**

### `config/anomaly_detector.yaml`
- **excluded_features**
  - `label`
  - `reason`
  - `derived_from`
  - `value`
- **early_window_days**
  - `value`
  - `label`
  - `reason`
  - `derived_from`
- **n_estimators**
  - `value`
  - `label`
  - `reason`
  - `derived_from`
- **max_samples**
  - `value`
  - `label`
  - `reason`
  - `derived_from`
- **contamination**
  - `value`
  - `label`
  - `reason`
  - `derived_from`
- **random_state**
  - `value`
  - `label`
  - `reason`
  - `derived_from`
- **allowed_training_splits**
  - `value`
  - `label`
  - `reason`
  - `derived_from`

### `config/cold_start.yaml`
- **min_positive_labels_per_bank**
  - `value`
  - `label`
  - `reason`
  - `derived_from`
- **counted_verdict**
  - `value`
  - `label`
  - `reason`
  - `derived_from`

### `config/rules_tuning.yaml`
- **label**
- **reason**
- **derived_from**
- **allowed_tuning_splits**
- **thresholds**
  - `large_credit_amount_paise`
  - `outflow_share`
  - `time_window_seconds`
  - `new_account_age_days`
  - `payer_count`
  - `near_zero_balance_paise`

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

- `tie_break_order`:
  - `[outside_events, planned_payments, fraud_payments, resumed_payments, gang_device_events, opened_routine_payments, cash_withdrawals, gang_observations]`
