# Level 15 Handoff Note

The section "Generated facts" is the verbatim output of `py -3.12 scripts/handoff_facts.py --part 15c` (signatures from `inspect.signature`, fields from `dataclasses.fields`, keys read from the config files). Regenerate it instead of editing it. The rest is written by hand.

## Where things stand
Level 15 is complete across all parts (15a, 15b, 15c). It was merged by pull request from `feature/level-15-live-mule-features`.
- Part 15a (PR #84): live account state in Redis keyed by run, bank and account; an idempotent updater and a point-in-time reader (the Section 22 future-leak rule); payment sides streamed from the Kafka topic `upi.payments` and account events from Postgres; atomic trimming with a Lua script. Amendment A-037.
- Part 15b (PRs #85 to #91): drawn account opening dates; balances, withdrawals (the `bank_withdrawals` table) and home regions in the bank feed; 62-day long-lived summaries with closed-day reads; the canonical 52-feature M1 list; labels from `confirmed_fraud` feedback on the receiving account; network features over each bank's own payment graph; registry intelligence features where unknown stays distinct from no match; mock registry answers as of a past time. Amendment A-038.
- Part 15c (PRs #93 to #99): ingestion watermark; `feature_log` and `account_snapshots` (migrations 0010 to 0014); shadow decision logging behind `decision_feature_logging_enabled`; recompute from history; snapshots and a size estimate; the latency work (YAML cache, batched reads, shared per-decision reads, parallel registry screening with a derived timeout, one pooled registry connection, an honest timeout); a Docker Compose `benchmark` profile. Amendments A-034, A-035, A-036.

Judged latency (HTTP mock registry with the default delay, inside the Linux API container, by the method in `config/evaluation.yaml` `latency_method`): decision median p99 49.7 ms (limit 100), feature compute median p99 20.9 ms (limit 50), 0 registry timeouts of 9,000 questions. Showcase check 15c-3 in full mode judges this run.

Register counts: built 71, partial 13, planned 10, deferred 1 (total 95). Current amendments: A-034 to A-038.

## Decisions and why
- The bank's arrival cutoff is the minimum over its payment stream and its four relational pollers (`get_bank_watermark`), and is None if any source has no watermark, so a logged decision records exactly what it could see.
- Labels are driven by `confirmed_fraud` feedback only and label the receiving (credit-side) account (code cites architect decision 9b). Registry features keep unknown distinct from no match (decision 9c).
- A logged row is recomputed in arrival order from Postgres history with `LoggedAnswerScreener`, so a logged registry timeout replays as "unknown" exactly.
- There is no registry-answer cache in Level 15; the architect ruled that a cache belongs in Level 19.
- Latency scope (A-034): the registry wait counts only against the 100 ms whole-decision limit; the 50 ms limit covers our own feature code. Latency method (A-035): 100 warm-up calls, 3 runs of 1,000, median of the three p99 values, judged where the service runs (the Linux container). Both were written into `config/evaluation.yaml` before the judged measurement.
- The registry timeout is derived: `max_ms - feature_computation_max_ms - other_overhead_p99 - margin - clock_resolution` = 100 - 50 - 6.806 - 10 - 0.000001 = 33.19 ms (`decision_registry` in `config/account_features.yaml`). A call is "unknown" only if it really took longer than the timeout, measured with `perf_counter`; the wait allows two clock ticks of slack because asyncio timers can fire up to two ticks early on Windows.
- M1 uses no geo-velocity (section 12 of `docs/interfaces/account_state_contract.md`).
- The API starts one shared registry connection pool only when `decision_feature_logging_enabled` is on and a DPIP participant key is configured; without a key the registry features stay unknown.

## Known gaps and future work (open items)
1. Timeout formula input: the overhead was measured with the fake registry (6.8 ms, giving 33.19 ms). In the judged HTTP setup the overhead is 15.8 to 17.8 ms, which the same formula turns into about 22 ms. The architect decides which input is right.
2. A recurring 14 to 16 ms stall shows up in the feature_log write or in "other" among the slowest calls. Level 19 should look at it.
3. The native Windows harness is not the verdict (its monotonic clock ticks every 15.6 ms). A Python 3.13 upgrade is a later, separate round.
4. The compose registry lists an entry 15 minutes after receiving it, so the benchmark's seeded "matches" read `no_match` in the container run. Timing is unaffected.
5. 15-6 is partial: Level 18 (leakage checklist for label-based features, run automatically), Level 22 (feedback-loop handling, because analyst labels exist only for cases ashFALL queued) and Level 28 (bias measured).
6. G2 is partial: add the "exchange" merchant category before Level 23, so crypto cash-out is not a person-to-person transfer and the tracer treats it as unrecoverable.

## Verification commands
- `py -3.12 -m showcase --level 15`: all Level 15 checks; 15c-3 in full mode runs the latency method in the Linux container and prints the native Windows run as "not judged".
- `py -3.12 -m benchmarks.decision_latency --where container --registry http --registry-delay default`: the judged latency run (needs the compose stack; `make bench-latency-container REGISTRY=http DELAY=default`).
- `py -3.12 -m core.issue_register_v2`: register state (built 71, partial 13, planned 10, deferred 1).
- `py -3.12 -m pytest tests/unit/test_issue_register_v2.py -q -p no:cacheprovider`: register integrity and the pinned counts.
- `py -3.12 -m pytest tests/unit/test_registry_prefetch.py tests/unit/test_decision_latency.py -q -p no:cacheprovider`: registry prefetch, timeout and latency method.
- `py -3.12 -B scripts/verify_world.py`: golden checks.
- `py -3.12 scripts/handoff_facts.py --part 15c`: regenerates the facts below. (`--check` looks for `level_15c.md`, so it does not apply to this file name.)

---

## Generated facts

## Register Changes
- 15-1: Velocity already uses event time; a 7-day window needs its own expiry (status: built)
- 15-2: The region feature broke bank visibility (status: built)
- 15-3: Recompute by event time would flap on late events (status: built)
- 15-5: Routine life needed purpose labels the bank cannot see (status: built)
- 15-6: Label-based network features risk leakage and feedback loops (status: partial)
- 15-7: Global event order; feature_log size (status: built)
- 15-8: Global event order; feature_log size (status: built)
- 15-9: Geo kernel fit; M1 versioning; device scope; world-computed novelty (status: built)
- 15-10: Geo kernel fit; M1 versioning; device scope; world-computed novelty (status: built)
- 15-11: Geo kernel fit; M1 versioning; device scope; world-computed novelty (status: built)
- 15-12: Geo kernel fit; M1 versioning; device scope; world-computed novelty (status: built)
- G1: Mule origin; rented mules keep normal life (status: built)
- G2: Cash-out channels including ATM limits and genuine shops (status: partial)
- R7: Failed payments in the transaction feed (status: built)
- R8: Device and routine-life feature fixes (status: built)
- E3: Regions (status: built)
- R5: OS updates do not trigger new-device rules (status: built)

---

## Public Interfaces

Extracted directly via `inspect.signature`:

### `features/ingest_watermark.py`

```python
get_bank_watermark(client: Any, bank_id: str) -> datetime.datetime | None
```

```python
get_payment_watermark(client: Any, partition: int) -> datetime.datetime | None
```

```python
get_poller_watermark(client: Any, bank_id: str, source: str) -> datetime.datetime | None
```

```python
record_payment_watermark(client: Any, partition: int, timestamp: datetime.datetime | int | float) -> None
```

```python
record_poller_watermark(client: Any, bank_id: str, source: str, timestamp: datetime.datetime | int | float) -> None
```

```python
to_watermark_seconds(ts: datetime.datetime | int | float) -> int
```

### `features/feature_log.py`

```python
insert_feature_log_row(conn: Any, row: features.feature_log.FeatureLogRow) -> None
```

```python
read_feature_log_row(conn: Any, bank_id: str, payment_id: str, decision_time: datetime.datetime | None = None) -> features.feature_log.FeatureLogRow | None
```

```python
to_feature_values_array(features_dict: dict[str, typing.Any]) -> list[typing.Any]
```

```python
to_ordered_values(features_dict: dict[str, typing.Any]) -> list[typing.Any]
```

```python
FeatureLogRow(bank_id: str, payment_id: str, run_id: str, account_id: str, decision_time: datetime.datetime, decided_at: datetime.datetime, arrival_cutoff: datetime.datetime | None, watermark_lag_ms: int | None, status: str, feature_values: list[typing.Any] | dict[str, typing.Any], feature_set_version: str = 'm1', feature_names_sha256: str = '', error_type: str | None = None, error_message: str | None = None, missing_request_fields: list[str] = <factory>, registry_answers: list[dict[str, typing.Any]] = <factory>, created_at: datetime.datetime | None = None) -> None
```
- dataclass

### `features/decision_features.py`

```python
compute_decision_row(redis_client: Any, bank_id: str, request: api.v1.models.DecisionRequest, clock: core.event_clock.Clock, settings: features.account_keys.AccountFeaturesSettings | None = None, screener: features.registry_screening.RegistryScreener | None = None, dpip_key: str | None = None, arrival_cutoff: datetime.datetime | None = None) -> features.feature_log.FeatureLogRow
```

```python
compute_feature_values(redis_client: Any, bank_id: str, run_id: str, request: api.v1.models.DecisionRequest, decision_time: datetime.datetime, arrival_cutoff: datetime.datetime, screener: features.registry_screening.RegistryScreener | None = None, dpip_key: str | None = None, settings: features.account_keys.AccountFeaturesSettings | None = None) -> features.decision_features.FeatureComputation
```

```python
FeatureComputation(feature_values: dict[str, typing.Any], status: str, missing_request_fields: list[str], registry_answers: list[dict[str, typing.Any]]) -> None
```
- dataclass

### `features/feature_replay.py`

```python
recompute_logged_row(pg_connection: Any, scratch_redis: Any, bank_id: str, payment_id: str, dpip_key: str | None = None, settings: features.account_keys.AccountFeaturesSettings | None = None) -> features.feature_replay.RecomputeResult
```

```python
replay_history_into_redis(pg_connection: Any, scratch_redis: Any, run_id: str, bank_id: str, arrival_cutoff: datetime.datetime, settings: features.account_keys.AccountFeaturesSettings | None = None) -> features.feature_replay.ReplayCounts
```

```python
RecomputeResult(logged: list[typing.Any], recomputed: list[typing.Any], mismatches: list[tuple[str, typing.Any, typing.Any]]) -> None
```
- dataclass

```python
ReplayCounts(transactions: int, account_events: int, withdrawals: int, feedback: int, complaints: int) -> None
```
- dataclass

### `features/account_snapshots.py`

```python
full_snapshot_due(day_number: int, settings: features.account_snapshots.SnapshotSettings) -> bool
```

```python
load_snapshot_settings(config_path: pathlib.Path | None = None) -> features.account_snapshots.SnapshotSettings
```

```python
select_all_accounts(pg: Any, run_id: str, bank_id: str, arrival_cutoff: datetime.datetime) -> list[str]
```

```python
select_daily_snapshot_accounts(pg: Any, run_id: str, bank_id: str, day_start: datetime.datetime, day_end: datetime.datetime) -> list[str]
```

```python
to_account_ordered_values(features_dict: dict[str, typing.Any]) -> list[typing.Any]
```

```python
write_account_snapshots(pg: Any, redis: Any, run_id: str, bank_id: str, snapshot_time: datetime.datetime, snapshot_kind: str, account_ids: collections.abc.Sequence[str], settings: features.account_keys.AccountFeaturesSettings | None = None) -> int
```

```python
SnapshotSettings(full_interval_days: int) -> None
```
- dataclass

### `features/registry_prefetch.py`

```python
clock_resolution_seconds() -> float
```

```python
compute_registry_timeout_ms(max_ms: float, feature_computation_max_ms: float, other_overhead_p99_ms: float, margin_ms: float, clock_resolution_ms: float, minimum_ms: float) -> float
```

```python
create_registry_gateway() -> features.registry_prefetch.RegistryGateway
```

```python
list_registry_questions(client: Any, run_id: str, bank_id: str, account_id: str, payee_bank_id: str | None, payee_vpa: str | None, decision_time: datetime.datetime, arrival_cutoff: datetime.datetime, dpip_key: str | None = None, settings: features.account_keys.AccountFeaturesSettings | None = None) -> list[tuple[str, str]]
```

```python
load_registry_timeout_seconds(config_path: str | pathlib.Path = 'config/account_features.yaml') -> float
```

```python
prefetch_registry_answers(client: features.registry_prefetch.AsyncScreenClient, questions: list[tuple[str, str]], timeout_seconds: float) -> dict[tuple[str, str], features.registry_screening.ScreenOutcome]
```

```python
prepare_registry_questions(redis_client: Any, bank_id: str, request: api.v1.models.DecisionRequest, dpip_key: str | None, settings: features.account_keys.AccountFeaturesSettings | None = None) -> tuple[datetime.datetime, list[tuple[str, str]]] | None
```

```python
run_registry_prefetch(client: features.registry_prefetch.AsyncScreenClient, questions: list[tuple[str, str]], timeout_seconds: float) -> dict[tuple[str, str], features.registry_screening.ScreenOutcome]
```

### `features/decision_reads.py`

```python
read_decision_reads(client: Any, run_id: str, bank_id: str, account_id: str, decision_time: datetime.datetime, arrival_cutoff: datetime.datetime, settings: features.account_keys.AccountFeaturesSettings | None = None) -> features.decision_reads.DecisionReads
```

```python
DecisionReads(run_id: str, bank_id: str, account_id: str, decision_time: datetime.datetime, arrival_cutoff: datetime.datetime, payer_snapshot: features.account_reader.AccountSnapshot, labelled_visible: frozenset[str], complained_visible: frozenset[str]) -> None
```
- dataclass

---

## Records and Fields

Extracted directly via `dataclasses.fields`:

- `FeatureLogRow`:
  - Fields: `bank_id: str`, `payment_id: str`, `run_id: str`, `account_id: str`, `decision_time: datetime`, `decided_at: datetime`, `arrival_cutoff: datetime.datetime | None`, `watermark_lag_ms: int | None`, `status: str`, `feature_values: list[Any] | dict[str, Any]`, `feature_set_version: str`, `feature_names_sha256: str`, `error_type: str | None`, `error_message: str | None`, `missing_request_fields: list[str]`, `registry_answers: list[dict[str, Any]]`, `created_at: datetime.datetime | None`
  - Path: `features/feature_log.py`

- `FeatureComputation`:
  - Fields: `feature_values: dict[str, Any]`, `status: str`, `missing_request_fields: list[str]`, `registry_answers: list[dict[str, Any]]`
  - Path: `features/decision_features.py`

- `RecomputeResult`:
  - Fields: `logged: list[Any]`, `recomputed: list[Any]`, `mismatches: list[tuple[str, Any, Any]]`
  - Path: `features/feature_replay.py`

- `ReplayCounts`:
  - Fields: `transactions: int`, `account_events: int`, `withdrawals: int`, `feedback: int`, `complaints: int`
  - Path: `features/feature_replay.py`

- `SnapshotSettings`:
  - Fields: `full_interval_days: int`
  - Path: `features/account_snapshots.py`

- `DecisionReads`:
  - Fields: `run_id: str`, `bank_id: str`, `account_id: str`, `decision_time: datetime`, `arrival_cutoff: datetime`, `payer_snapshot: AccountSnapshot`, `labelled_visible: frozenset[str]`, `complained_visible: frozenset[str]`
  - Path: `features/decision_reads.py`

---

## Configuration Keys

### `config/account_features.yaml`
- **account_features**
  - `windows_seconds`
  - `retention_seconds`
  - `allowed_lateness_seconds`
  - `key_expiry_margin_seconds`
  - `key_prefix`
  - `run_id_source`
  - `account_event_batch_size`
  - `account_event_overlap_rows`
  - `summary_horizon_days`
  - `routine_life_warm_up_days`
  - `fingerprint_warning_count`
  - `small_credit_max_paise`
  - `reporting_thresholds_paise`
  - `below_threshold_band_fraction`
  - `near_zero_balance_max_paise`
  - `routine_cycle_days`
  - `routine_rhythm_tolerance_days`
  - `routine_min_history_days`
  - `machine_timing_min_payments`
  - `device_change_window_seconds`
  - `network_max_hops`
  - `network_max_nodes_visited`
  - `commit_grace`
- **account_snapshots**
  - `full_interval_days`
- **decision_registry**
  - `timeout_margin_ms`
  - `timeout_minimum_ms`
  - `timeout_ms`

### `config/evaluation.yaml`
- **meta**
  - `level`
  - `author`
  - `status`
- **capacity**
  - `label`
  - `sweep_min_pct_active_accounts_per_day`
  - `sweep_max_pct_active_accounts_per_day`
  - `headline_pct_active_accounts_per_day`
  - `report_full_curve`
  - `reason`
- **fair_comparison**
  - `label`
  - `method`
  - `tuned_on`
  - `main_genuine_restriction_rate_pct`
  - `extra_genuine_restriction_rate_pct`
  - `report_full_tradeoff_curve`
  - `reason`
- **ml_beats_rules**
  - `label`
  - `all_conditions_must_hold`
  - `min_median_recall_difference_points`
  - `min_share_of_test_worlds_ml_wins_pct`
  - `confidence_level_pct`
  - `confidence_interval_method`
  - `interval_must_exclude_zero`
  - `reason`
  - `min_test_worlds`
- **latency**
  - `label`
  - `endpoint`
  - `percentile`
  - `max_ms`
  - `measured`
  - `world_for_payment_rate`
  - `warm_up_required`
  - `hardware`
  - `feature_computation_max_ms`
  - `reason`
- **latency_scope**
  - `label`
  - `registry_wait_counts_against`
  - `feature_computation_covers`
  - `fit_check`
  - `reason`
  - `derived_from`
- **latency_method**
  - `label`
  - `warm_up_calls`
  - `measured_calls`
  - `runs`
  - `judged_on`
  - `slowest_calls_reported`
  - `reason`
  - `derived_from`
- **money_held_and_returned**
  - `label`
  - `issue_ids`
  - `held_means`
  - `returned_means`
  - `always_report_both`
  - `reason`
- **base_rate_reweighting**
  - `label`
  - `issue_ids`
  - `applies_to`
  - `report`
  - `inflation_factor_comes_from`
  - `real_world_ratio_comes_from`
  - `reason`
- **genuine_harm**
  - `label`
  - `issue_ids`
  - `counts`
  - `reason`
- **held_out_tactics**
  - `label`
  - `used_only_in`
  - `tactics`
  - `deferred`
  - `kept_in_training`
  - `reason`
- **world_splits**
- **trivial_world_check**
  - `label`
  - `issue_ids_v2`
  - `derived_from`
  - `metric`
  - `fail_threshold`
  - `report_top_features`
  - `report_alongside`
  - `gate_on`
  - `report_only_on`
  - `report_held_out_per_family`
  - `routine_min_gap_days`
  - `reason`
- **validation_worlds**
  - `label`
  - `issue_ids_v2`
  - `derived_from`
  - `ranges`
  - `seeds`
  - `used_for`
  - `counts`
  - `reason`
- **earlier_work_note**
- **not_decided_here**

### `config/nationalsim.yaml`
- **nationalsim**
  - `label`
  - `reason`
  - `issue_ids_v2`
  - `derived_from`
  - `listing_delay_minutes`
  - `recycle_min_days`
  - `recycled_number_share`
  - `dispute_resolution_minutes`
  - `night_hours`
  - `night_multiplier`
  - `weekend_multiplier`
  - `banks`
  - `registry_base_url`
  - `client_max_connections`
  - `client_max_keepalive_connections`
  - `screen_added_delay_ms`
  - `screen_added_delay_ms_registry_slow`
  - `client_timeout_seconds`
  - `client_retry_backoff_seconds`

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

- `tie_break_order`:
  - `[outside_events, planned_payments, fraud_payments, resumed_payments, gang_device_events, opened_routine_payments, cash_withdrawals, gang_observations]`
