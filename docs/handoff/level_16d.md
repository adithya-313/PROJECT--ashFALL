# Level 16 Handoff Note

The section "Generated facts" is the verbatim output of `py -3.12 scripts/handoff_facts.py --part 16d` (signatures from `inspect.signature`, fields from `dataclasses.fields`, keys read from the config files). Regenerate it instead of editing it. The rest is written by hand.

## Where things stand
Level 16 covers historical simulation replay, four-path label ingestion, and point-in-time dataset construction across parts 16a, 16b, 16c, and 16d.
- Part 16a (PRs #103 to #108): ingestion clock and labels table; run-scoped watermarks; arrival times on payments, events, and withdrawals.
- Part 16b (PRs #109 to #114): replay workspaces with PostgreSQL schema isolation, Redis DB isolation, and single-partition Kafka topic; replay wiring and standalone decision request builder; world-time step ordering; determinism hash (`feature_log` SHA-256) and history recompute.
- Part 16c (PRs #115 to #120): per-event decision scoring (`decisions_per_event_payment=True`) and receiver VPA inclusion; labels and daily account snapshots in replay; idle-partition payment watermark advance; dataset builder producing PyArrow tables and Parquet files with cryptographic manifests; evaluation truth join.
- Part 16d (PRs #121 to #124 and current): phone visibility (KYC phone on account opening events); registry seed and world reports in replay through an in-process registry; in-process nationalsim screener and seeded screen delay; police bulk labels and hold request labels at receiving banks; committed Level 16 showcase world (seed 65) with 12 checks; dataset rebuild across splits with superseded marking.

Showcase: 12/12 checks passing (`python -m showcase --level 16`).
Register state: 104 issues checked (built 72, partial 13, planned 17, deferred 2).

## Decisions and why
- Sizing: Preset `tiny` (60 people, 12 shops, 61 days) used for multi-world split datasets; full scale remains Issue 16-4 for Level 18a.
- Single dataset per split: Extracts across constituent worlds are combined into one unified dataset with provenance preserved by `world_id`.
- Simulation horizon cutoff: Cutoff timestamp evaluated at midnight IST of clock start date plus simulation duration `n_days` (`midnight + timedelta(days=n_days)`), capturing all simulated events without lookahead bias.
- Unpinned gangs in build: Sampler resolves gang mixes dynamically per split.
- Direct complaint separation: Check 16-7 separates direct complaints (same-bank, via None) from cross-bank hold requests (via 'hold_request').
- Superseded datasets: Rebuilt datasets mark previous manifests of the same split with `superseded_by`; `load_dataset` and `join_truth` fail closed unless `allow_superseded=True` is explicitly passed.

## Level 16 Switches

Generated dynamically from `config/replay.yaml` via `python scripts/list_level_16_switches.py`:

| Switch Name | Configuration File | Default Live | Replay Value | What OFF Reproduces |
| :--- | :--- | :--- | :--- | :--- |
| `decisions_per_event_payment` | `config/replay.yaml` | `false` | `true` | Level 16b per-batch payment attempt scoring (decisions evaluated per attempt) |
| `idle_partition_advance` | `config/replay.yaml` | `false` | `true` | Level 16c-2 quiet payment partition watermark stalling |
| `kyc_phone_on_account_opened` | `config/replay.yaml` | `false` | `true` | Level 16c feeds without KYC phone payload on account_opened events |
| `world_registry_reporting` | `config/replay.yaml` | `false` | `true` | Level 16c replays without in-process DPIP registry labels |
| `replay_speed` | `config/replay.yaml` | `1.0` | `"max"` | Real-time paced replay execution (with inter-step sleep delays) |

## Known gaps and future work (open items)
1. 16-1: Live-world decisions per attempt and held-payment attempt minute deferred to Level 19.
2. 16-3: Decision API request has no device field (deferred to Level 19).
3. 16-4: Full-scale dataset building and a faster replay drain planned for Level 18a.
4. 16-5: Idle-partition watermark advance for the live payment consumer planned for Level 19.
5. 16-6: Phone changes during a run (SIM swap, linked-number switch) needed before Level 18a.
6. 16-7: World generates revoke/dispute for wrong listings planned for Level 21.
7. 16-8: Registry coverage by mule depth planned for Level 18a.
8. 16-9: Pre-world registry seed share calibration planned for Level 28.

## Verification commands
- `py -3.12 -m showcase --level 16`: all 12 Level 16 checks.
- `py -3.12 -B scripts/verify_world.py`: golden checks.
- `py -3.12 -m core.issue_register_v2`: issue register verification.
- `py -3.12 -m core.folder_check`: folder classification check.
- `py -3.12 -m core.truth_import_check`: truth isolation check.
- `py -3.12 scripts/handoff_facts.py --part 16d --check`: validates handoff note structure.
- Slow replay integration tests:
  - `tests/integration/test_dataset_end_to_end.py`
  - `tests/integration/test_replay_runner.py`
  - `tests/integration/test_replay_determinism.py`
  - `tests/integration/test_replay_labels_snapshots.py`

---

## Generated facts

## Register Changes
- 16-1: Live-world decisions per attempt and held-payment attempt minute (status: deferred)
- 16-2: Registry wired into replay (payee_registry_match always None in replay) (status: built)
- 16-3: Decision API request has no device field (Section 5 lists it) (status: planned)
- 16-4: Full-scale dataset building and a faster replay drain (status: planned)
- 16-5: Idle-partition watermark advance for the live payment consumer (status: planned)
- 16-6: Phone changes during a run (SIM swap, linked-number switch) are not modelled; needed before Level 18a (status: planned)
- 16-7: World generates revoke/dispute for wrong listings (status: planned)
- 16-8: Registry coverage by mule depth (N2) not implemented; one uniform reporting share (status: planned)
- 16-9: Pre-world registry seed share and uniform reporter are assumptions; calibrate (status: planned)

---

## Public Interfaces

Extracted directly via `inspect.signature`:

### `replay/runner.py`

```python
check_workspace_topic_partitions(consumer: aiokafka.consumer.consumer.AIOKafkaConsumer, topic: str) -> None
```

```python
compute_feature_log_sha256(rows: collections.abc.Sequence[asyncpg.protocol.record.Record]) -> str
```

```python
day_boundaries_between(prev_world_time: datetime.datetime, next_world_time: datetime.datetime) -> list[datetime.datetime]
```

```python
drain(step_index: int, workspace_conn: asyncpg.connection.Connection, producer: Any, consumer: aiokafka.consumer.consumer.AIOKafkaConsumer, workspace: world.replay_workspace.ReplayWorkspace, redis_client: Any, acct_settings: features.account_keys.AccountFeaturesSettings, banks: list[str], world_clock: core.event_clock.ManualClock, run_id: str, poller_cursors: dict[str, dict[str, int]], deadline: float, batch_size: int = 100, registry: replay.runner.RegistryDrain | None = None) -> float
```

```python
drain_registry(registry: replay.runner.RegistryDrain, conn: Any, timeout_ms: int = 20) -> bool
```

```python
idle_advance_target(clock_now: datetime.datetime, commit_grace_seconds: int | float, partition_drained: bool, next_unsent_arrival: datetime.datetime | None) -> tuple[datetime.datetime | None, str]
```

```python
load_default_decisions_per_event_payment() -> bool
```

```python
load_default_idle_partition_advance() -> bool
```

```python
load_default_kyc_phone_on_account_opened() -> bool
```

```python
load_default_replay_speed() -> str | float
```

```python
load_default_world_registry_reporting() -> bool
```

```python
pace_seconds(previous_world_time: datetime.datetime | None, next_world_time: datetime.datetime, speed: str | float) -> float
```

```python
pace_step(previous_world_time: datetime.datetime | None, next_world_time: datetime.datetime, speed: str | float, sleep: collections.abc.Callable[[float], collections.abc.Awaitable[None]] = <function sleep at 0x000001E4A35D9DA0>) -> float
```

```python
run_replay(recorded_world: world.replay_recording.RecordedWorld, feed: world.bank_feed.BankFeed, admin_pg: str, *, decisions_per_event_payment: bool | None = None, replay_speed: str | float | None = None, sleep: collections.abc.Callable[[float], collections.abc.Awaitable[None]] = <function sleep at 0x000001E4A35D9DA0>, replay_settings: world.replay_workspace.ReplaySettings | None = None, account_settings: features.account_keys.AccountFeaturesSettings | None = None, live_redis: Any = None, base_redis_url: str | None = None, redis_client_factory: collections.abc.Callable[[world.replay_workspace.ReplayWorkspace], typing.Any] | None = None, before_decision_hook: collections.abc.Callable[[replay.runner.ReplayHookContext], collections.abc.Awaitable[None]] | None = None, before_drop_hook: collections.abc.Callable[[replay.runner.ReplayDropContext], collections.abc.Awaitable[None]] | None = None, workspace_name: str | None = None, idle_partition_advance: bool | None = None, world_registry_reporting: bool | None = None, registry_settings: replay.registry_replay.RegistryReplaySettings | None = None) -> replay.runner.ReplayResult
```

```python
validate_replay_speed(speed: Any) -> None
```

```python
RegistryDrain(consumer: aiokafka.consumer.consumer.AIOKafkaConsumer, topic: str, handler: collections.abc.Callable[[list[typing.Any], typing.Any], collections.abc.Awaitable[None]] | None = None) -> None
```
- dataclass

```python
ReplayDropContext(workspace: world.replay_workspace.ReplayWorkspace, workspace_dsn: str, logged_payment_ids_by_bank: dict[str, list[str]], dpip_key: str, registry_wired: bool = False, registry_wired_reason: str = 'no nationalsim screener or DPIP key in replay until 16d') -> None
```
- dataclass

```python
ReplayHookContext(step_index: int, decision_step: replay.ordering.DecisionStep, consumer: aiokafka.consumer.consumer.AIOKafkaConsumer, workspace: world.replay_workspace.ReplayWorkspace) -> None
```
- dataclass

```python
ReplayResult(decisions_sent: int, feature_rows_by_status: dict[str, int], decisions_arrival_cutoff_behind: int, max_arrival_cutoff_gap_seconds: float, longest_drain_seconds: float, decisions_before_first_payment: int, unwatermarked_payment_ids: tuple[str, ...] = (), logged_features_by_payment_id: dict[str, dict[str, typing.Any]] = <factory>, missing_request_fields_by_payment_id: dict[str, list[str]] = <factory>, registry_wired: bool = False, registry_wired_reason: str = 'no nationalsim screener or DPIP key in replay until 16d', registry_screens: int = 0, registry_screens_unknown: int = 0, feature_log_sha256: str = '') -> None
```
- dataclass

### `replay/ordering.py`

```python
build_replay_steps(recorded_world: world.replay_recording.RecordedWorld | typing.Any, feed: world.bank_feed.BankFeed, decisions_per_event_payment: bool = False, registry_steps: collections.abc.Sequence[replay.ordering.FeedStep] = ()) -> list[replay.ordering.FeedStep | replay.ordering.DecisionStep]
```

```python
check_no_resumed_events(output: Any) -> None
```

```python
DecisionStep(world_time: datetime.datetime, planned_payment_id: str, call: world.bank_connectors.DecisionCall) -> None
```
- dataclass

```python
FeedStep(world_time: datetime.datetime, kind: str, bank_id: str, record_id: str, record: Any) -> None
```
- dataclass

### `replay/dataset_builder.py`

```python
build_dataset(extracts: collections.abc.Sequence[replay.dataset_builder.WorldExtract], split: str, cutoff: datetime.datetime, code_version: str | None = None) -> replay.dataset_builder.Dataset
```

```python
content_hash(dataset: replay.dataset_builder.Dataset) -> str
```

```python
extract_world(ctx: replay.runner.ReplayDropContext, recorded_world: world.replay_recording.RecordedWorld, replay_flags: dict[str, typing.Any]) -> replay.dataset_builder.WorldExtract
```

```python
load_dataset(dataset_dir: pathlib.Path, allow_superseded: bool = False) -> replay.dataset_builder.Dataset
```

```python
mark_superseded(old_dir: pathlib.Path, new_dir: pathlib.Path) -> None
```

```python
write_dataset(dataset: replay.dataset_builder.Dataset, root: pathlib.Path = WindowsPath('data/datasets')) -> pathlib.Path
```

```python
Dataset(payments: pyarrow.lib.Table, snapshots: pyarrow.lib.Table, labels: pyarrow.lib.Table, manifest: dict[str, typing.Any]) -> None
```
- dataclass

```python
WorldExtract(world_id: str, split: str, seed: int, run_id: str, gang_families: tuple[str, ...], decisions_per_event_payment: bool, registry_wired: bool, registry_wired_reason: str, payment_rows: list[dict[str, typing.Any]], snapshot_rows: list[dict[str, typing.Any]], label_rows: list[dict[str, typing.Any]], clock_start: str = '2026-09-01', label_status_rows: list[dict[str, typing.Any]] = <factory>) -> None
```
- dataclass

### `replay/registry_driver.py`

```python
build_registry_driver(recorded_world: world.replay_recording.RecordedWorld, clock: core.event_clock.ManualClock, settings: replay.registry_replay.RegistryReplaySettings | None = None) -> replay.registry_driver.RegistryDriver
```

```python
registry_event_payload(item: nationalsim.adapter.IngestionItem, entry: nationalsim.registry.RegistryEntry) -> dict[str, typing.Any]
```

```python
RegistryDriver(core: nationalsim.registry.RegistryCore, key: str, seed: list[replay.registry_replay.RegistryListing], world: list[replay.registry_replay.RegistryListing], world_start: datetime.date, bulk: list[replay.registry_replay.BulkEvent] = <factory>, holds: list[replay.hold_requests.HoldEvent] = <factory>, hold_missed: int = 0, run_id: str = '', clock: Any = None, screen_rng: Any = None, screen_delay_ms: tuple[float, float] = (2.0, 15.0), screen_timeout_ms: float = 33.19, stats: replay.registry_screener.ScreenStats = <factory>) -> None
```
- dataclass

### `replay/registry_replay.py`

```python
load_registry_replay_settings(split: str, path: pathlib.Path | None = None) -> replay.registry_replay.RegistryReplaySettings
```

```python
plan_registry_seed(phone_book: world.phone_book.PhoneBook, settings: replay.registry_replay.RegistryReplaySettings, rng: numpy.random._generator.Generator, run_id: str, key: str, world_start: datetime.date) -> list[replay.registry_replay.RegistryListing]
```

```python
replay_dpip_key(run_id: str) -> str
```

```python
visible_world_time(ingest_time: datetime.datetime, listing_delay: datetime.timedelta, world_start: datetime.date) -> datetime.datetime
```

```python
BulkEvent(item: nationalsim.adapter.IngestionItem, visible_time: datetime.datetime, batch_id: str, region: int) -> None
```
- dataclass

```python
RegistryListing(item: nationalsim.adapter.IngestionItem, visible_time: datetime.datetime, seed: bool) -> None
```
- dataclass

```python
RegistryReplaySettings(listing_delay: datetime.timedelta, dispute_resolution_delay: datetime.timedelta, bank_settings: dict[str, dict[str, typing.Any]], seed_share_range: tuple[float, float], reporter_choice: str, screen_delay_ms: tuple[float, float] = (2.0, 15.0)) -> None
```
- dataclass

### `replay/hold_requests.py`

```python
plan_hold_requests(complaints: collections.abc.Sequence[world.label_records.LabelEvent], world_ranges: dict[str, typing.Any], split: str, hold_cfg: dict[str, typing.Any], bank_factors: collections.abc.Sequence[float], rng: numpy.random._generator.Generator, run_id: str, key: str, world_start: datetime.date) -> tuple[list[replay.hold_requests.HoldEvent], int]
```

```python
HoldEvent(item: nationalsim.adapter.IngestionItem, visible_time: datetime.datetime, complaint_id: str, label_id: str, payment_ids: tuple[str, ...], latency_minutes: int, claimed_type: str | None) -> None
```
- dataclass

### `evaluation/dataset_truth_join.py`

```python
join_truth(dataset_dir: pathlib.Path, truth_root: pathlib.Path, allow_superseded: bool = False) -> tuple[pyarrow.lib.Table, pyarrow.lib.Table]
```

```python
roles_at(dataset_dir: pathlib.Path, truth_root: pathlib.Path, queries: list[tuple[str, str, datetime.datetime]], allow_superseded: bool = False) -> list[tuple[str, str | None]]
```

---

## Records and Fields

Extracted directly via `dataclasses.fields`:

- `RegistryDrain`:
  - Fields: `consumer: AIOKafkaConsumer`, `topic: str`, `handler: collections.abc.Callable[[list[Any], Any], collections.abc.Awaitable[None]] | None`
  - Path: `replay/runner.py`

- `ReplayDropContext`:
  - Fields: `workspace: ReplayWorkspace`, `workspace_dsn: str`, `logged_payment_ids_by_bank: dict[str, list[str]]`, `dpip_key: str`, `registry_wired: bool`, `registry_wired_reason: str`
  - Path: `replay/runner.py`

- `ReplayHookContext`:
  - Fields: `step_index: int`, `decision_step: DecisionStep`, `consumer: AIOKafkaConsumer`, `workspace: ReplayWorkspace`
  - Path: `replay/runner.py`

- `ReplayResult`:
  - Fields: `decisions_sent: int`, `feature_rows_by_status: dict[str, int]`, `decisions_arrival_cutoff_behind: int`, `max_arrival_cutoff_gap_seconds: float`, `longest_drain_seconds: float`, `decisions_before_first_payment: int`, `unwatermarked_payment_ids: tuple[str, ...]`, `logged_features_by_payment_id: dict[str, dict[str, Any]]`, `missing_request_fields_by_payment_id: dict[str, list[str]]`, `registry_wired: bool`, `registry_wired_reason: str`, `registry_screens: int`, `registry_screens_unknown: int`, `feature_log_sha256: str`
  - Path: `replay/runner.py`

- `DecisionStep`:
  - Fields: `world_time: datetime`, `planned_payment_id: str`, `call: DecisionCall`
  - Path: `replay/ordering.py`

- `FeedStep`:
  - Fields: `world_time: datetime`, `kind: str`, `bank_id: str`, `record_id: str`, `record: Any`
  - Path: `replay/ordering.py`

- `Dataset`:
  - Fields: `payments: Table`, `snapshots: Table`, `labels: Table`, `manifest: dict[str, Any]`
  - Path: `replay/dataset_builder.py`

- `WorldExtract`:
  - Fields: `world_id: str`, `split: str`, `seed: int`, `run_id: str`, `gang_families: tuple[str, ...]`, `decisions_per_event_payment: bool`, `registry_wired: bool`, `registry_wired_reason: str`, `payment_rows: list[dict[str, Any]]`, `snapshot_rows: list[dict[str, Any]]`, `label_rows: list[dict[str, Any]]`, `clock_start: str`, `label_status_rows: list[dict[str, Any]]`
  - Path: `replay/dataset_builder.py`

- `RegistryDriver`:
  - Fields: `core: RegistryCore`, `key: str`, `seed: list[replay.registry_replay.RegistryListing]`, `world: list[replay.registry_replay.RegistryListing]`, `world_start: date`, `bulk: list[replay.registry_replay.BulkEvent]`, `holds: list[replay.hold_requests.HoldEvent]`, `hold_missed: int`, `run_id: str`, `clock: Any`, `screen_rng: Any`, `screen_delay_ms: tuple[float, float]`, `screen_timeout_ms: float`, `stats: ScreenStats`
  - Path: `replay/registry_driver.py`

- `BulkEvent`:
  - Fields: `item: IngestionItem`, `visible_time: datetime`, `batch_id: str`, `region: int`
  - Path: `replay/registry_replay.py`

- `RegistryListing`:
  - Fields: `item: IngestionItem`, `visible_time: datetime`, `seed: bool`
  - Path: `replay/registry_replay.py`

- `RegistryReplaySettings`:
  - Fields: `listing_delay: timedelta`, `dispute_resolution_delay: timedelta`, `bank_settings: dict[str, dict[str, Any]]`, `seed_share_range: tuple[float, float], reporter_choice: str`, `screen_delay_ms: tuple[float, float]`
  - Path: `replay/registry_replay.py`

- `HoldEvent`:
  - Fields: `item: IngestionItem`, `visible_time: datetime`, `complaint_id: str`, `label_id: str`, `payment_ids: tuple[str, ...]`, `latency_minutes: int`, `claimed_type: str | None`
  - Path: `replay/hold_requests.py`

---

## Configuration Keys

### `config/replay.yaml`
- **replay_workspace**
  - `redis_workspace_db_min`
  - `redis_workspace_db_max`
  - `registry_key`
  - `registry_lock_seconds`
  - `drain_timeout_seconds`
  - `kafka_topic_partitions`
  - `decisions_per_event_payment`
  - `kyc_phone_on_account_opened`
  - `world_registry_reporting`
  - `replay_speed`
  - `idle_partition_advance`

### `config/datasets.yaml`
- **datasets**
  - `worlds_per_split`

### `config/showcase_level_16.yaml`
- **world**
  - `label`
  - `derived_from`
  - `reason`
  - `seed`
  - `n_people`
  - `n_shops`
  - `n_days`
  - `range_name`
- **pins**
  - `bulk_confirmation_events_per_year`
  - `victim_complaint_share`
  - `false_complaints_per_100000_genuine_payments`
- **registry**
  - `label`
  - `showcase_only`
  - `non_participants`
  - `reason`
- **minimums**
  - `label`
  - `bulk_confirmations_with_accounts`
  - `cross_bank_complaints`
  - `recycled_numbers_listed_before_day_0`
  - `g3_complaints_on_one_account`
  - `g3_spread_days`
  - `false_reported_accounts`
  - `hold_requests`
  - `innocent_holder_listings`
  - `reason`
- **scan**
  - `label`
  - `seeds_scanned`
  - `seeds_passing`
  - `chosen_seed`
  - `chosen_seed_counts`
  - `reason`

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
