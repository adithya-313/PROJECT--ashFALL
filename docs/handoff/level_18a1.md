# Level 18a1 Handoff Note — fast replay

All timing figures below are from tiny sampler worlds (SIMULATED) unless stated otherwise.

The section "Generated facts" is the verbatim output of
`py -3.12 scripts/handoff_facts.py --part 18a1` (signatures from `inspect.signature`,
fields from `dataclasses.fields`, keys read from the config files). Regenerate it instead
of editing it. Everything else is written by hand.

## Where things stand
Level 18a1 makes replay fast enough to build datasets and run evaluation at scale. The
drain no longer re-applies a 200-row overlap tail while advancing its watermark, and it
answers "is the consumer caught up?" from local produced/consumed counters instead of
asking Kafka on every drain. Branch: `feature/level-18a1-fast-replay`.
- 18a1-i: optional `overlap_rows` override on the four `apply_*_since` pollers; the
  drain's watermark-advance step passes `overlap_rows=0`. Catch-up loops keep the
  configured 200-row overlap.
- 18a1-ii: `DrainCounters` (payments and registry produced/consumed) and two switches,
  `local_offset_tracking` and `registry_poll_on_change`; broker cross-checks at every
  snapshot boundary and once at run end before the drop hook.
- The three switches are committed `true` in `config/replay.yaml` (`0636adf`), with the
  hook note corrected in `efd7c65`.
- `scripts/build_level_16_datasets.py` prints one plain line per dataset build before
  replay starts (`96251d3`):
  `Replay switches: watermark_advance_skip_overlap=<v> local_offset_tracking=<v> registry_poll_on_change=<v>`.
- `scripts/compare_datasets_normalised.py` (Commit C, `de19251`) is the committed
  run-id-normalised comparison used for the output-identity verdict.
- Commit A (`e116181`) fixed the fake consumer in the idle-advance run test so it matches
  aiokafka 0.13.0 (synchronous `partitions_for_topic`).

## Results — SIMULATED, tiny worlds (proves the speedup, not a full-scale result)
Staged measurement on `test-0000`, sampler seed 2:

| stage | wall | measured phase rows |
|---|---|---|
| baseline (both local switches off) | 4,681 s | `drain.watermark_advance` 1,684.5 s; `drain.offset_check` 1,510.9 s; `drain.registry_poll` 1,008.4 s; Redis commands 9.52 M |
| `watermark_advance_skip_overlap` on | 3,256 s (18a1-i headline; the 18a1-ii profile measured the same configuration at 3,325 s) | `drain.watermark_advance` 49.8 s; Redis commands 2.95 M |
| both local switches also on | 778 s | `drain.offset_check` 0.004 s; `drain.registry_poll` 0.105 s; remaining hot phases `run.decision_post` ~246 s, `run.snapshot_write` ~195 s |

Speedup 4,681 s -> 778 s (6.0x).

### Smoke run — second tiny world, all three switches on
- Seed: `fitting sampler seed: 3`.
- Switch line: `Replay switches: watermark_advance_skip_overlap=True local_offset_tracking=True registry_poll_on_change=True`.
- Exit code: `0` (every snapshot-boundary and run-end cross-check passed).
- Dataset: `test-20261031T1830-ca89025131f3`, world `test-0000`, rows payments=3075,
  snapshots=2780, labels=9, wall time 981.69 s.
- Profile: `replay_seconds` 979.07; `feature_log_sha256`
  `bf2040db7661950119296cba3ec8b1ec7b9f3b578364f6e214412996d90dd13d`;
  `record_seconds` 1.002.

## What each component does
- `watermark_advance_skip_overlap` (`config/replay.yaml`): the drain's watermark-advance
  step reads only rows after each poller cursor instead of re-applying the 200-row
  overlap tail. Catch-up loops still use the overlap. Fallback: with the switch off the
  advance step passes no override and the configured overlap applies everywhere.
- `local_offset_tracking`: the payments-topic caught-up check uses
  `DrainCounters.payments_produced` / `payments_consumed` instead of `end_offsets` plus
  `committed` on every drain. Fallback: broker offsets when the switch is off.
- `registry_poll_on_change`: while the local registry produced count equals the consumed
  count the registry drain skips `getmany` and offset calls; `relay_national_outbox`
  still runs every drain and reports `published_by_topic` per mapped topic.
  Fallback: the broker path when the switch is off.
- `before_decision_hook` override: `resolve_drain_switches` forces both local-counter
  switches OFF when a `before_decision_hook` is supplied, because that hook runs between
  drains and may publish outside the counters. `before_drop_hook` does NOT force them
  off: it runs at teardown, after the last drain and after the run-end cross-check.
- Cross-checks: `cross_check_counters` compares the local counters with the broker's end
  and committed offsets at every snapshot boundary, and `finish_run` repeats it once at
  run end before the drop hook; a mismatch raises `ReplayPipelineError` and skips the
  hook. Counting is always on, so the cross-check can run in either mode.

## Verification method
`scripts/compare_datasets_normalised.py` loads the payments, snapshots and labels parquet
tables of two dataset folders, replaces each world's own run_id with `RUNID` in every
string cell (including list cells such as `labels.payment_ids`), sorts by the
`content_hash` sort keys in `replay/dataset_builder.py`, and compares every remaining
cell exactly. It never ignores a column and never reads truth. Exit 0 = identical,
1 = differs, 2 = incomparable input.

The 18a1-ii run was compared against fixture
`test-20261031T1830-9a2856ce53b2`:

```
$ PYTHONPATH=. py -3.12 scripts/compare_datasets_normalised.py runs/profile_18a1ii_b/datasets/test-20261031T1830-5561dcd4db96 showcase/fixtures/level_17/datasets/test-20261031T1830-9a2856ce53b2
payments: rows_new=2895 rows_fixture=2895 differing_rows=0
snapshots: rows_new=2683 rows_fixture=2683 differing_rows=0
labels: rows_new=8 rows_fixture=8 differing_rows=0
VERDICT: IDENTICAL
```

The smoke run exercised both cross-checks at every boundary and at run end and exited 0;
it was not compared table-by-table to the seed-2 fixture because it is a different world.

## Decisions and why
- Make the speed switches default on: proven output-identical on the tiny world; the
  build line records the resolved values in every unattended build.
- Keep the cross-checks at snapshot boundaries and run end: local counters are only
  trusted while they agree with the broker.
- Force the switches off only for `before_decision_hook`, never for `before_drop_hook`.
- Print the switches from the build script rather than configure logging, so an
  unattended build log shows them in plain stdout.

## Known gaps and deferrals
- 18-1 (planned): dataset builds cannot be pinned to a code version —
  `build_level_16_datasets.py` ignores `ASHFALL_CODE_VERSION`, so a fixture cannot be
  reproduced byte-for-byte at a later commit.
- Scope is two tiny worlds (seed 2 verification, seed 3 smoke). Full-scale 680 small
  worlds still needs about another order of magnitude; parallel worlds or an in-process
  mode is the next part.
- Remaining hot phases in the 778 s run: `run.decision_post` ~246 s,
  `run.snapshot_write` ~195 s.
- 3 pre-existing mypy errors in `world/` (unchanged by 18a1).
- 17-4 CI status is unknown until CI runs (Level 16 showcase may still time out).

## How to verify
- `py -3.12 -m pytest -q -p no:cacheprovider tests/unit`
- `py -3.12 -B scripts/verify_world.py` -> ALL GOLDEN CHECKS PASSED.
- `py -3.12 -m showcase --level 17` (unaffected by 18a1; the Level 16 showcase is the
  slow one).
- Compare two dataset folders:
  `py -3.12 scripts/compare_datasets_normalised.py <new_dir> <fixture_dir>`.

## Generated facts

## Register Changes
- 16-4: Full-scale dataset building and a faster replay drain (status: partial)
- 17-4: CI level-showcase times out: the Level 16 showcase replays twice (~2.5 h) (status: planned)
- 18-1: Dataset builds cannot be pinned to a code version: build_level_16_datasets.py ignores ASHFALL_CODE_VERSION, so a fixture cannot be reproduced byte-for-byte at a later commit (status: planned)

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
cross_check_counters(consumer: Any, workspace: world.replay_workspace.ReplayWorkspace, registry: 'RegistryDrain | None', counters: replay.runner.DrainCounters) -> None
```

```python
day_boundaries_between(prev_world_time: datetime.datetime, next_world_time: datetime.datetime) -> list[datetime.datetime]
```

```python
drain(step_index: int, workspace_conn: asyncpg.connection.Connection, producer: Any, consumer: aiokafka.consumer.consumer.AIOKafkaConsumer, workspace: world.replay_workspace.ReplayWorkspace, redis_client: Any, acct_settings: features.account_keys.AccountFeaturesSettings, banks: list[str], world_clock: core.event_clock.ManualClock, run_id: str, poller_cursors: dict[str, dict[str, int]], deadline: float, batch_size: int = 100, registry: replay.runner.RegistryDrain | None = None, profile: replay.profiling.ReplayProfile | None = None, skip_overlap_on_advance: bool = False, counters: replay.runner.DrainCounters | None = None, local_offset_tracking: bool = False, registry_poll_on_change: bool = False) -> float
```

```python
drain_registry(registry: replay.runner.RegistryDrain, conn: Any, timeout_ms: int = 20) -> bool
```

```python
finish_run(switches_active: bool, consumer: Any, workspace: world.replay_workspace.ReplayWorkspace, registry: 'RegistryDrain | None', counters: replay.runner.DrainCounters, drop_hook: collections.abc.Callable[[], collections.abc.Awaitable[None]] | None) -> None
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
load_default_local_offset_tracking() -> bool
```

```python
load_default_registry_poll_on_change() -> bool
```

```python
load_default_replay_speed() -> str | float
```

```python
load_default_watermark_advance_skip_overlap() -> bool
```

```python
load_default_world_registry_reporting() -> bool
```

```python
pace_seconds(previous_world_time: datetime.datetime | None, next_world_time: datetime.datetime, speed: str | float) -> float
```

```python
pace_step(previous_world_time: datetime.datetime | None, next_world_time: datetime.datetime, speed: str | float, sleep: collections.abc.Callable[[float], collections.abc.Awaitable[None]] = <function sleep at 0x000001F7A67362A0>) -> float
```

```python
read_registry_batch(registry: replay.runner.RegistryDrain, conn: Any, timeout_ms: int = 20) -> int
```

```python
resolve_drain_switches(local_offset_tracking: bool, registry_poll_on_change: bool, decision_hook_present: bool) -> tuple[bool, bool]
```

```python
run_replay(recorded_world: world.replay_recording.RecordedWorld, feed: world.bank_feed.BankFeed, admin_pg: str, *, decisions_per_event_payment: bool | None = None, replay_speed: str | float | None = None, sleep: collections.abc.Callable[[float], collections.abc.Awaitable[None]] = <function sleep at 0x000001F7A67362A0>, replay_settings: world.replay_workspace.ReplaySettings | None = None, account_settings: features.account_keys.AccountFeaturesSettings | None = None, live_redis: Any = None, base_redis_url: str | None = None, redis_client_factory: collections.abc.Callable[[world.replay_workspace.ReplayWorkspace], typing.Any] | None = None, before_decision_hook: collections.abc.Callable[[replay.runner.ReplayHookContext], collections.abc.Awaitable[None]] | None = None, before_drop_hook: collections.abc.Callable[[replay.runner.ReplayDropContext], collections.abc.Awaitable[None]] | None = None, workspace_name: str | None = None, idle_partition_advance: bool | None = None, world_registry_reporting: bool | None = None, registry_settings: replay.registry_replay.RegistryReplaySettings | None = None) -> replay.runner.ReplayResult
```

```python
validate_replay_speed(speed: Any) -> None
```

```python
DrainCounters(payments_produced: int = 0, payments_consumed: int = 0, registry_produced: int = 0, registry_consumed: int = 0) -> None
```
- dataclass

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
ReplayResult(decisions_sent: int, feature_rows_by_status: dict[str, int], decisions_arrival_cutoff_behind: int, max_arrival_cutoff_gap_seconds: float, longest_drain_seconds: float, decisions_before_first_payment: int, unwatermarked_payment_ids: tuple[str, ...] = (), logged_features_by_payment_id: dict[str, dict[str, typing.Any]] = <factory>, missing_request_fields_by_payment_id: dict[str, list[str]] = <factory>, registry_wired: bool = False, registry_wired_reason: str = 'no nationalsim screener or DPIP key in replay until 16d', registry_screens: int = 0, registry_screens_unknown: int = 0, feature_log_sha256: str = '', profile: dict[str, typing.Any] = <factory>) -> None
```
- dataclass

### `nationalsim/national_outbox.py`

```python
consume_own_messages(bank_id: int | str, run_id: str, topic: str, bootstrap_servers: str | None = None, expected_count: int | None = None, idle_timeout_seconds: float = 3.0) -> nationalsim.national_outbox.BankConsumerResult
```

```python
create_national_envelope(event_type: str, event_id: uuid.UUID, run_id: str, occurred_at: str, data: dict[str, typing.Any]) -> dict[str, typing.Any]
```

```python
relay_national_outbox(conn: asyncpg.connection.Connection, producer: aiokafka.producer.producer.AIOKafkaProducer, batch_size: int = 100, topic_map: collections.abc.Mapping[str, str] | None = None, published_by_topic: dict[str, int] | None = None) -> int
```

```python
write_event(conn: asyncpg.connection.Connection, topic: str, run_id: str, bank_id: int | str, payload: dict[str, typing.Any]) -> uuid.UUID
```

```python
BankConsumerResult(accepted_messages: list[dict[str, typing.Any]], foreign_keys_count: int) -> None
```
- dataclass

### `scripts/compare_datasets_normalised.py`

```python
compare_datasets(new_dir: pathlib.Path, fixture_dir: pathlib.Path) -> int
```

```python
differing_rows(left: list[dict[str, typing.Any]], right: list[dict[str, typing.Any]]) -> int
```

```python
load(folder: pathlib.Path, table: str, run_ids: dict[str, str]) -> tuple[set[str], list[dict[str, typing.Any]]]
```

```python
main(argv: collections.abc.Sequence[str] | None = None) -> int
```

```python
normalise(value: Any, run_id: str) -> Any
```

```python
sort_value(value: Any) -> tuple[bool, str]
```

```python
world_run_ids(folder: pathlib.Path) -> dict[str, str]
```

---

## Records and Fields

Extracted directly via `dataclasses.fields`:

- `DrainCounters`:
  - Fields: `payments_produced: int`, `payments_consumed: int`, `registry_produced: int`, `registry_consumed: int`
  - Path: `replay/runner.py`

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
  - Fields: `decisions_sent: int`, `feature_rows_by_status: dict[str, int]`, `decisions_arrival_cutoff_behind: int`, `max_arrival_cutoff_gap_seconds: float`, `longest_drain_seconds: float`, `decisions_before_first_payment: int`, `unwatermarked_payment_ids: tuple[str, ...]`, `logged_features_by_payment_id: dict[str, dict[str, Any]]`, `missing_request_fields_by_payment_id: dict[str, list[str]]`, `registry_wired: bool`, `registry_wired_reason: str`, `registry_screens: int`, `registry_screens_unknown: int`, `feature_log_sha256: str`, `profile: dict[str, Any]`
  - Path: `replay/runner.py`

- `BankConsumerResult`:
  - Fields: `accepted_messages: list[dict[str, Any]]`, `foreign_keys_count: int`
  - Path: `nationalsim/national_outbox.py`

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
  - `watermark_advance_skip_overlap`
  - `local_offset_tracking`
  - `registry_poll_on_change`

### `config/datasets.yaml`
- **datasets**
  - `worlds_per_split`

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
