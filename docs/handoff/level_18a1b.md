# Level 18a1b Handoff Note — resumable builds and world replay retry

All fixture and laptop figures below are from tiny sampler worlds (SIMULATED) unless stated
otherwise.

The section "Generated facts" is the verbatim output of
`PYTHONPATH=. py -3.12 scripts/handoff_facts.py --part 18a1b` (signatures from
`inspect.signature`, fields from `dataclasses.fields`, keys read from the config files).
Regenerate it instead of editing it. Everything else is written by hand.

## Where things stand
Level 18a1b makes a long Level 16 build survivable: a world that finished is not repeated, and a
world whose Redis or Postgres connection drops is replayed instead of killing the split.
Branch: `feature/level-18a1b-resumable-builds`, merged into `feature/level-18-mule-models`.
- `scripts/dataset_checkpoints.py` holds `DatasetCheckpoint` and the save, load and identity
  checks. Pickle is used only for files written and read by this script on the owner's machine.
- `--checkpoint-dir PATH` on `scripts/build_level_16_datasets.py` turns resuming on. Without the
  flag the build keeps today's exact order (truth before replay, nothing written for resume).
- `config/dataset_build.yaml` holds the retry policy: the switch, the attempt count and the waits.
- `replay_world_with_retry` re-runs one world after a dropped connection and raises anything else
  at once. It replays the whole world, not just the failed command.
- `--world-shard K/N` on the same script builds only the worlds whose plan position is `K mod N`
  (and writes checkpoints only, never a dataset), so several processes can share one split.
- `--parallel-workers N` starts N children of the same script, one shard each, then assembles the
  split in the parent from the checkpoints they wrote.
- `--assemble-only` builds from checkpoints already on disk without recording or replaying.
- Branch for the parallel work: `feature/level-18a1b-parallel-worlds`, merged into
  `feature/level-18-mule-models`.

## What each component does
- `DatasetCheckpoint`: format version, split, world id, seed, preset name, sampler seed, code
  version, run id, world end, record seconds, replay seconds and the `WorldExtract`.
- `save_checkpoint`: writes `<checkpoint-dir>/<split>/<world_id>.pkl.tmp`, flushes, fsyncs, then
  `os.replace`s it onto the `.pkl`. A crash leaves only the `.tmp`, which no loader reads.
- `load_checkpoint`: reads only the exact `.pkl` path, refuses a different type or format
  version, and checks that the extract's run id equals the recorded identity run id.
- `verify_checkpoint_identity`: compares split, world id, seed, preset name, sampler seed and
  code version with the current spec and settings; a mismatch raises one line naming the field
  and both values.
- `world_end_for_recorded` and `cutoff_from_world_ends`: the cutoff rule moved out so a resumed
  build derives the same cutoff from saved world ends as from recorded worlds.
- `load_dataset_build_config`: reads `config/dataset_build.yaml`; nothing in the recording or
  run-id path reads it (A-041; `git grep -n "dataset_build.yaml"` lists only the build script).
- `replay_world_with_retry`: catches `ConnectionResetError`, `ConnectionError`,
  `redis.exceptions.ConnectionError`, `asyncpg.exceptions.ConnectionDoesNotExistError`,
  `ConnectionFailureError`, `CannotConnectNowError` and `ClientCannotConnectError`; waits the
  configured seconds between attempts and re-raises the original exception after the last one.
- `config/dataset_build.yaml`: `retry_world_on_connection_error: true`, `max_world_attempts: 3`,
  `retry_wait_seconds: [30, 120]`, all labelled `policy`.
- `scripts/dataset_shards.py`: `WorldShard` and `parse_world_shard` for `K/N`, `shard_specs` (a
  world belongs to shard `position % N`), the STOP file path and check, and
  `load_plan_checkpoints`, which loads and verifies every planned world's checkpoint in plan order
  or raises `MissingCheckpointsError` naming the missing world ids.
- `EXIT_STOPPED = 3` in `scripts/dataset_shards.py`: 0, 1 and 2 are already taken (success,
  failure, usage refusal); 3 means "stopped or incomplete, run the same command again".
- `checkpoint_split_worlds`: per world, loads an existing checkpoint and prints the resumed line,
  or checks for the STOP file before starting a world that still needs building, or records,
  replays and saves a checkpoint. Returns whether it stopped.
- `split_worker_commands`, `worker_argv`, `start_worker_process` and `run_worker_commands`: the
  parent builds one explicit argument list per child (same interpreter, same script, same flags
  minus `--parallel-workers` and `--splits`, plus this split and `--world-shard K/N`), starts all
  children without a shell, and waits for all of them.
- `validate_parallel_workers`: refuses a count above the replay workspace pool
  (`redis_workspace_db_max - redis_workspace_db_min + 1`, read from `config/replay.yaml`).
- `worker_failure_exit_code`: 3 when every failed child exited 3, otherwise 1. On any child
  failure the parent prints each exit code and the last 5 lines of each failing child's log and
  does not assemble.

## Laptop verification (SIMULATED, tiny, test split only)
Run at `0005729`, preset tiny, sampler seed 2, `--write-evaluation-inputs`, output rooted in
`runs/ckpt18a1b/` (datasets, truth, world cards and checkpoints).
- Run 1 (clean tree): exit 0, wall 619.59s; replay 617.94s, record 0.56s for test-0000;
  dataset `test-20261031T1830-3f409e3271ab`, rows payments 2895, snapshots 2688, labels 8.
- Run 2a (dirty tree, handoff draft uncommitted): exit 1 with
  `ValueError: checkpoint identity mismatch: code_version: checkpoint='0005729' current='0005729-dirty'`.
- Run 2b (tree clean again, same command): exit 0, wall 1.39s, printed
  `Resumed world test-0000 from runs\ckpt18a1b\checkpoints\test\test-0000.pkl (record 0.56s, replay 617.94s earlier)`.
- Comparison: run 1 and run 2b produced the same deterministic dataset id, and the folder
  compares `VERDICT: IDENTICAL` against `showcase/fixtures/level_17/datasets/test-20261031T1830-3910e3bea0d4`
  (payments 2895, snapshots 2688, labels 8, 0 differing rows each).
- run_id: `ff70a12d26e662e9`. The architect's brief expected `7ee68d1b5a539657`, which is only
  reproducible at the fixture's `code_version` 38151de because `make_run_id` hashes the code
  version; the ruling accepted the identical-rows comparison instead (see Decisions).

## Parallel worlds (Windows laptop, SIMULATED, tiny, training split only)
Run at `413e388`, preset tiny, sampler seed 2, `--write-evaluation-inputs`, `--parallel-workers 2`,
checkpoints and datasets rooted in `runs/par18a1b/`.
- Run 1: exit 0, wall 796.72s (split wall 793.67s). Worker 0, training-0000: record 0.66s, replay
  788.20s. Worker 1, training-0001: record 0.60s, replay 676.73s. The two worlds replayed at the
  same time, so the split took about one world's replay time instead of two.
- Dataset `training-20261031T1830-a0107f3a6a37`: payments 5990, snapshots 5527, labels 22.
  `scripts/compare_datasets_normalised.py` against
  `showcase/fixtures/level_17/datasets/training-20261031T1830-624c2ad40f4c` prints
  `VERDICT: IDENTICAL` with 0 differing rows in each table.
- Peak `docker stats --no-stream` during run 1, sampled every 30s: redis 22.40% CPU, 16.57MiB;
  postgres 16.91% CPU, 351.5MiB (postgres memory peaked at 351.5MiB, redis at 16.57MiB).
- Run 2 (same command): exit 0, wall 6.98s (split wall 4.3s). Both workers printed
  `Resumed world ...` and no replay ran. The dataset id is the same folder, which is deterministic,
  so run 2 rewrote it rather than creating a second tree; identity with run 1 and with the fixture
  rests on that folder name and the IDENTICAL comparison, not on diffing two folders.
- STOP check: `--world-shard 0/1` on the test split in a fresh `runs/par18a1b/stopcheck/` with the
  STOP file created before start printed `STOP file found; not starting test-0000` and exited 3;
  no world was recorded.

## Stopping and resuming
- Stop safely: create `<checkpoint-dir>/<split>/STOP` (for example
  `runs/par18a1b/checkpoints/training/STOP`) while a build runs. A running world is not killed: it
  finishes and saves its checkpoint. Before starting another world that still needs building, the
  process prints `STOP file found; not starting <world_id>` and exits 3 (stopped or incomplete).
  The parent then assembles nothing.
- Resume: delete the STOP file and run the same command again. Finished worlds print
  `Resumed world <world_id> from <path> (record Xs, replay Ys earlier)` and are not replayed; only
  missing worlds are built, and the split is then assembled in plan order.
- Open risk, Redis: Redis is single-threaded. Two workers used at most 22.40% CPU of the host, so
  it was not the limit here, but the lab-machine scaling test (1, 4, 8, 12 replays) must measure
  whether Redis becomes a per-core ceiling that flattens the speedup.
- Open risk, Postgres: `max_locks_per_transaction` is 1024, sized for 3 parallel replays'
  workspace `DROP SCHEMA ... CASCADE`; 8 and 12 parallel replays are unverified and must be
  watched for lock-table exhaustion on the lab machine.

## Decisions and why
- Retry a whole world, not a single command: `run_replay` drops its workspace in a `finally`
  block, so only a fresh replay is guaranteed to start clean; restarting one command would leave
  unknown state behind.
- Save truth after replay when checkpointing: the checkpoint is written last, so a crash between
  truth and checkpoint is visible as an existing truth folder. The build then stops and names the
  folder to delete, instead of letting a resumed world's truth look complete.
- Keep the retry policy out of the run-id inputs: `code_version` is part of `make_run_id`, and
  A-041 showed that a config read by the run-id path changes every run id. `config/dataset_build.yaml`
  is read only by the build script.
- Assemble in plan order, never completion order: `build_dataset` builds its tables in extract
  order and `write_dataset` writes those tables directly, so parquet row order follows the extract
  order. Checkpoints are therefore loaded by plan position. The dataset id is unaffected because
  the content hash sorts rows before hashing.
- Parallel workers are child processes of the same script, not threads: each replay owns an event
  loop, an asyncpg pool and a Redis workspace DB, and separate processes keep them apart. The pool
  size check makes more workers than workspace DBs a refusal instead of a hang on the lock.
- STOP is checked only before a world that still needs building, never before resuming, so creating
  the file does not throw away work already on disk.
- The parent starts every worker before waiting for any and never kills a healthy one, so one
  failed worker costs only its own worlds; the split is not assembled until every worker succeeds.

## Known gaps and deferrals
- Only 2 parallel workers were measured, on a Windows laptop, for 2 tiny worlds. The lab-machine
  scaling test (1, 4, 8, 12 replays) has not run; see the Redis and Postgres open risks above.
- `--world-shard` accepts `K < N` only; shards hold worlds by plan position, so uneven splits are
  possible when N does not divide the split's world count.
- `asyncpg.exceptions.InterfaceError` ("connection is closed", seen in the Level 15 flake) is not
  retried, deliberately, because `InterfaceError` also covers programming errors; revisit with
  evidence that it was raised by a lost connection.
- The parallel round resumed two worlds from checkpoints, but no run was interrupted in the middle
  of a split and then resumed from a partial set.

## How to verify
- `PYTHONPATH=. py -3.12 -B scripts/verify_world.py` -> ALL GOLDEN CHECKS PASSED.
- `py -3.12 -m pytest -q -p no:cacheprovider tests/unit`
- `py -3.12 -m showcase --level 16` and `py -3.12 -m showcase --level 17`
- `py -3.12 -m core.config_validator config/dataset_build.yaml`
- Resume check: build the test split with `--checkpoint-dir`, then run the same command again;
  the second run prints the resumed line and writes the same dataset id.
- Parallel check: `PYTHONPATH=. py -3.12 scripts/build_level_16_datasets.py --splits training
  --checkpoint-dir runs/par18a1b/checkpoints --parallel-workers 2` prints one line per worker and
  one dataset id; running it again resumes both worlds in seconds.
- STOP check: create `<checkpoint-dir>/<split>/STOP` before starting; the build prints
  `STOP file found; not starting <world_id>`, starts nothing and exits 3.

## Generated facts

## Register Changes
- 16-4: Full-scale dataset building and a faster replay drain (status: partial)

---

## Public Interfaces

Extracted directly via `inspect.signature`:

### `scripts/build_level_16_datasets.py`

```python
build_all_level_16_datasets(splits: collections.abc.Sequence[str] | None = None, preset_name: str = 'tiny', datasets_root: pathlib.Path = WindowsPath('C:/Users/sinne/Downloads/PROJECT-ashFALL/data/datasets'), sampler_seed: int = 1, write_evaluation_inputs: bool = False, world_cards_root: pathlib.Path = WindowsPath('C:/Users/sinne/Downloads/PROJECT-ashFALL/runs/world_cards'), replay_profile_dir: pathlib.Path | None = None, worlds_per_split: dict[str, int] | None = None, checkpoint_dir: pathlib.Path | None = None, world_shard: scripts.dataset_shards.WorldShard | None = None, assemble_only: bool = False, parallel_workers: int | None = None, base_argv: collections.abc.Sequence[str] = ()) -> dict[str, dict[str, typing.Any]]
```

```python
build_split_dataset(split: str, specs: list[world.world_sampler.WorldSpec], preset_name: str, sampler_cfg: dict[str, typing.Any], code_version: str, sampler_seed: int = 1, datasets_root: pathlib.Path = WindowsPath('C:/Users/sinne/Downloads/PROJECT-ashFALL/data/datasets'), admin_pg: str | None = None, base_redis_url: str | None = None, write_evaluation_inputs: bool = False, truth_root: pathlib.Path | None = None, world_cards_root: pathlib.Path = WindowsPath('C:/Users/sinne/Downloads/PROJECT-ashFALL/runs/world_cards'), replay_profile_dir: pathlib.Path | None = None, checkpoint_dir: pathlib.Path | None = None, assemble_only: bool = False) -> tuple[replay.dataset_builder.Dataset, pathlib.Path, list[pathlib.Path]]
```

```python
checkpoint_split_worlds(split: str, specs: collections.abc.Sequence[world.world_sampler.WorldSpec], preset_name: str, sampler_cfg: dict[str, typing.Any], code_version: str, *, sampler_seed: int, checkpoint_dir: pathlib.Path, db_url: str, redis_url: str, evaluation_truth_root: pathlib.Path | None, world_cards_root: pathlib.Path, replay_profile_dir: pathlib.Path | None) -> bool
```

```python
cutoff_from_world_ends(world_ends: collections.abc.Sequence[datetime.datetime]) -> datetime.datetime
```

```python
derive_dataset_cutoff(recorded_worlds: collections.abc.Sequence[world.replay_recording.RecordedWorld]) -> datetime.datetime
```

```python
find_fitting_sampler_seed(splits: collections.abc.Sequence[str], preset_name: str, start_seed: int, max_tries: int, datasets_cfg: dict[str, int] | None = None) -> int
```

```python
host_redis_factory(ws: world.replay_workspace.ReplayWorkspace) -> redis.client.Redis
```

```python
host_side_redis_url() -> str
```

```python
load_dataset_build_config(path: pathlib.Path = WindowsPath('C:/Users/sinne/Downloads/PROJECT-ashFALL/config/dataset_build.yaml')) -> dict[str, typing.Any]
```

```python
load_datasets_config(path: pathlib.Path | str | None = None) -> dict[str, int]
```

```python
log_tail(log_path: pathlib.Path, lines: int = 5) -> list[str]
```

```python
main() -> None
```

```python
parse_worlds_per_split(text: str) -> dict[str, int]
```

```python
plan_datasets(datasets_cfg: dict[str, int] | None = None, sampler_cfg: dict[str, typing.Any] | None = None, world_ranges: dict[str, typing.Any] | None = None, sampler_seed: int = 1) -> dict[str, list[world.world_sampler.WorldSpec]]
```

```python
record_world_for_spec(spec: world.world_sampler.WorldSpec, preset_name: str, sampler_cfg: dict[str, typing.Any], code_version: str, loop: asyncio.events.AbstractEventLoop) -> world.replay_recording.RecordedWorld
```

```python
redis_command_calls(client: redis.client.Redis) -> int
```

```python
replay_and_extract(recorded: world.replay_recording.RecordedWorld, admin_pg: str, base_redis_url: str, *, profile_dir: pathlib.Path | None = None) -> replay.dataset_builder.WorldExtract
```

```python
replay_world_with_retry(recorded: world.replay_recording.RecordedWorld, admin_pg: str, base_redis_url: str, *, config: dict[str, typing.Any], profile_dir: pathlib.Path | None = None, sleep: collections.abc.Callable[[float], collections.abc.Awaitable[None]] = <function sleep at 0x000002F30D8CF6A0>) -> replay.dataset_builder.WorldExtract
```

```python
run_worker_commands(commands: collections.abc.Sequence[scripts.build_level_16_datasets.WorkerCommand], *, spawn: collections.abc.Callable[[collections.abc.Sequence[str], pathlib.Path], scripts.build_level_16_datasets.ProcessHandle] = <function start_worker_process at 0x000002F3295EEF20>) -> list[int]
```

```python
save_evaluation_inputs(recorded: world.replay_recording.RecordedWorld, truth_root: pathlib.Path, world_cards_root: pathlib.Path) -> dict[str, typing.Any]
```

```python
split_worker_commands(base_argv: collections.abc.Sequence[str], split: str, specs: collections.abc.Sequence[world.world_sampler.WorldSpec], checkpoint_dir: pathlib.Path, workers: int) -> list[scripts.build_level_16_datasets.WorkerCommand]
```

```python
start_worker_process(argv: collections.abc.Sequence[str], log_path: pathlib.Path) -> subprocess.Popen[bytes]
```

```python
summarize_labels_by_source_and_via(dataset: replay.dataset_builder.Dataset) -> dict[str, int]
```

```python
supersede_prior_datasets(split: str, new_dir: pathlib.Path, datasets_root: pathlib.Path = WindowsPath('C:/Users/sinne/Downloads/PROJECT-ashFALL/data/datasets')) -> list[pathlib.Path]
```

```python
validate_parallel_workers(count: int) -> int
```

```python
worker_argv(base_argv: collections.abc.Sequence[str], shard: scripts.dataset_shards.WorldShard, split: str) -> list[str]
```

```python
worker_failure_exit_code(codes: collections.abc.Sequence[int]) -> int
```

```python
world_end_for_recorded(recorded: world.replay_recording.RecordedWorld) -> datetime.datetime
```

```python
world_shard_argument(text: str) -> scripts.dataset_shards.WorldShard
```

```python
WorkerCommand(index: int, world_ids: tuple[str, ...], argv: tuple[str, ...], log_path: pathlib.Path) -> None
```
- dataclass

### `scripts/dataset_checkpoints.py`

```python
checkpoint_path(checkpoint_dir: pathlib.Path, split: str, world_id: str) -> pathlib.Path
```

```python
checkpoint_temp_path(path: pathlib.Path) -> pathlib.Path
```

```python
load_checkpoint(path: pathlib.Path) -> scripts.dataset_checkpoints.DatasetCheckpoint
```

```python
save_checkpoint(path: pathlib.Path, checkpoint: scripts.dataset_checkpoints.DatasetCheckpoint) -> None
```

```python
verify_checkpoint_identity(checkpoint: scripts.dataset_checkpoints.DatasetCheckpoint, *, split: str, world_id: str, seed: int, preset_name: str, sampler_seed: int, code_version: str) -> None
```

```python
DatasetCheckpoint(format_version: int, split: str, world_id: str, seed: int, preset_name: str, sampler_seed: int, code_version: str, run_id: str, world_end: datetime.datetime, record_seconds: float, replay_seconds: float, extract: replay.dataset_builder.WorldExtract) -> None
```
- dataclass

### `scripts/dataset_shards.py`

```python
load_plan_checkpoints(checkpoint_dir: pathlib.Path, split: str, specs: collections.abc.Sequence[world.world_sampler.WorldSpec], *, preset_name: str, sampler_seed: int, code_version: str) -> tuple[list[replay.dataset_builder.WorldExtract], list[datetime.datetime]]
```

```python
parse_world_shard(text: str) -> scripts.dataset_shards.WorldShard
```

```python
shard_specs(specs: collections.abc.Sequence[world.world_sampler.WorldSpec], shard: scripts.dataset_shards.WorldShard) -> list[world.world_sampler.WorldSpec]
```

```python
stop_file_path(checkpoint_dir: pathlib.Path, split: str) -> pathlib.Path
```

```python
stop_requested(checkpoint_dir: pathlib.Path, split: str) -> bool
```

---

## Records and Fields

Extracted directly via `dataclasses.fields`:

- `WorkerCommand`:
  - Fields: `index: int`, `world_ids: tuple[str, ...]`, `argv: tuple[str, ...]`, `log_path: Path`
  - Path: `scripts/build_level_16_datasets.py`

- `DatasetCheckpoint`:
  - Fields: `format_version: int`, `split: str`, `world_id: str`, `seed: int`, `preset_name: str`, `sampler_seed: int`, `code_version: str`, `run_id: str`, `world_end: datetime`, `record_seconds: float`, `replay_seconds: float`, `extract: WorldExtract`
  - Path: `scripts/dataset_checkpoints.py`

---

## Configuration Keys

### `config/dataset_build.yaml`
- **parameters**
  - `retry_world_on_connection_error`
  - `max_world_attempts`
  - `retry_wait_seconds`

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

## Round 2c — several Redis servers
- Switch: `parameters.replay_redis_urls` in `config/dataset_build.yaml`. `value: []` is the OFF state
  and reproduces the single-Redis behaviour exactly; each entry must be a non-empty `redis://` URL
  and no address may repeat. `load_replay_redis_urls` refuses a bad entry by name.
- CLI override: `--replay-redis-urls "url1,url2"` replaces the config switch for one run, so the
  tracked config can stay at `[]` and the run's `code_version` does not gain a "-dirty" suffix. It is
  refused unless `--parallel-workers` is above 1 (otherwise it would do nothing), goes through the
  same `redis://` and duplicate checks as the config, and is stripped from the child commands.
  Example:
  `PYTHONPATH=. py -3.12 scripts/build_level_16_datasets.py --splits training --preset tiny --sampler-seed 2 --write-evaluation-inputs --parallel-workers 2 --replay-redis-urls "redis://localhost:6379/0,redis://localhost:6380/0" --checkpoint-dir runs/r2c/on/checkpoints --root runs/r2c/on/datasets --world-cards-root runs/r2c/on/world_cards`
- Routing: `redis_url_for_worker(k, urls)` returns `None` when the list is empty, otherwise
  `urls[k % len(urls)]`. Round-robin, so neighbouring workers start on different servers; with one
  URL every worker keeps the parent's address.
- Child environment: `start_worker_process(argv, log_path, redis_url=None)` passes no `env` when the
  address is None (today's call unchanged) and otherwise a copy of the parent environment with
  `TEST_REDIS_URL` set to that worker's URL. The parent's own environment is never modified.
- New worker cap: `validate_parallel_workers` allows `pool_size * max(1, len(urls))` workers, where
  `pool_size = redis_workspace_db_max - redis_workspace_db_min + 1` (15 here). With `[]` the cap is
  15 and the old message is unchanged; with 3 URLs the cap is 45 and the message names the 3 servers.
- Extra servers: `docker compose up -d redis-2 redis-3` starts the two new same-image services on
  host ports 6380 and 6381 (`docker-compose.yml`; the existing `redis` service is untouched). A run
  can then set `value: ["redis://localhost:6379/0", "redis://localhost:6380/0"]`.
- Replay profile command count: `redis_commands_during_replay` in each world's profile JSON is read
  from the server's own command stats, so it includes every worker on that server during the world;
  do not sum profiles.

### Laptop acceptance (Windows laptop, SIMULATED, tiny)
Preset tiny, sampler seed 2, training split only, `--parallel-workers 2`,
`--write-evaluation-inputs`, fresh folders under `runs/r2c/off/` and `runs/r2c/on/`.
| Run | Switch | Exit | Wall seconds | Comparison vs level_17 fixture |
| --- | --- | --- | --- | --- |
| OFF | `[]` | 0 | 818 | `VERDICT: IDENTICAL` (payments 5990, snapshots 5527, labels 22, 0 differing rows each) |
| ON | `["redis://localhost:6379/0", "redis://localhost:6380/0"]` | 0 | 807 | `VERDICT: IDENTICAL` (payments 5990, snapshots 5527, labels 22, 0 differing rows each) |
- Both runs wrote dataset `training-20261031T1830-a64b3975255d`; the split wall times were 814.41s
  (OFF) and 804.39s (ON). The ON parent printed
  `Worker 0: ...; redis redis://localhost:6379/0` and
  `Worker 1: ...; redis redis://localhost:6380/0`; the OFF parent printed no redis part.
- The worker logs themselves carry only the schema `apply` lines, so the per-worker address is
  recorded by the parent start line. `docker exec project-ashfall-redis-2-1 redis-cli info commandstats`
  shows server 2 really served work: `cmdstat_zrange:calls=1941222`, `cmdstat_set:calls=34673`,
  `cmdstat_evalsha:calls=21952`.
- The ON switch edit was local to that run and was reverted; the committed value is `[]`.
