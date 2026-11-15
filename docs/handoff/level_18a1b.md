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

## Known gaps and deferrals
- Parallel worlds are not implemented; worlds still run one at a time. That is the next speed
  lever, after `decision_post` and `snapshot_write` replay timings are measured.
- `asyncpg.exceptions.InterfaceError` ("connection is closed", seen in the Level 15 flake) is not
  retried, deliberately, because `InterfaceError` also covers programming errors; revisit with
  evidence that it was raised by a lost connection.
- The laptop check covered one world; a multi-world resume was not run in this round.

## How to verify
- `PYTHONPATH=. py -3.12 -B scripts/verify_world.py` -> ALL GOLDEN CHECKS PASSED.
- `py -3.12 -m pytest -q -p no:cacheprovider tests/unit`
- `py -3.12 -m showcase --level 16` and `py -3.12 -m showcase --level 17`
- `py -3.12 -m core.config_validator config/dataset_build.yaml`
- Resume check: build the test split with `--checkpoint-dir`, then run the same command again;
  the second run prints the resumed line and writes the same dataset id.

## Generated facts

## Register Changes
- 16-4: Full-scale dataset building and a faster replay drain (status: partial)

---

## Public Interfaces

Extracted directly via `inspect.signature`:

### `scripts/build_level_16_datasets.py`

```python
build_all_level_16_datasets(splits: collections.abc.Sequence[str] | None = None, preset_name: str = 'tiny', datasets_root: pathlib.Path = WindowsPath('C:/Users/sinne/Downloads/PROJECT-ashFALL/data/datasets'), sampler_seed: int = 1, write_evaluation_inputs: bool = False, world_cards_root: pathlib.Path = WindowsPath('C:/Users/sinne/Downloads/PROJECT-ashFALL/runs/world_cards'), replay_profile_dir: pathlib.Path | None = None, worlds_per_split: dict[str, int] | None = None, checkpoint_dir: pathlib.Path | None = None) -> dict[str, dict[str, typing.Any]]
```

```python
build_split_dataset(split: str, specs: list[world.world_sampler.WorldSpec], preset_name: str, sampler_cfg: dict[str, typing.Any], code_version: str, sampler_seed: int = 1, datasets_root: pathlib.Path = WindowsPath('C:/Users/sinne/Downloads/PROJECT-ashFALL/data/datasets'), admin_pg: str | None = None, base_redis_url: str | None = None, write_evaluation_inputs: bool = False, truth_root: pathlib.Path | None = None, world_cards_root: pathlib.Path = WindowsPath('C:/Users/sinne/Downloads/PROJECT-ashFALL/runs/world_cards'), replay_profile_dir: pathlib.Path | None = None, checkpoint_dir: pathlib.Path | None = None) -> tuple[replay.dataset_builder.Dataset, pathlib.Path, list[pathlib.Path]]
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
replay_world_with_retry(recorded: world.replay_recording.RecordedWorld, admin_pg: str, base_redis_url: str, *, config: dict[str, typing.Any], profile_dir: pathlib.Path | None = None, sleep: collections.abc.Callable[[float], collections.abc.Awaitable[None]] = <function sleep at 0x000001237B5974C0>) -> replay.dataset_builder.WorldExtract
```

```python
save_evaluation_inputs(recorded: world.replay_recording.RecordedWorld, truth_root: pathlib.Path, world_cards_root: pathlib.Path) -> dict[str, typing.Any]
```

```python
summarize_labels_by_source_and_via(dataset: replay.dataset_builder.Dataset) -> dict[str, int]
```

```python
supersede_prior_datasets(split: str, new_dir: pathlib.Path, datasets_root: pathlib.Path = WindowsPath('C:/Users/sinne/Downloads/PROJECT-ashFALL/data/datasets')) -> list[pathlib.Path]
```

```python
world_end_for_recorded(recorded: world.replay_recording.RecordedWorld) -> datetime.datetime
```

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

---

## Records and Fields

Extracted directly via `dataclasses.fields`:

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
