# Level 12b Handoff Note

## What 12b Delivered
Level 12b implements hidden ground truth, the procedural tactic language core, external gang devices, and the Part-B showcase checks (World Engine II Part B):
1. **Hidden Truth Store & Tamper Resistance**:
   - `world/truth_records.py`: Immutable truth dataclasses (`RoleSpan`, `DeviceControlSpan`, `PaymentStory`, `TruthFacts`) tracking account roles, device compromises, and attack narratives across half-open temporal validity windows `[valid_from, valid_to)`. Validates against overlapping roles or controller spans, negative times, invalid role names, and negative or zero mule depths.
   - `world/truth_writer.py`: Serializer generating the four `truth-v1` artifacts (`roles.jsonl`, `device_control.jsonl`, `payment_stories.jsonl`, `manifest.json`) in run directories under `ASHFALL_TRUTH_DIR`. Strictly refuses to overwrite existing truth artifacts.
   - `evaluation/truth.py`: Evaluation loader (`load_truth`, `TruthStore`) verifying SHA-256 checksums against `manifest.json` before loading, and inspecting the caller stack to permit only `evaluation/`, `showcase/`, and `tests/` callers while refusing production operational packages (`api`, `serving`, `features`, `models`, `online_learning`, `agent`, `core`).
   - `core/truth_import_check.py`: Static AST import guard verifying that no unauthorized Python file imports truth modules or references `ASHFALL_TRUTH_DIR`.
   - `world/interactive_events.py`: Events-v2 writer (`write_events_v2_file`) serializing complete simulation event streams to `events-v2.jsonl`.
2. **Procedural Tactic Language & Gang Planner**:
   - `world/gang_config.py`: Structured gang configuration (`GangConfig`, `GangBudget`, `FixtureSource`, `BlockStep`) with serialization (`gang_config_to_plain`, `gang_config_from_plain`) and strict validation (`validate_gang_config`).
   - `world/tactic_blocks.py`: Catalogued building blocks (`acquire`, `take_control`, `warm_up`, `route`, `disguise`) with strengths (strength 1 = fixed, strength 2 = camouflaged) and dividing lines separating training from held-out tactics.
   - `world/gang_planner.py`: Compiler (`plan_gang`) turning gang specifications into planned payments, truth facts, device events, and multi-dimensional cost summaries (`BlockCost` covering money, time, and cumulative risk). Enforces gang budgets, raising `GangBudgetExceeded(ValueError)` if any limit is breached.
3. **External Gang Devices & Compromise Events**:
   - `world/gang_devices.py`: External device generator (`make_gang_devices`) provisioning devices with neutral identifiers (`dev_ext_` + 6 digits) and standard fingerprints via child stream 6 (`gang_devices`). Round-robin device assignment generates `DeviceControlSpan` truth and `GangDeviceEvent` compromise events (`device_acquired`, `sim_swap`, `linked_number_switch`, `emulator_enabled`, `remote_access_enabled`).
   - `world/interactive_runner.py`: Runner integration queuing gang device events under new tie-break kind `gang_device_events` appended to `tie_break_order`. Added `device_event_stream` merging Level 11 device events and gang device events.
4. **Deterministic Run Identity & Child Stream Isolation**:
   - `world/run_identity.py` and `world/interactive_runner.py`: Deterministic run identity hashes `gang_configs` plain-data into `effective_config["gang_configs"]`, distinguishing different gang setups while leaving runs without gang configs bit-for-bit identical to Level 12a.
   - Stream isolation: child stream 6 for gang devices and child stream 7 for gang planning, preventing random draw shifts across simulation components.
5. **Issue Register v2 Progress**:
   - `config/issues_v2.yaml`: Updated 12-1 to `built`, 12-17 to `partial`, K5 to `partial`, and added dividing-line evidence to 12-8 (`built: 14, partial: 9, planned: 69, deferred: 1`).
6. **Level 12 Showcase (Part B & Complete 10-Check Suite)**:
   - `showcase/level_12_checks_b.py`, `showcase/level_12_support_b.py`, `showcase/level_12.py`: Added checks 6-10 covering gang budget enforcement, caller-based truth isolation, disguise camouflaging, merged device event streaming, and truth reconstruction.
   - `tests/unit/showcase/test_level_12_checks.py`: Pinned the total showcase check list to exactly 10 checks (first 5 Part A, last 5 Part B).

## Public Interfaces

### Functions
Extracted directly via `inspect.signature`:

```python
write_truth_files(output: world.interactive_records.InteractiveWorldOutput, truth_root: pathlib.Path) -> pathlib.Path
```
- Path: `world/truth_writer.py`

```python
truth_root_from_env(env: collections.abc.Mapping[str, str]) -> pathlib.Path
```
- Path: `world/truth_writer.py`

```python
write_events_v2_file(output: world.interactive_records.InteractiveWorldOutput, path: pathlib.Path) -> str
```
- Path: `world/interactive_events.py`

```python
load_truth(truth_root: pathlib.Path, run_id: str) -> evaluation.truth.TruthStore
```
- Path: `evaluation/truth.py`

```python
TruthStore.role_at(self, account_id: str, minute: int) -> str
```
- Path: `evaluation/truth.py`

```python
TruthStore.owner_at(self, account_id: str, minute: int) -> str | None
```
- Path: `evaluation/truth.py`

```python
TruthStore.depth_at(self, account_id: str, minute: int) -> int | None
```
- Path: `evaluation/truth.py`

```python
TruthStore.story_of(self, payment_id: str) -> dict[str, typing.Any]
```
- Path: `evaluation/truth.py`

```python
TruthStore.controllers_of(self, account_id: str, minute: int) -> list[str]
```
- Path: `evaluation/truth.py`

```python
find_truth_violations(repo_root: pathlib.Path = WindowsPath('C:/Users/sinne/Downloads/PROJECT-ashFALL')) -> list[str]
```
- Path: `core/truth_import_check.py`

```python
gang_config_to_plain(config: world.gang_config.GangConfig) -> dict[str, typing.Any]
```
- Path: `world/gang_config.py`

```python
gang_config_from_plain(data: dict[str, typing.Any]) -> world.gang_config.GangConfig
```
- Path: `world/gang_config.py`

```python
validate_gang_config(config: world.gang_config.GangConfig) -> None
```
- Path: `world/gang_config.py`

```python
plan_gang(config: world.gang_config.GangConfig, world: world.build_world.World, rng: numpy.random._generator.Generator, reserved_accounts: set[str] | None = None, gang_devices_rng: numpy.random._generator.Generator | None = None) -> world.gang_planner.GangPlan
```
- Path: `world/gang_planner.py`

```python
make_gang_devices(count: int, rng: numpy.random._generator.Generator, start_index: int = 1, prefix: str | None = None) -> list[world.devices.Device]
```
- Path: `world/gang_devices.py`

```python
device_event_stream(result: world.interactive_records.InteractiveRunResult) -> list[dict[str, typing.Any]]
```
- Path: `world/interactive_runner.py`

```python
run_interactive_world(fast_settings: world.fast_mode.FastRunSettings, runner_settings: world.interactive_records.RunnerSettings = RunnerSettings(fraud_enabled=False), fraud_inputs: world.interactive_records.FraudInputs = FraudInputs(payments=(), outside_events=(), truth=TruthFacts(roles=(), device_control=(), payment_stories=()), gang_device_events=()), hook: collections.abc.Callable[[world.interactive_records.PaymentView], typing.Literal['allow', 'hold', 'decline']] = <function allow_all at 0x...>, configs: world.world_configs.WorldConfigs | None = None, code_version: str | None = None, state_factory: Any = None, gang_configs: tuple[world.gang_config.GangConfig, ...] = ()) -> world.interactive_records.InteractiveWorldOutput
```
- Path: `world/interactive_runner.py`

### Records and Fields
Extracted directly via `dataclasses.fields`:

- `RoleSpan`:
  - Fields: `account_id: str`, `role: str`, `gang_id: str`, `depth: int | None`, `valid_from: int`, `valid_to: int | None`
  - Path: `world/truth_records.py`
- `DeviceControlSpan`:
  - Fields: `device_id: str`, `account_id: str`, `gang_id: str`, `valid_from: int`, `valid_to: int | None`, `sim_swap: bool`, `linked_number_switch: bool`, `emulator: bool`, `remote_access: bool`
  - Path: `world/truth_records.py`
- `PaymentStory`:
  - Fields: `gang_id: str`, `block: str`, `layer: int | None`, `description: str`
  - Path: `world/truth_records.py`
- `TruthFacts`:
  - Fields: `roles: tuple[world.truth_records.RoleSpan, ...]`, `device_control: tuple[world.truth_records.DeviceControlSpan, ...]`, `payment_stories: tuple[world.truth_records.PaymentStory, ...]`
  - Path: `world/truth_records.py`
- `GangBudget`:
  - Fields: `money_paise: int`, `time_minutes: int`, `max_risk: float`
  - Path: `world/gang_config.py`
- `FixtureSource`:
  - Fields: `account_id: str`, `bank_index: int`, `opening_balance_paise: int`, `start_minute: int`
  - Path: `world/gang_config.py`
- `BlockStep`:
  - Fields: `block: str`, `params: tuple[tuple[str, typing.Any], ...]`
  - Path: `world/gang_config.py`
- `GangConfig`:
  - Fields: `gang_id: str`, `strength: int`, `budget: world.gang_config.GangBudget`, `source: world.gang_config.FixtureSource`, `steps: tuple[world.gang_config.BlockStep, ...]`
  - Path: `world/gang_config.py`
- `BlockCost`:
  - Fields: `money_paise: int`, `time_minutes: int`, `risk: float`
  - Path: `world/gang_planner.py`
- `GangPlan`:
  - Fields: `payments: tuple[world.calendar_runner.PlannedPayment, ...]`, `truth: world.truth_records.TruthFacts`, `costs: tuple[world.gang_planner.BlockCost, ...]`, `totals: world.gang_planner.BlockCost`, `gang_device_events: tuple[world.gang_devices.GangDeviceEvent, ...]`
  - Path: `world/gang_planner.py`
- `GangDeviceEvent`:
  - Fields: `minute: int`, `device_id: str`, `account_id: str | None`, `kind: str`, `platform: str`, `os_generation: int`, `fingerprint: str`
  - Path: `world/gang_devices.py`

### Exceptions
- `GangBudgetExceeded(ValueError)`:
  - Raised when planned gang operations exceed any budget dimension (money, time, or risk), with an error message detailing the exceeded limit, the planned total, and the step that caused the violation.
  - Path: `world/gang_planner.py`

## Configuration Keys

### 1. `config/truth_store.yaml`
- `truth_store`:
  - `label`: `"policy"`
  - `reason`: Truth store records ground truth about illicit accounts, devices, and attack narratives separately from operational feeds so that only evaluation and benchmark tooling can inspect them.
  - `issue_ids_v2`: `[12-1, 12-17, K5]`
  - `derived_from`: `"ground rule 15; Level 12b brief task 5"`
  - `format`: `"truth-v1"`
  - `file_names`:
    - `roles`: `"roles.jsonl"`
    - `device_control`: `"device_control.jsonl"`
    - `payment_stories`: `"payment_stories.jsonl"`
    - `manifest`: `"manifest.json"`
  - `roles`:
    - `mule`: `"An intermediary account that receives and forwards laundered funds across payment hops"`
    - `collector`: `"An account aggregating stolen proceeds from multiple victims before laundering"`
    - `cash_out`: `"A terminal account used to withdraw illicit funds to cash or external instruments"`
    - `victim`: `"An ordinary target account deceived or compromised into initiating unauthorized payments"`
  - `env_var`: `ASHFALL_TRUTH_DIR`
  - `allowed_reader_folders`: `[evaluation, showcase, tests]`
  - `ashfall_packages`: `[api, serving, features, models, online_learning, agent, core]`

### 2. `config/tactic_blocks.yaml`
- `tactic_blocks`:
  - `label`: `"policy"`
  - `reason`: Catalogue of procedural building blocks used to specify gang laundering operations and attack tactics.
  - `issue_ids_v2`: `[12-8, 12-9]`
  - `derived_from`: `"amendment A-014; amendment A-016"`
  - `blocks`:
    - `acquire`: allowed strengths `[1, 2]`, `time_minutes_per_use: [60, 1440]`, parameter `count` references `world_ranges.yaml::mules_per_gang`.
    - `take_control`: allowed strengths `[1, 2]`, `time_minutes_per_use: [10, 120]`, parameters `devices: [1, 5]`, `sim_swap: bool`, `linked_number_switch: bool`, `emulator: bool`, `remote_access: bool`.
    - `warm_up`: allowed strengths `[1, 2]`, `time_minutes_per_use: "days x 1440"`, parameter `days` references `world_ranges.yaml::mule_warmup_days`.
    - `route`: allowed strengths `[1, 2]`, `time_minutes_per_use: "hop dwell"`, parameters `layers: [1, 5]`, `accounts_per_layer: [1, 10]`, `dwell_minutes` references `world_ranges.yaml::minutes_money_stays_in_mule_account`, `amount_inr` references `world_ranges.yaml::mule_transfer_amount_inr`.
    - `disguise`: allowed strengths `[2]`, `time_minutes_per_use: [0, 30]`, parameter `jitter_minutes: [0, 30]`.
- `held_out_dividing_lines`:
  - `label`: `"policy"`, `issue_ids_v2`: `[12-8]`, `derived_from`: `"amendment A-014; config/world_ranges.yaml"`
  - `training_max_dwell_minutes`: references upper bound of `minutes_money_stays_in_mule_account`.
  - `slow_drip_max_amount_inr`: references lower bound of `mule_transfer_amount_inr`.
  - `training_devices_retired_on_schedule`: `false`.
- `gang_payment_purpose`:
  - `label`: `"policy"`, `purpose`: `everyday`, `reason`: real scam purposes decided in Level 12c.
- `gang_devices`:
  - `label`: `"policy"`, `device_id_prefix`: `"dev_ext_"`, `device_event_kinds`: `[device_acquired, sim_swap, linked_number_switch, emulator_enabled, remote_access_enabled]`.

### 3. `config/tactic_costs.yaml` (`tactic_risk_per_use` block)
- `tactic_risk_per_use`:
  - `label`: `"assumption"`
  - `reason`: `"No published figure; it exists only to compare tactics, as amendment A-016 says."`
  - `issue_ids_v2`: `[12-9]`
  - `derived_from`: `"amendment A-016"`
  - `unit`: `probability_per_use`
  - `acquire`: `range: [0.001, 0.01]`
  - `warm_up`: `range: [0.0, 0.001]`
  - `route`: `range: [0.001, 0.02]`
  - `disguise`: `range: [0.0, 0.005]`
  - `take_control`: `range: [0.001, 0.01]`

### 4. `config/interactive_runner.yaml`
- `random_stream_children`:
  - `fraud_payment_failures`: `4`
  - `resumed_payment_failures`: `5`
  - `gang_devices`: `6` (child 6 reserved for gang device creation in B-2b)
  - `gang_planning`: `7` (child 7 provides planning randomness for gangs per D6)
- `tie_break_order`:
  - `[outside_events, planned_payments, fraud_payments, resumed_payments, gang_device_events]` (gang device events appended at the end so existing ranks never change).

## Decisions and Why

### B-1 Conflicts
1. **Conflict 1 (Run Identity Fix Timing)**:
   - The run-id fix was moved from B-1 to B-2. Because a `resume_payment` outside event names a payment by an ID that already contains the `run_id`, the `run_id` cannot be a hash of raw simulation inputs (which would require an input to contain its own cryptographic hash). Instead, B-2 incorporates gang configurations into `effective_config["gang_configs"]`. Until then, the truth writer refused to write into a run directory containing truth files, preventing accidental overwrites.
2. **Conflict 2 (Truth Environment Separation without Containers)**:
   - Production world and evaluation containers do not yet exist in this phase. Environment separation was implemented via `envs/truth.env.example` and verified with AST checks proving `api` and `outbox-relay` cannot access the truth environment variable. Mounting into containers is retained as a known gap (K5).
3. **Conflict 3 (Config-Driven Allowed Truth Readers)**:
   - Allowed truth readers (`evaluation/`, `showcase/`, `tests/`) and prohibited packages (`api`, `serving`, `features`, `models`, `online_learning`, `agent`, `core`) are defined in `config/truth_store.yaml` rather than hardcoded in Python modules.

### D1–D8 (Tactic Language Core & Gang Budgets)
- **D1 (Registered Dividing Lines)**: Dividing lines separating training tactics from held-out tactics are registered in `config/tactic_blocks.yaml` as references to bounds in `config/world_ranges.yaml`, never hardcoded numbers. A resolving function resolves them dynamically.
- **D2 (Per-Block Parameter Ranges & Assumptions)**: Block execution durations live in `tactic_blocks.yaml`. Probability of detection per use lives in `config/tactic_costs.yaml` under `tactic_risk_per_use` labelled `assumption` per amendment A-016.
- **D3 (Fixture Collector Account)**: Illicit money flows from a designated fixture collector account registered before the simulation starts (via `state_factory`), with role `"collector"` in ground truth.
- **D4 (Gang Payment Purpose)**: Gang payments use purpose `"everyday"` from `config/tactic_blocks.yaml`, pending specific scam typologies in Level 12c.
- **D5 (Run Identity Names Gang Configs)**: When gang configurations are supplied, `effective_config` gains `"gang_configs"` formatted as canonical plain-data dictionaries, altering the run ID deterministically. Runs without gangs preserve exact Level 12a run IDs.
- **D6 (Planning Random Stream Isolation)**: Gang planning draws exclusively from child stream 7 (`gang_planning`). Gang device generation draws exclusively from child stream 6 (`gang_devices`). Neither shifts the simulation run stream.
- **D7 (Sophistication Strengths)**: Strength 1 ("fixed") draws transfer amounts and dwell once per gang and reuses them across hops, rounding amounts down to whole Rs 1,000. Strength 2 ("camouflage") draws per-hop amounts and dwell, permitting the `disguise` block to jitter payments within dwell windows.
- **D8 (Cost Calculation and Multi-Dimensional Budget Enforcement)**: Money cost equals `account_supplier_fee_share` (drawn once per gang) times total money routed through acquired accounts in paise. Time cost is the sum of block durations. Risk is cumulative probability: $1 - \prod (1 - r_i)$. Exceeding any budget dimension raises `GangBudgetExceeded(ValueError)` naming the limit, planned total, and offending step.

### E1–E8 (Take Control & Gang Devices)
- **E1 (`take_control` Block Structure)**: Must follow `acquire`. Configures device counts (`[1, 5]`) and boolean compromise flags (`sim_swap`, `linked_number_switch`, `emulator`, `remote_access`). Acquired accounts are assigned devices round-robin.
- **E2 (Neutral Device Identifiers)**: Gang devices use neutral prefix `dev_ext_` with 6 digits from `config/tactic_blocks.yaml` so IDs do not leak illicit roles, following Level 11 fingerprinting conventions.
- **E3 (Device Control Truth Spans)**: Produces one `DeviceControlSpan` per (device, account) with `valid_to: None` carrying the configured compromise flags.
- **E4 (Gang Device Events)**: Emits one `device_acquired` event per provisioned device, followed by one event per active boolean compromise flag for each controlled account at the takeover minute.
- **E5 (Runner Device Stream Integration)**: Queued under `gang_device_events` at the end of `tie_break_order`. Output records `applied_gang_device_events`. `device_event_stream(result)` merges Level 11 and gang events in minute and rank order without role labels.
- **E6 (No Premature Device Retirement)**: Training dividing line `training_devices_retired_on_schedule: false` forbids retirement or rotation parameters in training `take_control` blocks; unknown parameters are rejected.
- **E7 (Payment Device Stamping Gap)**: Fraud payments currently retain the victim/owner phone device; stamping with gang devices is deferred to Level 12d.
- **E8 (Take Control Assumptions)**: Duration `[10, 120]` minutes and risk `[0.001, 0.01]` labelled `assumption`.

### Additional Architecture Decisions
- **Option-B Risk Block**: Added `tactic_risk_per_use` as a top-level block in `config/tactic_costs.yaml` (sibling of `tactic_cost_units`), keeping `tests/unit/test_tactic_families.py` frozen and avoiding schema collisions.
- **Run ID Gang Config Hashing vs. Resume-ID Circularity**: Incorporating plain gang configurations into `effective_config` guarantees deterministic, collision-free run IDs for gang campaigns without requiring inputs to self-reference circular hashes.

## Known Gaps
- 12c: route forwards planned amounts; if a receipt fails, the forward would use the owner's own money. Gangs must forward at most what arrived.
- 12c: acquire uses rented accounts only; borrowed-KYC and business accounts need register_account mid-run.
- 12c: victims, cash-out, scam purposes (gang payments use "everyday" for now).
- 12c/12d: fraud payments are stamped with the owner's phone, not the gang device.
- 12c/12d + Level 14: gang device ids ("dev_ext_") differ from genuine ("dev_person_"), and sim_swap / linked_number_switch / emulator / remote_access events exist only for gangs. Before Level 14 exposes account events, genuine people need these events at a background rate, and device ids must be projected so no prefix reveals a gang.
- 12d: held-out family generation and enforcement beyond the registered dividing lines.
- route layers [1, 5] and accounts_per_layer [1, 10] are unsourced assumptions chosen during B-2a.
- the run id names world config + gang configs; hand-made FraudInputs are not in it (the truth writer refuses to overwrite, so nothing is lost).
- world/evaluation containers do not exist; the truth env mount is pending (K5).
- from 12a, still open: the analyst as a pure function plus the 12e negative check (12e); world counts and seed disjointness (12f); the trivial-world check and inflation factor (12g); ASHFALL_CODE_VERSION in containers.

## Exact Verification Commands
```powershell
# 1. Golden checks verification
python -B scripts/verify_world.py

# 2. Full unit suite (excluding external dependencies)
python -m pytest tests/unit -q -p no:cacheprovider -m "not needs_stack and not gpu and not ml and not needs_dvc"

# 3. Targeted unit tests covering Level 12b components
python -m pytest -v -p no:cacheprovider tests/unit/world/test_truth_records.py tests/unit/world/test_truth_writer.py tests/unit/evaluation/test_truth_loader.py tests/unit/world/test_tactic_config.py tests/unit/world/test_gang_config.py tests/unit/world/test_gang_planner.py tests/unit/world/test_gang_devices.py tests/unit/world/test_gang_device_stream.py tests/unit/world/test_gang_runs.py tests/unit/showcase/test_level_12_checks_b.py tests/unit/showcase/test_level_12_checks.py tests/unit/test_truth_env_separation.py tests/unit/test_issue_register_v2.py

# 4. Showcase Part B execution (5/5 checks)
python -m showcase --level 12 --part b

# 5. Showcase Part A execution (5/5 checks)
python -m showcase --level 12 --part a

# 6. Showcase Level 12 quick mode (all 10 checks)
python -m showcase --level 12 --quick

# 7. Baseline Level 11 showcase invariance (16/16 checks)
python -m showcase --level 11 --quick

# 8. Issue register v2 validation
python -m core.issue_register_v2
python -m core.issue_register

# 9. Truth isolation static import check
python -m core.truth_import_check

# 10. YAML configuration schema validation
python -m core.config_validator config/truth_store.yaml config/tactic_blocks.yaml config/tactic_costs.yaml config/interactive_runner.yaml config/showcase_level_12.yaml

# 11. Code comment and standard validation
python -m pytest tests/unit/world/test_new_code_comment_standard.py

# 12. Structure and import hygiene checks
python -m core.folder_check
python -m core.imports_check

# 13. Git status and commit history
git log --oneline 1b6709a..HEAD
git status --porcelain
```
