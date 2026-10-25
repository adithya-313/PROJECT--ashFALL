# Level 12a Handoff Note

## What 12a Delivered
Level 12a establishes the foundational reactive simulation engine (World Engine II), expanding upon Level 11's deterministic schedule generator with reactive event processing capabilities:
1. **Interactive Event Loop & Runner**:
   - `world/interactive_loop.py` and `world/interactive_runner.py`: A priority-heap simulation loop executing transactions and outside interventions minute-by-minute.
   - Dual-mode operation governed by a fraud switch (`RunnerSettings.fraud_enabled`): when False, the engine transparently delegates to Level 11's `run_world`; when True, it runs interactive queuing, hold interception, and outside events.
   - Preserved true payment origin (`origin: "planned" | "fraud"`) distinct from stream execution state (`source: "planned" | "fraud" | "resumed"`), tracking resumed events in `events_resumed: list[bool]`.
   - Child random stream isolation preventing interference with baseline Level 11 draws: child 4 for fraud payment failures, child 5 for resumed payment failures.
2. **Outside Events & Encumbrance Book**:
   - `world/outside_events.py`: Explicit outside events (`lien`, `debit_hold`, `release`, `resume_payment`) validated strictly on entry.
   - `world/encumbrance.py`: Double-entry-safe encumbrance ledger tracking active liens and debit holds per account, computing available balance without moving funds or altering bank ledger entries.
3. **Dynamic Account Registration**:
   - `world/interactive_state.py`: Seam for registering new citizen or commercial accounts mid-run (`RunnerState.register_account`), updating routing tables and tracking money created for conservation audits.
4. **Run Identity & Events-v2 Serialization**:
   - `world/run_identity.py`: Deterministic 16-character hexadecimal run identifier generated from seed, effective configuration, and code version; zero-padded payment identifiers (`make_payment_id`) and dependency-injected git resolution (`find_code_version`).
   - `world/interactive_events.py`: Line serializer producing 16-field `events-v2` JSON lines appending `payment_id` and `planned_payment_id` to Level 11 event records.
5. **Issue Register v2**:
   - `core/issue_register_v2.py`: Verification harness for `config/issues_v2.yaml`, maintaining strict separation from the frozen Level 11 register (`config/issues.yaml`) while validating evidence for 93 issues (77 Revision 2 IDs and 16 continuing Level 11 IDs).
6. **Level 12 Showcase (Part A)**:
   - `showcase/level_12.py`, `showcase/level_12_checks.py`, `showcase/level_12_support.py`: Five automated showcase checks (consistency, normal, edge, edge, negative) runnable via `--level 12 --part a` and `--level 12 --quick`.

## Public Interfaces

### Functions
Extracted directly via `inspect.signature`:

```python
run_interactive_world(fast_settings: world.fast_mode.FastRunSettings, runner_settings: world.interactive_records.RunnerSettings = RunnerSettings(fraud_enabled=False), fraud_inputs: world.interactive_records.FraudInputs = FraudInputs(payments=(), outside_events=()), hook: collections.abc.Callable[[world.interactive_records.PaymentView], typing.Literal['allow', 'hold', 'decline']] = <function allow_all>, configs: world.world_configs.WorldConfigs | None = None, code_version: str | None = None, state_factory: Any = None) -> world.interactive_records.InteractiveWorldOutput
```
- Path: `world/interactive_runner.py`

```python
RunnerState.register_account(self, account_id: str, bank_index: int, person: Any = None, opening_balance_paise: int = 0, minute: int = 0) -> None
```
- Path: `world/interactive_state.py`

```python
find_code_version(env: collections.abc.Mapping[str, str], repo_root: pathlib.Path, timeout_seconds: float, *, env_var: str | None = None, unknown: str | None = None, git_executable: str = 'git', run_git: collections.abc.Callable[[str, pathlib.Path, collections.abc.Sequence[str], float], str] = <function _run_git>) -> str
```
- Path: `world/run_identity.py`

```python
make_run_id(seed: int, effective_config: collections.abc.Mapping[str, typing.Any], code_version: str, length: int) -> str
```
- Path: `world/run_identity.py`

```python
make_payment_id(run_id: str, position: int, digits: int) -> str
```
- Path: `world/run_identity.py`

```python
interactive_event_lines(output: world.interactive_records.InteractiveWorldOutput) -> list[str]
```
- Path: `world/interactive_events.py`

```python
load_v2_register(path: pathlib.Path) -> dict[str, typing.Any]
```
- Path: `core/issue_register_v2.py`

```python
check_register(register: dict[str, typing.Any], root: pathlib.Path) -> list[str]
```
- Path: `core/issue_register_v2.py`

### Records and Fields
Extracted directly via `dataclasses.fields`:

- `PaymentView`:
  - Fields: `planned_payment_id`, `minute`, `from_account`, `to_account`, `amount_paise`
  - Path: `world/interactive_records.py`
- `FraudInputs`:
  - Fields: `payments`, `outside_events`
  - Path: `world/interactive_records.py`
- `OutsideEvent`:
  - Fields: `minute`, `kind`, `account_id`, `amount_paise`, `encumbrance_id`, `planned_payment_id`
  - Path: `world/outside_events.py`
- `InteractiveRunResult`:
  - Fields: `run_result`, `payment_ids`, `planned_payment_ids`, `event_sources`, `events_resumed`, `applied_device_events`, `held_payments`, `encumbrance_log`, `run_id`, `money_created_paise`
  - Path: `world/interactive_records.py`
- `InteractiveWorldOutput`:
  - Fields: `world`, `plan`, `episodes`, `episode_report`, `device_events`, `result`, `effective_config`, `code_version`
  - Path: `world/interactive_records.py`
- `RunnerSettings`:
  - Fields: `fraud_enabled`
  - Path: `world/interactive_records.py`

### Constants
```python
OUTSIDE_EVENT_KINDS = ('lien', 'debit_hold', 'release', 'resume_payment')
```
- Path: `world/outside_events.py`

### EncumbranceBook Public Methods and Fields
- Path: `world/encumbrance.py`
- Public Methods:
  ```python
  EncumbranceBook.available_balance(self, account_id: str, balance_paise: int) -> int
  EncumbranceBook.blocks_debit(self, account_id: str, balance_paise: int, amount_paise: int) -> bool
  EncumbranceBook.place(self, event: world.outside_events.OutsideEvent, encumbrance_id: str) -> world.encumbrance.Encumbrance
  EncumbranceBook.release(self, encumbrance_id: str, minute: int) -> world.encumbrance.Encumbrance
  ```
- Public Fields:
  - `placed`: Dictionary mapping encumbrance ID to placed Encumbrance record.
  - `active_by_account`: Dictionary mapping account ID to list of active Encumbrance records.
  - `log`: List of encumbrance actions recorded as `(minute, action, encumbrance_id)`.

### Framework Part Filtering
- Path: `showcase/framework.py`
- The `--part` flag (e.g., `python -m showcase --level 12 --part a`) enables running individual parts of multi-part levels while evaluating framework check coverage across the entire level definition before filtering.

## Configuration Keys

### 1. `config/interactive_runner.yaml`
- `interactive_runner`:
  - `label`: `"policy"`
  - `reason`: Policy constraints for stream isolation, tie-breaking, and identifier conventions.
  - `issue_ids_v2`: `[K1, K3, 12-2, 12-11]`
  - `derived_from`: `"policy; run id shape matches Level 11 live-mode event_id"`
  - `random_stream_children`:
    - `fraud_payment_failures`: `4`
    - `resumed_payment_failures`: `5`
  - `random_stream_children_reason`: Explains preservation of Level 11 streams (children 0-3).
  - `tie_break_order`: `[outside_events, planned_payments, fraud_payments, resumed_payments]`
  - `tie_break_order_reason`: Ordering rationale ensuring outside events precede payments at the same minute.
  - `run_id_hex_length`: `16`
  - `position_digits`: `6`
  - `id_shape_reason`: Matching Level 11 live-mode conventions.
  - `code_version_env_var`: `ASHFALL_CODE_VERSION`
  - `code_version_git_timeout_seconds`: `5`
  - `unknown_code_version`: `"unknown"`
  - `code_version_reason`: Explains container fallback logic.

### 2. `config/showcase_level_12.yaml`
- `split_world`:
  - `label`: `"policy"`
  - `issue_ids_v2`: `[K1, K3]`
  - `derived_from`: `"seed scan, B-4: seeds 1-12 at 120 people, 20 shops, 61 days, training; seed 1 is the first with at least 3 genuine split payments (21)"`
  - `seed`: `1`, `n_people`: `120`, `n_shops`: `20`, `n_days`: `61`, `range_name`: `"training"`
  - `min_split_payments`: `3`
  - `min_split_payments_reason`: Rationale requiring split payments to be repeatedly exercised.
  - `reason`: Benchmark selection rationale.
- `showcase_run`:
  - `label`: `"policy"`
  - `code_version`: `"showcase-12a"`
  - `reason`: Fixed version for deterministic showcase identifier generation.
- `check_3_lien`:
  - `label`: `"policy"`
  - `search_start`: `1000`
  - `reason`: Ensures candidate debit occurs well into the run with genuine payment history.
- `check_4_hold`:
  - `label`: `"policy"`
  - `target_index`: `100`
  - `reason`: Early successful debit cleanly illustrating transaction isolation.

## Decisions and Why
1. **Fraud Switch Fallback (Ground Rule 13)**:
   - When `runner_settings.fraud_enabled` is False, `run_interactive` directly delegates execution to Level 11's `run_world`, guaranteeing 100% backward compatibility and bit-for-bit repeatability unless fraud is requested.
2. **Queue Key Ordering**:
   - The queue key is `(minute, tie-break rank, order within rank, insertion number)` so events and outside events at the same minute execute in a strictly deterministic sequence.
3. **Stream Draw Consumption on Hold**:
   - A held PLANNED payment still consumes its failure draws at its planned slot, so later genuine draws never shift. A held fraud or resumed payment consumes none.
4. **Encumbrance Accounting without Money Movement**:
   - Active liens and debit holds are recorded in an `EncumbranceBook` outside bank ledgers, reducing available balance (`available_balance = ledger_balance - total_liens`) without moving funds into escrow or modifying bank ledger entries, preserving double-entry money conservation rules across all banks.
5. **Dynamic Account Registration Seam**:
   - `RunnerState.register_account` opens accounts mid-simulation and tracks `money_created_paise` for accurate total money conservation audits.
6. **PaymentView Information Barrier**:
   - `PaymentView` exposes only 5 outward-facing fields (`planned_payment_id`, `minute`, `from_account`, `to_account`, `amount_paise`) without leaking simulator origin, channel, device, or internal random seeds, upholding Ground Rules 15 and 29.
7. **Frozen Immutable Containers**:
   - Simulation records (`RunnerSettings`, `FraudInputs`, `PaymentView`, `OutsideEvent`, `InteractiveRunResult`, `InteractiveWorldOutput`) are defined as frozen dataclasses to prevent unintentional state mutation across checks or pipelines.
8. **Dependency-Injected Git Runner (`run_git`)**:
   - `find_code_version` accepts an optional `run_git` callable defaulting to `_run_git`, enabling fast, deterministic unit testing of timeout and dirty-state handling without OS process spawns.
9. **Two-Level Issue Register Architecture**:
   - `config/issues_v2.yaml` holds 93 IDs = 77 Revision 2 IDs + 16 continuing Level 11 IDs, leaving `config/issues.yaml` (31 issues) strictly frozen to avoid breaking pinned Level 11 assertions.
10. **Origin vs. Source Separation**:
    - `QueuedPayment` tracks both `origin: "planned" | "fraud"` (true provenance) and `source: "planned" | "fraud" | "resumed"` (queue execution state), recording `events_resumed: list[bool]`; separating them prevents resumed payments from double-drawing from genuine streams.
11. **Split World Selected by Seed Scan**:
    - Seed 1 (120 people, 20 shops, 61 days) was chosen from a systematic scan of seeds 1–12 as the first configuration producing genuine split payments (21 splits over UPI limits), verifying payment splitting parity across switch boundaries.
12. **Issue R9 Verified Unchanged**:
    - Left R9 deferred in both `issues.yaml` and `issues_v2.yaml` without editing, preserving the cross-register link without unnecessary modifications.
13. **Check Minimums Apply Per Level**:
    - The framework enforces check coverage (at least 1 normal, 2 edge, 1 negative) across the whole level definition before filtering by `--part`; 12b needs an extra edge check and 12e needs an extra negative check to maintain full-level coverage as parts are added.
14. **Check 3 Asserts Invariant Properties Rather Than Fixed Counts**:
    - Asserts that the candidate debit fails with `funds_on_hold`, every blocked event belongs to the liened account and postdates the lien, prior events are identical, and money rules hold, ensuring test honesty if upstream world generation changes.
15. **Branch Names**:
    - Work stays on `feature/level-12a-protections-and-runner` targeting `feature/level-12-fraud-world-engine` (not `main`), keeping each level part reviewed and isolated before landing on the level integration branch.
16. **Hook Decisions Apply After the Leg's Draw**:
    - Evaluating the hook after the payment leg's random stream draw guarantees that genuine random stream progression is strictly preserved whether the bank allows or holds the payment.
17. **Events-v2 Has No Source/Fraud Field**:
    - Ground rules 15 and 29 prohibit leaking ground-truth fraud or simulator origin labels into event feeds consumed by bank models or analysts.
18. **A-012: Inflation Factor is a Lower Bound**:
    - The world fraud inflation factor represents a lower bound on transaction volume inflation, to be empirically measured and documented in Level 12g.
19. **A-013: Trivial-World Check Gate**:
    - Fails at a best single-feature AUPRC >= 0.90 on validation worlds only, requiring world generator adjustments rather than loosening evaluation thresholds.
20. **A-014: Tactic Families Division**:
    - 7 training + 4 held-out tactic families; `config/evaluation.yaml` owns the held-out list to evaluate out-of-distribution model generalization.
21. **A-015: Validation Worlds Independence**:
    - Validation worlds use training ranges and independent seeds to prevent distribution shift, with exact world counts decided based on measured runtime in 12f.
22. **A-016: Measurement Units and Economic Ranges**:
    - Money in Rs (stored as paise), time in simulated minutes, risk as a probability per use; supplier fee [0.03, 0.06] and operator share [0.20, 0.30], labelled assumption.

## Known Gaps for 12b–12g
- 12b: gang device events must join the runner's final device flush (today it runs to the last scheduled Level 11 device event)
- 12b: write the truth file set (K4/K5, rule 15) with its separate env file and mount, plus a CI import check that only evaluation/ can load truth
- 12b: add the events-v2 file writer together with the truth file set
- 12b: register the held-out family boundary parameters (dwell time, amount bands, device lifetime) BEFORE any gang world runs (A-014)
- 12b: set the risk values (A-016)
- 12b: add an edge check to the 12b showcase
- 12b: make_run_id ignores the fraud switch and fraud inputs, so two different fraud-on runs of one world share payment_ids; fix this before truth files are written
- 12c: register_account needs a mid-run trigger (an outside event or a scheduled action) for mule accounts opened during the run
- 12c: fraud payments must use a purpose from PAYMENT_PURPOSES; decide which purposes gangs use
- 12d: the gang reaction interface (observation feed) stays outside the hook; the hook stays payment -> decision
- 12e: the simulated analyst must be built as a pure function (Level 27 measures triage lift against it); add a negative check to the 12e showcase
- 12f: commit world counts per split from measured run time (A-015) and enforce seed disjointness
- 12g: run the trivial-world check on validation worlds (A-013); put the measured inflation factor on fraud-on world cards with its lower-bound note (A-012)
- Containers: world runs inside containers need ASHFALL_CODE_VERSION set, or the run id records "unknown"

## Exact Verification Commands
```powershell
# 1. Full unit suite (excluding external dependencies)
python -m pytest tests/unit -q -p no:cacheprovider -m "not needs_stack and not gpu and not ml and not needs_dvc"

# 2. Targeted unit tests covering Level 12a components (using real test names)
python -m pytest -v -p no:cacheprovider tests/unit/showcase/test_level_12_checks.py tests/unit/world/test_frozen_anchors.py tests/unit/world/test_interactive_runner_fraud_on.py tests/unit/world/test_run_identity.py tests/unit/world/test_new_code_comment_standard.py tests/unit/test_issue_register_v2.py

# 3. Showcase Part A execution
python -m showcase --level 12 --part a

# 4. Showcase Level 12 quick mode (whole level)
python -m showcase --level 12 --quick

# 5. Baseline Level 11 showcase invariance
python -m showcase --level 11 --quick

# 6. Issue register validation
python -m core.issue_register_v2
python -m core.issue_register

# 7. YAML configuration schema validation
python -m core.config_validator config/showcase_level_12.yaml config/interactive_runner.yaml

# 8. Linting and static type checking
ruff check showcase api core
mypy showcase api core --ignore-missing-imports --no-strict-optional

# 9. Structure and import hygiene checks
python -m core.folder_check
python -m core.imports_check

# 10. Level 10 pre-registration proof
python -c "from core.level10_check import preregistration_proof as p; print(p()['result'])"

# 11. Git status and commit history
git log --oneline b5fab81..HEAD
git status --porcelain
```
