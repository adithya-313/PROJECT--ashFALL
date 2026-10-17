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
   - `world/encumbrance.py`: Double-entry-safe encumbrance ledger tracking active liens and debit holds per account, computing spendable balance without moving funds or altering bank ledger entries.
3. **Dynamic Account Registration**:
   - `world/interactive_state.py`: Seam for registering new citizen or commercial accounts mid-run (`RunnerState.register_account`), updating routing tables and tracking money created for conservation audits.
4. **Run Identity & Events-v2 Serialization**:
   - `world/run_identity.py`: Deterministic 16-character hexadecimal run identifier generated from seed, effective configuration, and code version; zero-padded payment identifiers (`make_payment_id`) and dependency-injected git resolution (`find_code_version`).
   - `world/interactive_events.py`: Line serializer producing 16-field `events-v2` JSON lines appending `payment_id` and `planned_payment_id` to Level 11 event records.
5. **Issue Register v2**:
   - `core/issue_register_v2.py`: Verification harness for `config/issues_v2.yaml`, maintaining strict separation from the frozen Level 11 register (`config/issues.yaml`) while validating evidence for 93 issues.
6. **Level 12 Showcase (Part A)**:
   - `showcase/level_12.py`, `showcase/level_12_checks.py`, `showcase/level_12_support.py`: Five automated showcase checks (consistency, normal, edge, edge, negative) runnable via `--level 12 --part a` and `--level 12 --quick`.

## Public Interfaces

### 1. `run_interactive_world`
- **Path**: `world/interactive_runner.py`
- **Signature**:
  ```python
  def run_interactive_world(
      fast_settings: FastRunSettings,
      runner_settings: RunnerSettings = RunnerSettings(),
      fraud_inputs: FraudInputs = FraudInputs(),
      hook: Callable[[PaymentView], HookDecision] = allow_all,
      configs: WorldConfigs | None = None,
      code_version: str | None = None,
      state_factory: Any = None,
  ) -> InteractiveWorldOutput
  ```

### 2. `RunnerSettings`
- **Path**: `world/interactive_records.py`
- **Signature**:
  ```python
  @dataclass(frozen=True)
  class RunnerSettings:
      fraud_enabled: bool = False
  ```

### 3. `FraudInputs`
- **Path**: `world/interactive_records.py`
- **Signature**:
  ```python
  @dataclass(frozen=True)
  class FraudInputs:
      fraud_payments: Sequence[PlannedPayment] = ()
      outside_events: Sequence[OutsideEvent] = ()
  ```

### 4. `PaymentView`
- **Path**: `world/interactive_records.py`
- **Signature**:
  ```python
  @dataclass(frozen=True)
  class PaymentView:
      planned_payment_id: str
      minute: int
      from_account: str
      to_account: str
      amount_paise: int
      channel: str
      device_id: str | None
      bank_id: str
  ```

### 5. `allow_all`
- **Path**: `world/interactive_records.py`
- **Signature**:
  ```python
  def allow_all(payment: PaymentView) -> HookDecision:
      return "allow"
  ```

### 6. `InteractiveRunResult`
- **Path**: `world/interactive_records.py`
- **Signature**:
  ```python
  @dataclass(frozen=True)
  class InteractiveRunResult:
      run_result: RunResult
      payment_ids: list[str]
      planned_payment_ids: list[str]
      event_sources: list[str]
      events_resumed: list[bool]
      device_events: list[DeviceEvent]
      held_payments: list[QueuedPayment]
      encumbrance_log: list[tuple[int, str, str]]
  ```

### 7. `InteractiveWorldOutput`
- **Path**: `world/interactive_records.py`
- **Signature**:
  ```python
  @dataclass(frozen=True)
  class InteractiveWorldOutput:
      world: World
      plan: WorldPlan
      episodes: list[Episode]
      episode_report: dict[str, Any]
      device_events: list[DeviceEvent]
      result: InteractiveRunResult
      effective_config: dict[str, Any]
      code_version: str
  ```

### 8. `OutsideEvent` Kinds
- **Path**: `world/outside_events.py`
- **Kinds**: `("lien", "debit_hold", "release", "resume_payment")`
- **Record**:
  ```python
  @dataclass(frozen=True)
  class OutsideEvent:
      minute: int
      kind: str
      account_id: str | None = None
      amount_paise: int = 0
      encumbrance_id: str | None = None
      planned_payment_id: str | None = None
  ```

### 9. `EncumbranceBook` Methods
- **Path**: `world/encumbrance.py`
- **Methods**:
  - `place(event: OutsideEvent, fallback_id: str | None = None) -> Encumbrance`
  - `release(event: OutsideEvent) -> Encumbrance`
  - `is_debit_blocked(account_id: str, balance_paise: int, debit_amount_paise: int) -> tuple[bool, str]`
  - `total_liens_for(account_id: str) -> int`
  - `has_debit_hold(account_id: str) -> bool`
  - `active_for(account_id: str) -> list[Encumbrance]`
  - `active_ids() -> set[str]`

### 10. `RunnerState.register_account`
- **Path**: `world/interactive_state.py`
- **Signature**:
  ```python
  def register_account(
      self,
      account_id: str,
      bank_index: int,
      person: Any = None,
      opening_balance_paise: int = 0,
      minute: int = 0,
  ) -> None
  ```

### 11. `find_code_version`
- **Path**: `world/run_identity.py`
- **Signature**:
  ```python
  def find_code_version(
      env: Mapping[str, str],
      repo_root: Path,
      timeout_seconds: float,
      *,
      env_var: str | None = None,
      unknown: str | None = None,
      git_executable: str = "git",
      run_git: Callable[[str, Path, Sequence[str], float], str] = _run_git,
  ) -> str
  ```

### 12. `make_run_id` and `make_payment_id`
- **Path**: `world/run_identity.py`
- **Signatures**:
  ```python
  def make_run_id(seed: int, effective_config: Mapping[str, Any], code_version: str, length: int) -> str: ...
  def make_payment_id(run_id: str, position: int, digits: int) -> str: ...
  ```

### 13. `interactive_event_lines`
- **Path**: `world/interactive_events.py`
- **Signature**:
  ```python
  def interactive_event_lines(output: InteractiveWorldOutput) -> list[str]: ...
  ```

### 14. `load_v2_register` and `check_register`
- **Path**: `core/issue_register_v2.py`
- **Signatures**:
  ```python
  def load_v2_register(path: Path) -> dict[str, Any]: ...
  def check_register(register: dict[str, Any], root: Path) -> list[str]: ...
  ```

### 15. The `--part` Flag
- **Path**: `showcase/framework.py`
- **Usage**:
  ```powershell
  python -m showcase --level 12 --part a
  ```
  Enables running individual parts of multi-part levels while evaluating framework check coverage across the entire level definition.

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
   - *Decision*: When `runner_settings.fraud_enabled` is False, `run_interactive` directly delegates execution to Level 11's `run_world`.
   - *Why*: Guarantees zero regression risk and exact byte-for-byte equivalence for callers requesting standard baseline runs without fraud mechanics.
2. **Priority Heap Ordering**:
   - *Decision*: Items are queued in a min-heap ordered by `(minute, tie_break_rank, sequence)`.
   - *Why*: Provides deterministic, single-pass interleaved execution of payments and outside events without maintaining multiple synchronized clocks.
3. **Three-Way Hook Decision Interface**:
   - *Decision*: Bank callbacks return `"allow"`, `"hold"`, or `"decline"`.
   - *Why*: Captures realistic transactional authorization outcomes. Holds freeze execution cleanly without advancing random streams or modifying account balances until resumed.
4. **Encumbrance Accounting without Money Movement**:
   - *Decision*: Active liens and debit holds are recorded in an `EncumbranceBook` outside bank ledgers.
   - *Why*: Encumbrances reduce spendable funds (`available_balance = ledger_balance - total_liens`) without creating synthetic escrow transfers, preserving double-entry money conservation rules across all banks.
5. **Dynamic Account Registration Seam**:
   - *Decision*: Provided `RunnerState.register_account` to open accounts mid-simulation and track `money_created_paise`.
   - *Why*: Allows future levels (12c, 12d) to introduce new mule and genuine accounts dynamically while preserving accurate total money accounting.
6. **PaymentView Information Barrier**:
   - *Decision*: Hooks inspect `PaymentView` records providing amount, minute, accounts, channel, device, and bank id without exposing simulator origin or internal random seeds.
   - *Why*: Prevents bank decision logic or fraud detection models from peeking at ground truth simulator state (enforcing Ground Rules 15 and 29).
7. **Frozen Immutable Containers**:
   - *Decision*: `RunnerSettings`, `FraudInputs`, `PaymentView`, `OutsideEvent`, `InteractiveRunResult`, and `InteractiveWorldOutput` are defined as frozen dataclasses.
   - *Why*: Prevents unintentional state mutation across showcase checks, test suites, or analysis pipelines.
8. **Dependency-Injected Git Runner (`run_git`)**:
   - *Decision*: `find_code_version` accepts an optional `run_git` callable defaulting to `_run_git`.
   - *Why*: Decouples code version resolution from OS-level git execution, allowing fast, deterministic unit testing of timeout and dirty-state handling without OS process spawns.
9. **Two-Level Issue Register Separation**:
   - *Decision*: `config/issues_v2.yaml` tracks 93 Revision 2 issues, leaving `config/issues.yaml` (31 issues) strictly frozen.
   - *Why*: Level 11 tests pin the exact 31 IDs of `issues.yaml`. Introducing `issues_v2.yaml` allows comprehensive tracking of later work without invalidating Level 11 assertions.
10. **Origin vs. Source Separation**:
    - *Decision*: `QueuedPayment` tracks both `origin: "planned" | "fraud"` (true historical provenance) and `source: "planned" | "fraud" | "resumed"` (current queue execution state), recording `events_resumed: list[bool]`.
    - *Why*: `source` dictates child stream allocation and draw consumption on hold; resetting `source` to `planned` on resumption would cause the payment to draw from the run stream twice, shifting all downstream random draws.
11. **Split World Selected by Seed Scan**:
    - *Decision*: Benchmark world (seed 1 at 120 people, 20 shops, 61 days) was chosen via an automated scan of seeds 1–12.
    - *Why*: Seed 1 was empirically demonstrated to produce 21 genuine split payments (exceeding the Rs 51,377 UPI limit), providing a concrete benchmark for payment splitting equivalence without arbitrary synthetic tuning.
12. **Issue R9 Verified Unchanged**:
    - *Decision*: Left R9 deferred in both `issues.yaml` and `issues_v2.yaml` without editing.
    - *Why*: The status of R9 is confirmed deferred in both registers, satisfying the cross-register link check without modifying existing files.
13. **Showcase Check Minimums Apply Per Level**:
    - *Decision*: The framework applies check coverage rules (at least 1 normal, 2 edge, 1 negative) to the whole level definition before filtering by `--part`.
    - *Why*: Ensures architectural integrity and comprehensive verification across the complete level contract while allowing modular execution of individual parts during incremental development.
14. **Check 3 Asserts Invariant Properties Rather Than Fixed Counts**:
    - *Decision*: Check 3 asserts that the candidate debit fails with `(status, failure_domain, reason) == ("failed", "sender_bank", "funds_on_hold")`, at least one debit is blocked, every blocked debit belongs to the liened account and postdates the lien, prior events are identical, and double-entry money balances, rather than asserting `blocked_count == 30`.
    - *Why*: Eliminates brittle magic numbers, ensuring the verification remains honest and accurate even if upstream simulation parameters change.

## Known Gaps for 12b–12g
1. **Gang Device Events Joining Final Device Flush**: Device events emitted by gang operations must be integrated into the final runtime flush in Level 12c.
2. **Truth File Set + Env File + Mount + CI Import Check**: Truth file generation, environment variables, filesystem mount points, and CI verification to be implemented in Level 12b.
3. **Events-v2 File Writer**: Dedicated writer utility for emitting `events-v2` files to disk to be added in Level 12b.
4. **Held-Out Family Boundary Parameters**: Pinned hyperparameter configurations for held-out fraud typology families before generating gang worlds.
5. **Risk Values (Amendment A-016)**: Integration of formal risk scoring and calibrated probability assignments.
6. **Level 12b Edge Check**: Showcase edge check covering temporal truth generation and account rental transitions.
7. **Register Account Mid-Run Trigger**: Automated trigger mechanism invoking `register_account` dynamically during event execution.
8. **Fraud Purposes from PAYMENT_PURPOSES**: Drawing fraud payment purposes from official `PAYMENT_PURPOSES` distributions.
9. **Observation Feed Outside Hook**: Emitting bank observations to external consumer queues rather than directly within the synchronous bank hook.
10. **Analyst as a Pure Function**: Refactoring transaction analyst logic into a side-effect-free pure function in Level 12e.
11. **Level 12e Negative Check**: Validation check proving error handling for invalid analyst inputs.
12. **World Counts and Seed Disjointness (12f)**: Sampler validation ensuring distinct seeds across train, validation, and test splits.
13. **Trivial-World Check & Inflation Factor (12g)**: Trivial-world gate rejecting degenerate scenarios and bounding fraud inflation factors.
14. **ASHFALL_CODE_VERSION in Containers**: Standardizing container build arguments to inject Git version identifiers into runtime environments.
15. **Run Identifier Independence from Fraud Inputs**: Currently `make_run_id` derives only from seed, effective configuration, and code version; in Level 12b it must incorporate the fraud switch and inputs so distinct runs share no colliding `payment_id` values.

## Exact Verification Commands
```powershell
# 1. Full unit suite (excluding external dependencies)
python -m pytest tests/unit -q -p no:cacheprovider -m "not needs_stack and not gpu and not ml and not needs_dvc"

# 2. Targeted unit tests covering Level 12a components
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
