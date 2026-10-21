# Level 12c Handoff Note

## What 12c Delivered
Level 12c implements victim money routing, mule lifecycle transitions from warm-up to burned, rented owner activity, diverse cash-out channels, and Part-C showcase checks (World Engine II Part C):

1. **Victim Inflows & Scam Planning**:
   - `world/scam_config.py` & `world/scam_planner.py`: Configures and plans dynamic victim fraud generation across scam categories (`investment_or_job`, `unauthorised_debit`, and general scams). Evaluates power-law victim distributions per receiving mule, multi-payment victim schedules with escalation multiples, and optional bait payouts.
   - `world/truth_records.py`: Tracks victim roles (`RoleSpan` with role `"victim"`) and forensic payment narratives (`PaymentStory` tagging scam, route hops, and cash-outs).
   - Money routing: Gangs forward at most what arrived from victims; unreceived or outage-blocked inflows cleanly suppress downstream forwards via `IllicitMoneyRule`.

2. **Mule Lifecycle & State Machine**:
   - Lifecycle stages: Intermediate mule accounts advance through three temporal phases: `warm_up` -> `active` -> `burned`.
   - `world/truth_records.py`: `RoleSpan` enriched with `active_from` and `burned_from` timestamps, validated for temporal ordering and strict non-overlap.
   - `evaluation/truth.py`: `TruthStore.phase_at(account_id, minute)` resolves the operational lifecycle phase at any simulation minute.

3. **Mule Recruitment Diversity & Rented Owner Activity**:
   - `world/mule_origins.py`: Provisions mules across three distinct recruitment channels: `rented`, `borrowed_kyc`, and `business`. Stamped into truth via `RoleSpan.origin`.
   - `world/opening_handles.py`: Generates placeholder handles for accounts created mid-run (`__opening_...__`), resolved before simulation dispatch.
   - `world/rented_owner_activity.py`: Models authentic account owner behavior during rental intervals. Genuine incoming and outgoing transactions continue at a configured kept share, while conflicting outgoing transfers are paused cleanly under child stream 12 without shifting other draws.
   - `world/account_openings_config.py` & `world/opened_routine.py`: Supports background account creation for genuine citizens and businesses with routine daily transaction schedules.

4. **Cash-Out Channels & Safe Withdrawals**:
   - `world/cash_out.py`: Sinks laundered mule funds across four realistic exit vectors: crypto exchanges (`crypto`), ATM cash withdrawals (`atm`), bank branch counter withdrawals (`branch`), and retail merchant purchases (`genuine_shop`).
   - `world/split_below_threshold.py`: Implements structuring tactics splitting large amounts into smaller sub-threshold transfers.
   - `world/cash_withdrawals.py`: Processes ATM and branch counter withdrawals. Strictly enforces all-or-nothing debit posting against real ledger balances, recording short balances as failed debits using domain `"customer"` and reason `"insufficient_balance"` matching Level 11 conventions. Requires positive daily limits for ATM and None for branch withdrawals. Takes failure draws before balance evaluation to maintain stream invariance.

5. **Showcase Part C & 15-Check Suite**:
   - `showcase/fixtures/level_12/gang_victims_cashout.yaml`: Realistic scam syndicate fixture combining acquisition, warm-up, victim scams, multi-layer routing, and cash-out channels.
   - `showcase/level_12_checks_c.py` & `showcase/level_12_support_c.py`: Checks 11 to 15 proving end-to-end victim-to-cashout flow, rented mule normal life, lifecycle phase transitions, innocent downstream merchant receipts, and bank outage resilience. Uses cached execution for checks 11-14 to preserve quick-mode performance.
   - `tests/unit/showcase/test_level_12_checks.py`: Pinned check registry updated to exactly 15 checks (5 Part A, 5 Part B, 5 Part C).

---

## Public Interfaces

### Functions
Extracted directly via `inspect.signature`:

```python
apply_illicit_cap(context: Any, planned_payment_id: str, minute: int, from_account: str, to_account: str, planned_paise: int, available_paise: int) -> int | None
```
- Path: `world/cash_withdrawals.py`

```python
apply_split_below_threshold(payments: collections.abc.Sequence[world.calendar_runner.PlannedPayment], rules: collections.abc.Sequence[world.interactive_records.IllicitMoneyRule], stories: collections.abc.Sequence[world.truth_records.PaymentStory], threshold_paise: int, gang_id: str, split: str = 'training', config_path: pathlib.Path | None = None, world_ranges_path: pathlib.Path | None = None) -> tuple[list[world.calendar_runner.PlannedPayment], list[world.interactive_records.IllicitMoneyRule], list[world.truth_records.PaymentStory]]
```
- Path: `world/split_below_threshold.py`

```python
build_opening_event(gang_id: str, opening_number: int, origin: str, minute: int) -> world.outside_events.OutsideEvent
```
- Path: `world/mule_origins.py`

```python
draw_atm_daily_limit_paise(split: str, rng: numpy.random._generator.Generator, world_ranges_path: pathlib.Path | None = None) -> int
```
- Path: `world/cash_out.py`

```python
draw_branch_delay_minutes(split: str, rng: numpy.random._generator.Generator, config_path: pathlib.Path | None = None) -> int
```
- Path: `world/cash_out.py`

```python
draw_cash_out_shares(split: str, rng: numpy.random._generator.Generator, config_path: pathlib.Path | None = None, world_ranges_path: pathlib.Path | None = None) -> dict[str, float]
```
- Path: `world/cash_out.py`

```python
draw_mule_origins(count: int, split: str, rng: numpy.random._generator.Generator) -> list[str]
```
- Path: `world/mule_origins.py`

```python
draw_threshold_paise(split: str, threshold_param: int | float | None, rng: numpy.random._generator.Generator, config_path: pathlib.Path | None = None) -> int
```
- Path: `world/split_below_threshold.py`

```python
plan_cash_out_step(world: world.build_world.World, gang_id: str, strength: int, terminal_mules: list[tuple[str, int, int]], split: str, layers: int, opening_start_index: int, has_disguise: bool, jitter: int | None, fixed_dwell: int | None, dwell_param: Any, rng: numpy.random._generator.Generator, config_path: pathlib.Path | None = None, world_ranges_path: pathlib.Path | None = None) -> world.cash_out.CashOutPlan
```
- Path: `world/cash_out.py`

```python
plan_opened_routine(world: Any, plan: world.world_plan.WorldPlan, person: Any, open_minute: int, rng: numpy.random._generator.Generator, start_index: int) -> tuple[list[world.calendar_runner.PlannedPayment], int]
```
- Path: `world/opened_routine.py`

```python
plan_owner_activity(roles: collections.abc.Sequence[world.truth_records.RoleSpan], split: str, rng: numpy.random._generator.Generator) -> dict[str, world.rented_owner_activity.OwnerActivity]
```
- Path: `world/rented_owner_activity.py`

```python
plan_scam_step(world: world.build_world.World, config: world.gang_config.GangConfig, step: world.gang_config.BlockStep, mules: list[str], start_minute: int, reserved_accounts: set[str] | None, rng: numpy.random._generator.Generator) -> world.scam_planner.ScamPlan
```
- Path: `world/scam_planner.py`

```python
process_cash_withdrawal(context: Any, withdrawal: world.cash_withdrawals.CashWithdrawal) -> None
```
- Path: `world/cash_withdrawals.py`

```python
resolve_opening_handles(fraud_inputs: world.interactive_records.FraudInputs, next_person_index: int, next_shop_index: int) -> tuple[world.interactive_records.FraudInputs, dict[str, str]]
```
- Path: `world/opening_handles.py`

```python
schedule_genuine_account_openings(n_people: int, n_days: int, split: str, rng: numpy.random._generator.Generator, config: dict[str, typing.Any] | None = None) -> tuple[world.outside_events.OutsideEvent, ...]
```
- Path: `world/account_openings_config.py`

```python
validate_cash_withdrawal(withdrawal: world.cash_withdrawals.CashWithdrawal, state: Any = None) -> None
```
- Path: `world/cash_withdrawals.py`

```python
validate_role_lifecycle(role_span: world.truth_records.RoleSpan) -> None
```
- Path: `world/truth_records.py`

```python
validate_role_origin(role_span: world.truth_records.RoleSpan, allowed_origins: collections.abc.Sequence[str]) -> None
```
- Path: `world/truth_records.py`

```python
TruthStore.origin_at(self, account_id: str, minute: int) -> str | None
```
- Path: `evaluation/truth.py`

```python
TruthStore.phase_at(self, account_id: str, minute: int) -> str | None
```
- Path: `evaluation/truth.py`

---

## Records and Fields
Extracted directly via `dataclasses.fields`:

- `RoleSpan`:
  - Fields: `account_id: str`, `role: str`, `gang_id: str`, `depth: int | None`, `valid_from: int`, `valid_to: int | None`, `origin: str | None`, `active_from: int | None`, `burned_from: int | None`
  - Path: `world/truth_records.py`

- `MuleInflow`:
  - Fields: `last_victim_payment_minute: int`, `net_paise: int`, `inflow_window_end: int`
  - Path: `world/scam_planner.py`

- `ScamPayment`:
  - Fields: `minute: int`, `amount_paise: int`, `is_payout: bool`
  - Path: `world/scam_planner.py`

- `ScamPlan`:
  - Fields: `payments: tuple[world.calendar_runner.PlannedPayment, ...]`, `stories: tuple[world.truth_records.PaymentStory, ...]`, `rules: tuple[world.interactive_records.IllicitMoneyRule, ...]`, `roles: tuple[world.truth_records.RoleSpan, ...]`, `mule_inflows: dict[str, world.scam_planner.MuleInflow]`, `span_minutes: int`, `risk: float`, `dropped_payments_count: int`
  - Path: `world/scam_planner.py`

- `OwnerActivity`:
  - Fields: `kept_share: float`, `valid_from: int`, `valid_to: int | None`
  - Path: `world/rented_owner_activity.py`

- `PausedOwnerPayment`:
  - Fields: `minute: int`, `from_account: str`, `to_account: str`, `amount_paise: int`
  - Path: `world/rented_owner_activity.py`

- `CashWithdrawal`:
  - Fields: `minute: int`, `account_id: str`, `amount_paise: int`, `channel: str`, `card_daily_limit_paise: int | None`
  - Path: `world/cash_withdrawals.py`

- `CashWithdrawalRecord`:
  - Fields: `id: str`, `minute: int`, `account: str`, `channel: str`, `requested: int`, `withdrawn: int`, `status: str`, `reason: str | None`, `domain: str | None`
  - Path: `world/cash_withdrawals.py`

- `CashOutPlan`:
  - Fields: `payments: list[world.calendar_runner.PlannedPayment]`, `roles: list[world.truth_records.RoleSpan]`, `stories: list[world.truth_records.PaymentStory]`, `withdrawals: list[world.cash_withdrawals.CashWithdrawal]`, `outside_events: list[world.outside_events.OutsideEvent]`, `money_rules: list[world.interactive_records.IllicitMoneyRule]`, `time_cost: int`, `risk_cost: float`
  - Path: `world/cash_out.py`

---

## Config Files and Keys

### 1. `config/interactive_runner.yaml`
- `random_stream_children`:
  - `0-3`: Reserved for Level 11 simulation baseline (events, episodes, failures, device events).
  - `4`: `fraud_payment_failures`
  - `5`: `resumed_payment_failures`
  - `6`: `gang_devices`
  - `7`: `gang_planning`
  - `8`: `genuine_account_openings` (mid-run opening events for genuine citizens)
  - `9`: `gang_account_openings` (mid-run accounts for borrowed-KYC and business mules)
  - `10`: `opened_account_routine` (decision and planning of newly opened person routine)
  - `11`: `opened_routine_failures` (failure draws of newly opened person payments)
  - `12`: `rented_owner_activity` (draws deciding kept share and paused outgoing owner payments)
  - `13`: `cash_withdrawal_failures` (failure draws for ATM and branch counter withdrawals)
- `tie_break_order`:
  - `[outside_events, planned_payments, fraud_payments, resumed_payments, gang_device_events, opened_routine_payments, cash_withdrawals]`

### 2. `config/scam_schedules.yaml`
- Defines parameters for victim fraud planning:
  - `scam_types`: `[investment_or_job, unauthorised_debit]`
  - `victims_per_mule`: Power-law distribution configurations (`typical`, `largest`, `tail_exponent`).
  - `payments_per_victim`: Payment frequency bounds and escalation factors.
  - `bait_payout`: Probability and fraction of initial funds returned to victims.
  - `timing`: Dwell intervals and min spacing between fraud debits.

### 3. `config/account_openings.yaml`
- Background population growth and genuine account openings:
  - `enabled`: Feature flag controlling background opening injection.
  - `opening_rate`: Daily new accounts per capita.
  - `shop_share`: Proportion of new accounts registered as retail merchant shops.
  - `no_routine_share`: Proportion of accounts opened without active calendar schedules.

### 4. `config/cash_out.yaml`
- Extraction channel definitions and limits:
  - `channels`: `[crypto, atm, branch, genuine_shop]`
  - `channel_shares`: Proportional split across channels.
  - `atm_daily_limit_inr`: Daily cash ceiling per payment card.
  - `branch_delay_minutes`: Turnaround latency for physical bank teller counter withdrawals.
  - `genuine_shop_kinds`: Merchant category codes receiving commercial retail cash-outs.

### 5. `config/tactic_blocks.yaml` & `config/tactic_costs.yaml`
- `scam`: Parameter validation, block duration bounds, and empirical detection risk per use.
- `cash_out`: Parameter validation, block duration bounds, and extraction risk per use.
- `split_below_threshold`: Structuring parameters splitting large sums under reporting thresholds.

### 6. `config/truth_store.yaml`
- `allowed_origins`: `[rented, borrowed_kyc, business]` registered under ground truth schema.

---

## Decisions and Why

1. **All-or-Nothing Cash Withdrawals (C5-1)**:
   - When processing ATM or counter cash withdrawals, real ledger balances must be evaluated before posting any leg. If the balance is insufficient, zero legs are posted and the withdrawal is marked failed using domain `"customer"` and reason `"insufficient_balance"`, matching Level 11 convention.
   - Stream invariance: The failure draw (child stream 13) is evaluated before the ledger balance check, ensuring identical draw positions regardless of balance state.
   - Channel validation: Channel `"atm"` strictly requires `card_daily_limit_paise > 0`; channel `"branch"` requires `card_daily_limit_paise` to be `None`.

2. **Scam Inflows as Primary Gang Funding Source (C-1)**:
   - Replaced static fixture funding with procedural victim fraud campaigns. Illicit money arriving at receiving mules is bounded by real victim payments.
   - Forwarding safety: Mules forward only funds that actually arrive; outages or held payments automatically suppress dependent forwarding hops via `IllicitMoneyRule`.

3. **Temporal Mule Lifecycle (C-2)**:
   - Mule accounts transition deterministically through `warm_up`, `active`, and `burned` states.
   - `active_from` marks when laundering hops begin; `burned_from` marks when the gang abandons or rotates the account. Evaluated via `TruthStore.phase_at`.

4. **Preservation of Authentic Life for Rented Accounts (C-3)**:
   - Rented mule owners keep normal routine transactions (incoming salary/remittances and essential outgoing expenses). Conflicting outgoing transfers are paused cleanly during the rental span.
   - Mid-run account creation resolves placeholder handles (`__opening_...__`) before simulation start, ensuring deterministic account IDs without pre-simulation ledger collisions.

5. **Downstream Commercial Cash-Outs (C-4)**:
   - Terminal funds sink into crypto exchanges or legitimate retail shops.
   - Legal innocence: Commercial shops receiving payments preserve `role_at = genuine` in truth, reflecting real-world laundering via innocent retail commerce.

6. **Showcase Part C Cached Execution (C5-2)**:
   - Checks 11–14 share a single simulation run via `get_showcase_c_run()` in `showcase/level_12_support_c.py`, preserving rapid showcase execution (<35s in quick mode).

---

## Known Gaps

1. **Lien-Triggered Burn (12d)**:
   - Outside liens placed on active mule accounts currently freeze debits without immediately advancing the mule's lifecycle phase to `burned`. Lien-induced early burning is scheduled for Level 12d.
2. **Withdrawals Reaching Banks (Level 14)**:
   - Cash withdrawals debit customer accounts in the simulator ledger, but interbank settlement or physical currency replenishment at bank branches is not yet modelled until Level 14.
3. **Genuine New Shops Get No Customers**:
   - Accounts created mid-run as genuine merchant shops receive opening balances, but existing citizens do not dynamically update their payment schedules to patronize them.
4. **Route Layers and Accounts per Layer Unsourced**:
   - Parameter bounds for route depth `[1, 5]` and layer width `[1, 10]` remain heuristic policy choices pending empirical grounding.
5. **Device Compromise Anomaly Uniformity**:
   - Gang device events (`sim_swap`, `linked_number_switch`, `emulator`, `remote_access`) are currently emitted exclusively for criminal syndicates. Genuine citizens need background event rates before Level 14 exposes raw device logs to detection models.
6. **Payment Device Stamping**:
   - Fraudulent transactions emitted by gangs currently reflect the account owner's registered phone identifier rather than the gang's controlling device identifier.

---

## Exact Verification Commands

```powershell
# 1. Golden checks verification
python -B scripts/verify_world.py

# 2. Full unit suite (excluding external dependencies)
python -m pytest tests/unit -q -p no:cacheprovider -m "not needs_stack and not gpu and not ml and not needs_dvc"

# 3. Targeted Level 12c unit test suite
python -m pytest tests/unit/world/test_cash_withdrawals.py tests/unit/world/test_cash_out.py tests/unit/world/test_scam_gangs.py tests/unit/world/test_mule_lifecycle.py tests/unit/world/test_rented_owner_activity.py tests/unit/world/test_mid_run_accounts.py tests/unit/world/test_split_below_threshold.py tests/unit/showcase/test_level_12_checks_c.py tests/unit/showcase/test_level_12_checks.py -q

# 4. Showcase Part C execution (5/5 checks)
python -m showcase --level 12 --part c

# 5. Showcase Part B execution (5/5 checks)
python -m showcase --level 12 --part b

# 6. Showcase Part A execution (5/5 checks)
python -m showcase --level 12 --part a

# 7. Showcase Level 12 quick mode (all 15 checks)
python -m showcase --level 12 --quick

# 8. Baseline Level 11 showcase invariance (16/16 checks)
python -m showcase --level 11 --quick

# 9. Dry-run make shortcut
make -n show LEVEL=12 PART=c

# 10. Issue register v2 validation
python -m core.issue_register_v2
python -m core.issue_register

# 11. Truth isolation static import check
python -m core.truth_import_check

# 12. YAML configuration schema validation
python -m core.config_validator config/truth_store.yaml config/tactic_blocks.yaml config/tactic_costs.yaml config/scam_schedules.yaml config/cash_out.yaml config/account_openings.yaml config/interactive_runner.yaml config/showcase_level_12.yaml

# 13. Code comment and documentation standards
python -m pytest tests/unit/world/test_new_code_comment_standard.py

# 14. Code quality and type safety
ruff check showcase/ world/ tests/unit/ evaluation/
mypy showcase api core evaluation --ignore-missing-imports --no-strict-optional

# 15. Git status and commit history
git log --oneline feature/level-12-fraud-world-engine..HEAD
git status --porcelain
```
