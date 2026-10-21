# Level 12c Handoff Note

## Round History
- C-1: forward at most the illicit money that arrived; mid-run account openings in existing id formats; genuine openings behind a switch; new documentation guard.
- C-2: victims and scam schedules (investment, job, unauthorised debit) as the gang's money source; route forwards victim money only.
- C-3a: mule origins in truth; borrowed-KYC and business mules opened mid-run via handles; rented owners keep part of their normal life; routine life for new genuine people.
- C-3b: omitted step sizes drawn from config; heavy-tailed victims per mule; lifecycle (warm-up, active, burned); payouts tied to their own victim.
- C-4: cash withdrawals with ATM daily limits; cash-out channels (crypto collector, ATM, branch, genuine shops); split-below-threshold block.
- C-5: withdrawals fail cleanly on a short balance; showcase part c (checks 11-15); handoff.

---

## Register Changes
- 12-19: New mule accounts with new ID formats would reveal their role (status: built)
- G1: Mule origin; rented mules keep normal life (status: built)
- G2: Cash-out channels including ATM limits and genuine shops (status: partial)
- G3: Mule lifecycle; heavy-tailed victims per mule (status: partial)
- G6: Scam-type payment schedules (status: partial)
- A2: Genuine shops downstream of mules (status: built)
- 12-17: Mule depth was not recorded; gang routing must obey the same limits (status: built)

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
choose_victims(world: world.build_world.World, count: int, excluded: set[str], rng: numpy.random._generator.Generator) -> list[str]
```
- Path: `world/scam_planner.py`

```python
compute_power_law_probabilities(cap: int, exponent: float) -> list[float]
```
- Path: `world/scam_planner.py`

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
draw_no_routine_share(split: str, config: dict[str, typing.Any], rng: numpy.random._generator.Generator) -> float
```
- Path: `world/opened_routine.py`

```python
draw_origin_shares(split: str, rng: numpy.random._generator.Generator) -> dict[str, float]
```
- Path: `world/mule_origins.py`

```python
draw_scam_type(chances: world.scam_planner.TypeChances, rng: numpy.random._generator.Generator) -> str
```
- Path: `world/scam_planner.py`

```python
draw_threshold_paise(split: str, threshold_param: int | float | None, rng: numpy.random._generator.Generator, config_path: pathlib.Path | None = None) -> int
```
- Path: `world/split_below_threshold.py`

```python
draw_type_chances(split: str, rng: numpy.random._generator.Generator) -> world.scam_planner.TypeChances
```
- Path: `world/scam_planner.py`

```python
get_cash_out_channel_share_ranges(split: str, config_path: pathlib.Path | None = None, world_ranges_path: pathlib.Path | None = None) -> dict[str, tuple[float, float]]
```
- Path: `world/cash_out.py`

```python
get_genuine_shop_kinds(config_path: pathlib.Path | None = None) -> tuple[str, ...]
```
- Path: `world/cash_out.py`

```python
get_threshold_inr_range(split: str = 'training', config_path: pathlib.Path | None = None) -> tuple[int, int]
```
- Path: `world/split_below_threshold.py`

```python
is_idle(no_routine_share: float, rng: numpy.random._generator.Generator) -> bool
```
- Path: `world/opened_routine.py`

```python
is_opening_handle(account_id: str) -> bool
```
- Path: `world/mule_origins.py`

```python
load_account_openings_config(config_path: pathlib.Path | None = None) -> dict[str, typing.Any]
```
- Path: `world/account_openings_config.py`

```python
load_cash_out_config(path: pathlib.Path | None = None) -> dict[str, typing.Any]
```
- Path: `world/cash_out.py`

```python
load_scam_schedules(config_path: pathlib.Path | None = None) -> dict[str, typing.Any]
```
- Path: `world/scam_config.py`

```python
load_truth_config(path: pathlib.Path = WindowsPath('C:/Users/sinne/Downloads/PROJECT-ashFALL/config/truth_store.yaml')) -> dict[str, typing.Any]
```
- Path: `world/truth_records.py`

```python
make_opening_handle(gang_id: str, opening_number: int) -> str
```
- Path: `world/mule_origins.py`

```python
plan_cash_out_step(world: world.build_world.World, gang_id: str, strength: int, terminal_mules: list[tuple[str, int, int]], split: str, layers: int, opening_start_index: int, has_disguise: bool, jitter: int | None, fixed_dwell: int | None, dwell_param: Any, rng: numpy.random._generator.Generator, config_path: pathlib.Path | None = None, world_ranges_path: pathlib.Path | None = None) -> world.cash_out.CashOutPlan
```
- Path: `world/cash_out.py`

```python
plan_investment_or_job_payments(first_minute: int, split: str, rng: numpy.random._generator.Generator) -> list[world.scam_planner.ScamPayment]
```
- Path: `world/scam_planner.py`

```python
plan_opened_routine(world: Any, plan: world.world_plan.WorldPlan, person: Any, open_minute: int, rng: numpy.random._generator.Generator, start_index: int) -> tuple[list[world.calendar_runner.PlannedPayment], int]
```
- Path: `world/opened_routine.py`

```python
plan_owner_activity(roles: Sequence[world.truth_records.RoleSpan], split: str, rng: numpy.random._generator.Generator) -> dict[str, world.rented_owner_activity.OwnerActivity]
```
- Path: `world/rented_owner_activity.py`

```python
plan_payout_back(payments: list[world.scam_planner.ScamPayment], split: str, rng: numpy.random._generator.Generator) -> world.scam_planner.ScamPayment | None
```
- Path: `world/scam_planner.py`

```python
plan_scam_step(world: world.build_world.World, config: world.gang_config.GangConfig, step: world.gang_config.BlockStep, mules: list[str], start_minute: int, reserved_accounts: set[str] | None, rng: numpy.random._generator.Generator) -> world.scam_planner.ScamPlan
```
- Path: `world/scam_planner.py`

```python
plan_unauthorised_debit_payments(first_minute: int, split: str, rng: numpy.random._generator.Generator) -> list[world.scam_planner.ScamPayment]
```
- Path: `world/scam_planner.py`

```python
plan_victim_schedule(victim: str, mule: str, chances: world.scam_planner.TypeChances, first_minute: int, split: str, rng: numpy.random._generator.Generator) -> world.scam_planner.VictimSchedule
```
- Path: `world/scam_planner.py`

```python
process_cash_withdrawal(context: Any, withdrawal: world.cash_withdrawals.CashWithdrawal) -> None
```
- Path: `world/cash_withdrawals.py`

```python
replay_openings(outside_events: tuple[world.outside_events.OutsideEvent, ...], next_person_index: int, next_shop_index: int) -> tuple[dict[str, str], dict[str, int]]
```
- Path: `world/opening_handles.py`

```python
resolve_debit_amount_inr(split: str, config_path: pathlib.Path | None = None, world_ranges_path: pathlib.Path | None = None) -> tuple[float, float]
```
- Path: `world/scam_config.py`

```python
resolve_debit_payment_count(split: str, config_path: pathlib.Path | None = None) -> tuple[int, int]
```
- Path: `world/scam_config.py`

```python
resolve_debit_pays_back(config_path: pathlib.Path | None = None) -> bool
```
- Path: `world/scam_config.py`

```python
resolve_debit_spread_minutes(split: str, config_path: pathlib.Path | None = None) -> tuple[int, int]
```
- Path: `world/scam_config.py`

```python
resolve_first_payment_amount_inr(split: str, config_path: pathlib.Path | None = None, world_ranges_path: pathlib.Path | None = None) -> tuple[float, float]
```
- Path: `world/scam_config.py`

```python
resolve_investment_or_job_share(split: str, config_path: pathlib.Path | None = None, world_ranges_path: pathlib.Path | None = None) -> tuple[float, float]
```
- Path: `world/scam_config.py`

```python
resolve_investment_share_of_investment_or_job(split: str, config_path: pathlib.Path | None = None) -> tuple[float, float]
```
- Path: `world/scam_config.py`

```python
resolve_min_minutes_between_victim_payments(config_path: pathlib.Path | None = None) -> int
```
- Path: `world/scam_config.py`

```python
resolve_mule_active_days_before_dropped(split: str, world_ranges_path: pathlib.Path | None = None) -> tuple[int, int]
```
- Path: `world/scam_config.py`

```python
resolve_no_routine_share_range(split: str, config: dict[str, typing.Any]) -> tuple[float, float]
```
- Path: `world/opened_routine.py`

```python
resolve_opening_handles(fraud_inputs: world.interactive_records.FraudInputs, next_person_index: int, next_shop_index: int) -> tuple[world.interactive_records.FraudInputs, dict[str, str]]
```
- Path: `world/opening_handles.py`

```python
resolve_opening_rate_range(split: str = 'training', config: dict[str, typing.Any] | None = None) -> tuple[float, float]
```
- Path: `world/account_openings_config.py`

```python
resolve_payment_escalation_multiple(split: str, config_path: pathlib.Path | None = None) -> tuple[float, float]
```
- Path: `world/scam_config.py`

```python
resolve_payment_spread_days(split: str, config_path: pathlib.Path | None = None, world_ranges_path: pathlib.Path | None = None) -> tuple[int, int]
```
- Path: `world/scam_config.py`

```python
resolve_payments_per_victim(split: str, config_path: pathlib.Path | None = None, world_ranges_path: pathlib.Path | None = None) -> tuple[int, int]
```
- Path: `world/scam_config.py`

```python
resolve_payout_back_share(split: str, config_path: pathlib.Path | None = None, world_ranges_path: pathlib.Path | None = None) -> tuple[float, float]
```
- Path: `world/scam_config.py`

```python
resolve_remainder_scam_type(config_path: pathlib.Path | None = None) -> str
```
- Path: `world/scam_config.py`

```python
resolve_shop_share_range(split: str = 'training', config: dict[str, typing.Any] | None = None) -> tuple[float, float]
```
- Path: `world/account_openings_config.py`

```python
resolve_victims_per_mule_largest(split: str, world_ranges_path: pathlib.Path | None = None) -> tuple[int, int]
```
- Path: `world/scam_config.py`

```python
resolve_victims_per_mule_tail_exponent(split: str, config_path: pathlib.Path | None = None) -> tuple[float, float]
```
- Path: `world/scam_config.py`

```python
resolve_victims_per_mule_typical(split: str, world_ranges_path: pathlib.Path | None = None) -> tuple[int, int]
```
- Path: `world/scam_config.py`

```python
schedule_genuine_account_openings(n_people: int, n_days: int, split: str, rng: numpy.random._generator.Generator, config: dict[str, typing.Any] | None = None) -> tuple[world.outside_events.OutsideEvent, ...]
```
- Path: `world/account_openings_config.py`

```python
validate_cash_withdrawal(withdrawal: world.cash_withdrawals.CashWithdrawal, state: Any = None) -> None
```
- Path: `world/cash_withdrawals.py`

```python
validate_mule_depth(role_span: world.truth_records.RoleSpan) -> None
```
- Path: `world/truth_records.py`

```python
validate_no_overlapping_device_control_spans(spans: Sequence[world.truth_records.DeviceControlSpan]) -> None
```
- Path: `world/truth_records.py`

```python
validate_no_overlapping_role_spans(spans: Sequence[world.truth_records.RoleSpan]) -> None
```
- Path: `world/truth_records.py`

```python
validate_payment_stories_length(payment_stories: Sequence[world.truth_records.PaymentStory], payment_count: int) -> None
```
- Path: `world/truth_records.py`

```python
validate_role_in_config(role_span: world.truth_records.RoleSpan, allowed_roles: Sequence[str]) -> None
```
- Path: `world/truth_records.py`

```python
validate_role_lifecycle(role_span: world.truth_records.RoleSpan) -> None
```
- Path: `world/truth_records.py`

```python
validate_role_origin(role_span: world.truth_records.RoleSpan, allowed_origins: Sequence[str]) -> None
```
- Path: `world/truth_records.py`

```python
validate_time_span(span: world.truth_records.RoleSpan | world.truth_records.DeviceControlSpan) -> None
```
- Path: `world/truth_records.py`

```python
validate_truth_facts(truth: world.truth_records.TruthFacts, payment_count: int, allowed_roles: Optional[Sequence[str]] = None, allowed_origins: Optional[Sequence[str]] = None) -> None
```
- Path: `world/truth_records.py`

```python
TruthStore.controllers_of(self, account_id: str, minute: int) -> list[str]
```
- Path: `evaluation/truth.py`

```python
TruthStore.depth_at(self, account_id: str, minute: int) -> int | None
```
- Path: `evaluation/truth.py`

```python
TruthStore.origin_at(self, account_id: str, minute: int) -> str | None
```
- Path: `evaluation/truth.py`

```python
TruthStore.owner_at(self, account_id: str, minute: int) -> str | None
```
- Path: `evaluation/truth.py`

```python
TruthStore.phase_at(self, account_id: str, minute: int) -> str | None
```
- Path: `evaluation/truth.py`

```python
TruthStore.role_at(self, account_id: str, minute: int) -> str
```
- Path: `evaluation/truth.py`

```python
TruthStore.story_of(self, payment_id: str) -> dict[str, typing.Any]
```
- Path: `evaluation/truth.py`

```python
load_truth(truth_root: pathlib.Path, run_id: str) -> evaluation.truth.TruthStore
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

## Configuration Keys

### `config/scam_schedules.yaml`
- **scam_type_shares** (label: `policy`)
  - `label`
  - `reason`
  - `issue_ids_v2`
  - `derived_from`
  - `investment_or_job_share` (label: `policy`)
  - `investment_share_of_investment_or_job` (label: `assumption`)
  - `remainder_type` (label: `policy`)
- **investment_and_job_schedule** (label: `policy`)
  - `label`
  - `reason`
  - `issue_ids_v2`
  - `derived_from`
  - `payments_per_victim` (label: `policy`)
  - `payment_spread_days` (label: `policy`)
  - `first_payment_amount_inr` (label: `policy`)
  - `payment_escalation_multiple` (label: `assumption`)
  - `payout_back_share` (label: `policy`)
  - `min_minutes_between_victim_payments` (label: `policy`)
- **unauthorised_debit_schedule** (label: `policy`)
  - `label`
  - `reason`
  - `issue_ids_v2`
  - `derived_from`
  - `debit_payments` (label: `assumption`)
  - `debit_spread_minutes` (label: `assumption`)
  - `debit_amount_inr` (label: `policy`)
  - `payout_back` (label: `policy`)
- **victims_per_mule_tail_exponent** (label: `assumption`)
  - `label`
  - `unit`
  - `training`
  - `test`
  - `stress`
  - `issue_ids_v2`
  - `derived_from`
  - `reason`

### `config/account_openings.yaml`
- **parameters**
  - `genuine_account_openings_per_1000_people_per_30_days` (label: `assumption`)
  - `share_of_openings_that_are_shops` (label: `assumption`)
  - `share_of_new_genuine_people_with_no_routine` (label: `assumption`)

### `config/cash_out.yaml`
- **cash_out** (label: `policy`)
  - `label`
  - `reason`
  - `issue_ids_v2`
  - `derived_from`
  - `channel_shares` (label: `policy`)
  - `genuine_shop_kinds` (label: `policy`)
  - `branch_withdrawal_delay_hours` (label: `assumption`)
  - `atm_daily_limit_inr` (label: `policy`)

### `config/truth_store.yaml`
- **truth_store** (label: `policy`)
  - `label`
  - `reason`
  - `issue_ids_v2`
  - `derived_from`
  - `format`
  - `file_names`
  - `file_names_reason`
  - `roles`
  - `roles_reason`
  - `mule_origins`
  - `mule_origins_reason`
  - `env_var`
  - `env_var_reason`
  - `allowed_reader_folders`
  - `allowed_reader_folders_reason`
  - `ashfall_packages`
  - `ashfall_packages_reason`

### `config/interactive_runner.yaml`
- **interactive_runner** (label: `policy`)
  - `label`
  - `reason`
  - `issue_ids_v2`
  - `derived_from`
  - `random_stream_children`
  - `random_stream_children_reason`
  - `tie_break_order`
  - `tie_break_order_reason`
  - `run_id_hex_length`
  - `position_digits`
  - `id_shape_reason`
  - `code_version_env_var`
  - `code_version_git_timeout_seconds`
  - `unknown_code_version`
  - `code_version_reason`

### `config/tactic_blocks.yaml` (scam, cash_out, split_below_threshold)
- **scam** (label: `assumption`)
  - `label`
  - `reason`
  - `meaning`
  - `allowed_strengths`
  - `time_minutes_per_use`
  - `parameters`:
    - `receiving_mules` (label: `assumption`)
    - `victims_per_receiving_mule` (label: `assumption`)
- **cash_out** (label: `assumption`)
  - `label`
  - `reason`
  - `meaning`
  - `allowed_strengths`
  - `time_minutes_per_use`
  - `parameters`:
    - `channels` (label: `policy`)
- **split_below_threshold** (label: `assumption`)
  - `label`
  - `reason`
  - `meaning`
  - `allowed_strengths`
  - `time_minutes_per_use`
  - `parameters`:
    - `threshold_inr` (label: `assumption`)

---

## Random Stream Children and Tie-Break Order

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
