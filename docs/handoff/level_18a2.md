# Level 18a2 Handoff Note — takeover realism

All fixture figures below are from tiny sampler worlds (SIMULATED, four worlds) unless stated
otherwise.

The section "Generated facts" is the verbatim output of
`PYTHONPATH=. py -3.12 scripts/handoff_facts.py --part 18a2` (signatures from
`inspect.signature`, fields from `dataclasses.fields`, keys read from the config files).
Regenerate it instead of editing it. Everything else is written by hand.

## Where things stand
Level 18a2 gives generated gangs a device take-over trail, so the bank feed can see the
same phone events on a gang-controlled account that it sees on a genuine one. Branch:
`feature/level-18a2-takeover-devices`.
- `config/takeover_realism.yaml` holds four take-control share ranges (SIM swap,
  linked-number switch, emulator, remote access), drawn per world and per gang only when
  `takeover_realism` is true.
- Stream child 27 (`gang_take_control_flags`) draws the flags so the gang-mix stream
  (child 22) never shifts (`f412b59`).
- A take-control parameter `bind_control_device` makes each controlled account emit one
  `control_binding` gang device event at take-control and at every rotation generation
  (`6c3a902`, `world/gang_devices.py`).
- The bank feed maps the gang kind `control_binding` to a bank-visible
  `device_binding_change` whose details carry exactly the genuine device fields
  (`4cda524`, `world/bank_feed.py`).
- The shares moved out of `config/world_ranges.yaml` into their own file so the run-id
  effective config stays unchanged with the switch off (`7b5345c`, A-041), and the switch
  is committed `true` for generated gangs (`bf57274`, `config/gang_mix.yaml`).

## What each component does
- `config/takeover_realism.yaml`: one share range per flag for training, test and stress
  worlds; training sits inside test and test inside stress. Read with WORLD scope:
  stress worlds use the stress column; training, validation and test use the training
  column, because these flags are not route settings.
- `takeover_realism` (`config/gang_mix.yaml`): when true, generated gangs draw the four
  flags and set `bind_control_device`; when false, generated gangs are built exactly as
  before 18a2.
- `bind_control_device` (`config/tactic_blocks.yaml`): when true, `build_gang_device_events`
  emits one `control_binding` event per controlled account at take-control, and
  `build_rotation_events` emits one at each generation whose device differs from the
  previous one.
- `control_binding` mapping (`config/bank_feed.yaml`): projected as `device_binding_change`
  with the genuine fields (`fingerprint_before`, `fingerprint_after`), so the Level 15
  `new_device_r5` feature and rule_5 can see it.
- Deliberately bank-invisible in v1: `linked_number_switch`, `emulator` and
  `remote_access`. They stay in the client telemetry and the truth store only; a phone
  field on gang events only would make them distinguishable from genuine events (rule 29).

## Fixture rebuild (SIMULATED, tiny worlds, proves plumbing only)
The Level 17 fixtures were recorded at `cd712d3`, before 18a2, so generated gangs left no
device-binding trail. They are re-recorded at `38151de` with `takeover_realism` on, preset
tiny, sampler seed 2, splits training,validation,test (worlds 2/1/1 from
`config/datasets.yaml`), `--write-evaluation-inputs`.

New ids: `training-20261031T1830-624c2ad40f4c`,
`validation-20261031T1830-e5593f4c3f8e`, `test-20261031T1830-3910e3bea0d4`.
New run ids: `79d161df9ecf4172` (training-0000), `6b20a184cb1a5efe` (training-0001),
`d9a663e9838a6eb4` (validation-0000), `7ee68d1b5a539657` (test-0000).

`device_binding_change` trail, counted from each dataset's `snapshots.parquet` as rows
with `device_hardware_changed == 1` (a `device_binding_change` in the last 7 days) and the
distinct accounts behind them:

| world | old accounts | new accounts | old rows | new rows |
|---|---|---|---|---|
| training-0000 | 3 | 11 | 11 | 37 |
| training-0001 | 1 | 17 | 4 | 56 |
| validation-0000 | 4 | 9 | 18 | 41 |
| test-0000 | 6 | 14 | 29 | 57 |

Level 11 frozen anchors are unaffected: the fraud-off world never reads the switch.

## Fixture evaluation (tiny fixture worlds; not a real result)
`reports/level_17_evaluation/e9cf3ce2a657/result.json`:
- rules recall at capacity 0.50; anomaly 0.28; rules and anomaly 0.61.
- genuine restriction: rules 2.86%, anomaly 7.14%, rules and anomaly 8.57%.
- rule_5 standalone on validation: recall 0.4615 at 2.7778% genuine restriction.

## Decisions and why
- Draw the four take-control flags per generated gang on stream child 27: the flags need
  draws that never shift child 22, and a gang, not an account, decides its take-over
  method (A-040a, A-040b).
- Project `control_binding` as a genuine-shaped `device_binding_change`: the bank feed must
  see the same event shape on a gang phone as on a genuine phone, so a model cannot learn
  the difference from the event itself (A-040c, A-040d).
- Keep `linked_number_switch`, `emulator` and `remote_access` bank-invisible in v1: a
  bank-visible field written only for gangs would be a giveaway (A-040f, rule 29).
- Turn `takeover_realism` on only after moving the shares out of the run-id effective
  config: adding them to `world_ranges` changed every fraud-on run id even with the switch
  off (A-041). Level 11's fraud-off world is byte-identical.

## Known gaps and deferrals
- 17-1: rule_5 mule recall is still unmeasured at scale; the four tiny fixtures prove the
  plumbing, not the detection rate on >= 20 worlds.
- The three bank-invisible flags are a v1 ruling, not an oversight; re-open A-040(f) if a
  later level needs them in the feed.
- The fixtures' spreads are not meaningful with one test world.

## How to verify
- `PYTHONPATH=. py -3.12 -B scripts/verify_world.py` -> ALL GOLDEN CHECKS PASSED.
- `py -3.12 -m pytest -q -p no:cacheprovider tests/unit`
- `py -3.12 -m showcase --level 17` -> 7 checks, Overall: PASSED.
- Fixture evaluation: `PYTHONPATH=. py -3.12 -m evaluation.run_level_17 --training showcase/fixtures/level_17/datasets/training-20261031T1830-624c2ad40f4c --validation showcase/fixtures/level_17/datasets/validation-20261031T1830-e5593f4c3f8e --test showcase/fixtures/level_17/datasets/test-20261031T1830-3910e3bea0d4 --truth-root showcase/fixtures/level_17/truth --world-cards-root showcase/fixtures/level_17/world_cards`
- Manifest: `PYTHONPATH=. py -3.12 scripts/write_fixture_manifest.py --fixtures-root showcase/fixtures/level_17 --code-version 38151de --preset tiny --sampler-seed 2 --purpose "Level 17 showcase only; tiny worlds; numbers are not a real result"`

## Generated facts

## Register Changes
- 16-6: Phone changes during a run: generated gangs draw take-control flags; linked-number switch, emulator and remote access stay bank-invisible in v1 (status: partial)

---

## Public Interfaces

Extracted directly via `inspect.signature`:

### `world/gang_mix.py`

```python
family_for_gang(rules: list[dict[str, typing.Any]], layers: int) -> str | None
```

```python
gang_mix_rng(seed: int) -> numpy.random._generator.Generator
```

```python
generate_gang_mix(spec: world.world_sampler.WorldSpec, sampler_cfg: dict[str, typing.Any], mix_cfg: dict[str, typing.Any], eligible_people: int, world_ranges: dict[str, typing.Any] | None = None) -> tuple[world.gang_config.GangConfig, ...]
```

```python
load_gang_mix_config(path: pathlib.Path | None = None) -> dict[str, typing.Any]
```

```python
load_takeover_shares(path: pathlib.Path = WindowsPath('C:/Users/sinne/Downloads/PROJECT-ashFALL/config/takeover_realism.yaml')) -> dict[str, typing.Any]
```

```python
take_control_flags_rng(seed: int) -> numpy.random._generator.Generator
```

```python
validate_gang_mix_config(mix_cfg: dict[str, typing.Any], training_families: set[str]) -> None
```

### `world/gang_config.py`

```python
gang_config_from_plain(data: dict[str, typing.Any]) -> world.gang_config.GangConfig
```

```python
gang_config_to_plain(config: world.gang_config.GangConfig) -> dict[str, typing.Any]
```

```python
gang_start_minute(config: world.gang_config.GangConfig) -> int
```

```python
validate_gang_config(config: world.gang_config.GangConfig) -> None
```

```python
BlockStep(block: str, params: tuple[tuple[str, typing.Any], ...] = ()) -> None
```
- dataclass

```python
FixtureSource(account_id: str, bank_index: int, opening_balance_paise: int, start_minute: int) -> None
```
- dataclass

```python
GangBudget(money_paise: int, time_minutes: int, max_risk: float) -> None
```
- dataclass

```python
GangConfig(gang_id: str, strength: int, budget: world.gang_config.GangBudget, source: world.gang_config.FixtureSource | None = None, steps: tuple[world.gang_config.BlockStep, ...] = (), start_minute: int | None = None, family: str | None = None) -> None
```
- dataclass

### `world/gang_devices.py`

```python
assign_devices_round_robin(accounts: Sequence[str], devices: Sequence[world.devices.Device]) -> dict[str, world.devices.Device]
```

```python
build_device_control_spans(gang_id: str, account_to_device: dict[str, world.devices.Device], take_control_minute: int, sim_swap: bool, linked_number_switch: bool, emulator: bool, remote_access: bool) -> list[world.truth_records.DeviceControlSpan]
```

```python
build_gang_device_events(devices: Sequence[world.devices.Device], account_to_device: dict[str, world.devices.Device], take_control_minute: int, sim_swap: bool, linked_number_switch: bool, emulator: bool, remote_access: bool, *, bind_control_device: bool = False) -> list[world.gang_devices.GangDeviceEvent]
```

```python
build_rotation_events(accounts: Sequence[str], devices: Sequence[world.devices.Device], take_control_minute: int, period_minutes: int, rotations: int, sim_swap: bool, linked_number_switch: bool, emulator: bool, remote_access: bool, *, bind_control_device: bool = False) -> list[world.gang_devices.GangDeviceEvent]
```

```python
build_rotation_spans(gang_id: str, accounts: Sequence[str], devices: Sequence[world.devices.Device], take_control_minute: int, period_minutes: int, rotations: int, sim_swap: bool, linked_number_switch: bool, emulator: bool, remote_access: bool) -> list[world.truth_records.DeviceControlSpan]
```

```python
make_gang_devices(count: int, rng: numpy.random._generator.Generator, start_index: int = 1, prefix: str | None = None) -> list[world.devices.Device]
```

```python
GangDeviceEvent(minute: int, device_id: str, account_id: str | None, kind: str, platform: str, os_generation: int, fingerprint: str) -> None
```
- dataclass

### `world/bank_feed.py`

```python
bank_visible_vpa(account_id: str) -> str
```

```python
check_bank_visible(record: dict[str, typing.Any], allowed: frozenset[str]) -> None
```

```python
project_payment(event: world.failures.PaymentEvent, payment_id: str, stamp: world.device_runtime.DeviceStamp | None, clock_start: datetime.date, config: world.bank_feed_config.BankFeedConfig, balance_replay: world.bank_feed.BalanceReplay | None = None) -> list[world.bank_feed.BankFeedPaymentSide]
```

```python
project_run(output: world.interactive_records.InteractiveWorldOutput, label_events: Sequence[world.label_records.LabelEvent] = (), phone_book: world.phone_book.PhoneBook | None = None, kyc_phone: bool | None = None) -> world.bank_feed.BankFeed
```

```python
replay_balances(world: world.build_world.World) -> world.bank_feed.BalanceReplay
```

```python
resolve_account_type(account_id: str, config: world.bank_feed_config.BankFeedConfig) -> str
```

```python
resolve_bank_id(bank_index: int, config: world.bank_feed_config.BankFeedConfig) -> str
```

```python
BankFeed(payments_by_bank: dict[str, tuple[world.bank_feed.BankFeedPaymentSide, ...]], account_events_by_bank: dict[str, tuple[world.bank_feed.BankFeedAccountEvent, ...]], complaints_by_bank: dict[str, tuple[world.bank_feed.BankFeedComplaint, ...]], dropped_device_events_by_kind: dict[str, int], withdrawals_by_bank: dict[str, tuple[world.bank_feed.BankFeedWithdrawal, ...]] = <factory>) -> None
```
- dataclass

```python
BankFeedAccountEvent(bank_id: str, event_id: str, account_id: str, kind: str, event_time: datetime.datetime, details: dict[str, typing.Any]) -> None
```
- dataclass

```python
BankFeedComplaint(bank_id: str, complaint_id: str, complainant_account: str, reported_account: str, reported_bank_id: str, payment_ids: tuple[str, ...], claimed_type: str, received_at: datetime.datetime) -> None
```
- dataclass

```python
BankFeedPaymentSide(bank_id: str, payment_id: str, side: str, account_id: str, counterparty_bank_id: str, sender_bank_id: str, receiver_bank_id: str, amount_paise: int, event_time: datetime.datetime, status: str, failure_domain: str | None, payee_type: str, channel: str, counterparty_vpa: str | None = None, sender_device_fingerprint: str | None = None, sender_os_generation: int | None = None, balance_after_paise: int | None = None) -> None
```
- dataclass

```python
BankFeedWithdrawal(withdrawal_id: str, bank_id: str, account_id: str, channel: str, amount_paise: int, status: str, failure_domain: str | None, event_time: datetime.datetime, balance_after_paise: int | None) -> None
```
- dataclass

---

## Records and Fields

Extracted directly via `dataclasses.fields`:

- `BlockStep`:
  - Fields: `block: str`, `params: tuple[tuple[str, Any], ...]`
  - Path: `world/gang_config.py`

- `FixtureSource`:
  - Fields: `account_id: str`, `bank_index: int`, `opening_balance_paise: int`, `start_minute: int`
  - Path: `world/gang_config.py`

- `GangBudget`:
  - Fields: `money_paise: int`, `time_minutes: int`, `max_risk: float`
  - Path: `world/gang_config.py`

- `GangConfig`:
  - Fields: `gang_id: str`, `strength: int`, `budget: GangBudget`, `source: world.gang_config.FixtureSource | None`, `steps: tuple[world.gang_config.BlockStep, ...]`, `start_minute: int | None`, `family: str | None`
  - Path: `world/gang_config.py`

- `GangDeviceEvent`:
  - Fields: `minute: int`, `device_id: str`, `account_id: str | None`, `kind: str`, `platform: str`, `os_generation: int`, `fingerprint: str`
  - Path: `world/gang_devices.py`

- `BankFeed`:
  - Fields: `payments_by_bank: dict[str, tuple[world.bank_feed.BankFeedPaymentSide, ...]]`, `account_events_by_bank: dict[str, tuple[world.bank_feed.BankFeedAccountEvent, ...]]`, `complaints_by_bank: dict[str, tuple[world.bank_feed.BankFeedComplaint, ...]]`, `dropped_device_events_by_kind: dict[str, int]`, `withdrawals_by_bank: dict[str, tuple[world.bank_feed.BankFeedWithdrawal, ...]]`
  - Path: `world/bank_feed.py`

- `BankFeedAccountEvent`:
  - Fields: `bank_id: str`, `event_id: str`, `account_id: str`, `kind: str`, `event_time: datetime`, `details: dict[str, Any]`
  - Path: `world/bank_feed.py`

- `BankFeedComplaint`:
  - Fields: `bank_id: str`, `complaint_id: str`, `complainant_account: str`, `reported_account: str`, `reported_bank_id: str`, `payment_ids: tuple[str, ...]`, `claimed_type: str`, `received_at: datetime`
  - Path: `world/bank_feed.py`

- `BankFeedPaymentSide`:
  - Fields: `bank_id: str`, `payment_id: str`, `side: str`, `account_id: str`, `counterparty_bank_id: str`, `sender_bank_id: str`, `receiver_bank_id: str`, `amount_paise: int`, `event_time: datetime`, `status: str`, `failure_domain: str | None`, `payee_type: str`, `channel: str`, `counterparty_vpa: str | None`, `sender_device_fingerprint: str | None`, `sender_os_generation: int | None`, `balance_after_paise: int | None`
  - Path: `world/bank_feed.py`

- `BankFeedWithdrawal`:
  - Fields: `withdrawal_id: str`, `bank_id: str`, `account_id: str`, `channel: str`, `amount_paise: int`, `status: str`, `failure_domain: str | None`, `event_time: datetime`, `balance_after_paise: int | None`
  - Path: `world/bank_feed.py`

---

## Configuration Keys

### `config/takeover_realism.yaml`
- **parameters**
  - `take_control_share_sim_swap`
  - `take_control_share_linked_number_switch`
  - `take_control_share_emulator`
  - `take_control_share_remote_access`

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
