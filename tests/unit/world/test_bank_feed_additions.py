# File: tests/unit/world/test_bank_feed_additions.py
# Purpose: Unit tests for Level 15b bank feed additions: balance replay, cash withdrawals, and account home regions.

from dataclasses import asdict
from datetime import date, datetime, timezone
from types import SimpleNamespace
from typing import Any, cast

import pytest

from world.bank_feed import (
    ACCOUNT_OPENED_DETAIL_FIELDS,
    BANK_VISIBLE_WITHDRAWAL_FIELDS,
    ProjectionLeakError,
    check_bank_visible,
    project_payment,
    project_run,
    replay_balances,
)
from world.bank_feed_config import BankFeedConfig, load_bank_feed_config
from world.cash_withdrawals import CashWithdrawalRecord
from world.failures import PaymentEvent
from world.ledger import Ledger

CLOCK_START = date(2026, 10, 28)
CLOCK_START_DT = datetime(2026, 10, 28, 0, 0, 0, tzinfo=timezone.utc)


@pytest.fixture(scope="module")
def feed_config() -> BankFeedConfig:
    """Provides a verified BankFeedConfig instance for the test module."""
    return load_bank_feed_config()


def test_balance_replay_matches_ledger_balance_after_1000_operations() -> None:
    """Test 1: Balance replay matches ledger.balance_of for every account after 1000 mixed operations."""
    ledger_0 = Ledger(bank_index=0, upi_limit_paise=10000000)
    ledger_1 = Ledger(bank_index=1, upi_limit_paise=10000000)

    # Initialize 10 accounts across 2 bank ledgers with initial opening balances.
    accounts_0 = [f"acc_person_{i:06d}" for i in range(5)]
    accounts_1 = [f"acc_shop_{i:06d}" for i in range(5)]

    for acc in accounts_0:
        ledger_0.add_account(acc, opening_balance_paise=10_000_000, minute=0)
    for acc in accounts_1:
        ledger_1.add_account(acc, opening_balance_paise=10_000_000, minute=0)

    world = SimpleNamespace(ledgers=[ledger_0, ledger_1])

    # Execute 1000 mixed payments and withdrawals across accounts.
    for op_idx in range(1, 1001):
        minute = op_idx
        sender = accounts_0[op_idx % 5]
        receiver = accounts_1[(op_idx + 1) % 5]
        amount = 1000 + (op_idx % 500)

        tx_id = f"tx_{minute}_{sender}_{receiver}_{amount}"
        if op_idx % 4 == 0:
            # Simulates a cash withdrawal debit
            w_id = f"cw_{minute}_{sender}_atm_0"
            ledger_0.post_debit(sender, amount, minute, w_id)
        else:
            # Payment transfer debit and credit
            ledger_0.post_debit(sender, amount, minute, tx_id)
            ledger_1.post_credit(receiver, amount, minute, tx_id)

    # Walk the ledgers using replay_balances.
    replay = replay_balances(world)

    # Verify every account's replayed final balance equals authoritative ledger.balance_of.
    for acc in accounts_0:
        assert replay.get_final_balance(acc) == ledger_0.balance_of(acc)
    for acc in accounts_1:
        assert replay.get_final_balance(acc) == ledger_1.balance_of(acc)


def test_two_identical_payments_in_same_minute_get_monotonic_balances(feed_config: BankFeedConfig) -> None:
    """Test 2: Two identical payments in the same minute get distinct, monotonic balances via occurrence tracking."""
    ledger_0 = Ledger(bank_index=0, upi_limit_paise=10000000)
    ledger_1 = Ledger(bank_index=1, upi_limit_paise=10000000)

    sender = "acc_person_000001"
    receiver = "acc_shop_000001"
    initial_sender_balance = 500000
    initial_receiver_balance = 100000

    ledger_0.add_account(sender, opening_balance_paise=initial_sender_balance, minute=0)
    ledger_1.add_account(receiver, opening_balance_paise=initial_receiver_balance, minute=0)

    world = SimpleNamespace(ledgers=[ledger_0, ledger_1])

    # Two identical payments in minute 15 for amount 50000 paise.
    minute = 15
    amount = 50000
    tx_id = f"tx_{minute}_{sender}_{receiver}_{amount}"

    # Payment 1
    ledger_0.post_debit(sender, amount, minute, tx_id)
    ledger_1.post_credit(receiver, amount, minute, tx_id)

    # Payment 2 (identical parameters and minute)
    ledger_0.post_debit(sender, amount, minute, tx_id)
    ledger_1.post_credit(receiver, amount, minute, tx_id)

    replay = replay_balances(world)

    event1 = PaymentEvent(
        minute=minute,
        from_account=sender,
        to_account=receiver,
        amount_paise=amount,
        from_bank_index=0,
        to_bank_index=1,
        status="success",
        reason="ok",
        failure_domain=None,
    )
    event2 = PaymentEvent(
        minute=minute,
        from_account=sender,
        to_account=receiver,
        amount_paise=amount,
        from_bank_index=0,
        to_bank_index=1,
        status="success",
        reason="ok",
        failure_domain=None,
    )

    sides_1 = project_payment(event1, "pmt_001", None, CLOCK_START, feed_config, replay)
    sides_2 = project_payment(event2, "pmt_002", None, CLOCK_START, feed_config, replay)

    debit_1 = next(s for s in sides_1 if s.side == "debit")
    credit_1 = next(s for s in sides_1 if s.side == "credit")
    debit_2 = next(s for s in sides_2 if s.side == "debit")
    credit_2 = next(s for s in sides_2 if s.side == "credit")

    # Assert distinct balances
    assert debit_1.balance_after_paise == initial_sender_balance - amount
    assert debit_2.balance_after_paise == initial_sender_balance - (2 * amount)
    assert debit_1.balance_after_paise != debit_2.balance_after_paise
    assert debit_1.balance_after_paise > debit_2.balance_after_paise  # Strictly decreasing debit balances

    assert credit_1.balance_after_paise == initial_receiver_balance + amount
    assert credit_2.balance_after_paise == initial_receiver_balance + (2 * amount)
    assert credit_1.balance_after_paise != credit_2.balance_after_paise
    assert credit_1.balance_after_paise < credit_2.balance_after_paise  # Strictly increasing credit balances


def test_failed_payment_has_balance_after_none_and_ledger_unchanged(feed_config: BankFeedConfig) -> None:
    """Test 3: Failed payment has balance_after_paise = None; ledger balance remains completely unchanged."""
    ledger_0 = Ledger(bank_index=0, upi_limit_paise=10000000)
    sender = "acc_person_000001"
    receiver = "acc_shop_000001"
    initial_balance = 500000
    ledger_0.add_account(sender, opening_balance_paise=initial_balance, minute=0)
    world = SimpleNamespace(ledgers=[ledger_0])

    replay = replay_balances(world)

    failed_event = PaymentEvent(
        minute=20,
        from_account=sender,
        to_account=receiver,
        amount_paise=100000,
        from_bank_index=0,
        to_bank_index=1,
        status="failed",
        reason="technical_decline",
        failure_domain="sender_bank",
    )

    sides = project_payment(failed_event, "pmt_fail_001", None, CLOCK_START, feed_config, replay)
    assert len(sides) == 1
    debit_side = sides[0]

    assert debit_side.status == "failed"
    assert debit_side.balance_after_paise is None
    # Ledger balance unchanged
    assert ledger_0.balance_of(sender) == initial_balance


def test_cross_bank_payments_show_independent_balances(feed_config: BankFeedConfig) -> None:
    """Test 4: Cross-bank payments show the correct balance on debit and credit sides independently."""
    ledger_0 = Ledger(bank_index=0, upi_limit_paise=10000000)
    ledger_1 = Ledger(bank_index=1, upi_limit_paise=10000000)

    sender = "acc_person_000010"
    receiver = "acc_shop_000020"
    ledger_0.add_account(sender, opening_balance_paise=800000, minute=0)
    ledger_1.add_account(receiver, opening_balance_paise=250000, minute=0)

    world = SimpleNamespace(ledgers=[ledger_0, ledger_1])

    minute = 35
    amount = 150000
    tx_id = f"tx_{minute}_{sender}_{receiver}_{amount}"
    ledger_0.post_debit(sender, amount, minute, tx_id)
    ledger_1.post_credit(receiver, amount, minute, tx_id)

    replay = replay_balances(world)

    event = PaymentEvent(
        minute=minute,
        from_account=sender,
        to_account=receiver,
        amount_paise=amount,
        from_bank_index=0,
        to_bank_index=1,
        status="success",
        reason="ok",
        failure_domain=None,
    )

    sides = project_payment(event, "pmt_xb_001", None, CLOCK_START, feed_config, replay)
    debit = next(s for s in sides if s.side == "debit")
    credit = next(s for s in sides if s.side == "credit")

    assert debit.bank_id == "bank_0"
    assert debit.balance_after_paise == 650000
    assert credit.bank_id == "bank_1"
    assert credit.balance_after_paise == 400000


def test_withdrawal_projection_success_failed_skipped(feed_config: BankFeedConfig) -> None:
    """Test 5: Withdrawal projection: success shows reduced balance; failed shows unchanged;
    skipped omitted; reason dropped.
    """
    ledger_0 = Ledger(bank_index=0, upi_limit_paise=10000000)
    acc = "acc_person_000001"
    start_balance = 500000
    ledger_0.add_account(acc, opening_balance_paise=start_balance, minute=0)

    # 1. Successful withdrawal at minute 10 of 100000 paise
    w_succ_id = "cw_10_acc_person_000001_atm_0"
    ledger_0.post_debit(acc, 100000, 10, w_succ_id)
    rec_succ = CashWithdrawalRecord(
        id=w_succ_id,
        minute=10,
        account=acc,
        channel="atm",
        requested=100000,
        withdrawn=100000,
        status="success",
        reason=None,
        domain=None,
    )

    # 2. Failed withdrawal at minute 20 for 1000000 paise (insufficient funds, no ledger debit)
    w_fail_id = "cw_20_acc_person_000001_branch_0"
    rec_fail = CashWithdrawalRecord(
        id=w_fail_id,
        minute=20,
        account=acc,
        channel="branch",
        requested=1000000,
        withdrawn=0,
        status="failed",
        reason="insufficient_balance",
        domain="customer",
    )

    # 3. Skipped withdrawal at minute 30 (encumbrance hold, never touched bank)
    rec_skip = CashWithdrawalRecord(
        id="cw_30_acc_person_000001_atm_0",
        minute=30,
        account=acc,
        channel="atm",
        requested=50000,
        withdrawn=0,
        status="skipped",
        reason="funds_on_hold",
        domain=None,
    )

    world = SimpleNamespace(
        ledgers=[ledger_0],
        people=[SimpleNamespace(person_id="person_000001", account_id=acc, region=2)],
        shops=[],
    )

    result = SimpleNamespace(
        run_id="run_test_w",
        run_result=SimpleNamespace(events=[], event_sender_devices=[]),
        payment_ids=[],
        applied_device_events=[],
        applied_gang_device_events=[],
        opened_accounts=[],
        cash_withdrawals=[rec_succ, rec_fail, rec_skip],
    )
    plan = SimpleNamespace(clock=SimpleNamespace(start_date=CLOCK_START))
    output = SimpleNamespace(world=world, plan=plan, result=result)

    feed = project_run(cast(Any, output))
    withdrawals = feed.withdrawals_by_bank["bank_0"]

    # Skipped withdrawal must be omitted; only success and failed remain.
    assert len(withdrawals) == 2
    ids = [w.withdrawal_id for w in withdrawals]
    assert w_succ_id in ids
    assert w_fail_id in ids
    assert "cw_30_acc_person_000001_atm_0" not in ids

    # Successful withdrawal shows reduced balance (500000 - 100000 = 400000)
    w_succ = next(w for w in withdrawals if w.withdrawal_id == w_succ_id)
    assert w_succ.status == "success"
    assert w_succ.amount_paise == 100000
    assert w_succ.balance_after_paise == 400000
    assert w_succ.failure_domain is None

    # Failed withdrawal shows unchanged balance at that moment (400000) and failure_domain='customer'
    w_fail = next(w for w in withdrawals if w.withdrawal_id == w_fail_id)
    assert w_fail.status == "failed"
    assert w_fail.amount_paise == 1000000
    assert w_fail.balance_after_paise == 400000
    assert w_fail.failure_domain == "customer"

    # Reason field is NEVER projected in bank-visible record.
    for w in withdrawals:
        w_dict = asdict(w)
        assert "reason" not in w_dict
        check_bank_visible(w_dict, BANK_VISIBLE_WITHDRAWAL_FIELDS)


def test_account_opened_home_region_and_config_switch() -> None:
    """Test 6: Account opened includes home_region; config switch controls start-of-run existing account emission."""
    ledger_0 = Ledger(bank_index=0, upi_limit_paise=10000000)
    people_accounts = [f"acc_person_{i:06d}" for i in range(10)]
    shop_accounts = [f"acc_shop_{i:06d}" for i in range(2)]
    for acc in people_accounts:
        ledger_0.add_account(acc, opening_balance_paise=100000, minute=0)
    for acc in shop_accounts:
        ledger_0.add_account(acc, opening_balance_paise=500000, minute=0)

    world = SimpleNamespace(
        ledgers=[ledger_0],
        people=[
            SimpleNamespace(person_id=f"person_{i:06d}", account_id=acc, region=1, bank_index=0)
            for i, acc in enumerate(people_accounts)
        ],
        shops=[SimpleNamespace(account_id=acc, region=2, bank_index=0) for acc in shop_accounts],
    )
    result = SimpleNamespace(
        run_id="run_test_open",
        run_result=SimpleNamespace(events=[], event_sender_devices=[]),
        payment_ids=[],
        applied_device_events=[],
        applied_gang_device_events=[],
        opened_accounts=[],
        cash_withdrawals=[],
    )
    plan = SimpleNamespace(clock=SimpleNamespace(start_date=CLOCK_START))
    out = SimpleNamespace(world=world, plan=plan, result=result)

    feed_enabled = project_run(cast(Any, out))
    all_events = [ev for events in feed_enabled.account_events_by_bank.values() for ev in events]
    open_events = [ev for ev in all_events if ev.kind == "account_opened"]

    # With emit_opening_for_existing_accounts=True, all 12 accounts (10 people + 2 shops) have open events.
    assert len(open_events) == 12
    for ev in open_events:
        assert "home_region" in ev.details
        assert isinstance(ev.details["home_region"], int)
        check_bank_visible(ev.details, ACCOUNT_OPENED_DETAIL_FIELDS)


def test_withdrawal_projection_leak_check_raises_projection_leak_error() -> None:
    """Test 7: Attempting to project or audit a withdrawal with extra unapproved fields raises ProjectionLeakError."""
    valid_record = {
        "withdrawal_id": "cw_01",
        "bank_id": "bank_0",
        "account_id": "acc_01",
        "channel": "atm",
        "amount_paise": 50000,
        "status": "success",
        "failure_domain": None,
        "event_time": CLOCK_START_DT,
        "balance_after_paise": 450000,
    }
    # Clean record passes audit without error
    check_bank_visible(valid_record, BANK_VISIBLE_WITHDRAWAL_FIELDS)

    # Attempting to serialize simulation reason raises ProjectionLeakError
    leaky_record = dict(valid_record)
    leaky_record["reason"] = "insufficient_balance"
    with pytest.raises(ProjectionLeakError, match="Record contains bank-invisible fields: \\['reason'\\]"):
        check_bank_visible(leaky_record, BANK_VISIBLE_WITHDRAWAL_FIELDS)

    # Attempting to serialize internal simulation flag raises ProjectionLeakError
    leaky_flag = dict(valid_record)
    leaky_flag["illicit_flow_id"] = "flow_xyz"
    with pytest.raises(ProjectionLeakError, match="Record contains bank-invisible fields: \\['illicit_flow_id'\\]"):
        check_bank_visible(leaky_flag, BANK_VISIBLE_WITHDRAWAL_FIELDS)
