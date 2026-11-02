# File: tests/unit/test_replay_ordering.py
# Purpose: Unit tests for world-time replay step interleaving and ordering invariants.
"""
WHAT: Unit tests covering causal replay interleaving rules, tie-breaking, and resumed-event rejection.
WHY: Architect decisions 16b Q2, P1, S1, S2, and 16c-1 condition-2 guard:
Ensures decision steps execute strictly after preceding feed events and immediately before their own sides,
with zero leakage of hold-period information or future simulation state.
"""

from dataclasses import dataclass
from datetime import date

import pytest

from core.event_clock import world_minute_to_ist
from replay.ordering import (
    DecisionStep,
    FeedStep,
    build_replay_steps,
    check_no_resumed_events,
)
from world.bank_connectors import DecisionCall
from world.bank_feed import (
    BankFeed,
    BankFeedAccountEvent,
    BankFeedPaymentSide,
    BankFeedWithdrawal,
)
from world.failures import PaymentEvent
from world.interactive_records import PaymentView
from world.replay_recording import RecordedWorld

START_DATE = date(2026, 10, 28)


@dataclass(frozen=True)
class DummyRunResult:
    events: tuple[PaymentEvent, ...]


@dataclass(frozen=True)
class DummyInteractiveResult:
    run_id: str
    payment_ids: tuple[str, ...]
    planned_payment_ids: tuple[str, ...]
    events_resumed: tuple[bool, ...]
    run_result: DummyRunResult


@dataclass(frozen=True)
class DummyClock:
    start_date: date


@dataclass(frozen=True)
class DummyWorldPlan:
    clock: DummyClock


@dataclass(frozen=True)
class DummyInteractiveOutput:
    plan: DummyWorldPlan
    result: DummyInteractiveResult


def make_payment_side(
    payment_id: str,
    side: str,
    minute: int,
    bank_id: str = "bank_0",
    account_id: str = "acc_01",
) -> BankFeedPaymentSide:
    t = world_minute_to_ist(minute, START_DATE)
    return BankFeedPaymentSide(
        bank_id=bank_id,
        payment_id=payment_id,
        side=side,
        account_id=account_id,
        counterparty_bank_id="bank_1",
        sender_bank_id=bank_id,
        receiver_bank_id="bank_1",
        amount_paise=10000,
        event_time=t,
        status="success",
        failure_domain=None,
        payee_type="person",
        channel="upi_app",
    )


def make_dummy_output(
    payment_ids: tuple[str, ...],
    planned_ids: tuple[str, ...],
    events_resumed: tuple[bool, ...],
    minutes: tuple[int, ...],
) -> DummyInteractiveOutput:
    events = tuple(
        PaymentEvent(
            minute=m,
            from_account="acc_01",
            to_account="acc_02",
            amount_paise=10000,
            from_bank_index=0,
            to_bank_index=1,
            status="success",
            reason="ok",
        )
        for m in minutes
    )
    res = DummyInteractiveResult(
        run_id="6cadb2486265507d",
        payment_ids=payment_ids,
        planned_payment_ids=planned_ids,
        events_resumed=events_resumed,
        run_result=DummyRunResult(events=events),
    )
    return DummyInteractiveOutput(
        plan=DummyWorldPlan(clock=DummyClock(start_date=START_DATE)),
        result=res,
    )


def test_own_sides_come_after_decision_and_unrelated_at_same_minute_comes_before() -> None:
    """
    1. Unrelated record at the same minute comes BEFORE the decision (Rule 1).
    2. Decision's own sides come immediately AFTER the decision (Rule 2).
    """
    output = make_dummy_output(
        payment_ids=("pmt_001",),
        planned_ids=("plan_001",),
        events_resumed=(False,),
        minutes=(10,),
    )
    call = DecisionCall("bank_0", "bank_1", b"{}", "dec:plan_001")
    payment = PaymentView("plan_001", 10, "acc_01", "acc_02", 10000)
    rec_world = RecordedWorld(output=output, calls=((payment, call),))  # type: ignore[arg-type]

    # Unrelated account event at minute 10
    t10 = world_minute_to_ist(10, START_DATE)
    unrelated_acct = BankFeedAccountEvent(
        bank_id="bank_0",
        event_id="evt_unrelated_01",
        account_id="acc_99",
        kind="account_opened",
        event_time=t10,
        details={},
    )
    # Own transaction side for pmt_001 at minute 10
    own_side = make_payment_side("pmt_001", "debit", 10)

    feed = BankFeed(
        payments_by_bank={"bank_0": (own_side,)},
        account_events_by_bank={"bank_0": (unrelated_acct,)},
        withdrawals_by_bank={},
        complaints_by_bank={},
        dropped_device_events_by_kind={},
    )

    steps = build_replay_steps(rec_world, feed)
    assert len(steps) == 3

    # Step 0: Unrelated record at minute 10 comes BEFORE decision
    assert isinstance(steps[0], FeedStep)
    assert steps[0].record_id == "evt_unrelated_01"

    # Step 1: Decision
    assert isinstance(steps[1], DecisionStep)
    assert steps[1].planned_payment_id == "plan_001"

    # Step 2: Own side comes immediately AFTER decision
    assert isinstance(steps[2], FeedStep)
    assert steps[2].record_id == "pmt_001:debit"


def test_two_decisions_in_one_minute_keep_hook_order_with_own_sides_between_them() -> None:
    """
    Decisions with the same world minute keep hook order; each is followed by its own sides
    before the next decision (Rule 3).
    """
    output = make_dummy_output(
        payment_ids=("pmt_001", "pmt_002"),
        planned_ids=("plan_001", "plan_002"),
        events_resumed=(False, False),
        minutes=(10, 10),
    )
    call1 = DecisionCall("bank_0", "bank_1", b"{}", "dec:plan_001")
    pay1 = PaymentView("plan_001", 10, "acc_01", "acc_02", 10000)

    call2 = DecisionCall("bank_0", "bank_1", b"{}", "dec:plan_002")
    pay2 = PaymentView("plan_002", 10, "acc_03", "acc_04", 20000)

    rec_world = RecordedWorld(
        output=output,  # type: ignore[arg-type]
        calls=((pay1, call1), (pay2, call2)),
    )

    side1 = make_payment_side("pmt_001", "debit", 10)
    side2 = make_payment_side("pmt_002", "debit", 10)

    feed = BankFeed(
        payments_by_bank={"bank_0": (side1, side2)},
        account_events_by_bank={},
        withdrawals_by_bank={},
        complaints_by_bank={},
        dropped_device_events_by_kind={},
    )

    steps = build_replay_steps(rec_world, feed)
    assert len(steps) == 4

    # 1. Decision 1
    assert isinstance(steps[0], DecisionStep)
    assert steps[0].planned_payment_id == "plan_001"

    # 2. Side 1
    assert isinstance(steps[1], FeedStep)
    assert steps[1].record_id == "pmt_001:debit"

    # 3. Decision 2
    assert isinstance(steps[2], DecisionStep)
    assert steps[2].planned_payment_id == "plan_002"

    # 4. Side 2
    assert isinstance(steps[3], FeedStep)
    assert steps[3].record_id == "pmt_002:debit"


def test_record_one_minute_later_comes_after() -> None:
    """A feed record occurring at minute 11 comes after a minute 10 decision and its sides."""
    output = make_dummy_output(
        payment_ids=("pmt_001",),
        planned_ids=("plan_001",),
        events_resumed=(False,),
        minutes=(10,),
    )
    call = DecisionCall("bank_0", "bank_1", b"{}", "dec:plan_001")
    pay = PaymentView("plan_001", 10, "acc_01", "acc_02", 10000)
    rec_world = RecordedWorld(output=output, calls=((pay, call),))  # type: ignore[arg-type]

    side1 = make_payment_side("pmt_001", "debit", 10)

    t11 = world_minute_to_ist(11, START_DATE)
    later_withdrawal = BankFeedWithdrawal(
        withdrawal_id="wth_later_01",
        bank_id="bank_0",
        account_id="acc_01",
        channel="atm",
        amount_paise=5000,
        status="success",
        failure_domain=None,
        event_time=t11,
        balance_after_paise=5000,
    )

    feed = BankFeed(
        payments_by_bank={"bank_0": (side1,)},
        account_events_by_bank={},
        withdrawals_by_bank={"bank_0": (later_withdrawal,)},
        complaints_by_bank={},
        dropped_device_events_by_kind={},
    )

    steps = build_replay_steps(rec_world, feed)
    assert len(steps) == 3

    assert isinstance(steps[0], DecisionStep)
    assert steps[0].planned_payment_id == "plan_001"

    assert isinstance(steps[1], FeedStep)
    assert steps[1].record_id == "pmt_001:debit"

    assert isinstance(steps[2], FeedStep)
    assert steps[2].record_id == "wth_later_01"


def test_ties_are_deterministic() -> None:
    """Building replay steps twice from the same inputs yields identical ordered lists."""
    output = make_dummy_output(
        payment_ids=("pmt_001",),
        planned_ids=("plan_001",),
        events_resumed=(False,),
        minutes=(10,),
    )
    call = DecisionCall("bank_0", "bank_1", b"{}", "dec:plan_001")
    pay = PaymentView("plan_001", 10, "acc_01", "acc_02", 10000)
    rec_world = RecordedWorld(output=output, calls=((pay, call),))  # type: ignore[arg-type]

    t10 = world_minute_to_ist(10, START_DATE)
    acct1 = BankFeedAccountEvent("bank_1", "evt_b", "acc_02", "sim_swap", t10, {})
    acct2 = BankFeedAccountEvent("bank_0", "evt_a", "acc_01", "sim_swap", t10, {})
    side1 = make_payment_side("pmt_001", "debit", 10)

    feed = BankFeed(
        payments_by_bank={"bank_0": (side1,)},
        account_events_by_bank={"bank_0": (acct2,), "bank_1": (acct1,)},
        withdrawals_by_bank={},
        complaints_by_bank={},
        dropped_device_events_by_kind={},
    )

    steps1 = build_replay_steps(rec_world, feed)
    steps2 = build_replay_steps(rec_world, feed)
    assert steps1 == steps2


def test_resumed_event_raises_value_error_naming_id_and_minute() -> None:
    """check_no_resumed_events and build_replay_steps raise ValueError naming id and minute on resumed events."""
    output = make_dummy_output(
        payment_ids=("pmt_resumed_99",),
        planned_ids=("plan_resumed_99",),
        events_resumed=(True,),
        minutes=(42,),
    )
    call = DecisionCall("bank_0", "bank_1", b"{}", "dec:plan_resumed_99")
    pay = PaymentView("plan_resumed_99", 42, "acc_01", "acc_02", 10000)
    rec_world = RecordedWorld(output=output, calls=((pay, call),))  # type: ignore[arg-type]

    feed = BankFeed(
        payments_by_bank={},
        account_events_by_bank={},
        withdrawals_by_bank={},
        complaints_by_bank={},
        dropped_device_events_by_kind={},
    )

    with pytest.raises(ValueError, match=r"pmt_resumed_99.*42"):
        check_no_resumed_events(output)

    with pytest.raises(ValueError, match=r"pmt_resumed_99.*42"):
        build_replay_steps(rec_world, feed)
