# File: tests/unit/test_event_decisions.py
# Purpose: Unit tests for per-event decisions and receiver_vpa builder integration (Decisions D1-D4).
"""
WHAT: Unit tests covering bank_visible_vpa, receiver_vpa inclusion in build_decision_request,
and per-event decision call derivation.

WHY: Verifies architect decisions P2(b), 16c-1 conditions 1-3, and receiver_vpa (a) ensuring
deterministic payload construction and byte-level backward compatibility.
"""
import asyncio
from dataclasses import dataclass
from datetime import date

import orjson
import pytest

from core.event_clock import world_minute_to_ist
from replay.event_decisions import build_event_decision_calls
from world.bank_connectors import MockBankConnectors
from world.bank_feed import (
    bank_visible_vpa,
    project_payment,
)
from world.bank_feed_config import BankFeedConfig
from world.failures import PaymentEvent
from world.interactive_records import PaymentView


@pytest.fixture
def sample_feed_config() -> BankFeedConfig:
    """Provides a deterministic feed configuration for tests."""
    return BankFeedConfig(
        bank_ids_by_index={0: "bank_0", 1: "bank_1"},
        payee_type_by_account_prefix={"acc_person_": "person", "acc_shop_": "merchant"},
        channel_value="upi_app",
        decision_timeout_seconds=2.0,
        failed_payment_visibility={"sender_bank": {"sender": True, "receiver": False}},
        device_event_kind_map={"phone_change": "device_binding_change"},
        unmapped_gang_device_kinds=("emulator_enabled",),
        allowed_account_event_kinds=("account_opened",),
        emit_opening_for_existing_accounts=True,
    )


@pytest.fixture
def connectors(sample_feed_config: BankFeedConfig) -> MockBankConnectors:
    """Creates a MockBankConnectors instance with test accounts mapped."""
    loop = asyncio.new_event_loop()
    return MockBankConnectors(
        transports={},
        modes={"bank_0": "shadow", "bank_1": "shadow"},
        account_bank_map={
            "acc_person_001": "bank_0",
            "acc_shop_001": "bank_1",
        },
        clock_start=date(2026, 10, 28),
        feed_config=sample_feed_config,
        decision_timeout_seconds=2.0,
        loop=loop,
        max_identifier_length=128,
    )


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
class DummyInteractiveOutput:
    result: DummyInteractiveResult


@dataclass(frozen=True)
class DummyRecordedWorld:
    output: DummyInteractiveOutput


def make_handbuilt_recorded_world(
    events: tuple[PaymentEvent, ...],
    payment_ids: tuple[str, ...],
    planned_payment_ids: tuple[str, ...],
    events_resumed: tuple[bool, ...],
    run_id: str = "test_run_001",
) -> DummyRecordedWorld:
    """Constructs a small hand-built RecordedWorld for isolated unit tests."""
    return DummyRecordedWorld(
        output=DummyInteractiveOutput(
            result=DummyInteractiveResult(
                run_id=run_id,
                payment_ids=payment_ids,
                planned_payment_ids=planned_payment_ids,
                events_resumed=events_resumed,
                run_result=DummyRunResult(events=events),
            )
        )
    )


def test_bank_visible_vpa_matches_feed_counterparty_vpa(
    sample_feed_config: BankFeedConfig,
) -> None:
    """
    D1: Verifies bank_visible_vpa returns the opaque stand-in VPA exactly matching
    the bank feed's counterparty_vpa on both debit and credit sides.
    """
    account_id = "acc_shop_001"
    vpa = bank_visible_vpa(account_id)
    assert vpa == account_id

    event = PaymentEvent(
        minute=15,
        from_account="acc_person_001",
        to_account="acc_shop_001",
        amount_paise=250000,
        from_bank_index=0,
        to_bank_index=1,
        status="success",
        reason="ok",
        failure_domain=None,
    )

    sides = project_payment(
        event=event,
        payment_id="pay_001",
        stamp=None,
        clock_start=date(2026, 10, 28),
        config=sample_feed_config,
    )

    assert len(sides) == 2
    debit, credit = sides[0], sides[1]
    assert debit.counterparty_vpa == bank_visible_vpa(event.to_account)
    assert credit.counterparty_vpa == bank_visible_vpa(event.from_account)


def test_build_decision_request_receiver_vpa_flag(
    connectors: MockBankConnectors,
    sample_feed_config: BankFeedConfig,
) -> None:
    """
    D2: Verifies include_receiver_vpa=False produces today's exact bytes with no receiver_vpa key,
    while include_receiver_vpa=True adds only receiver_vpa matching bank_visible_vpa(to_account)
    and the feed's counterparty_vpa.
    """
    payment = PaymentView(
        planned_payment_id="plan_shop_001",
        minute=15,
        from_account="acc_person_001",
        to_account="acc_shop_001",
        amount_paise=250000,
    )

    # 1. Default (False) matches explicit include_receiver_vpa=False
    call_default = connectors.build_decision_request(payment)
    call_false = connectors.build_decision_request(payment, include_receiver_vpa=False)
    assert call_default.body_bytes == call_false.body_bytes

    body_false = orjson.loads(call_false.body_bytes)
    assert "receiver_vpa" not in body_false

    # 2. include_receiver_vpa=True adds only "receiver_vpa"
    call_true = connectors.build_decision_request(payment, include_receiver_vpa=True)
    body_true = orjson.loads(call_true.body_bytes)

    assert "receiver_vpa" in body_true
    expected_vpa = bank_visible_vpa(payment.to_account)
    assert body_true["receiver_vpa"] == expected_vpa

    # 3. No other keys added or changed
    added_keys = set(body_true.keys()) - set(body_false.keys())
    assert added_keys == {"receiver_vpa"}
    for k in body_false:
        assert body_true[k] == body_false[k]

    # 4. Equals the feed's counterparty_vpa for the same payment debit record
    event = PaymentEvent(
        minute=15,
        from_account=payment.from_account,
        to_account=payment.to_account,
        amount_paise=payment.amount_paise,
        from_bank_index=0,
        to_bank_index=1,
        status="success",
        reason="ok",
        failure_domain=None,
    )
    sides = project_payment(
        event=event,
        payment_id=payment.planned_payment_id,
        stamp=None,
        clock_start=date(2026, 10, 28),
        config=sample_feed_config,
    )
    debit_side = sides[0]
    assert body_true["receiver_vpa"] == debit_side.counterparty_vpa


def test_build_event_decision_calls_one_call_per_event_attributes(
    connectors: MockBankConnectors,
) -> None:
    """
    D3: Verifies build_event_decision_calls produces exactly one DecisionCall per executed event,
    with payment_id equal to the event's payment id, amount equal to piece amount, and event_time
    equal to event minute.
    """
    events = (
        PaymentEvent(
            minute=20,
            from_account="acc_person_001",
            to_account="acc_shop_001",
            amount_paise=150000,
            from_bank_index=0,
            to_bank_index=1,
            status="success",
            reason="ok",
        ),
        PaymentEvent(
            minute=25,
            from_account="acc_person_001",
            to_account="acc_shop_001",
            amount_paise=200000,
            from_bank_index=0,
            to_bank_index=1,
            status="success",
            reason="ok",
        ),
    )
    payment_ids = ("evt_pay_001", "evt_pay_002")
    planned_ids = ("plan_001", "plan_002")
    events_resumed = (False, False)

    world = make_handbuilt_recorded_world(events, payment_ids, planned_ids, events_resumed)
    calls = build_event_decision_calls(world, connectors)

    assert len(calls) == 2
    for i, (event_pid, view, call) in enumerate(calls):
        ev = events[i]
        assert event_pid == payment_ids[i]
        assert view.planned_payment_id == event_pid
        assert view.amount_paise == ev.amount_paise
        assert view.minute == ev.minute

        body = orjson.loads(call.body_bytes)
        assert body["payment_id"] == event_pid
        assert body["amount_paise"] == ev.amount_paise
        expected_iso = world_minute_to_ist(ev.minute, connectors.clock_start).isoformat()
        assert body["event_time"] == expected_iso
        assert body["receiver_vpa"] == bank_visible_vpa(ev.to_account)


def test_build_event_decision_calls_condition_1_attempt_time_only(
    connectors: MockBankConnectors,
) -> None:
    """
    D3 (Condition 1): Verifies a FAILED piece and a SUCCESSFUL piece produce bodies with
    exactly the same key set, and no post-attempt key (status, reason, failure_domain,
    balance_after_paise) appears anywhere in the request.
    """
    events = (
        PaymentEvent(
            minute=30,
            from_account="acc_person_001",
            to_account="acc_shop_001",
            amount_paise=100000,
            from_bank_index=0,
            to_bank_index=1,
            status="failed",
            reason="insufficient_funds",
            failure_domain="sender_bank",
        ),
        PaymentEvent(
            minute=35,
            from_account="acc_person_001",
            to_account="acc_shop_001",
            amount_paise=100000,
            from_bank_index=0,
            to_bank_index=1,
            status="success",
            reason="ok",
            failure_domain=None,
        ),
    )
    payment_ids = ("evt_fail_001", "evt_succ_002")
    planned_ids = ("plan_fail_001", "plan_succ_002")
    events_resumed = (False, False)

    world = make_handbuilt_recorded_world(events, payment_ids, planned_ids, events_resumed)
    calls = build_event_decision_calls(world, connectors)

    assert len(calls) == 2
    body_failed = orjson.loads(calls[0][2].body_bytes)
    body_success = orjson.loads(calls[1][2].body_bytes)

    # Identical key sets regardless of outcome
    assert set(body_failed.keys()) == set(body_success.keys())

    # Strictly attempt-time only: no post-attempt fields
    forbidden_keys = {"status", "reason", "failure_domain", "balance_after_paise"}
    assert not (forbidden_keys & set(body_failed.keys()))
    assert not (forbidden_keys & set(body_success.keys()))


def test_build_event_decision_calls_split_payment(
    connectors: MockBankConnectors,
) -> None:
    """
    D3: Verifies a split payment (multiple events sharing one planned_payment_id)
    produces separate decision calls with distinct event payment ids.
    """
    events = (
        PaymentEvent(
            minute=40,
            from_account="acc_person_001",
            to_account="acc_shop_001",
            amount_paise=50000,
            from_bank_index=0,
            to_bank_index=1,
            status="success",
            reason="ok",
        ),
        PaymentEvent(
            minute=40,
            from_account="acc_person_001",
            to_account="acc_shop_001",
            amount_paise=75000,
            from_bank_index=0,
            to_bank_index=1,
            status="success",
            reason="ok",
        ),
    )
    payment_ids = ("evt_piece_001", "evt_piece_002")
    planned_ids = ("plan_split_001", "plan_split_001")  # Shared planned id
    events_resumed = (False, False)

    world = make_handbuilt_recorded_world(events, payment_ids, planned_ids, events_resumed)
    calls = build_event_decision_calls(world, connectors)

    assert len(calls) == 2
    assert calls[0][0] == "evt_piece_001"
    assert calls[1][0] == "evt_piece_002"
    assert calls[0][0] != calls[1][0]

    body_0 = orjson.loads(calls[0][2].body_bytes)
    body_1 = orjson.loads(calls[1][2].body_bytes)
    assert body_0["payment_id"] == "evt_piece_001"
    assert body_1["payment_id"] == "evt_piece_002"
    assert body_0["amount_paise"] == 50000
    assert body_1["amount_paise"] == 75000


def test_build_event_decision_calls_resumed_event_raises(
    connectors: MockBankConnectors,
) -> None:
    """
    D3: Verifies build_event_decision_calls rejects resumed events and raises ValueError
    naming the offending payment id and minute.
    """
    events = (
        PaymentEvent(
            minute=42,
            from_account="acc_person_001",
            to_account="acc_shop_001",
            amount_paise=100000,
            from_bank_index=0,
            to_bank_index=1,
            status="success",
            reason="ok",
        ),
    )
    payment_ids = ("evt_resumed_001",)
    planned_ids = ("plan_resumed_001",)
    events_resumed = (True,)

    world = make_handbuilt_recorded_world(events, payment_ids, planned_ids, events_resumed)

    with pytest.raises(ValueError) as exc_info:
        build_event_decision_calls(world, connectors)

    msg = str(exc_info.value)
    assert "evt_resumed_001" in msg
    assert "42" in msg


def test_make_hook_decisions_per_event_payment_raises(
    connectors: MockBankConnectors,
) -> None:
    """
    D4: Verifies make_hook(decisions_per_event_payment=True) raises ValueError with the exact D4 message.
    """
    with pytest.raises(ValueError) as exc_info:
        connectors.make_hook(decisions_per_event_payment=True)

    expected_msg = (
        "decisions per event payment are replay-only: "
        "the event payment id does not exist at hook time in the live world loop (deferred to Level 19)"
    )
    assert str(exc_info.value) == expected_msg
