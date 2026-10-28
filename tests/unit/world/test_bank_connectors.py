# File: tests/unit/world/test_bank_connectors.py
# Purpose: Unit tests verifying MockBankConnectors logic, mode policies, timeouts, and feed routing.
"""
WHAT: Unit tests verifying MockBankConnectors logic, mode policies, timeouts, and feed routing.
WHY: Exercises MockBankConnectors with in-memory FakeTransport instances to prove that payment
decision requests are constructed correctly, shadow/live modes behave as specified, timeouts
fall back to allow, and bank feed events are dispatched to the appropriate banks in order.
"""

import asyncio
from datetime import date, datetime
from typing import Any

import orjson
import pytest

from core.event_clock import IST
from world.bank_connectors import (
    BankTransport,
    MockBankConnectors,
    TransportOutcome,
    make_account_event_idempotency_key,
    make_complaint_idempotency_key,
    make_decision_idempotency_key,
    make_transaction_idempotency_key,
)
from world.bank_feed import (
    BankFeed,
    BankFeedAccountEvent,
    BankFeedComplaint,
    BankFeedPaymentSide,
)
from world.bank_feed_config import BankFeedConfig
from world.interactive_records import PaymentView

pytestmark = pytest.mark.anyio


class FakeOutcome:
    """In-memory outcome container fulfilling the TransportOutcome structural protocol."""

    def __init__(self, status: int = 200, body: dict[str, Any] | None = None, replayed: bool = False) -> None:
        """Initialises the fake outcome container with response status, body, and replayed flag."""
        self.status = status
        self.body = body if body is not None else {}
        self.replayed = replayed


class FakeTransport:
    """In-memory fake transport recording requests and returning configured mode/decision outcomes."""

    def __init__(self, mode: str = "shadow", decision: str = "unscored", delay: float = 0.0) -> None:
        """Initialises the fake transport with configured mode, simulated decision, and optional delay."""
        self.mode = mode
        self.decision = decision
        self.delay = delay
        self.calls: list[tuple[str, bytes, str]] = []

    async def post(self, path: str, body_bytes: bytes, idempotency_key: str) -> TransportOutcome:
        """Records the outgoing request and returns a simulated outcome."""
        if self.delay > 0:
            await asyncio.sleep(self.delay)
        self.calls.append((path, body_bytes, idempotency_key))
        if path == "/v1/decisions":
            return FakeOutcome(
                status=200,
                body={"mode": self.mode, "decision": self.decision, "reason": "simulated"},
                replayed=False,
            )
        return FakeOutcome(status=200, body={"status": "ok"}, replayed=False)


@pytest.fixture
def test_feed_config() -> BankFeedConfig:
    """Constructs a deterministic BankFeedConfig instance for connector unit tests."""
    return BankFeedConfig(
        bank_ids_by_index={0: "bank_0", 1: "bank_1", 2: "bank_2"},
        payee_type_by_account_prefix={"acc_person_": "person", "acc_shop_": "merchant"},
        channel_value="upi_app",
        decision_timeout_seconds=2.0,
        failed_payment_visibility={"sender_bank": {"sender": True, "receiver": False}},
        device_event_kind_map={"phone_change": "device_binding_change"},
        unmapped_gang_device_kinds=("emulator_enabled",),
        allowed_account_event_kinds=(
            "device_binding_change",
            "sim_swap",
            "app_reinstall",
            "os_update",
            "account_opened",
            "account_reactivated",
        ),
    )


async def test_one_decision_per_hook_call_with_planned_id_and_right_sender_bank(
    test_feed_config: BankFeedConfig,
) -> None:
    """Proves that invoking the hook issues a single decision request to the sender bank transport."""
    loop = asyncio.get_running_loop()
    t0 = FakeTransport(mode="shadow", decision="unscored")
    t1 = FakeTransport(mode="shadow", decision="unscored")
    transports: dict[str, BankTransport] = {"bank_0": t0, "bank_1": t1}
    modes = {"bank_0": "shadow", "bank_1": "shadow"}
    account_map = {"acc_person_001": "bank_0", "acc_shop_001": "bank_1"}

    connectors = MockBankConnectors(
        transports=transports,
        modes=modes,
        account_bank_map=account_map,
        clock_start=date(2026, 10, 28),
        feed_config=test_feed_config,
        decision_timeout_seconds=2.0,
        loop=loop,
        max_identifier_length=128,
    )
    hook = connectors.make_hook()
    payment = PaymentView("plan_dec_001", 15, "acc_person_001", "acc_shop_001", 250000)

    decision = await asyncio.to_thread(hook, payment)
    assert decision == "allow"
    assert len(t0.calls) == 1
    assert len(t1.calls) == 0

    path, body, key = t0.calls[0]
    assert path == "/v1/decisions"
    assert key == "dec:plan_dec_001"
    parsed = orjson.loads(body)
    assert parsed["payment_id"] == "plan_dec_001"
    assert parsed["sender_account_id"] == "acc_person_001"
    assert parsed["receiver_bank_id"] == "bank_1"
    assert parsed["payee_type"] == "merchant"
    assert parsed["amount_paise"] == 250000


async def test_shadow_mode_returns_allow_even_when_decision_declined(
    test_feed_config: BankFeedConfig,
) -> None:
    """Proves that a bank operating in shadow mode always returns allow regardless of decision content."""
    loop = asyncio.get_running_loop()
    t0 = FakeTransport(mode="shadow", decision="decline")
    transports: dict[str, BankTransport] = {"bank_0": t0}
    modes = {"bank_0": "shadow"}
    account_map = {"acc_person_001": "bank_0", "acc_shop_001": "bank_0"}

    connectors = MockBankConnectors(
        transports=transports,
        modes=modes,
        account_bank_map=account_map,
        clock_start=date(2026, 10, 28),
        feed_config=test_feed_config,
        decision_timeout_seconds=2.0,
        loop=loop,
        max_identifier_length=128,
    )
    hook = connectors.make_hook()
    payment = PaymentView("plan_dec_002", 20, "acc_person_001", "acc_shop_001", 100000)

    decision = await asyncio.to_thread(hook, payment)
    assert decision == "allow"
    assert connectors.live_fallbacks == 0


async def test_live_mode_counts_fallback_for_unscored(
    test_feed_config: BankFeedConfig,
) -> None:
    """Proves that live mode increments live_fallbacks when the decision returned is unscored."""
    loop = asyncio.get_running_loop()
    t0 = FakeTransport(mode="live", decision="unscored")
    transports: dict[str, BankTransport] = {"bank_0": t0}
    modes = {"bank_0": "live"}
    account_map = {"acc_person_001": "bank_0", "acc_shop_001": "bank_0"}

    connectors = MockBankConnectors(
        transports=transports,
        modes=modes,
        account_bank_map=account_map,
        clock_start=date(2026, 10, 28),
        feed_config=test_feed_config,
        decision_timeout_seconds=2.0,
        loop=loop,
        max_identifier_length=128,
    )
    hook = connectors.make_hook()
    payment = PaymentView("plan_dec_003", 25, "acc_person_001", "acc_shop_001", 150000)

    decision = await asyncio.to_thread(hook, payment)
    assert decision == "allow"
    assert connectors.live_fallbacks == 1


async def test_timeout_returns_allow_and_records_failure(
    test_feed_config: BankFeedConfig,
) -> None:
    """Proves that a transport delay exceeding the decision timeout yields allow and records failure."""
    loop = asyncio.get_running_loop()
    t0 = FakeTransport(mode="shadow", decision="unscored", delay=0.3)
    transports: dict[str, BankTransport] = {"bank_0": t0}
    modes = {"bank_0": "shadow"}
    account_map = {"acc_person_001": "bank_0", "acc_shop_001": "bank_0"}

    connectors = MockBankConnectors(
        transports=transports,
        modes=modes,
        account_bank_map=account_map,
        clock_start=date(2026, 10, 28),
        feed_config=test_feed_config,
        decision_timeout_seconds=0.05,
        loop=loop,
        max_identifier_length=128,
    )
    hook = connectors.make_hook()
    payment = PaymentView("plan_dec_timeout", 30, "acc_person_001", "acc_shop_001", 50000)

    decision = await asyncio.to_thread(hook, payment)
    assert decision == "allow"
    assert len(connectors.decision_failures) == 1
    assert connectors.decision_failures[0][0] == "plan_dec_timeout"


async def test_mode_mismatch_raises_value_error(
    test_feed_config: BankFeedConfig,
) -> None:
    """Proves that receiving a response mode differing from configured bank mode halts execution."""
    loop = asyncio.get_running_loop()
    t0 = FakeTransport(mode="shadow", decision="unscored")
    transports: dict[str, BankTransport] = {"bank_0": t0}
    modes = {"bank_0": "live"}
    account_map = {"acc_person_001": "bank_0", "acc_shop_001": "bank_0"}

    connectors = MockBankConnectors(
        transports=transports,
        modes=modes,
        account_bank_map=account_map,
        clock_start=date(2026, 10, 28),
        feed_config=test_feed_config,
        decision_timeout_seconds=2.0,
        loop=loop,
        max_identifier_length=128,
    )
    hook = connectors.make_hook()
    payment = PaymentView("plan_dec_mismatch", 35, "acc_person_001", "acc_shop_001", 75000)

    with pytest.raises(ValueError, match="Mode mismatch"):
        await asyncio.to_thread(hook, payment)


async def test_unknown_account_raises_value_error(
    test_feed_config: BankFeedConfig,
) -> None:
    """Proves that a payment referencing an account missing from account_bank_map raises ValueError."""
    loop = asyncio.get_running_loop()
    t0 = FakeTransport(mode="shadow", decision="unscored")
    transports: dict[str, BankTransport] = {"bank_0": t0}
    modes = {"bank_0": "shadow"}
    account_map = {"acc_person_001": "bank_0"}

    connectors = MockBankConnectors(
        transports=transports,
        modes=modes,
        account_bank_map=account_map,
        clock_start=date(2026, 10, 28),
        feed_config=test_feed_config,
        decision_timeout_seconds=2.0,
        loop=loop,
        max_identifier_length=128,
    )
    hook = connectors.make_hook()
    payment = PaymentView("plan_dec_unknown", 40, "acc_person_001", "acc_unregistered", 80000)

    with pytest.raises(ValueError, match="Receiver account 'acc_unregistered'"):
        hook(payment)


async def test_send_feed_routes_each_record_to_own_bank_in_documented_order(
    test_feed_config: BankFeedConfig,
) -> None:
    """Proves that send_feed sorts records globally by event time and bank id, dispatching to correct transports."""
    loop = asyncio.get_running_loop()
    t0 = FakeTransport(mode="shadow")
    t1 = FakeTransport(mode="shadow")
    transports: dict[str, BankTransport] = {"bank_0": t0, "bank_1": t1}
    modes = {"bank_0": "shadow", "bank_1": "shadow"}

    connectors = MockBankConnectors(
        transports=transports,
        modes=modes,
        account_bank_map={},
        clock_start=date(2026, 10, 28),
        feed_config=test_feed_config,
        decision_timeout_seconds=2.0,
        loop=loop,
        max_identifier_length=128,
    )

    t10 = datetime(2026, 10, 28, 10, 0, 0, tzinfo=IST)
    t05 = datetime(2026, 10, 28, 9, 5, 0, tzinfo=IST)
    t20 = datetime(2026, 10, 28, 10, 20, 0, tzinfo=IST)

    feed = BankFeed(
        payments_by_bank={
            "bank_0": (
                BankFeedPaymentSide(
                    bank_id="bank_0",
                    payment_id="pmt_001",
                    side="debit",
                    account_id="acc_p_1",
                    counterparty_bank_id="bank_1",
                    sender_bank_id="bank_0",
                    receiver_bank_id="bank_1",
                    amount_paise=1000,
                    event_time=t10,
                    status="success",
                    failure_domain=None,
                    payee_type="merchant",
                    channel="upi_app",
                ),
            ),
            "bank_1": (
                BankFeedPaymentSide(
                    bank_id="bank_1",
                    payment_id="pmt_001",
                    side="credit",
                    account_id="acc_s_1",
                    counterparty_bank_id="bank_0",
                    sender_bank_id="bank_0",
                    receiver_bank_id="bank_1",
                    amount_paise=1000,
                    event_time=t10,
                    status="success",
                    failure_domain=None,
                    payee_type="merchant",
                    channel="upi_app",
                ),
            ),
        },
        account_events_by_bank={
            "bank_0": (
                BankFeedAccountEvent(
                    bank_id="bank_0",
                    event_id="evt_001",
                    account_id="acc_p_1",
                    kind="sim_swap",
                    event_time=t05,
                    details={},
                ),
            ),
            "bank_1": (),
        },
        complaints_by_bank={
            "bank_0": (),
            "bank_1": (
                BankFeedComplaint(
                    bank_id="bank_1",
                    complaint_id="cmp_001",
                    complainant_account="acc_s_1",
                    reported_account="acc_p_1",
                    reported_bank_id="bank_0",
                    payment_ids=("pmt_001",),
                    claimed_type="fraud",
                    received_at=t20,
                ),
            ),
        },
        dropped_device_events_by_kind={},
    )

    await connectors.send_feed(feed)

    assert len(t0.calls) == 2
    assert t0.calls[0][0] == "/v1/account-events"
    assert t0.calls[1][0] == "/v1/transactions"

    assert len(t1.calls) == 2
    assert t1.calls[0][0] == "/v1/transactions"
    assert t1.calls[1][0] == "/v1/complaints"

    assert connectors.sent_by_kind["transaction"] == 2
    assert connectors.sent_by_kind["account_event"] == 1
    assert connectors.sent_by_kind["complaint"] == 1


def test_idempotency_keys_match_documented_format() -> None:
    """Proves that deterministic idempotency keys follow the exact documented prefixes and enforce bounds."""
    assert make_decision_idempotency_key("plan_1", 128) == "dec:plan_1"
    assert make_transaction_idempotency_key("pmt_1", "debit", 128) == "tx:pmt_1:debit"
    assert make_account_event_idempotency_key("evt_1", 128) == "acct:evt_1"
    assert make_complaint_idempotency_key("cmp_1", 128) == "cmp:cmp_1"

    with pytest.raises(ValueError, match="exceeds max length"):
        make_decision_idempotency_key("x" * 200, 128)

