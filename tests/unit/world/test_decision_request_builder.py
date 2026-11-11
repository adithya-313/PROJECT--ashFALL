# File: tests/unit/world/test_decision_request_builder.py
# Purpose: Unit tests proving MockBankConnectors.build_decision_request reproduces pre-move payload bytes.
import asyncio
from datetime import date
from typing import Any

import pytest

from world.bank_connectors import DecisionCall, MockBankConnectors, TransportOutcome
from world.bank_feed_config import BankFeedConfig
from world.interactive_records import PaymentView

pytestmark = pytest.mark.anyio


class RecordingFakeOutcome:
    """In-memory outcome container fulfilling the TransportOutcome structural protocol."""

    def __init__(self, status: int = 200, body: dict[str, Any] | None = None, replayed: bool = False) -> None:
        self.status = status
        self.body = body if body is not None else {"mode": "shadow", "decision": "unscored"}
        self.replayed = replayed


class RecordingFakeTransport:
    """In-memory fake transport recording all post requests."""

    def __init__(self, mode: str = "shadow") -> None:
        self.mode = mode
        self.calls: list[tuple[str, bytes, str]] = []

    async def post(self, path: str, body_bytes: bytes, idempotency_key: str) -> TransportOutcome:
        self.calls.append((path, body_bytes, idempotency_key))
        return RecordingFakeOutcome()


@pytest.fixture
def connectors_feed_config() -> BankFeedConfig:
    """Deterministic configuration matching feed payee classifications."""
    return BankFeedConfig(
        bank_ids_by_index={0: "bank_0", 1: "bank_1", 2: "bank_2"},
        payee_type_by_account_prefix={"acc_person_": "person", "acc_shop_": "merchant"},
        channel_value="upi_app",
        decision_timeout_seconds=2.0,
        failed_payment_visibility={"sender_bank": {"sender": True, "receiver": False}},
        device_event_kind_map={"phone_change": "device_binding_change"},
        unmapped_gang_device_kinds=("emulator_enabled",),
        allowed_account_event_kinds=("account_opened",),
        emit_opening_for_existing_accounts=True,
    )


async def test_build_decision_request_matches_pre_move_literals_and_hook_payloads(
    connectors_feed_config: BankFeedConfig,
) -> None:
    """
    Proves that build_decision_request generates byte-for-byte identical payloads to the
    pre-move implementation and that make_hook dispatches the exact DecisionCall output.
    """
    loop = asyncio.get_running_loop()
    t0 = RecordingFakeTransport()
    t1 = RecordingFakeTransport()
    account_map = {
        "acc_person_001": "bank_0",
        "acc_person_002": "bank_1",
        "acc_shop_001": "bank_1",
    }

    connectors = MockBankConnectors(
        transports={"bank_0": t0, "bank_1": t1},
        modes={"bank_0": "shadow", "bank_1": "shadow"},
        account_bank_map=account_map,
        clock_start=date(2026, 10, 28),
        feed_config=connectors_feed_config,
        decision_timeout_seconds=2.0,
        loop=loop,
        max_identifier_length=128,
    )
    hook = connectors.make_hook()

    # Pre-move literal baseline payloads captured on unchanged code before refactoring:
    literal_shop_bytes = (
        b'{"payment_id":"plan_shop_001","sender_account_id":"acc_person_001",'
        b'"receiver_bank_id":"bank_1","amount_paise":250000,"event_time":"2026-10-28T00:15:00+05:30",'
        b'"channel":"upi_app","payee_type":"merchant"}'
    )
    literal_shop_idempotency = "dec:plan_shop_001"

    literal_person_bytes = (
        b'{"payment_id":"plan_person_001","sender_account_id":"acc_person_001",'
        b'"receiver_bank_id":"bank_1","amount_paise":150000,"event_time":"2026-10-28T00:30:00+05:30",'
        b'"channel":"upi_app","payee_type":"person"}'
    )
    literal_person_idempotency = "dec:plan_person_001"

    literal_same_bank_bytes = (
        b'{"payment_id":"plan_same_bank_001","sender_account_id":"acc_person_002",'
        b'"receiver_bank_id":"bank_1","amount_paise":50000,"event_time":"2026-10-28T00:45:00+05:30",'
        b'"channel":"upi_app","payee_type":"merchant"}'
    )
    literal_same_bank_idempotency = "dec:plan_same_bank_001"

    cases: list[tuple[PaymentView, DecisionCall]] = [
        (
            PaymentView("plan_shop_001", 15, "acc_person_001", "acc_shop_001", 250000),
            DecisionCall("bank_0", "bank_1", literal_shop_bytes, literal_shop_idempotency),
        ),
        (
            PaymentView("plan_person_001", 30, "acc_person_001", "acc_person_002", 150000),
            DecisionCall("bank_0", "bank_1", literal_person_bytes, literal_person_idempotency),
        ),
        (
            PaymentView("plan_same_bank_001", 45, "acc_person_002", "acc_shop_001", 50000),
            DecisionCall("bank_1", "bank_1", literal_same_bank_bytes, literal_same_bank_idempotency),
        ),
    ]

    for payment, expected_call in cases:
        # 1. build_decision_request produces exact pre-move literal bytes and idempotency key
        call = connectors.build_decision_request(payment)
        assert call.sender_bank_id == expected_call.sender_bank_id
        assert call.receiver_bank_id == expected_call.receiver_bank_id
        assert call.body_bytes == expected_call.body_bytes
        assert call.idempotency_key == expected_call.idempotency_key

        # 2. Hook invokes transport with exactly what build_decision_request returned
        transport = t0 if call.sender_bank_id == "bank_0" else t1
        prev_calls = len(transport.calls)
        decision = await asyncio.to_thread(hook, payment)
        assert decision == "allow"
        assert len(transport.calls) == prev_calls + 1
        path, sent_bytes, sent_key = transport.calls[-1]
        assert path == "/v1/decisions"
        assert sent_bytes == call.body_bytes
        assert sent_key == call.idempotency_key
