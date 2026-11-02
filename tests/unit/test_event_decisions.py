# File: tests/unit/test_event_decisions.py
# Purpose: Unit tests for per-event decisions and receiver_vpa builder integration (Decisions D1-D4).
"""
WHAT: Unit tests covering bank_visible_vpa, receiver_vpa inclusion in build_decision_request,
and per-event decision call derivation.

WHY: Verifies architect decisions P2(b), 16c-1 conditions 1-3, and receiver_vpa (a) ensuring
deterministic payload construction and byte-level backward compatibility.
"""
import asyncio
from datetime import date

import orjson
import pytest

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
