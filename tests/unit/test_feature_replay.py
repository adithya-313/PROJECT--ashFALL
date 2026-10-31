# File: tests/unit/test_feature_replay.py
# Purpose: Unit tests for historical feature replay and row conversion functions.
# Issue: Level 15c Round 2a (recompute a logged feature row from history).
"""
WHAT: Unit tests for historical feature replay and schema-2 payment side parsing.

WHY: Verifies that payment_side_from_transaction_row converts database rows from
the transactions table into PaymentSideRecord instances identical to those produced
by the live Kafka ingress path (parse_upi_event with UpiEventPayload), and enforces
strict KeyError behavior on missing required columns.
"""

from datetime import datetime, timezone

import pytest

from api.v1.upi_outbox import (
    EVENT_TYPE_UPI_PAYMENT_RECORDED,
    UPI_SCHEMA_VERSION,
    UpiEventData,
    UpiEventPayload,
)
from features.account_records import (
    parse_upi_event,
    parse_upi_payload,
    payment_side_from_transaction_row,
)

pytestmark = pytest.mark.anyio


def test_payment_side_from_transaction_row_matches_live_upi_event_payload() -> None:
    """
    D1a: Verifies that payment_side_from_transaction_row produces a PaymentSideRecord
    identical to parse_upi_event decoding a live UpiEventPayload for the same transaction side.
    """
    event_time = datetime(2026, 10, 28, 10, 0, 0, tzinfo=timezone.utc)
    created_at = datetime(2026, 10, 28, 10, 0, 5, tzinfo=timezone.utc)
    payment_id = "0123456789abcdef-pay-001"
    bank_id = "bank_0"
    account_id = "acc_payer_001"
    side = "debit"
    counterparty_bank_id = "bank_1"
    counterparty_vpa = "acc_payee@bank1"
    amount_paise = 250000
    status = "success"
    failure_domain = None
    payee_type = "merchant"
    channel = "in_store"
    sender_device_fingerprint = "fp_abc123"
    sender_os_generation = 14
    balance_after_paise = 500000

    row = {
        "payment_id": payment_id,
        "bank_id": bank_id,
        "account_id": account_id,
        "side": side,
        "counterparty_bank_id": counterparty_bank_id,
        "counterparty_vpa": counterparty_vpa,
        "amount_paise": amount_paise,
        "status": status,
        "failure_domain": failure_domain,
        "payee_type": payee_type,
        "channel": channel,
        "sender_device_fingerprint": sender_device_fingerprint,
        "sender_os_generation": sender_os_generation,
        "balance_after_paise": balance_after_paise,
        "event_time": event_time,
        "created_at": created_at,
    }

    # Record from transaction row
    record_from_row = payment_side_from_transaction_row(row)

    # Real live UpiEventPayload model as built by api/v1/upi_outbox.py
    payload_model = UpiEventPayload(
        event_type=EVENT_TYPE_UPI_PAYMENT_RECORDED,
        schema_version=UPI_SCHEMA_VERSION,
        event_id="00000000-0000-0000-0000-000000000001",
        transaction_id="00000000-0000-0000-0000-000000000002",
        payment_id=payment_id,
        side=side,
        event_time=event_time.isoformat(),
        recorded_at=created_at.isoformat(),
        source="world",
        data=UpiEventData(
            bank_id=bank_id,
            account_id=account_id,
            counterparty_vpa=counterparty_vpa,
            counterparty_bank_id=counterparty_bank_id,
            sender_bank_id=bank_id,
            receiver_bank_id=counterparty_bank_id,
            amount_paise=amount_paise,
            status=status,
            failure_domain=failure_domain,
            payee_type=payee_type,
            channel=channel,
            sender_device_fingerprint=sender_device_fingerprint,
            sender_os_generation=sender_os_generation,
            balance_after_paise=balance_after_paise,
        ),
    )
    raw_bytes = payload_model.model_dump_json().encode("utf-8")
    record_from_event = parse_upi_event(raw_bytes)
    record_from_payload = parse_upi_payload(payload_model.model_dump())

    # Assert byte-for-byte / field-for-field equality
    assert record_from_row == record_from_event
    assert record_from_row == record_from_payload
    assert record_from_row.arrival_time == created_at
    assert record_from_row.event_time == event_time
    assert record_from_row.run_id == "0123456789abcdef"


def test_payment_side_from_transaction_row_missing_created_at_raises_key_error() -> None:
    """
    D1a: Verifies that a row missing 'created_at' raises KeyError without silent fallback.
    """
    event_time = datetime(2026, 10, 28, 10, 0, 0, tzinfo=timezone.utc)
    row = {
        "payment_id": "0123456789abcdef-pay-001",
        "bank_id": "bank_0",
        "account_id": "acc_001",
        "side": "debit",
        "counterparty_bank_id": "bank_1",
        "counterparty_vpa": "acc_002",
        "amount_paise": 10000,
        "status": "success",
        "failure_domain": None,
        "payee_type": "person",
        "channel": "app",
        "sender_device_fingerprint": None,
        "sender_os_generation": None,
        "balance_after_paise": None,
        "event_time": event_time,
        # 'created_at' intentionally omitted
    }
    with pytest.raises(KeyError) as exc_info:
        payment_side_from_transaction_row(row)
    assert "created_at" in str(exc_info.value)
