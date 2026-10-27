# File: tests/integration/test_upi_outbox_writer.py
# Purpose: Proves Level 14a UPI outbox writer: schema "2" envelopes, UUIDv5 event identities,
#          per-bank isolation, replay idempotency, IST timestamps, and backward compatibility with Level 7.
from datetime import datetime, timezone
from decimal import Decimal
import uuid

import orjson
import pytest

from api.routers.scoring import TransactionRequest
from api.services.outbox import record_transaction_with_outbox
from api.v1.bank_scope import bank_scoped_transaction
from api.v1.upi_outbox import (
    EVENT_TYPE_UPI_PAYMENT_RECORDED,
    UPI_SCHEMA_VERSION,
    UpiPaymentSide,
    record_upi_side,
)
from core.event_clock import IST

pytestmark = pytest.mark.anyio


async def count_rows(conn, table: str) -> int:
    """Helper to count rows in the isolated test schema."""
    return await conn.fetchval(f"SELECT count(*) FROM {table}")


async def test_cross_bank_payment(isolated_schema) -> None:
    """
    Recording a cross-bank payment: bank_0 debit under bank_0 scope and bank_2 credit under bank_2 scope
    produces 2 transactions rows, 2 outbox rows, 2 different event_ids, both schema_version '2', and
    both UUID version 5.
    """
    connection, _ = isolated_schema
    now = datetime.now(timezone.utc)
    pmt_id = "pmt_cross_bank_001"

    debit_side = UpiPaymentSide(
        payment_id=pmt_id,
        bank_id="bank_0",
        side="debit",
        account_id="acc_debit_0",
        counterparty_bank_id="bank_2",
        sender_bank_id="bank_0",
        receiver_bank_id="bank_2",
        amount_paise=150000,
        event_time=now,
        status="success",
        counterparty_vpa="user2@bank2",
    )

    credit_side = UpiPaymentSide(
        payment_id=pmt_id,
        bank_id="bank_2",
        side="credit",
        account_id="acc_credit_2",
        counterparty_bank_id="bank_0",
        sender_bank_id="bank_0",
        receiver_bank_id="bank_2",
        amount_paise=150000,
        event_time=now,
        status="success",
        counterparty_vpa="user0@bank0",
    )

    async with bank_scoped_transaction(connection, "bank_0") as scoped_0:
        res_debit = await record_upi_side(scoped_0, debit_side)
        assert res_debit.was_replay is False

    async with bank_scoped_transaction(connection, "bank_2") as scoped_2:
        res_credit = await record_upi_side(scoped_2, credit_side)
        assert res_credit.was_replay is False

    assert await count_rows(connection, "transactions") == 2
    assert await count_rows(connection, "outbox") == 2

    assert res_debit.event_id != res_credit.event_id

    # Verify both event_ids are UUID version 5
    uuid_debit = uuid.UUID(res_debit.event_id)
    uuid_credit = uuid.UUID(res_credit.event_id)
    assert uuid_debit.version == 5
    assert uuid_credit.version == 5

    # Check outbox rows in database
    outbox_rows = await connection.fetch("SELECT event_id, event_type, payload FROM outbox ORDER BY id")
    assert len(outbox_rows) == 2
    for r in outbox_rows:
        assert r["event_type"] == EVENT_TYPE_UPI_PAYMENT_RECORDED
        payload = orjson.loads(r["payload"])
        assert payload["schema_version"] == UPI_SCHEMA_VERSION
        assert payload["source"] == "world"
        assert payload["data"]["amount_paise"] == 150000


async def test_same_side_sent_twice_is_replay(isolated_schema) -> None:
    """
    Submitting the exact same payment side twice results in exactly 1 transaction row and 1 outbox row.
    The second call returns was_replay=True with the original transaction_id and identical event_id.
    """
    connection, _ = isolated_schema
    now = datetime.now(timezone.utc)
    pmt_id = "pmt_replay_002"

    side = UpiPaymentSide(
        payment_id=pmt_id,
        bank_id="bank_1",
        side="debit",
        account_id="acc_1_payer",
        counterparty_bank_id="bank_2",
        sender_bank_id="bank_1",
        receiver_bank_id="bank_2",
        amount_paise=7500,
        event_time=now,
        status="success",
    )

    async with bank_scoped_transaction(connection, "bank_1") as scoped:
        first_res = await record_upi_side(scoped, side)
        assert first_res.was_replay is False

    assert await count_rows(connection, "transactions") == 1
    assert await count_rows(connection, "outbox") == 1

    async with bank_scoped_transaction(connection, "bank_1") as scoped:
        second_res = await record_upi_side(scoped, side)
        assert second_res.was_replay is True

    # Row counts must not change after replay
    assert await count_rows(connection, "transactions") == 1
    assert await count_rows(connection, "outbox") == 1

    assert second_res.transaction_id == first_res.transaction_id
    assert second_res.event_id == first_res.event_id


async def test_ist_event_time_preserves_offset(isolated_schema) -> None:
    """
    An IST-aware event_time (core.event_clock.IST) preserves its exact instant in the database
    and the serialized payload event_time explicitly includes '+05:30'.
    """
    connection, _ = isolated_schema
    ist_time = datetime(2026, 10, 27, 14, 30, 0, tzinfo=IST)
    pmt_id = "pmt_ist_003"

    side = UpiPaymentSide(
        payment_id=pmt_id,
        bank_id="bank_0",
        side="debit",
        account_id="acc_0_ist",
        counterparty_bank_id="bank_1",
        sender_bank_id="bank_0",
        receiver_bank_id="bank_1",
        amount_paise=25000,
        event_time=ist_time,
        status="success",
    )

    async with bank_scoped_transaction(connection, "bank_0") as scoped:
        res = await record_upi_side(scoped, side)

    # Read back from transactions
    tx_row = await connection.fetchrow(
        "SELECT event_time, amount_paise FROM transactions WHERE transaction_id = $1::uuid",
        uuid.UUID(res.transaction_id),
    )
    assert tx_row is not None
    assert tx_row["amount_paise"] == 25000
    # Instant in time matches
    assert tx_row["event_time"] == ist_time

    # Inspect outbox payload
    outbox_row = await connection.fetchrow(
        "SELECT payload FROM outbox WHERE event_id = $1::uuid",
        uuid.UUID(res.event_id),
    )
    assert outbox_row is not None
    payload = orjson.loads(outbox_row["payload"])
    assert "+05:30" in payload["event_time"]


async def test_naive_datetime_rejected_without_db_write(isolated_schema) -> None:
    """
    A naive event_time (missing tzinfo) raises ValueError before any database write occurs.
    The database remains completely empty.
    """
    connection, _ = isolated_schema
    naive_time = datetime(2026, 10, 27, 14, 30, 0)  # naive!

    side = UpiPaymentSide(
        payment_id="pmt_naive_004",
        bank_id="bank_0",
        side="debit",
        account_id="acc_0_naive",
        counterparty_bank_id="bank_1",
        sender_bank_id="bank_0",
        receiver_bank_id="bank_1",
        amount_paise=1000,
        event_time=naive_time,
        status="success",
    )

    with pytest.raises(ValueError, match="event_time must be timezone-aware"):
        async with bank_scoped_transaction(connection, "bank_0") as scoped:
            await record_upi_side(scoped, side)

    assert await count_rows(connection, "transactions") == 0
    assert await count_rows(connection, "outbox") == 0


async def test_failed_payment_with_failure_domain(isolated_schema) -> None:
    """A failed payment with failure_domain 'sender_bank' is stored and reflected in the outbox payload."""
    connection, _ = isolated_schema
    now = datetime.now(timezone.utc)
    pmt_id = "pmt_failed_005"

    side = UpiPaymentSide(
        payment_id=pmt_id,
        bank_id="bank_0",
        side="debit",
        account_id="acc_0_fail",
        counterparty_bank_id="bank_1",
        sender_bank_id="bank_0",
        receiver_bank_id="bank_1",
        amount_paise=5000,
        event_time=now,
        status="failed",
        failure_domain="sender_bank",
    )

    async with bank_scoped_transaction(connection, "bank_0") as scoped:
        res = await record_upi_side(scoped, side)

    tx_row = await connection.fetchrow(
        "SELECT status, failure_domain FROM transactions WHERE transaction_id = $1::uuid",
        uuid.UUID(res.transaction_id),
    )
    assert tx_row is not None
    assert tx_row["status"] == "failed"
    assert tx_row["failure_domain"] == "sender_bank"

    outbox_row = await connection.fetchrow(
        "SELECT payload FROM outbox WHERE event_id = $1::uuid",
        uuid.UUID(res.event_id),
    )
    assert outbox_row is not None
    payload = orjson.loads(outbox_row["payload"])
    assert payload["data"]["status"] == "failed"
    assert payload["data"]["failure_domain"] == "sender_bank"


async def test_level_7_regression_schema_1_ecommerce(isolated_schema) -> None:
    """
    Regression check: api.services.outbox.record_transaction_with_outbox continues to write
    schema_version '1' events with source 'ecommerce' without modification after migration 0006.
    """
    connection, _ = isolated_schema
    req = TransactionRequest(
        amount=99.95,
        account_id="acc_l7_reg",
        merchant_id="merch_l7",
        device_fingerprint="fp_123",
        geo_coordinates="12.97,77.59",
    )

    stored = await record_transaction_with_outbox(connection, req, fraud_score=0.15, model_version="m1")
    assert stored.was_replay is False

    tx_row = await connection.fetchrow(
        "SELECT source, amount, bank_id FROM transactions WHERE transaction_id = $1::uuid",
        uuid.UUID(stored.transaction_id),
    )
    assert tx_row is not None
    assert tx_row["source"] == "ecommerce"
    assert tx_row["amount"] == Decimal("99.95")
    assert tx_row["bank_id"] is None  # Level 7 does not populate bank_id

    outbox_row = await connection.fetchrow(
        "SELECT payload FROM outbox WHERE transaction_id = $1::uuid",
        uuid.UUID(stored.transaction_id),
    )
    assert outbox_row is not None
    payload = orjson.loads(outbox_row["payload"])
    assert payload["schema_version"] == "1"
    assert payload["source"] == "ecommerce"
