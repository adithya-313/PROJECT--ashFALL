# File: api/v1/upi_outbox.py
# Purpose: Level 14a transactional outbox writer for UPI payments (schema_version "2").
# Writes transactions and corresponding outbox event in a single atomic database transaction.
import uuid
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from typing import NamedTuple

import asyncpg
from pydantic import BaseModel

from api.v1.bank_scope import UnknownBankError, load_known_banks
from world.failures import FAILURE_DOMAINS

# UPI payment event constants
UPI_SCHEMA_VERSION: str = "2"
EVENT_TYPE_UPI_PAYMENT_RECORDED: str = "upi.payment.recorded"

# A dedicated fixed UUID namespace for deterministic UPI event_id generation via UUIDv5.
# This value is permanent and must never change, and is deliberately different from Level 7's namespace
# (6f4a1c52-9d3e-4b87-a2c5-0e7d1b9f3a68).
UPI_EVENT_ID_NAMESPACE: uuid.UUID = uuid.UUID("3b8d234a-e4ef-4c6e-8a21-9e8c459f2e71")

PAISE_PER_RUPEE: Decimal = Decimal("100")


@dataclass(frozen=True)
class UpiPaymentSide:
    """
    Represents one side (debit or credit) of a UPI payment to be recorded.

    Fields:
      payment_id: The unique UPI payment identifier.
      bank_id: The bank on whose books this side is posted.
      side: 'debit' for sender bank, 'credit' for receiver bank.
      account_id: The customer account identifier within bank_id.
      counterparty_bank_id: The bank identifier of the other party.
      sender_bank_id: The originator bank identifier.
      receiver_bank_id: The beneficiary bank identifier.
      amount_paise: The payment amount in integer paise (> 0).
      event_time: The timezone-aware timestamp of the payment event.
      status: 'success' or 'failed'.
      counterparty_vpa: Optional Virtual Payment Address of counterparty.
      failure_domain: Mandatory if status is 'failed', None if 'success'.
      payee_type: Optional payee classification (projected in Level 14c).
      channel: Optional channel identifier (projected in Level 14c).
    """

    payment_id: str
    bank_id: str
    side: str
    account_id: str
    counterparty_bank_id: str
    sender_bank_id: str
    receiver_bank_id: str
    amount_paise: int
    event_time: datetime
    status: str
    counterparty_vpa: str | None = None
    failure_domain: str | None = None
    payee_type: str | None = None
    channel: str | None = None


class UpiEventData(BaseModel):
    """The business payload inside a schema-2 UPI outbox event."""

    bank_id: str
    account_id: str
    counterparty_vpa: str | None = None
    counterparty_bank_id: str
    sender_bank_id: str
    receiver_bank_id: str
    amount_paise: int
    status: str
    failure_domain: str | None = None
    payee_type: str | None = None
    channel: str | None = None


class UpiEventPayload(BaseModel):
    """The outer envelope of a schema-2 UPI outbox event."""

    event_type: str = EVENT_TYPE_UPI_PAYMENT_RECORDED
    schema_version: str = UPI_SCHEMA_VERSION
    event_id: str
    transaction_id: str
    payment_id: str
    side: str
    event_time: str
    recorded_at: str
    source: str = "world"
    data: UpiEventData


class StoredUpiSide(NamedTuple):
    """Outcome of recording a UPI payment side."""

    transaction_id: str
    event_id: str
    was_replay: bool


def validate_side(upi_side: UpiPaymentSide) -> None:
    """
    Validate all fields of a UpiPaymentSide before database persistence.

    Enforces:
      - event_time must be timezone-aware with explicit UTC offset (no naive datetimes).
      - amount_paise must be a strictly positive integer.
      - side must be 'debit' or 'credit'.
      - status must be 'success' or 'failed'.
      - failure_domain must be one of the 5 canonical failure domains if provided.
      - status and failure_domain correlation: 'failed' requires failure_domain, 'success' forbids it.
      - bank_id, counterparty_bank_id, sender_bank_id, receiver_bank_id must be known configured banks.
      - bank_id must match sender_bank_id for debit, and receiver_bank_id for credit.
    """
    # 1. Timezone contract: explicit offsets only
    if upi_side.event_time.tzinfo is None or upi_side.event_time.tzinfo.utcoffset(upi_side.event_time) is None:
        raise ValueError("event_time must be timezone-aware with an explicit offset")

    # 2. Strict positive integer paise
    if not isinstance(upi_side.amount_paise, int) or upi_side.amount_paise <= 0:
        raise ValueError(f"amount_paise must be a positive integer, got {upi_side.amount_paise}")

    # 3. Side validation
    if upi_side.side not in ("debit", "credit"):
        raise ValueError(f"side must be 'debit' or 'credit', got {upi_side.side!r}")

    # 4. Status validation
    if upi_side.status not in ("success", "failed"):
        raise ValueError(f"status must be 'success' or 'failed', got {upi_side.status!r}")

    # 5. Failure domain validation
    if upi_side.failure_domain is not None and upi_side.failure_domain not in FAILURE_DOMAINS:
        raise ValueError(f"failure_domain must be one of {FAILURE_DOMAINS}, got {upi_side.failure_domain!r}")

    # 6. Status/domain correlation
    if upi_side.status == "failed" and upi_side.failure_domain is None:
        raise ValueError("failure_domain must be provided when status is 'failed'")
    if upi_side.status == "success" and upi_side.failure_domain is not None:
        raise ValueError(f"failure_domain must be None when status is 'success', got {upi_side.failure_domain!r}")

    # 7. Known bank IDs
    known_banks = load_known_banks()
    for b_label, b_val in [
        ("bank_id", upi_side.bank_id),
        ("counterparty_bank_id", upi_side.counterparty_bank_id),
        ("sender_bank_id", upi_side.sender_bank_id),
        ("receiver_bank_id", upi_side.receiver_bank_id),
    ]:
        if b_val not in known_banks:
            raise UnknownBankError(f"Unknown bank for {b_label}: {b_val!r} (known: {sorted(known_banks)})")

    # 8. Bank role matches payment side
    if upi_side.side == "debit" and upi_side.bank_id != upi_side.sender_bank_id:
        raise ValueError(
            f"For debit side, bank_id ({upi_side.bank_id}) must match sender_bank_id ({upi_side.sender_bank_id})"
        )
    if upi_side.side == "credit" and upi_side.bank_id != upi_side.receiver_bank_id:
        raise ValueError(
            f"For credit side, bank_id ({upi_side.bank_id}) must match receiver_bank_id ({upi_side.receiver_bank_id})"
        )


def make_upi_event_id(payment_id: str, side: str) -> uuid.UUID:
    """
    Generate a deterministic UUIDv5 event_id from payment_id and side.

    Given that each payment side is distinct, (payment_id, side) uniquely identifies the event.
    Replaying the same payment side produces the exact same event_id.
    """
    return uuid.uuid5(UPI_EVENT_ID_NAMESPACE, f"{payment_id}:{side}")


async def record_upi_side(conn: asyncpg.Connection, upi_side: UpiPaymentSide) -> StoredUpiSide:
    """
    Atomically insert a UPI payment side and its schema-2 outbox event.

    Precondition: The caller must already be inside a bank_scoped_transaction for upi_side.bank_id.
    This ensures PostgreSQL Row Level Security enforces that the bank role only writes its own bank_id.

    Replay behaviour:
      If a row for (bank_id, payment_id, side) already exists (violating uq_transactions_bank_payment_side),
      the insert performs DO NOTHING. We then look up the existing transaction_id and return was_replay=True,
      without writing a duplicate outbox event.
      Note: Comparing a replay's body to the stored one and returning 409 Conflict is handled in Level 14b,
      not here.

    Returns:
      StoredUpiSide with transaction_id, event_id string, and was_replay flag.
    """
    # Validate fields prior to database operations
    validate_side(upi_side)

    # Decimal amount in Rupees matching the paise integer: amount * 100 == amount_paise
    amount_rupees = (Decimal(upi_side.amount_paise) / PAISE_PER_RUPEE).quantize(Decimal("0.01"))
    event_id = make_upi_event_id(upi_side.payment_id, upi_side.side)
    event_id_str = str(event_id)

    # Use a nested transaction (SAVEPOINT in asyncpg if transaction is already open)
    async with conn.transaction():
        # Attempt to insert transactions row
        row = await conn.fetchrow(
            """
            INSERT INTO transactions (
                source, account_id, amount, payment_id, bank_id, side,
                counterparty_vpa, counterparty_bank_id, amount_paise,
                event_time, status, failure_domain, payee_type, channel
            )
            VALUES (
                'world', $1, $2, $3, $4, $5,
                $6, $7, $8,
                $9, $10, $11, $12, $13
            )
            ON CONFLICT ON CONSTRAINT uq_transactions_bank_payment_side DO NOTHING
            RETURNING transaction_id, created_at
            """,
            upi_side.account_id,
            amount_rupees,
            upi_side.payment_id,
            upi_side.bank_id,
            upi_side.side,
            upi_side.counterparty_vpa,
            upi_side.counterparty_bank_id,
            upi_side.amount_paise,
            upi_side.event_time,
            upi_side.status,
            upi_side.failure_domain,
            upi_side.payee_type,
            upi_side.channel,
        )

        if row is not None:
            # First-time insertion: construct schema-2 payload and insert outbox row
            transaction_id = str(row["transaction_id"])
            created_at_iso = row["created_at"].isoformat()

            payload = UpiEventPayload(
                event_type=EVENT_TYPE_UPI_PAYMENT_RECORDED,
                schema_version=UPI_SCHEMA_VERSION,
                event_id=event_id_str,
                transaction_id=transaction_id,
                payment_id=upi_side.payment_id,
                side=upi_side.side,
                event_time=upi_side.event_time.isoformat(),
                recorded_at=created_at_iso,
                source="world",
                data=UpiEventData(
                    bank_id=upi_side.bank_id,
                    account_id=upi_side.account_id,
                    counterparty_vpa=upi_side.counterparty_vpa,
                    counterparty_bank_id=upi_side.counterparty_bank_id,
                    sender_bank_id=upi_side.sender_bank_id,
                    receiver_bank_id=upi_side.receiver_bank_id,
                    amount_paise=upi_side.amount_paise,
                    status=upi_side.status,
                    failure_domain=upi_side.failure_domain,
                    payee_type=upi_side.payee_type,
                    channel=upi_side.channel,
                ),
            )

            await conn.execute(
                """
                INSERT INTO outbox (transaction_id, event_id, event_type, payload)
                VALUES ($1::uuid, $2::uuid, $3, $4::jsonb)
                """,
                transaction_id,
                event_id_str,
                EVENT_TYPE_UPI_PAYMENT_RECORDED,
                payload.model_dump_json(),
            )

            return StoredUpiSide(
                transaction_id=transaction_id,
                event_id=event_id_str,
                was_replay=False,
            )

        # Replay: row already exists, query transaction_id and do NOT write outbox event
        existing_row = await conn.fetchrow(
            """
            SELECT transaction_id
            FROM transactions
            WHERE bank_id = $1 AND payment_id = $2 AND side = $3
            """,
            upi_side.bank_id,
            upi_side.payment_id,
            upi_side.side,
        )
        if existing_row is None:
            raise ValueError(
                f"Existing transaction not found for replay: bank_id={upi_side.bank_id}, "
                f"payment_id={upi_side.payment_id}, side={upi_side.side}"
            )

        return StoredUpiSide(
            transaction_id=str(existing_row["transaction_id"]),
            event_id=event_id_str,
            was_replay=True,
        )
