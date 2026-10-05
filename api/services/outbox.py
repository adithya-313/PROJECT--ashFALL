"""
File: api/services/outbox.py
Purpose: The "transactional outbox" write. It saves a transaction AND the event announcing it in ONE database
         transaction, so either both exist or neither does.

The story in plain English:
  When /score receives a payment we must do two things: store the payment, and (later) tell other systems about it
  through Kafka. If we did these as two separate steps, a crash in between would leave us with a stored payment that
  nobody hears about, or an announcement about a payment that was never stored. The outbox pattern avoids this: we
  write the payment row and an "event to publish" row into the SAME database transaction. A separate background relay
  (a later commit) reads the unpublished event rows and sends them to Kafka. This file never talks to Kafka.

The outbox payload schema below is LOCKED as of Level 7: downstream consumers rely on it, so changing it needs a
versioned new event type, not an edit.
"""

from datetime import datetime
from decimal import ROUND_HALF_UP, Decimal
from typing import TYPE_CHECKING

import asyncpg
from pydantic import BaseModel

if TYPE_CHECKING:  # only for type hints; importing the router here would create a circular import
    from api.routers.scoring import TransactionRequest

# The one event type this commit produces. A constant, so the payload and the outbox row can never disagree.
EVENT_TYPE_TRANSACTION_CREATED = "transaction.created"

# The two decimal places NUMERIC(18,2) stores. Money is rounded to cents once, here, and never handled as a float again.
CENTS = Decimal("0.01")


class OutboxEventData(BaseModel):
    """The business facts inside the event."""

    account_id: str
    merchant_id: str | None
    amount: str  # a string like "42.50", so no consumer ever reads money as a float
    fraud_score: float | None
    model_version: str


class OutboxEventPayload(BaseModel):
    """The outer envelope of every outbox event (LOCKED as of Level 7)."""

    event_type: str = EVENT_TYPE_TRANSACTION_CREATED
    transaction_id: str
    occurred_at: str  # ISO-8601 time the transaction row was created, taken from the database clock
    source: str
    data: OutboxEventData


def amount_to_decimal(amount: float) -> Decimal:
    """
    Turn the float from the request into an exact Decimal rounded to cents.

    Going through str() first matters: Decimal(42.5) would copy the float's tiny binary rounding errors, while
    Decimal("42.5") is exact. The request model hands us a float, so this is the one place it becomes money.
    """
    return Decimal(str(amount)).quantize(CENTS, rounding=ROUND_HALF_UP)


def build_payload(transaction_id: str, occurred_at: datetime, request: "TransactionRequest", amount: Decimal,
                  fraud_score: float | None, model_version: str, source: str) -> OutboxEventPayload:
    """Fill in the locked event envelope for one transaction."""
    return OutboxEventPayload(
        transaction_id=transaction_id,
        occurred_at=occurred_at.isoformat(),
        source=source,
        data=OutboxEventData(
            account_id=request.account_id,
            merchant_id=request.merchant_id,
            amount=str(amount),
            fraud_score=fraud_score,
            model_version=model_version,
        ),
    )


async def record_transaction_with_outbox(
    conn: asyncpg.Connection,
    request: "TransactionRequest",
    fraud_score: float | None,
    model_version: str,
) -> str:
    """
    Save one e-commerce transaction and its outbox event atomically. Returns the new transaction_id.

    Everything happens inside one "async with conn.transaction()" block: if the second insert (or anything else in
    the block) fails, the database undoes the first insert too, so we never keep half of the pair.
    """
    source = "ecommerce"
    amount = amount_to_decimal(request.amount)

    async with conn.transaction():
        # Step 1: store the transaction. PaySim-only columns (txn_type and the four balances) are left out, so the
        # database fills them with NULL. RETURNING hands back the generated ID and creation time in the same trip.
        row = await conn.fetchrow(
            """
            INSERT INTO transactions
              (source, account_id, merchant_id, amount, device_fingerprint, geo_coordinates, fraud_score, model_version)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            RETURNING transaction_id, created_at
            """,
            source, request.account_id, request.merchant_id, amount, request.device_fingerprint,
            request.geo_coordinates, fraud_score, model_version,
        )
        transaction_id = str(row["transaction_id"])

        # Step 2: build the event announcing it, in the locked shape.
        payload = build_payload(transaction_id, row["created_at"], request, amount, fraud_score, model_version, source)

        # Step 3: store the event in the outbox. processed_at stays NULL, which means "not published yet".
        await conn.execute(
            """
            INSERT INTO outbox (transaction_id, event_type, payload)
            VALUES ($1::uuid, $2, $3::jsonb)
            """,
            transaction_id, EVENT_TYPE_TRANSACTION_CREATED, payload.model_dump_json(),
        )

    return transaction_id
