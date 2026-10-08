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

Event identity (Level 7): every event has an event_id. If the client sends an idempotency_key, the event_id is
computed from it, so retrying the same request gives the same event_id and the database refuses the second copy.
Without a key, the event_id is random.

The outbox payload schema below is LOCKED as of Level 7, schema_version "1": the envelope is event_type,
schema_version, event_id, transaction_id, occurred_at, source and data. Downstream consumers rely on it, so changing
it needs a new schema_version, not an edit.
"""

import uuid
from datetime import datetime
from decimal import ROUND_HALF_UP, Decimal
from typing import TYPE_CHECKING, NamedTuple

import asyncpg
from pydantic import BaseModel

if TYPE_CHECKING:  # only for type hints; importing the router here would create a circular import
    from api.routers.scoring import TransactionRequest

# The one event type this module produces. A constant, so the payload and the outbox row can never disagree.
EVENT_TYPE_TRANSACTION_CREATED = "transaction.created"

# The payload version. Bump it (as a new string) only when the envelope shape changes.
SCHEMA_VERSION = "1"

# A FIXED namespace for turning an idempotency key into an event_id (UUIDv5). It must never change: the same key
# only maps to the same event_id as long as this value stays the same. The value itself is arbitrary.
EVENT_ID_NAMESPACE = uuid.UUID("6f4a1c52-9d3e-4b87-a2c5-0e7d1b9f3a68")

# The name of the database rule that forbids two events with the same event_id (see migrations/0002).
EVENT_ID_UNIQUE_CONSTRAINT = "uq_outbox_event_id"

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
    """The outer envelope of every outbox event (LOCKED as of Level 7, schema_version "1")."""

    event_type: str = EVENT_TYPE_TRANSACTION_CREATED
    schema_version: str = SCHEMA_VERSION
    event_id: str
    transaction_id: str
    occurred_at: str  # ISO-8601 time the transaction row was created, taken from the database clock
    source: str
    data: OutboxEventData


class StoredTransaction(NamedTuple):
    """What the writer hands back: the stored transaction, whether it is new or the original of a replay."""

    transaction_id: str
    fraud_score: float | None
    model_version: str
    was_replay: bool


def make_event_id(idempotency_key: str | None) -> uuid.UUID:
    """
    Choose the event_id.

    With a key: uuid5(namespace, key), which is deterministic. The same key always gives the same ID, on any machine
    and at any time, so a retried request can be recognised. Without a key there is nothing to be consistent with, so
    we fall back to a random uuid4 and every request is treated as new.
    """
    if idempotency_key:
        return uuid.uuid5(EVENT_ID_NAMESPACE, idempotency_key)
    return uuid.uuid4()


def amount_to_decimal(amount: float) -> Decimal:
    """
    Turn the float from the request into an exact Decimal rounded to cents.

    Going through str() first matters: Decimal(42.5) would copy the float's tiny binary rounding errors, while
    Decimal("42.5") is exact. The request model hands us a float, so this is the one place it becomes money.
    """
    return Decimal(str(amount)).quantize(CENTS, rounding=ROUND_HALF_UP)


def build_payload(event_id: uuid.UUID, transaction_id: str, occurred_at: datetime, request: "TransactionRequest",
                  amount: Decimal, fraud_score: float | None, model_version: str, source: str) -> OutboxEventPayload:
    """Fill in the locked event envelope for one transaction."""
    return OutboxEventPayload(
        event_id=str(event_id),
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


async def insert_transaction_and_event(
    conn: asyncpg.Connection,
    request: "TransactionRequest",
    fraud_score: float | None,
    model_version: str,
    event_id: uuid.UUID,
) -> str:
    """
    Insert the transaction row and its outbox row inside ONE database transaction. Returns the new transaction_id.

    If anything in the block fails (including the database refusing a duplicate event_id), the database undoes the
    first insert too, so we never keep half of the pair.
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
        payload = build_payload(event_id, transaction_id, row["created_at"], request, amount, fraud_score,
                                model_version, source)

        # Step 3: store the event in the outbox. processed_at stays NULL, which means "not published yet".
        await conn.execute(
            """
            INSERT INTO outbox (transaction_id, event_id, event_type, payload)
            VALUES ($1::uuid, $2::uuid, $3, $4::jsonb)
            """,
            transaction_id, event_id, EVENT_TYPE_TRANSACTION_CREATED, payload.model_dump_json(),
        )
    return transaction_id


async def find_original_by_event_id(conn: asyncpg.Connection, event_id: uuid.UUID) -> StoredTransaction:
    """Look up the transaction that already owns this event_id (the original of a replayed request)."""
    row = await conn.fetchrow(
        """
        SELECT t.transaction_id, t.fraud_score, t.model_version
        FROM outbox o JOIN transactions t ON t.transaction_id = o.transaction_id
        WHERE o.event_id = $1
        """,
        event_id,
    )
    return StoredTransaction(str(row["transaction_id"]), row["fraud_score"], row["model_version"], was_replay=True)


async def record_transaction_with_outbox(
    conn: asyncpg.Connection,
    request: "TransactionRequest",
    fraud_score: float | None,
    model_version: str,
) -> StoredTransaction:
    """
    Save one e-commerce transaction and its outbox event atomically, or recognise a replay of an earlier request.

    A replay (same idempotency_key, so the same event_id) is a SUCCESS, not an error. The second insert into the
    outbox hits the UNIQUE rule on event_id, the database rolls back the whole attempt (no new rows are kept), and we
    return the ORIGINAL transaction. This is correct because the same key means "the same event": the first attempt
    already stored it, so doing it again would only create a duplicate. A client that retried because it never saw the
    first response gets the answer it missed. Any other database error is a real failure and is raised as before.
    """
    event_id = make_event_id(request.idempotency_key)
    try:
        transaction_id = await insert_transaction_and_event(conn, request, fraud_score, model_version, event_id)
    except asyncpg.UniqueViolationError as error:
        if error.constraint_name != EVENT_ID_UNIQUE_CONSTRAINT:
            raise  # a different uniqueness problem is a genuine failure, not a replay
        return await find_original_by_event_id(conn, event_id)
    return StoredTransaction(transaction_id, fraud_score, model_version, was_replay=False)
