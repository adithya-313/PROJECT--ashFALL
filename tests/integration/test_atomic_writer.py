# File: tests/integration/test_atomic_writer.py
# Purpose: Proves record_transaction_with_outbox saves the transaction row and its outbox row together or not at all,
#          and that a replayed idempotency_key is recognised.
#
# Real Postgres, not mocks: "both rows or neither" is a promise the DATABASE keeps, so only a real database can prove
# it. Each test runs in its own throwaway schema (see the isolated_schema fixture in conftest.py).
import uuid

import asyncpg
import orjson
import pytest

from api.routers.scoring import TransactionRequest
from api.services.outbox import EVENT_TYPE_TRANSACTION_CREATED, make_event_id, record_transaction_with_outbox

pytestmark = pytest.mark.anyio

MODEL_VERSION = "test-model-1"
FRAUD_SCORE = 0.25


def make_request(idempotency_key: str | None = None) -> TransactionRequest:
    """Build a normal request. Only the real API fields are used."""
    return TransactionRequest(
        amount=42.5,
        account_id="acct-1",
        merchant_id="merchant-1",
        device_fingerprint="device-1",
        geo_coordinates="40.7,-74.0",
        idempotency_key=idempotency_key,
    )


async def count_rows(connection: asyncpg.Connection, table_name: str) -> int:
    """Count every row in one table of the test schema (the schema is fresh, so these are all ours)."""
    return await connection.fetchval(f"SELECT count(*) FROM {table_name}")  # table_name is a fixed literal below


async def test_success_writes_one_transaction_and_one_outbox_row(isolated_schema):
    """One successful write stores exactly one transaction and one unpublished outbox row for it."""
    connection, _ = isolated_schema

    # Do one normal write.
    stored = await record_transaction_with_outbox(connection, make_request(), FRAUD_SCORE, MODEL_VERSION)

    # Exactly one row in each table, and the outbox row belongs to this transaction.
    assert await count_rows(connection, "transactions") == 1
    assert await count_rows(connection, "outbox") == 1
    outbox_row = await connection.fetchrow(
        "SELECT processed_at FROM outbox WHERE transaction_id = $1::uuid", stored.transaction_id)
    assert outbox_row is not None
    assert outbox_row["processed_at"] is None  # NULL means "not published yet"
    assert stored.was_replay is False


async def test_outbox_payload_matches_locked_schema_v1(isolated_schema):
    """The stored outbox payload has exactly the locked version-1 envelope and data keys."""
    connection, _ = isolated_schema

    stored = await record_transaction_with_outbox(connection, make_request(), FRAUD_SCORE, MODEL_VERSION)

    # Read the payload back as text and decode it with orjson.
    row = await connection.fetchrow(
        "SELECT event_id, event_type, payload::text AS payload_text FROM outbox WHERE transaction_id = $1::uuid",
        stored.transaction_id)
    payload = orjson.loads(row["payload_text"])

    # Top-level keys must be exactly the locked envelope.
    assert set(payload) == {"event_type", "schema_version", "event_id", "transaction_id", "occurred_at", "source",
                            "data"}
    assert payload["schema_version"] == "1"
    assert payload["event_type"] == EVENT_TYPE_TRANSACTION_CREATED
    assert row["event_type"] == EVENT_TYPE_TRANSACTION_CREATED
    assert payload["transaction_id"] == stored.transaction_id

    # The data block must be exactly these keys, and money travels as a string, never a float.
    assert set(payload["data"]) == {"account_id", "merchant_id", "amount", "fraud_score", "model_version"}
    assert isinstance(payload["data"]["amount"], str)

    # The event_id inside the payload must match the dedicated column the UNIQUE rule protects.
    assert payload["event_id"] == str(row["event_id"])


async def test_same_idempotency_key_returns_original_and_writes_nothing_new(isolated_schema):
    """Sending the same idempotency_key twice returns the original transaction and stores no second copy."""
    connection, _ = isolated_schema
    key = "key-" + uuid.uuid4().hex

    first = await record_transaction_with_outbox(connection, make_request(key), FRAUD_SCORE, MODEL_VERSION)
    second = await record_transaction_with_outbox(connection, make_request(key), 0.99, "other-model")

    # The second call is a recognised replay that hands back the FIRST transaction (and its original score).
    assert first.was_replay is False
    assert second.was_replay is True
    assert second.transaction_id == first.transaction_id
    assert second.fraud_score == FRAUD_SCORE
    assert second.model_version == MODEL_VERSION

    # Nothing new was kept.
    assert await count_rows(connection, "transactions") == 1
    assert await count_rows(connection, "outbox") == 1


def test_same_idempotency_key_gives_same_event_id_every_time():
    """The same key always maps to the same version-5 UUID, while no key gives a new random id each time."""
    key = "key-" + uuid.uuid4().hex

    # Same key twice: identical, and a name-based (version 5) UUID.
    first_id = make_event_id(key)
    second_id = make_event_id(key)
    assert first_id == second_id
    assert first_id.version == 5

    # No key: every call gives a different id, so unrelated requests never collide.
    assert make_event_id(None) != make_event_id(None)


async def test_database_error_mid_write_leaves_no_rows(isolated_schema):
    """If the outbox insert fails after the transaction insert worked, the database undoes both."""
    connection, _ = isolated_schema

    # Make every outbox insert fail. NOT VALID means existing rows are not checked (there are none), but new rows
    # still must satisfy CHECK (false), which nothing can.
    await connection.execute("ALTER TABLE outbox ADD CONSTRAINT force_fail_for_test CHECK (false) NOT VALID")

    # The writer inserts into transactions FIRST (that succeeds), then into outbox (that fails). Because both
    # happen in one database transaction, Postgres rolls back BOTH. This is the core Level 7 promise.
    with pytest.raises(asyncpg.CheckViolationError):
        await record_transaction_with_outbox(connection, make_request(), FRAUD_SCORE, MODEL_VERSION)

    # The schema was fresh, so any row at all would be a leftover from this failed attempt.
    assert await count_rows(connection, "transactions") == 0
    assert await count_rows(connection, "outbox") == 0
