# File: tests/integration/test_bank_idempotency.py
# Purpose: Proves commercial bank API idempotency key claims, replay response caching,
#          hash conflict detection, expiration renewal, tenant isolation, and transactional rollback.
from datetime import datetime, timedelta

import asyncpg
import pytest

from api.v1.bank_scope import bank_scoped_transaction
from api.v1.idempotency import (
    Outcome,
    claim_idempotency_key,
    hash_request,
    store_idempotent_response,
)
from core.event_clock import IST, ManualClock
from tests.integration.conftest import get_test_database_url

pytestmark = pytest.mark.anyio


@pytest.fixture
async def db_pool(isolated_schema):
    """Provides a connection pool scoped to the isolated test schema."""
    _, schema_name = isolated_schema
    pool = await asyncpg.create_pool(
        dsn=get_test_database_url(),
        min_size=1,
        max_size=2,
        server_settings={"search_path": schema_name},
    )
    assert pool is not None
    try:
        yield pool
    finally:
        await pool.close()


async def test_new_key_yields_outcome_new(db_pool):
    """Claiming an unused idempotency key returns Outcome.NEW."""
    clock = ManualClock(datetime(2026, 10, 28, 12, 0, 0, tzinfo=IST))
    req_hash = hash_request("POST", "/api/v1/decide", b'{"amount": 100}')

    async with db_pool.acquire() as conn:
        async with bank_scoped_transaction(conn, "bank_0"):
            res = await claim_idempotency_key(conn, "bank_0", "idem_new_001", req_hash, clock.now_ist())

    assert res.outcome == Outcome.NEW
    assert res.stored_status is None
    assert res.stored_body is None


async def test_same_key_and_hash_after_storing_yields_replay(db_pool):
    """Re-submitting the same key and request hash returns Outcome.REPLAY with the exact cached response."""
    clock = ManualClock(datetime(2026, 10, 28, 12, 0, 0, tzinfo=IST))
    req_hash = hash_request("POST", "/api/v1/decide", b'{"amount": 100}')
    key = "idem_replay_002"
    stored_payload = {"decision": "unscored", "score": None, "approved": True}

    async with db_pool.acquire() as conn:
        async with bank_scoped_transaction(conn, "bank_0"):
            first_res = await claim_idempotency_key(conn, "bank_0", key, req_hash, clock.now_ist())
            assert first_res.outcome == Outcome.NEW

            await store_idempotent_response(conn, "bank_0", key, 200, stored_payload)

    # Subsequent claim in a separate transaction returns REPLAY with identical payload
    async with db_pool.acquire() as conn:
        async with bank_scoped_transaction(conn, "bank_0"):
            second_res = await claim_idempotency_key(conn, "bank_0", key, req_hash, clock.now_ist())

    assert second_res.outcome == Outcome.REPLAY
    assert second_res.stored_status == 200
    assert second_res.stored_body == stored_payload


async def test_same_key_with_different_hash_yields_conflict(db_pool):
    """Re-submitting the same idempotency key with a mutated request body returns Outcome.CONFLICT."""
    clock = ManualClock(datetime(2026, 10, 28, 12, 0, 0, tzinfo=IST))
    hash_original = hash_request("POST", "/api/v1/decide", b'{"amount": 100}')
    hash_different = hash_request("POST", "/api/v1/decide", b'{"amount": 999}')
    key = "idem_conflict_003"

    async with db_pool.acquire() as conn:
        async with bank_scoped_transaction(conn, "bank_0"):
            first_res = await claim_idempotency_key(conn, "bank_0", key, hash_original, clock.now_ist())
            assert first_res.outcome == Outcome.NEW
            await store_idempotent_response(conn, "bank_0", key, 200, {"ok": True})

    # Different body on same key returns CONFLICT
    async with db_pool.acquire() as conn:
        async with bank_scoped_transaction(conn, "bank_0"):
            conflict_res = await claim_idempotency_key(conn, "bank_0", key, hash_different, clock.now_ist())

    assert conflict_res.outcome == Outcome.CONFLICT


async def test_expired_key_renews_as_outcome_new(db_pool):
    """When a key's retention window (24h) has elapsed, re-using it resets the row and returns Outcome.NEW."""
    now_start = datetime(2026, 10, 28, 12, 0, 0, tzinfo=IST)
    clock = ManualClock(now_start)
    hash_v1 = hash_request("POST", "/api/v1/decide", b'{"amount": 100}')
    hash_v2 = hash_request("POST", "/api/v1/decide", b'{"amount": 500}')
    key = "idem_expired_004"

    async with db_pool.acquire() as conn:
        async with bank_scoped_transaction(conn, "bank_0"):
            first_res = await claim_idempotency_key(conn, "bank_0", key, hash_v1, clock.now_ist())
            assert first_res.outcome == Outcome.NEW
            await store_idempotent_response(conn, "bank_0", key, 200, {"old": "data"})

    # Advance clock past the 24-hour lifetime (24h + 1s)
    clock.advance(timedelta(hours=24, seconds=1))

    async with db_pool.acquire() as conn:
        async with bank_scoped_transaction(conn, "bank_0"):
            expired_res = await claim_idempotency_key(conn, "bank_0", key, hash_v2, clock.now_ist())

    assert expired_res.outcome == Outcome.NEW
    assert expired_res.stored_status is None
    assert expired_res.stored_body is None


async def test_different_banks_using_same_key_string_are_independent(db_pool):
    """bank_0 and bank_1 can simultaneously claim the same key string without conflict."""
    clock = ManualClock(datetime(2026, 10, 28, 12, 0, 0, tzinfo=IST))
    req_hash = hash_request("POST", "/api/v1/decide", b'{"amount": 100}')
    shared_key_name = "idem_shared_key_string"

    async with db_pool.acquire() as conn:
        async with bank_scoped_transaction(conn, "bank_0"):
            res_0 = await claim_idempotency_key(conn, "bank_0", shared_key_name, req_hash, clock.now_ist())
            assert res_0.outcome == Outcome.NEW

    async with db_pool.acquire() as conn:
        async with bank_scoped_transaction(conn, "bank_1"):
            res_1 = await claim_idempotency_key(conn, "bank_1", shared_key_name, req_hash, clock.now_ist())
            assert res_1.outcome == Outcome.NEW


async def test_bank_0_key_row_is_invisible_under_bank_1_scope(db_pool):
    """Row Level Security on bank_idempotency_keys prevents bank_1 from viewing bank_0's keys."""
    clock = ManualClock(datetime(2026, 10, 28, 12, 0, 0, tzinfo=IST))
    req_hash = hash_request("POST", "/api/v1/decide", b'{"amount": 100}')
    key_bank_0 = "idem_bank_0_private_key"

    async with db_pool.acquire() as conn:
        async with bank_scoped_transaction(conn, "bank_0"):
            await claim_idempotency_key(conn, "bank_0", key_bank_0, req_hash, clock.now_ist())

    # Under bank_1 scope, querying bank_idempotency_keys returns 0 rows
    async with db_pool.acquire() as conn:
        async with bank_scoped_transaction(conn, "bank_1"):
            rows = await conn.fetch(
                "SELECT idempotency_key FROM bank_idempotency_keys WHERE idempotency_key = $1",
                key_bank_0,
            )
            assert len(rows) == 0


async def test_claim_inside_failed_transaction_leaves_no_row(db_pool):
    """If a business transaction rolls back, the claimed idempotency key is rolled back atomically."""
    clock = ManualClock(datetime(2026, 10, 28, 12, 0, 0, tzinfo=IST))
    req_hash = hash_request("POST", "/api/v1/decide", b'{"amount": 100}')
    key_rollback = "idem_rollback_test_006"

    with pytest.raises(RuntimeError):
        async with db_pool.acquire() as conn:
            async with bank_scoped_transaction(conn, "bank_0"):
                res = await claim_idempotency_key(conn, "bank_0", key_rollback, req_hash, clock.now_ist())
                assert res.outcome == Outcome.NEW
                raise RuntimeError("Simulated failure in endpoint business write")

    # In a fresh transaction, the key was not persisted and can be claimed anew as NEW
    async with db_pool.acquire() as conn:
        async with bank_scoped_transaction(conn, "bank_0"):
            res_after = await claim_idempotency_key(conn, "bank_0", key_rollback, req_hash, clock.now_ist())
            assert res_after.outcome == Outcome.NEW
