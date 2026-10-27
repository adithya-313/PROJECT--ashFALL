# File: api/v1/idempotency.py
# Purpose: Commercial bank API idempotency key claim, conflict detection, expiration, and response caching.
import hashlib
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

import asyncpg
import orjson

from api.v1.signing import load_bank_api_policy


class Outcome(str, Enum):
    """Result of attempting to claim an idempotency key."""

    NEW = "NEW"
    REPLAY = "REPLAY"
    CONFLICT = "CONFLICT"


@dataclass(frozen=True)
class IdempotencyResult:
    """Outcome of an idempotency claim along with any previously cached response."""

    outcome: Outcome
    stored_status: int | None
    stored_body: Any | None


class IdempotencyStateError(Exception):
    """Raised when an idempotency operation encounters an impossible or corrupted database state."""


def hash_request(method: str, path: str, body_bytes: bytes) -> str:
    """
    Computes SHA-256 digest over the HTTP method, path, and request body.

    WHY: Reusing an Idempotency-Key across different endpoints or with a different HTTP method
    is a semantic conflict. Including method and path ensures that identical bodies sent to
    different routes are treated as distinct request signatures.
    """
    hasher = hashlib.sha256()
    hasher.update(method.upper().encode("utf-8"))
    hasher.update(b"\n")
    hasher.update(path.encode("utf-8"))
    hasher.update(b"\n")
    hasher.update(body_bytes)
    return hasher.hexdigest()


async def claim_idempotency_key(
    conn: asyncpg.Connection,
    bank_id: str,
    key: str,
    request_hash: str,
    now: datetime,
) -> IdempotencyResult:
    """
    Claims an idempotency key within an active bank-scoped database transaction.

    Transaction and Atomicity Contract:
      MUST be called inside `bank_scoped_transaction(conn, bank_id)` in the SAME transaction
      as the endpoint's business write. If the business write fails and rolls back, the claim
      rolls back atomically as well, preventing an unhandled server error from permanently blocking
      future client retries. If concurrent requests arrive simultaneously with the same key,
      PostgreSQL primary key locking on (bank_id, idempotency_key) forces the second transaction
      to wait until the first commits or rolls back.

    Time Contract:
      The `now` parameter must be the wall-clock time (e.g. from Clock.now_ist()).
      Per Ground Rule 30, idempotency cache expiration is a transport concern and must be
      calculated against real wall-clock time rather than simulated event_time.

    Logic:
      1. Attempt INSERT into bank_idempotency_keys ON CONFLICT (bank_id, idempotency_key) DO NOTHING.
      2. If inserted successfully, returns Outcome.NEW.
      3. If conflict exists, SELECT the existing row FOR UPDATE:
         - If expires_at <= now, the previous key has expired: reset response_status=NULL,
           response_body=NULL, update request_hash to new hash, set new expires_at, and return Outcome.NEW.
         - Else if existing row's request_hash != current request_hash: return Outcome.CONFLICT.
         - Else (same hash and unexpired): return Outcome.REPLAY with stored_status and stored_body.
    """
    policy = load_bank_api_policy()
    expires_at = now + timedelta(hours=policy.idempotency_lifetime_hours)

    row = await conn.fetchrow(
        """
        INSERT INTO bank_idempotency_keys (
            bank_id, idempotency_key, request_hash, created_at, expires_at
        )
        VALUES ($1, $2, $3, $4, $5)
        ON CONFLICT (bank_id, idempotency_key) DO NOTHING
        RETURNING idempotency_key
        """,
        bank_id,
        key,
        request_hash,
        now,
        expires_at,
    )

    if row is not None:
        return IdempotencyResult(outcome=Outcome.NEW, stored_status=None, stored_body=None)

    existing = await conn.fetchrow(
        """
        SELECT request_hash, response_status, response_body, expires_at
        FROM bank_idempotency_keys
        WHERE bank_id = $1 AND idempotency_key = $2
        FOR UPDATE
        """,
        bank_id,
        key,
    )

    if existing is None:
        # This state is logically impossible: the INSERT on conflict DO NOTHING was bypassed because
        # a row with PK (bank_id, idempotency_key) already exists. Under bank isolation and row-level
        # locking (FOR UPDATE), the conflicting row must exist and be visible to this transaction.
        raise IdempotencyStateError(
            f"Idempotency row for bank {bank_id!r} key {key!r} conflicted on insert but was not found on select"
        )

    stored_expires_at = existing["expires_at"]
    if stored_expires_at <= now:
        await conn.execute(
            """
            UPDATE bank_idempotency_keys
            SET request_hash = $3,
                response_status = NULL,
                response_body = NULL,
                created_at = $4,
                expires_at = $5
            WHERE bank_id = $1 AND idempotency_key = $2
            """,
            bank_id,
            key,
            request_hash,
            now,
            expires_at,
        )
        return IdempotencyResult(outcome=Outcome.NEW, stored_status=None, stored_body=None)

    stored_hash = existing["request_hash"]
    stored_body = existing["response_body"]
    if stored_body is not None and isinstance(stored_body, str):
        stored_body = orjson.loads(stored_body)

    if stored_hash != request_hash:
        return IdempotencyResult(
            outcome=Outcome.CONFLICT,
            stored_status=existing["response_status"],
            stored_body=stored_body,
        )

    return IdempotencyResult(
        outcome=Outcome.REPLAY,
        stored_status=existing["response_status"],
        stored_body=stored_body,
    )


async def store_idempotent_response(
    conn: asyncpg.Connection,
    bank_id: str,
    key: str,
    status: int,
    body_dict: Any,
) -> None:
    """
    Caches the final HTTP status and response payload for an idempotent transaction.

    Args:
        conn: Bank-scoped database connection.
        bank_id: Identifier of the bank.
        key: Idempotency-Key string.
        status: HTTP status code returned to client (e.g. 200, 201).
        body_dict: Response payload dictionary or structure to cache.
    """
    body_json = orjson.dumps(body_dict).decode("utf-8")
    result_status = await conn.execute(
        """
        UPDATE bank_idempotency_keys
        SET response_status = $3,
            response_body = $4::jsonb
        WHERE bank_id = $1 AND idempotency_key = $2
        """,
        bank_id,
        key,
        status,
        body_json,
    )
    if result_status != "UPDATE 1":
        raise IdempotencyStateError(
            f"Expected exactly 1 row updated for bank {bank_id!r} key {key!r}, got {result_status!r}"
        )
