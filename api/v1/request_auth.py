# File: api/v1/request_auth.py
# Purpose: Bank API request authentication dependency verifying headers, signatures, freshness, nonces, and keys.
import re
from dataclasses import dataclass
from datetime import timedelta

import asyncpg
from fastapi import Request
from fastapi.responses import JSONResponse

from api.v1.bank_keys import resolve_bank_for_key
from api.v1.signing import (
    SigningNotConfiguredError,
    build_canonical_request,
    get_signing_secret,
    is_fresh,
    load_bank_api_policy,
    sign,
    signatures_match,
)
from core.event_clock import Clock

HEADER_KEY = "X-Ashfall-Key"
HEADER_TIMESTAMP = "X-Ashfall-Timestamp"
HEADER_NONCE = "X-Ashfall-Nonce"
HEADER_SIGNATURE = "X-Ashfall-Signature"
HEADER_IDEMPOTENCY_KEY = "Idempotency-Key"

NONCE_PATTERN = re.compile(r"^[A-Za-z0-9_-]{16,64}$")
IDEMPOTENCY_KEY_PATTERN = re.compile(r"^[A-Za-z0-9_.:-]{8,128}$")


@dataclass(frozen=True)
class BankRequestContext:
    """Authenticated context for an incoming commercial bank request."""

    bank_id: str
    body_bytes: bytes
    idempotency_key: str | None


class BankAuthError(Exception):
    """
    Raised when bank request authentication fails.

    Attributes:
        http_status: HTTP status code to return (e.g. 400, 401, 413, 503).
        code: Machine-readable short error code.
        message: Safe user-facing error message (never reveals internal secret or key details).
    """

    def __init__(self, http_status: int, code: str, message: str) -> None:
        super().__init__(message)
        self.http_status = http_status
        self.code = code
        self.message = message


def make_auth_error_response(error: BankAuthError) -> JSONResponse:
    """Converts a BankAuthError into a standardized JSON response envelope."""
    return JSONResponse(
        status_code=error.http_status,
        content={"error": {"code": error.code, "message": error.message}},
    )


def raw_request_path(request: Request) -> str:
    """
    Extracts and decodes the raw ASGI request path as ASCII.

    Signatures and idempotency hashes must cover the exact wire-format
    raw path rather than a decoded path so that alternate encodings of
    the same decoded text do not collide or produce valid signatures across
    different paths.

    Args:
        request: The incoming HTTP request containing the ASGI scope.

    Returns:
        The raw request path decoded as an ASCII string.

    Raises:
        BankAuthError: 400 'bad_request_path' if raw_path is missing from scope,
            is not bytes, or cannot be decoded as ASCII.
    """
    raw_path = request.scope.get("raw_path")
    if not isinstance(raw_path, bytes):
        raise BankAuthError(400, "bad_request_path", "Missing raw request path in scope")
    try:
        return raw_path.decode("ascii")
    except UnicodeDecodeError as exc:
        raise BankAuthError(400, "bad_request_path", "Raw request path is not valid ASCII") from exc


async def authenticate_bank_request(
    request: Request,
    pool: asyncpg.Pool,
    clock: Clock,
) -> BankRequestContext:
    """
    Authenticates an incoming bank HTTP request.

    Execution Order and Security Architecture:
      1. Content-Length / body size check (cheap DOS protection before reading memory).
      2. Header presence check (cheap validation before cryptographic parsing).
      3. Timestamp validation & freshness check (cheap parsing before database lookup).
      4. Nonce format regex check (cheap validation before stateful database lookup).
      5. API key resolution on owner connection (proves tenant identity and active key status).
      6. Secret retrieval for resolved bank (retrieves pre-shared HMAC secret from app config).
      7. Rebuild canonical request and verify HMAC signature (verifies integrity and authenticity).
      8. Nonce recording & deduplication in ONE owner transaction (nonce is consumed ONLY after
         cryptographic signature is proven valid, preventing nonce-exhaustion DOS attacks).
      9. Idempotency-Key validation for POST methods (verifies idempotency contract before business logic).

    Note:
      This function deliberately does NOT parse or validate the JSON body contents. Schema and payload
      validation is the responsibility of endpoint handlers.
    """
    policy = load_bank_api_policy()

    # 1. Content-Length / body size check
    cl_header = request.headers.get("content-length")
    if cl_header is not None:
        try:
            cl_val = int(cl_header)
            if cl_val > policy.max_request_body_bytes:
                raise BankAuthError(413, "body_too_large", "Request body exceeds maximum allowed size")
        except ValueError:
            pass

    body_bytes = await request.body()
    if len(body_bytes) > policy.max_request_body_bytes:
        raise BankAuthError(413, "body_too_large", "Request body exceeds maximum allowed size")

    # 2. Header presence check
    key = request.headers.get(HEADER_KEY)
    ts_raw = request.headers.get(HEADER_TIMESTAMP)
    nonce = request.headers.get(HEADER_NONCE)
    signature = request.headers.get(HEADER_SIGNATURE)

    if not key or not ts_raw or not nonce or not signature:
        raise BankAuthError(401, "missing_auth_headers", "Missing required authentication headers")

    # 3. Timestamp integer and freshness check
    try:
        ts = int(ts_raw)
    except ValueError as exc:
        raise BankAuthError(401, "bad_timestamp", "Timestamp header must be an integer") from exc

    now = clock.now_ist()
    if not is_fresh(ts, now, policy.signature_max_age_seconds):
        raise BankAuthError(401, "stale_signature", "Request timestamp is outside allowable window")

    # 4. Nonce format check
    if not NONCE_PATTERN.match(nonce):
        raise BankAuthError(401, "bad_nonce", "Nonce format is invalid")

    # 5. API key resolution on owner connection from the pool
    async with pool.acquire() as conn:
        bank_id = await resolve_bank_for_key(conn, key)

    if bank_id is None:
        raise BankAuthError(401, "invalid_key", "Invalid or revoked API key")

    # 6. Retrieve signing secret
    try:
        secret_bytes = get_signing_secret(bank_id)
    except SigningNotConfiguredError as exc:
        raise BankAuthError(503, "signing_not_configured", "Signing secret not configured") from exc

    # 7. Rebuild canonical request and compare signature
    path = raw_request_path(request)
    raw_query = request.url.query or ""
    canonical = build_canonical_request(
        method=request.method,
        path=path,
        query=raw_query,
        timestamp=ts,
        nonce=nonce,
        body_bytes=body_bytes,
    )
    expected_sig = sign(secret_bytes, canonical)
    if not signatures_match(expected_sig, signature):
        raise BankAuthError(401, "bad_signature", "Signature verification failed")

    # 8. Record nonce in ONE owner transaction, pruning expired nonces
    cutoff_time = now - timedelta(seconds=policy.nonce_retention_seconds)
    async with pool.acquire() as conn:
        async with conn.transaction():
            await conn.execute(
                "DELETE FROM bank_request_nonces WHERE created_at < $1",
                cutoff_time,
            )
            inserted = await conn.fetchval(
                """
                INSERT INTO bank_request_nonces (bank_id, nonce, created_at)
                VALUES ($1, $2, $3)
                ON CONFLICT (bank_id, nonce) DO NOTHING
                RETURNING nonce
                """,
                bank_id,
                nonce,
                now,
            )
            if inserted is None:
                raise BankAuthError(401, "nonce_reused", "Nonce has already been used")

    # 9. Idempotency-Key validation on POST
    idempotency_key: str | None = None
    if request.method.upper() == "POST":
        idempotency_key = request.headers.get(HEADER_IDEMPOTENCY_KEY)
        if not idempotency_key or not IDEMPOTENCY_KEY_PATTERN.match(idempotency_key):
            raise BankAuthError(400, "bad_idempotency_key", "Missing or invalid Idempotency-Key header on POST")

    return BankRequestContext(
        bank_id=bank_id,
        body_bytes=body_bytes,
        idempotency_key=idempotency_key,
    )
