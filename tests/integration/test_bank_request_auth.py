# File: tests/integration/test_bank_request_auth.py
# Purpose: Proves bank API request authentication pipeline: size limits, header validation,
#          timestamp freshness, nonce replay prevention, key resolution, HMAC signature verification,
#          and Idempotency-Key validation.
from datetime import datetime
import secrets

import asyncpg
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import httpx
from pydantic import SecretStr
import pytest

from api.v1.bank_keys import issue_api_key
from api.v1.request_auth import (
    BankAuthError,
    authenticate_bank_request,
    make_auth_error_response,
)
from api.v1.signing import build_canonical_request, sign
from core.config import config
from core.event_clock import IST, ManualClock
from tests.integration.conftest import get_test_database_url

pytestmark = pytest.mark.anyio

SECRET_BANK_0 = b"secret_bytes_bank_0_32_characters_min!"
SECRET_BANK_1 = b"secret_bytes_bank_1_32_characters_min!"


def create_test_app() -> FastAPI:
    """Builds an isolated FastAPI application instrumented with the bank authentication dependency."""
    app = FastAPI()

    @app.exception_handler(BankAuthError)
    async def auth_error_handler(request: Request, exc: BankAuthError) -> JSONResponse:
        return make_auth_error_response(exc)

    @app.post("/api/v1/test-post")
    async def post_route(request: Request) -> dict[str, str | None]:
        ctx = await authenticate_bank_request(request, request.app.state.pool, request.app.state.clock)
        return {"bank_id": ctx.bank_id, "idempotency_key": ctx.idempotency_key}

    @app.get("/api/v1/test-get")
    async def get_route(request: Request) -> dict[str, str]:
        ctx = await authenticate_bank_request(request, request.app.state.pool, request.app.state.clock)
        return {"bank_id": ctx.bank_id}

    return app


def make_signed_headers(
    key: str,
    secret: bytes,
    method: str,
    path: str,
    body: bytes,
    clock: ManualClock,
    nonce: str,
    query: str = "",
    idempotency_key: str | None = None,
    timestamp_override: int | str | None = None,
) -> dict[str, str]:
    """Helper to assemble and sign HTTP authentication headers."""
    ts = timestamp_override if timestamp_override is not None else int(clock.now_ist().timestamp())
    canonical = build_canonical_request(method, path, query, ts, nonce, body)
    sig = sign(secret, canonical)
    headers = {
        "X-Ashfall-Key": key,
        "X-Ashfall-Timestamp": str(ts),
        "X-Ashfall-Nonce": nonce,
        "X-Ashfall-Signature": sig,
    }
    if idempotency_key is not None:
        headers["Idempotency-Key"] = idempotency_key
    return headers


@pytest.fixture
async def auth_setup(isolated_schema, monkeypatch: pytest.MonkeyPatch):
    """Sets up database connection pool, active API keys, and test clock."""
    connection, schema_name = isolated_schema
    pool = await asyncpg.create_pool(
        dsn=get_test_database_url(),
        min_size=1,
        max_size=2,
        server_settings={"search_path": schema_name},
    )
    assert pool is not None

    monkeypatch.setattr(config, "bank_signing_secret_bank_0", SecretStr(SECRET_BANK_0.decode("utf-8")))
    monkeypatch.setattr(config, "bank_signing_secret_bank_1", SecretStr(SECRET_BANK_1.decode("utf-8")))

    # Issue API keys for bank_0 and bank_1
    _, key_bank_0 = await issue_api_key(connection, "bank_0")
    _, key_bank_1 = await issue_api_key(connection, "bank_1")

    now = datetime(2026, 10, 28, 12, 0, 0, tzinfo=IST)
    clock = ManualClock(now)

    app = create_test_app()
    app.state.pool = pool
    app.state.clock = clock

    try:
        yield {
            "app": app,
            "clock": clock,
            "key_bank_0": key_bank_0,
            "key_bank_1": key_bank_1,
        }
    finally:
        await pool.close()


async def test_good_request_authenticates_successfully(auth_setup):
    """A valid POST request authenticates cleanly, returning 200 with the correct bank_id."""
    app = auth_setup["app"]
    clock = auth_setup["clock"]
    key = auth_setup["key_bank_0"]

    body = b'{"amount": 500, "payment_id": "pmt_auth_001"}'
    headers = make_signed_headers(
        key=key,
        secret=SECRET_BANK_0,
        method="POST",
        path="/api/v1/test-post",
        body=body,
        clock=clock,
        nonce="valid_nonce_12345678",
        idempotency_key="idem_key_001_valid",
    )

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/v1/test-post", content=body, headers=headers)

    assert resp.status_code == 200
    data = resp.json()
    assert data["bank_id"] == "bank_0"
    assert data["idempotency_key"] == "idem_key_001_valid"


async def test_step_1_body_too_large(auth_setup):
    """A request whose body exceeds max_request_body_bytes (65536) returns 413 body_too_large."""
    app = auth_setup["app"]
    clock = auth_setup["clock"]
    key = auth_setup["key_bank_0"]

    oversized_body = b"x" * 65537
    headers = make_signed_headers(
        key=key,
        secret=SECRET_BANK_0,
        method="POST",
        path="/api/v1/test-post",
        body=oversized_body,
        clock=clock,
        nonce="valid_nonce_12345678",
        idempotency_key="idem_key_001_valid",
    )

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/v1/test-post", content=oversized_body, headers=headers)

    assert resp.status_code == 413
    assert resp.json()["error"]["code"] == "body_too_large"


async def test_step_2_missing_auth_headers(auth_setup):
    """Missing any of the 4 authentication headers returns 401 missing_auth_headers."""
    app = auth_setup["app"]
    body = b'{"data": 1}'

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        # Omits X-Ashfall-Signature
        headers = {
            "X-Ashfall-Key": "some_key",
            "X-Ashfall-Timestamp": "1700000000",
            "X-Ashfall-Nonce": "valid_nonce_12345678",
            "Idempotency-Key": "idem_valid_123",
        }
        resp = await client.post("/api/v1/test-post", content=body, headers=headers)

    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "missing_auth_headers"


async def test_step_3_bad_timestamp_and_stale_signature(auth_setup):
    """Non-integer timestamp returns 401 bad_timestamp; timestamp outside window returns 401 stale_signature."""
    app = auth_setup["app"]
    clock = auth_setup["clock"]
    key = auth_setup["key_bank_0"]
    body = b'{"data": 1}'

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        # Non-integer timestamp
        headers_bad_ts = {
            "X-Ashfall-Key": key,
            "X-Ashfall-Timestamp": "not-an-integer",
            "X-Ashfall-Nonce": "valid_nonce_12345678",
            "X-Ashfall-Signature": "fakesig",
            "Idempotency-Key": "idem_valid_123",
        }
        resp = await client.post("/api/v1/test-post", content=body, headers=headers_bad_ts)
        assert resp.status_code == 401
        assert resp.json()["error"]["code"] == "bad_timestamp"

        # Stale timestamp (> 300s old)
        stale_ts = int(clock.now_ist().timestamp()) - 301
        headers_stale = make_signed_headers(
            key=key,
            secret=SECRET_BANK_0,
            method="POST",
            path="/api/v1/test-post",
            body=body,
            clock=clock,
            nonce="valid_nonce_12345678",
            idempotency_key="idem_valid_123",
            timestamp_override=stale_ts,
        )
        resp_stale = await client.post("/api/v1/test-post", content=body, headers=headers_stale)
        assert resp_stale.status_code == 401
        assert resp_stale.json()["error"]["code"] == "stale_signature"


async def test_step_4_bad_nonce_format(auth_setup):
    """Nonce not matching ^[A-Za-z0-9_-]{16,64}$ returns 401 bad_nonce."""
    app = auth_setup["app"]
    clock = auth_setup["clock"]
    key = auth_setup["key_bank_0"]
    body = b'{"data": 1}'

    # Too short (< 16 chars)
    short_nonce = "short_nonce"
    headers = make_signed_headers(
        key=key,
        secret=SECRET_BANK_0,
        method="POST",
        path="/api/v1/test-post",
        body=body,
        clock=clock,
        nonce=short_nonce,
        idempotency_key="idem_valid_123",
    )

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/v1/test-post", content=body, headers=headers)

    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "bad_nonce"


async def test_step_5_invalid_key(auth_setup):
    """An unknown or malformed API key returns 401 invalid_key."""
    app = auth_setup["app"]
    clock = auth_setup["clock"]
    body = b'{"data": 1}'

    fake_key = "ak_0123456789abcdef.fake_secret_token_value_32_bytes"
    headers = make_signed_headers(
        key=fake_key,
        secret=SECRET_BANK_0,
        method="POST",
        path="/api/v1/test-post",
        body=body,
        clock=clock,
        nonce="valid_nonce_12345678",
        idempotency_key="idem_valid_123",
    )

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/v1/test-post", content=body, headers=headers)

    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "invalid_key"


async def test_step_6_signing_not_configured(auth_setup, monkeypatch: pytest.MonkeyPatch):
    """If no secret is configured for the resolved bank, returns 503 signing_not_configured."""
    app = auth_setup["app"]
    clock = auth_setup["clock"]
    key = auth_setup["key_bank_0"]
    body = b'{"data": 1}'

    # Unset signing secret for bank_0
    monkeypatch.setattr(config, "bank_signing_secret_bank_0", None)

    headers = make_signed_headers(
        key=key,
        secret=SECRET_BANK_0,
        method="POST",
        path="/api/v1/test-post",
        body=body,
        clock=clock,
        nonce="valid_nonce_12345678",
        idempotency_key="idem_valid_123",
    )

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/v1/test-post", content=body, headers=headers)

    assert resp.status_code == 503
    assert resp.json()["error"]["code"] == "signing_not_configured"


async def test_step_7_bad_signature(auth_setup):
    """A signature that does not match the canonical calculation returns 401 bad_signature."""
    app = auth_setup["app"]
    clock = auth_setup["clock"]
    key = auth_setup["key_bank_0"]
    body = b'{"data": 1}'

    headers = make_signed_headers(
        key=key,
        secret=SECRET_BANK_0,
        method="POST",
        path="/api/v1/test-post",
        body=body,
        clock=clock,
        nonce="valid_nonce_12345678",
        idempotency_key="idem_valid_123",
    )
    # Corrupt the signature
    headers["X-Ashfall-Signature"] = "0" * 64

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/v1/test-post", content=body, headers=headers)

    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "bad_signature"


async def test_step_8_nonce_reused(auth_setup):
    """Reusing the same nonce returns 401 nonce_reused on the second submission."""
    app = auth_setup["app"]
    clock = auth_setup["clock"]
    key = auth_setup["key_bank_0"]
    body = b'{"data": 1}'
    fixed_nonce = "fixed_nonce_12345678"

    headers_1 = make_signed_headers(
        key=key,
        secret=SECRET_BANK_0,
        method="POST",
        path="/api/v1/test-post",
        body=body,
        clock=clock,
        nonce=fixed_nonce,
        idempotency_key="idem_key_001",
    )

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        # First request succeeds
        resp1 = await client.post("/api/v1/test-post", content=body, headers=headers_1)
        assert resp1.status_code == 200

        # Second request with identical nonce is refused
        headers_2 = make_signed_headers(
            key=key,
            secret=SECRET_BANK_0,
            method="POST",
            path="/api/v1/test-post",
            body=body,
            clock=clock,
            nonce=fixed_nonce,
            idempotency_key="idem_key_002",
        )
        resp2 = await client.post("/api/v1/test-post", content=body, headers=headers_2)
        assert resp2.status_code == 401
        assert resp2.json()["error"]["code"] == "nonce_reused"


async def test_step_9_bad_idempotency_key_on_post(auth_setup):
    """Missing or invalid Idempotency-Key on POST returns 400 bad_idempotency_key."""
    app = auth_setup["app"]
    clock = auth_setup["clock"]
    key = auth_setup["key_bank_0"]
    body = b'{"data": 1}'

    # Missing Idempotency-Key
    headers_missing = make_signed_headers(
        key=key,
        secret=SECRET_BANK_0,
        method="POST",
        path="/api/v1/test-post",
        body=body,
        clock=clock,
        nonce="valid_nonce_12345678",
        idempotency_key=None,
    )

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/v1/test-post", content=body, headers=headers_missing)
        assert resp.status_code == 400
        assert resp.json()["error"]["code"] == "bad_idempotency_key"

        # Invalid format (too short < 8 chars)
        headers_short = make_signed_headers(
            key=key,
            secret=SECRET_BANK_0,
            method="POST",
            path="/api/v1/test-post",
            body=body,
            clock=clock,
            nonce="valid_nonce_12345679",
            idempotency_key="short",
        )
        resp_short = await client.post("/api/v1/test-post", content=body, headers=headers_short)
        assert resp_short.status_code == 400
        assert resp_short.json()["error"]["code"] == "bad_idempotency_key"


async def test_fast_replay_with_old_event_time_succeeds(auth_setup):
    """A fast replay (old event_time in JSON body, but fresh transport timestamp and nonce) returns 200."""
    app = auth_setup["app"]
    clock = auth_setup["clock"]
    key = auth_setup["key_bank_0"]
    body = b'{"event_time": "2020-01-01T00:00:00+05:30", "amount": 100}'

    headers = make_signed_headers(
        key=key,
        secret=SECRET_BANK_0,
        method="POST",
        path="/api/v1/test-post",
        body=body,
        clock=clock,
        nonce="fresh_nonce_12345678",
        idempotency_key="idem_replay_test_001",
    )

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/v1/test-post", content=body, headers=headers)

    assert resp.status_code == 200
    assert resp.json()["bank_id"] == "bank_0"


async def test_bank_0_secret_with_bank_1_key_fails(auth_setup):
    """A signature computed with bank_0's secret but presented with bank_1's key returns 401 bad_signature."""
    app = auth_setup["app"]
    clock = auth_setup["clock"]
    key_bank_1 = auth_setup["key_bank_1"]
    body = b'{"amount": 100}'

    # Signed with SECRET_BANK_0, presented under key_bank_1
    headers = make_signed_headers(
        key=key_bank_1,
        secret=SECRET_BANK_0,
        method="POST",
        path="/api/v1/test-post",
        body=body,
        clock=clock,
        nonce="mismatch_nonce_12345678",
        idempotency_key="idem_mismatch_001",
    )

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.post("/api/v1/test-post", content=body, headers=headers)

    assert resp.status_code == 401
    assert resp.json()["error"]["code"] == "bad_signature"


async def test_get_without_idempotency_key_succeeds(auth_setup):
    """A GET request without an Idempotency-Key header is accepted and returns 200."""
    app = auth_setup["app"]
    clock = auth_setup["clock"]
    key = auth_setup["key_bank_0"]

    headers = make_signed_headers(
        key=key,
        secret=SECRET_BANK_0,
        method="GET",
        path="/api/v1/test-get",
        body=b"",
        clock=clock,
        nonce="get_nonce_12345678",
        idempotency_key=None,
    )

    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
        resp = await client.get("/api/v1/test-get", headers=headers)

    assert resp.status_code == 200
    assert resp.json()["bank_id"] == "bank_0"


async def test_missing_raw_path_raises_bad_request_path(auth_setup):
    """If raw_path is missing from ASGI scope, authenticate_bank_request raises 400 bad_request_path."""
    app = auth_setup["app"]
    clock = auth_setup["clock"]
    key = auth_setup["key_bank_0"]

    headers = make_signed_headers(
        key=key,
        secret=SECRET_BANK_0,
        method="GET",
        path="/api/v1/test-get",
        body=b"",
        clock=clock,
        nonce="raw_path_nonce_12",
        idempotency_key=None,
    )

    class DropRawPathTransport(httpx.ASGITransport):
        async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
            async def wrapped_app(scope, receive, send):
                scope.pop("raw_path", None)
                await self.app(scope, receive, send)

            transport = httpx.ASGITransport(app=wrapped_app)
            return await transport.handle_async_request(request)

    async with httpx.AsyncClient(transport=DropRawPathTransport(app=app), base_url="http://test") as client:
        resp = await client.get("/api/v1/test-get", headers=headers)

    assert resp.status_code == 400
    assert resp.json()["error"]["code"] == "bad_request_path"

