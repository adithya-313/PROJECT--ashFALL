# File: tests/contract/test_bank_api_fuzz.py
# Purpose: Fuzzes all /v1 commercial bank API endpoints using schemathesis-generated cases.
#          Proves that unsigned requests are strictly rejected with 401 or 413, and signed requests
#          never produce 500 internal server errors (crashing the application).
import secrets
import uuid
from datetime import datetime
from typing import Any

import asyncpg
import httpx
import orjson
import pytest
import schemathesis.core
import schemathesis.openapi
from hypothesis import given, settings
from pydantic import SecretStr

from api.main import app
from api.v1.bank_keys import issue_api_key
from api.v1.signing import load_bank_api_policy, sign_outgoing
from core.config import config
from core.event_clock import IST, ManualClock
from tests.integration.conftest import get_test_database_url

pytest_plugins = ["tests.integration.conftest"]
pytestmark = pytest.mark.anyio

FUZZ_BANK_ID = "bank_0"
FUZZ_SECRET = "test_fuzz_secret_32_characters_minimum!"
START_TIME = datetime(2026, 10, 28, 12, 0, 0, tzinfo=IST)


def load_v1_operations() -> list[Any]:
    """Extracts all API operations from the OpenAPI schema matching the /v1 path prefix."""
    schema = schemathesis.openapi.from_dict(app.openapi())
    raw_ops: Any = schema.get_all_operations()
    ops: list[Any] = []
    for item in raw_ops:
        op: Any = item.ok() if hasattr(item, "ok") else item
        if hasattr(op, "path") and str(op.path).startswith("/v1"):
            ops.append(op)
    return ops


def generate_fuzz_cases(max_examples: int) -> list[Any]:
    """Generates a reproducible list of test cases for every /v1 operation using Hypothesis."""
    ops = load_v1_operations()
    all_cases: list[Any] = []
    for op in ops:
        strategy = op.as_strategy()

        @settings(max_examples=max_examples, derandomize=True, deadline=None)
        @given(case=strategy)
        def collect(case: Any) -> None:
            all_cases.append(case)

        collect()
    return all_cases


async def run_unsigned_fuzz_pass(client: httpx.AsyncClient, cases: list[Any]) -> dict[int, int]:
    """
    Executes an unsigned pass across all generated cases.

    Enforces that without authentication headers, requests always return 401
    (unauthenticated) or 413 (body too large), never 2xx or 5xx.
    """
    histogram: dict[int, int] = {}
    for case in cases:
        if isinstance(case.body, schemathesis.core.NotSet) or case.body is None:
            body_bytes = b""
        else:
            body_bytes = orjson.dumps(case.body)

        dummy_req = client.build_request(case.method, case.formatted_path, content=body_bytes)
        encoded_path = dummy_req.url.raw_path.decode("ascii")

        headers: dict[str, str] = {}
        if body_bytes:
            headers["Content-Type"] = "application/json"

        resp = await client.request(
            method=case.method,
            url=encoded_path,
            content=body_bytes,
            headers=headers,
        )

        status = resp.status_code
        histogram[status] = histogram.get(status, 0) + 1

        if status not in (401, 413):
            error_code: str | None = None
            try:
                data = resp.json()
                if isinstance(data, dict):
                    err = data.get("error")
                    if isinstance(err, dict):
                        error_code = err.get("code")
            except Exception:
                pass
            raise AssertionError(
                f"Unsigned fuzz request returned unexpected status: method={case.method} "
                f"path={encoded_path} status={status} error_code={error_code}"
            )

    return histogram


async def run_signed_fuzz_pass(
    client: httpx.AsyncClient,
    clock: ManualClock,
    bank_id: str,
    api_key: str,
    cases: list[Any],
) -> dict[int, int]:
    """
    Executes a signed pass across all generated cases.

    Enforces that every signed request with valid auth headers and idempotency key
    returns a client error (4xx) or success (2xx), but never an unhandled server crash (5xx).
    """
    histogram: dict[int, int] = {}
    for case in cases:
        if isinstance(case.body, schemathesis.core.NotSet) or case.body is None:
            body_bytes = b""
        else:
            body_bytes = orjson.dumps(case.body)

        dummy_req = client.build_request(case.method, case.formatted_path, content=body_bytes)
        encoded_path = dummy_req.url.raw_path.decode("ascii")

        nonce = f"fuzz_nonce_{secrets.token_hex(16)}"
        headers = sign_outgoing(
            bank_id=bank_id,
            method=case.method,
            path=encoded_path,
            body_bytes=body_bytes,
            clock=clock,
            nonce=nonce,
        )
        headers["X-Ashfall-Key"] = api_key
        headers["Content-Type"] = "application/json"
        if case.method.upper() == "POST":
            headers["Idempotency-Key"] = f"idem_{uuid.uuid4().hex}"

        resp = await client.request(
            method=case.method,
            url=encoded_path,
            content=body_bytes,
            headers=headers,
        )

        status = resp.status_code
        histogram[status] = histogram.get(status, 0) + 1

        if status >= 500:
            error_code: str | None = None
            try:
                data = resp.json()
                if isinstance(data, dict):
                    err = data.get("error")
                    if isinstance(err, dict):
                        error_code = err.get("code")
            except Exception:
                pass
            raise AssertionError(
                f"Signed fuzz request failed with server error: method={case.method} "
                f"path={encoded_path} status={status} error_code={error_code}"
            )

    return histogram


@pytest.fixture
async def fuzz_env(isolated_schema, monkeypatch: pytest.MonkeyPatch):
    """Sets up an isolated database schema, test clock, API key, and restores app state."""
    connection, schema_name = isolated_schema
    pool = await asyncpg.create_pool(
        dsn=get_test_database_url(),
        min_size=1,
        max_size=3,
        server_settings={"search_path": schema_name},
    )
    assert pool is not None

    monkeypatch.setattr(config, f"bank_signing_secret_{FUZZ_BANK_ID}", SecretStr(FUZZ_SECRET))
    _, api_key = await issue_api_key(connection, FUZZ_BANK_ID)

    clock = ManualClock(START_TIME)
    saved_pool = getattr(app.state, "db_pool", None)
    saved_clock = getattr(app.state, "transport_clock", None)
    app.state.db_pool = pool
    app.state.transport_clock = clock

    try:
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as client:
            yield {
                "client": client,
                "clock": clock,
                "api_key": api_key,
                "bank_id": FUZZ_BANK_ID,
            }
    finally:
        await pool.close()
        app.state.db_pool = saved_pool
        app.state.transport_clock = saved_clock


async def test_bank_api_fuzz_unsigned(fuzz_env: dict[str, Any]) -> None:
    """Unsigned requests generated by schemathesis are all rejected with 401 or 413, never 2xx/5xx."""
    policy = load_bank_api_policy()
    cases = generate_fuzz_cases(policy.fuzz_max_examples_per_endpoint)
    histogram = await run_unsigned_fuzz_pass(fuzz_env["client"], cases)
    assert sum(histogram.values()) == len(cases)
    for code in histogram:
        assert code in (401, 413)


async def test_bank_api_fuzz_signed(fuzz_env: dict[str, Any]) -> None:
    """Signed requests generated by schemathesis are handled cleanly with status < 500."""
    policy = load_bank_api_policy()
    cases = generate_fuzz_cases(policy.fuzz_max_examples_per_endpoint)
    histogram = await run_signed_fuzz_pass(
        fuzz_env["client"],
        fuzz_env["clock"],
        fuzz_env["bank_id"],
        fuzz_env["api_key"],
        cases,
    )
    assert sum(histogram.values()) == len(cases)
    for code in histogram:
        assert code < 500
