"""
File: tests/test_health_contract.py
Purpose: Contract test for the /health endpoint.
Connects: Tests the FastAPI application defined in src/main.py using httpx's
          AsyncClient in test mode (no real server started — it calls the ASGI
          app directly through FastAPI's test interface).

What a "contract test" means here:
A contract test locks down the observable interface of an endpoint: the exact
JSON keys present in the response, their types, and the HTTP status code.
It does NOT test values — it tests shape.

This test passes whether the response is serialised by stdlib json (Phase 1a)
or by orjson (Phase 1b), because both produce the same JSON keys and value types.
That is exactly the requirement: the optimisation must be invisible to callers.

Why httpx instead of requests:
FastAPI's recommended test client (fastapi.testclient.TestClient) uses httpx
under the hood. Using httpx directly with ASGITransport gives us async test
support and zero network overhead — no port binding, no socket, no server process.

Requires pytest and httpx — these must be in requirements.txt or installed
separately for test runs. They are added as dev dependencies.
"""

import pytest
import httpx

from src.main import app


@pytest.mark.anyio
async def test_health_response_shape() -> None:
    """
    Verifies that GET /health returns HTTP 200 with the correct JSON response shape.

    This is a contract test, not a value test. It confirms:
      - HTTP status is 200
      - The response body can be parsed as JSON
      - The response contains exactly the keys: 'status' and 'environment'
      - 'status' is a string
      - 'environment' is a string

    It does NOT check what value 'status' or 'environment' hold, because those
    values are environment-dependent and would make the test brittle.

    Args:
        None (pytest fixtures are injected automatically)

    Returns:
        None (the test asserts internally; pytest reports pass/fail)
    """
    # ASGITransport makes httpx call the ASGI application directly,
    # bypassing the TCP stack entirely — no server needs to be running.
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        response = await client.get("/health")

    # The HTTP status code must be 200 OK — any other code means the endpoint
    # is broken or the route is missing.
    assert response.status_code == 200, (
        f"Expected HTTP 200, got {response.status_code}. "
        "This would break any client depending on this endpoint."
    )

    # Parse the response body as JSON. If this raises, the response is not valid JSON —
    # which would mean the serialiser (orjson or stdlib json) is broken.
    body = response.json()

    # The response must be a dictionary (JSON object), not a list, string, or other type.
    assert isinstance(body, dict), (
        f"Expected a JSON object (dict), got {type(body).__name__}. "
        "The response shape contract requires an object at the top level."
    )

    # Assert the exact set of keys. Extra or missing keys would break callers
    # that depend on this response shape.
    expected_keys = {"status", "environment"}
    actual_keys = set(body.keys())
    assert actual_keys == expected_keys, (
        f"Response keys mismatch. Expected {expected_keys}, got {actual_keys}. "
        "Any key change is a breaking change to the API contract."
    )

    # Assert the types of each field. Type changes are also breaking for typed clients.
    assert isinstance(body["status"], str), (
        f"'status' must be a string, got {type(body['status']).__name__}."
    )
    assert isinstance(body["environment"], str), (
        f"'environment' must be a string, got {type(body['environment']).__name__}."
    )
