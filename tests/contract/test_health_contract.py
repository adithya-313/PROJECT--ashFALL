"""
File: tests/contract/test_health_contract.py
Purpose: Contract test for the /health endpoint.
Connects: Tests the FastAPI application defined in api/main.py using httpx's
          AsyncClient in test mode (no real server started — it calls the ASGI
          app directly through FastAPI's test interface).

What a "contract test" means here:
A contract test locks down the observable interface of an endpoint: the exact
JSON keys present in the response, their types, and the HTTP status code.
It does NOT test values — it tests shape.

This test passes whether the response is serialised by stdlib json (Phase 1a)
or by orjson (Phase 1b), because both produce the same JSON keys and value types.
That is exactly the requirement: the optimisation must be invisible to callers.
"""

import pytest
import httpx

from api.main import app


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
    """
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        response = await client.get("/health")

    assert response.status_code == 200, (
        f"Expected HTTP 200, got {response.status_code}. "
        "This would break any client depending on this endpoint."
    )

    body = response.json()

    assert isinstance(body, dict), (
        f"Expected a JSON object (dict), got {type(body).__name__}. "
        "The response shape contract requires an object at the top level."
    )

    expected_keys = {"status", "environment"}
    actual_keys = set(body.keys())
    assert actual_keys == expected_keys, (
        f"Response keys mismatch. Expected {expected_keys}, got {actual_keys}. "
        "Any key change is a breaking change to the API contract."
    )

    assert isinstance(body["status"], str), (
        f"'status' must be a string, got {type(body['status']).__name__}."
    )
    assert isinstance(body["environment"], str), (
        f"'environment' must be a string, got {type(body['environment']).__name__}."
    )
