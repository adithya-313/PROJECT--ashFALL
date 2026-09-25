"""
File: tests/contract/test_structured_log_output.py
Purpose: Contract test verifying that HTTP requests emit structured JSON logs with all required fields.
Connects: Exercises api/main.py, api/middleware/request_id.py, and core/logging.py together.

Contract requirement:
  Every log line emitted during a request must contain:
    - timestamp (ISO-8601 string)
    - level (e.g., 'info')
    - event (log message)
    - environment (e.g., 'development')
    - request_id (UUID or caller-provided trace ID)
"""

import json
import pytest
import httpx

from api.main import app
from core.config import config
from core.logging import configure_logging


@pytest.mark.anyio
async def test_structured_log_output_contains_required_fields(capsys: pytest.CaptureFixture) -> None:
    """
    Sends a GET request to /health, captures the emitted stdout lines,
    parses each line as JSON, and verifies the required log schema contract.
    """
    # Ensure stdout handler is bound to capsys's current stdout capture stream
    configure_logging(config.log_level)

    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        response = await client.get("/health", headers={"X-Request-ID": "test-trace-id-12345"})

        assert response.status_code == 200

    # Read capsys AFTER the AsyncClient context manager exits.
    # Exiting the context manager triggers the FastAPI shutdown lifespan,
    # which calls stop_log_listener(), gracefully draining the background
    # queue to sys.stdout before returning.
    captured = capsys.readouterr()
    stdout_lines = [line.strip() for line in captured.out.splitlines() if line.strip()]
    assert stdout_lines, "Expected at least one log line on stdout during health check request"

    found_matching_log = False
    for line in stdout_lines:
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            # Ignore any non-JSON output that may be emitted by third-party libraries
            continue

        if record.get("event") == "health check requested":
            found_matching_log = True

            # Assert all required contract fields are present
            assert "timestamp" in record, f"Missing 'timestamp' in log line: {record}"
            assert "level" in record, f"Missing 'level' in log line: {record}"
            assert "event" in record, f"Missing 'event' in log line: {record}"
            assert "environment" in record, f"Missing 'environment' in log line: {record}"
            assert "request_id" in record, f"Missing 'request_id' in log line: {record}"

            # Verify request_id propagated from the request header via middleware
            assert record["request_id"] == "test-trace-id-12345", (
                f"Expected request_id 'test-trace-id-12345', got {record.get('request_id')}"
            )

    assert found_matching_log, f"Could not find 'health check requested' JSON record in: {stdout_lines}"
