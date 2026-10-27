"""
tests/unit/nationalsim/test_bank_client.py

Unit tests for the resilient bank-side nationalsim HTTP client.
Verifies explicit timeout handling, single-retry behavior, outage window tracking,
and guarantees that offline national services never report false 'no match' screening verdicts.
"""

import threading
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Any

import pytest

from core.event_clock import IST, ManualClock
from nationalsim.bank_client import (
    NationalsimBankClient,
    OutageEvent,
    ScreenResult,
    load_bank_client_config,
)


class DummyNationalsimHandler(BaseHTTPRequestHandler):
    """Minimal HTTP handler simulating responses from nationalsim."""

    def do_POST(self) -> None:
        if self.path == "/screen":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"match": true, "risk_score": 0.85, "distinct_accounts_linked": 2}')
        else:
            self.send_response(404)
            self.end_headers()

    def do_GET(self) -> None:
        if self.path.startswith("/i4c/feed"):
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"entries": [{"entry_id": "test_e1", "category": "fraud"}]}')
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format: str, *args: Any) -> None:
        """Silences HTTP server output during test execution."""
        pass


@pytest.fixture
def dummy_server():
    """Provides a local test HTTP server for live request handling."""
    server = HTTPServer(("127.0.0.1", 0), DummyNationalsimHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    port = server.server_address[1]
    yield f"http://127.0.0.1:{port}"
    server.shutdown()
    server.server_close()


@pytest.mark.anyio
async def test_bank_client_records_outage_when_service_unreachable() -> None:
    """Proves that network failures open an outage event and never report match=False.

    When the national registry goes down, payment operations must treat screening as
    temporarily unavailable rather than assuming the payee is clean. Returning match=False
    during an outage would allow known mule accounts to transact undetected.
    """
    fixed_time = datetime(2026, 9, 1, 10, 0, tzinfo=IST)
    clock = ManualClock(fixed_time)
    # Port 59999 is intentionally unreachable
    client = NationalsimBankClient(
        base_url="http://127.0.0.1:59999",
        bank_id="bank_0",
        run_id="run_test_001",
        clock=clock,
        timeout_seconds=0.05,
        retry_backoff_seconds=0.01,
    )

    test_hash = "a" * 64
    result: ScreenResult = await client.screen(identifier_hash=test_hash)

    # 1. Result must explicitly reflect service unavailability
    assert result.status == "unavailable"
    assert result.is_unavailable is True
    assert result.match is None
    # Crucial guarantee: unavailable must NEVER be interpreted as 'no match'
    assert result.match is not False

    # 2. OutageEvent must be recorded with started_at and failure count
    assert len(client.outages) == 1
    outage: OutageEvent = client.outages[0]
    assert outage.bank_id == "bank_0"
    assert outage.run_id == "run_test_001"
    assert outage.started_at == fixed_time
    assert outage.ended_at is None
    assert outage.calls_failed == 1
    assert client.active_outage is outage

    # 3. Subsequent failure increments calls_failed on the same active outage
    clock.advance(timedelta(seconds=10))
    result_second = await client.screen(identifier_hash=test_hash)
    assert result_second.status == "unavailable"
    assert result_second.match is None
    assert outage.calls_failed == 2
    assert outage.ended_at is None

    # 4. Feed request failure also increments calls_failed
    clock.advance(timedelta(seconds=10))
    feed_result = await client.get_i4c_feed()
    assert feed_result.status == "unavailable"
    assert feed_result.entries == []
    assert feed_result.entries_count == 0
    assert feed_result.is_unavailable is True
    assert outage.calls_failed == 3
    assert outage.ended_at is None


@pytest.mark.anyio
async def test_bank_client_recovery_closes_active_outage(dummy_server: str) -> None:
    """Proves that a successful call closes an ongoing outage with an ended_at timestamp.

    When national registry connectivity is restored, the client must automatically
    mark the outage window as closed so audit logs reflect exact downtime duration.
    """
    fixed_time = datetime(2026, 9, 1, 10, 0, tzinfo=IST)
    clock = ManualClock(fixed_time)
    # Start pointing to offline port
    client = NationalsimBankClient(
        base_url="http://127.0.0.1:59999",
        bank_id="bank_1",
        run_id="run_recovery_001",
        clock=clock,
        timeout_seconds=0.05,
        retry_backoff_seconds=0.01,
    )

    test_hash = "b" * 64
    # Call fails and opens outage
    first_res = await client.screen(identifier_hash=test_hash)
    assert first_res.status == "unavailable"
    assert len(client.outages) == 1
    assert client.active_outage is not None
    assert client.active_outage.ended_at is None

    # Restore service by pointing base_url to running dummy server
    client.base_url = dummy_server
    recovery_time = datetime(2026, 9, 1, 10, 15, tzinfo=IST)
    clock.set_time(recovery_time)

    # Next call succeeds and closes outage
    recovered_res = await client.screen(identifier_hash=test_hash)
    assert recovered_res.status == "match"
    assert recovered_res.match is True
    assert recovered_res.risk_score == 0.85
    assert recovered_res.distinct_accounts_linked == 2
    assert recovered_res.is_unavailable is False

    # Outage must now be closed
    assert client.active_outage is None
    assert client.outages[0].ended_at == recovery_time
    assert client.outages[0].calls_failed == 1

    # Feed query also succeeds against recovered service
    feed_res = await client.get_i4c_feed()
    assert feed_res.status == "ok"
    assert feed_res.entries_count == 1
    assert feed_res.entries[0]["entry_id"] == "test_e1"
    assert feed_res.is_unavailable is False


def test_bank_client_sync_wrappers() -> None:
    """Proves synchronous wrappers screen_sync and get_i4c_feed_sync function correctly."""
    fixed_time = datetime(2026, 9, 1, 11, 0, tzinfo=IST)
    clock = ManualClock(fixed_time)
    client = NationalsimBankClient(
        base_url="http://127.0.0.1:59999",
        bank_id="bank_2",
        run_id="run_sync_001",
        clock=clock,
        timeout_seconds=0.05,
        retry_backoff_seconds=0.01,
    )

    screen_res = client.screen_sync(identifier_hash="c" * 64)
    assert screen_res.status == "unavailable"
    assert screen_res.match is None

    feed_res = client.get_i4c_feed_sync()
    assert feed_res.status == "unavailable"
    assert feed_res.entries == []


def test_load_bank_client_config_validates_inputs(tmp_path: Path) -> None:
    """Proves configuration parsing strictly validates timeout and backoff settings.

    Ensures no silent fallbacks are used if configuration keys are malformed or missing.
    """
    # 1. Valid real config
    timeout, backoff = load_bank_client_config("config/nationalsim.yaml")
    assert timeout == 2.0
    assert backoff == 0.2

    # 2. Missing config file raises ValueError
    with pytest.raises(ValueError, match="Missing configuration file"):
        load_bank_client_config("config/nonexistent_config.yaml")

    # 3. Missing nationalsim section
    bad_cfg1 = tmp_path / "bad1.yaml"
    bad_cfg1.write_text("other_key: 123\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Missing 'nationalsim' key"):
        load_bank_client_config(bad_cfg1)

    # 4. Missing timeout
    bad_cfg2 = tmp_path / "bad2.yaml"
    bad_cfg2.write_text("nationalsim:\n  client_retry_backoff_seconds: 0.2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Missing key 'client_timeout_seconds'"):
        load_bank_client_config(bad_cfg2)

    # 5. Invalid timeout value
    bad_cfg3 = tmp_path / "bad3.yaml"
    bad_cfg3.write_text(
        "nationalsim:\n  client_timeout_seconds: -1.0\n  client_retry_backoff_seconds: 0.2\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="Invalid 'client_timeout_seconds'"):
        load_bank_client_config(bad_cfg3)
