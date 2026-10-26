"""
tests/unit/test_raw_identifier_scan.py

Unit test suite for core.raw_identifier_scan.
Proves that leaked plaintext identifiers are detected across audit targets, and that
unreachable data planes are explicitly reported as NOT CHECKED, never as clean.
"""

from typing import Any

import pytest

from core.raw_identifier_scan import (
    find_raw_identifiers_in_text,
    scan_all_five_places,
    scan_logs_and_loki,
    scan_nationalsim_store,
    scan_tempo_spans,
)


def test_planting_one_raw_account_id_in_fake_log_line_is_found() -> None:
    """Proves that a planted raw account identifier in a log line is detected."""
    planted_account = "acc_planted_victim_01"
    known_ids = {planted_account}

    fake_log_lines = [
        '{"timestamp": "2026-09-01T10:00:00Z", "level": "info", "event": "service started"}',
        f'{{"timestamp": "2026-09-01T10:01:00Z", "level": "info", "event": "transfer", "target": "{planted_account}"}}',
        '{"timestamp": "2026-09-01T10:02:00Z", "level": "info", "event": "health ok"}',
    ]

    # 1. Direct text search finds the planted identifier
    found_in_line = find_raw_identifiers_in_text(fake_log_lines[1], known_ids)
    assert planted_account in found_in_line

    # 2. Store scanning also finds planted item if planted in store
    fake_store = {
        "entries": [{"identifier_hash": "h1", "raw_ref": planted_account}],
        "feedback": [],
        "resolutions": [],
    }
    store_res = scan_nationalsim_store(fake_store, known_ids)
    assert store_res.status == "CHECKED"
    assert store_res.raw_hits == 1
    assert planted_account in store_res.details


@pytest.mark.anyio
async def test_logs_scan_detects_planted_raw_account() -> None:
    """Proves scan_logs_and_loki detects planted account in local log lines."""
    planted_account = "acc_mule_secret_99"
    known_ids = {planted_account}
    log_lines = [
        f"Processing batch for {planted_account} at minute 50",
        "Standard system message with no identifiers",
    ]

    result = await scan_logs_and_loki(
        run_id="run-test-01",
        known_raw_identifiers=known_ids,
        local_log_lines=log_lines,
        loki_url="http://127.0.0.1:59999",  # Unreachable Loki port
    )

    assert result.status == "CHECKED"
    assert result.items_checked == 2
    assert result.raw_hits == 1
    assert planted_account in result.details
    assert not result.is_clean()


@pytest.mark.anyio
async def test_unreachable_place_is_not_checked_not_clean() -> None:
    """Proves an unreachable service (e.g. Tempo or Loki offline) is reported as NOT CHECKED, not clean."""
    # Point Tempo to an unreachable localhost port
    tempo_res = await scan_tempo_spans(
        run_id="run-test-02",
        known_raw_identifiers={"acc_test_01"},
        tempo_url="http://127.0.0.1:59999",
    )

    # Must be marked NOT CHECKED, not clean, never 0 hits / clean
    assert tempo_res.status == "NOT CHECKED"
    assert tempo_res.reason is not None
    assert "unreachable" in tempo_res.reason.lower()
    assert not tempo_res.is_clean()

    # When aggregating across all places, a NOT CHECKED place prevents all_clean
    fake_clean_store: dict[str, Any] = {"entries": [], "feedback": [], "resolutions": []}
    report = await scan_all_five_places(
        run_id="run-test-02",
        known_raw_identifiers=set(),
        store_export=fake_clean_store,
        local_log_lines=["Clean log line"],
        tempo_url="http://127.0.0.1:59999",  # Unreachable Tempo
        loki_url="http://127.0.0.1:59999",
    )

    assert report.places["tempo_spans"].status == "NOT CHECKED"
    # Even though store and logs have 0 hits, overall status MUST NOT be clean
    assert report.all_clean is False
