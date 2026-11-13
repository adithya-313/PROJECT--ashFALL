# File: tests/unit/test_watermark_advance_overlap.py
# Purpose: Unit tests for the overlap_rows override on the four apply_*_since pollers (Level 18a1-i).
"""
WHAT: Verifies that overlap_rows=None keeps the configured 200-row overlap, overlap_rows=0 reads only rows
after the cursor, a negative value raises, and the watermark is recorded in every valid case. Also checks
that the committed config leaves the drain's skip switch off.

WHY: The drain's watermark-advance step only needs the watermark moved, not the overlap tail re-applied.
The override must change the SQL start id and nothing else.
"""

from collections.abc import Awaitable, Callable
from datetime import datetime
from typing import Any

import fakeredis
import pytest

import features.account_labels as account_labels
import features.account_stream as account_stream
from core.event_clock import IST, ManualClock
from replay.runner import load_default_watermark_advance_skip_overlap

pytestmark = pytest.mark.anyio

RUN_ID = "66db9aa439b092ee"
AFTER_ID = 1000
CONFIG_OVERLAP = 200


class RecordingConnection:
    """Fake connection that returns no rows and records the SQL arguments of every fetch."""

    def __init__(self) -> None:
        self.fetch_args: list[tuple[Any, ...]] = []

    async def fetch(self, query: str, *args: Any) -> list[dict[str, Any]]:
        self.fetch_args.append(args)
        return []


POLLERS: list[tuple[str, Any, Any]] = [
    ("events", account_stream.apply_account_events_since, account_stream),
    ("withdrawals", account_stream.apply_withdrawals_since, account_stream),
    ("feedback", account_labels.apply_feedback_since, account_labels),
    ("complaints", account_labels.apply_complaints_since, account_labels),
]
POLLER_IDS = [name for name, _, _ in POLLERS]


async def run_poller(
    poller: Callable[..., Awaitable[int]],
    module: Any,
    monkeypatch: pytest.MonkeyPatch,
    **overrides: Any,
) -> tuple[RecordingConnection, list[tuple[Any, ...]]]:
    """Runs one poller against the recording connection and returns the connection and recorded watermarks."""
    recorded: list[tuple[Any, ...]] = []

    def fake_record(client: Any, bank_id: str, source: str, ts: int, run_id: str | None = None) -> None:
        recorded.append((bank_id, source, ts, run_id))

    monkeypatch.setattr(module, "record_poller_watermark", fake_record)
    conn = RecordingConnection()
    clock = ManualClock(datetime(2026, 9, 5, 12, 0, 0, tzinfo=IST))
    await poller(
        conn,
        fakeredis.FakeRedis(),
        bank_id="bank_0",
        after_id=AFTER_ID,
        clock=clock,
        run_id=RUN_ID,
        **overrides,
    )
    return conn, recorded


@pytest.mark.parametrize(("name", "poller", "module"), POLLERS, ids=POLLER_IDS)
async def test_default_overlap_uses_config_value(
    name: str, poller: Callable[..., Awaitable[int]], module: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifies that overlap_rows=None re-reads the configured overlap tail and records the watermark."""
    conn, recorded = await run_poller(poller, module, monkeypatch)
    assert conn.fetch_args[0][1] == max(0, AFTER_ID - CONFIG_OVERLAP)
    assert [(r[0], r[1]) for r in recorded] == [("bank_0", name)]


@pytest.mark.parametrize(("name", "poller", "module"), POLLERS, ids=POLLER_IDS)
async def test_zero_overlap_reads_only_after_cursor(
    name: str, poller: Callable[..., Awaitable[int]], module: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifies that overlap_rows=0 starts at the cursor and still records the watermark."""
    conn, recorded = await run_poller(poller, module, monkeypatch, overlap_rows=0)
    assert conn.fetch_args[0][1] == AFTER_ID
    assert [(r[0], r[1]) for r in recorded] == [("bank_0", name)]


@pytest.mark.parametrize(("name", "poller", "module"), POLLERS, ids=POLLER_IDS)
async def test_zero_and_default_record_same_watermark(
    name: str, poller: Callable[..., Awaitable[int]], module: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifies that the override does not change the watermark timestamp that gets recorded."""
    _, default_recorded = await run_poller(poller, module, monkeypatch)
    _, zero_recorded = await run_poller(poller, module, monkeypatch, overlap_rows=0)
    assert default_recorded == zero_recorded


@pytest.mark.parametrize(("name", "poller", "module"), POLLERS, ids=POLLER_IDS)
async def test_negative_overlap_raises(
    name: str, poller: Callable[..., Awaitable[int]], module: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Verifies that a negative overlap_rows raises ValueError before any query or watermark write."""
    recorded: list[tuple[Any, ...]] = []
    monkeypatch.setattr(module, "record_poller_watermark", lambda *a, **k: recorded.append(a))
    conn = RecordingConnection()
    clock = ManualClock(datetime(2026, 9, 5, 12, 0, 0, tzinfo=IST))
    with pytest.raises(ValueError, match="overlap_rows"):
        await poller(
            conn, fakeredis.FakeRedis(), bank_id="bank_0", after_id=AFTER_ID, clock=clock, run_id=RUN_ID,
            overlap_rows=-1,
        )
    assert conn.fetch_args == []
    assert recorded == []


def test_skip_overlap_switch_is_off_in_committed_config() -> None:
    """Verifies that the committed config leaves watermark_advance_skip_overlap off."""
    assert load_default_watermark_advance_skip_overlap() is False
