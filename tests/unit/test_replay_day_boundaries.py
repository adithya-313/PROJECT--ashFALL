# File: tests/unit/test_replay_day_boundaries.py
# Purpose: Unit tests for pure function day_boundaries_between in replay runner.
# Issue: Level 16c round 2a architect decision D2.
"""
WHAT: Unit tests verifying day_boundaries_between correctly computes IST midnight boundaries.

WHY: Architect decision D2: identifies completed simulation world days in IST to trigger
daily account snapshot computation during replay.
"""

from datetime import datetime, timezone

import pytest

from replay.runner import IST, day_boundaries_between


def test_day_boundaries_same_day() -> None:
    """Same IST day returns empty list."""
    prev = datetime(2026, 9, 1, 10, 0, 0, tzinfo=IST)
    nxt = datetime(2026, 9, 1, 18, 0, 0, tzinfo=IST)
    assert day_boundaries_between(prev, nxt) == []


def test_day_boundaries_crosses_one_midnight() -> None:
    """Crossing one IST midnight returns exactly that midnight boundary."""
    prev = datetime(2026, 9, 1, 22, 0, 0, tzinfo=IST)
    nxt = datetime(2026, 9, 2, 2, 0, 0, tzinfo=IST)
    expected = [datetime(2026, 9, 2, 0, 0, 0, tzinfo=IST)]
    assert day_boundaries_between(prev, nxt) == expected


def test_day_boundaries_crosses_two_midnights() -> None:
    """Crossing two IST midnights returns both midnight boundaries in chronological order."""
    prev = datetime(2026, 9, 1, 22, 0, 0, tzinfo=IST)
    nxt = datetime(2026, 9, 3, 2, 0, 0, tzinfo=IST)
    expected = [
        datetime(2026, 9, 2, 0, 0, 0, tzinfo=IST),
        datetime(2026, 9, 3, 0, 0, 0, tzinfo=IST),
    ]
    assert day_boundaries_between(prev, nxt) == expected


def test_day_boundaries_utc_midnight_not_ist_midnight() -> None:
    """UTC midnight crossing that falls on the same IST day returns empty list."""
    # 23:30 UTC = 05:00 IST on 2026-09-02; 01:00 UTC = 06:30 IST on 2026-09-02
    prev = datetime(2026, 9, 1, 23, 30, 0, tzinfo=timezone.utc)
    nxt = datetime(2026, 9, 2, 1, 0, 0, tzinfo=timezone.utc)
    assert day_boundaries_between(prev, nxt) == []


def test_day_boundaries_backwards_time_raises_value_error() -> None:
    """next_world_time earlier than prev_world_time raises ValueError."""
    prev = datetime(2026, 9, 2, 10, 0, 0, tzinfo=IST)
    nxt = datetime(2026, 9, 1, 10, 0, 0, tzinfo=IST)
    with pytest.raises(ValueError, match="is earlier than"):
        day_boundaries_between(prev, nxt)

