# File: tests/unit/test_replay_speed.py
# Purpose: Unit tests for replay speed validation, pacing calculation, and per-step pacing wiring.
# Issue: Level 16 task 2 (replay-speed control, architect decision D2).
"""
WHAT: Unit tests for replay speed validation, pace_seconds, and the per-step pacing helper pace_step.

WHY: Verifies:
1. pace_seconds returns 0.0 for speed "max".
2. pace_seconds scales world gaps: speed 60 with 120s world gap yields 2.0s real sleep.
3. pace_seconds returns 0.0 for identical timestamps (zero gap) and for the initial step (None previous).
4. pace_seconds raises ValueError on backwards world time gaps.
5. validate_replay_speed accepts "max" and positive numbers; rejects 0, negative numbers, and invalid strings.
6. load_default_replay_speed reads policy setting "max" from config/replay.yaml.
7. pace_step correctly invokes fake sleep with calculated pace across successive steps.
"""

from datetime import datetime, timezone
from typing import Any

import pytest

from replay.runner import (
    load_default_replay_speed,
    pace_seconds,
    pace_step,
    validate_replay_speed,
)


def test_pace_seconds_max_returns_zero() -> None:
    """Verifies that speed 'max' never sleeps regardless of world time gap."""
    t0 = datetime(2026, 9, 1, 10, 0, 0, tzinfo=timezone.utc)
    t1 = datetime(2026, 9, 1, 10, 5, 0, tzinfo=timezone.utc)
    assert pace_seconds(t0, t1, "max") == 0.0
    assert pace_seconds(None, t0, "max") == 0.0


def test_pace_seconds_numeric_speed_scales_gap() -> None:
    """Verifies that numeric speed divides world time gap (speed 60 with 120s gap -> 2.0s)."""
    t0 = datetime(2026, 9, 1, 10, 0, 0, tzinfo=timezone.utc)
    t1 = datetime(2026, 9, 1, 10, 2, 0, tzinfo=timezone.utc)  # 120s gap
    assert pace_seconds(t0, t1, 60) == 2.0
    assert pace_seconds(t0, t1, 60.0) == 2.0

    t2 = datetime(2026, 9, 1, 10, 2, 30, tzinfo=timezone.utc)  # 30s gap
    assert pace_seconds(t1, t2, 10) == 3.0


def test_pace_seconds_zero_and_initial_gap_returns_zero() -> None:
    """Verifies that zero gap and initial step (previous is None) return 0.0."""
    t0 = datetime(2026, 9, 1, 10, 0, 0, tzinfo=timezone.utc)
    assert pace_seconds(None, t0, 60) == 0.0
    assert pace_seconds(t0, t0, 60) == 0.0


def test_pace_seconds_backwards_gap_raises_value_error() -> None:
    """Verifies that backwards world time raises ValueError (steps are ordered; negative gap is a bug)."""
    t0 = datetime(2026, 9, 1, 10, 5, 0, tzinfo=timezone.utc)
    t1 = datetime(2026, 9, 1, 10, 0, 0, tzinfo=timezone.utc)  # 5 minutes earlier
    with pytest.raises(ValueError, match="Backwards world time gap"):
        pace_seconds(t0, t1, 60)


def test_validate_replay_speed_valid_inputs() -> None:
    """Verifies accepted values for replay speed: 'max' and numbers > 0."""
    validate_replay_speed("max")
    validate_replay_speed(1)
    validate_replay_speed(60)
    validate_replay_speed(0.5)
    validate_replay_speed(1000.0)


@pytest.mark.parametrize("invalid_val", [0, 0.0, -1, -0.5, "fast", "slow", "", None, True, False])
def test_validate_replay_speed_rejects_invalid_values(invalid_val: Any) -> None:
    """Verifies that 0, negative numbers, boolean values, and invalid strings are rejected."""
    with pytest.raises(ValueError):
        validate_replay_speed(invalid_val)


def test_load_default_replay_speed_matches_config() -> None:
    """Verifies that default replay speed loaded from config/replay.yaml is 'max'."""
    val = load_default_replay_speed()
    assert val == "max"


@pytest.mark.anyio
async def test_pace_step_wiring_with_recording_fake_sleep() -> None:
    """
    Verifies the per-step pacing helper (pace_step) invoked by run_replay before each step.
    Tested with a recording fake sleep to isolate from real time and database infrastructure.
    """
    sleep_calls: list[float] = []

    async def fake_sleep(duration: float) -> None:
        sleep_calls.append(duration)

    t0 = datetime(2026, 9, 1, 10, 0, 0, tzinfo=timezone.utc)
    t1 = datetime(2026, 9, 1, 10, 2, 0, tzinfo=timezone.utc)  # +120s
    t2 = datetime(2026, 9, 1, 10, 2, 30, tzinfo=timezone.utc)  # +30s
    t3 = datetime(2026, 9, 1, 10, 2, 30, tzinfo=timezone.utc)  # +0s

    steps_times = [t0, t1, t2, t3]
    prev: datetime | None = None

    # Run with speed=60: gaps are 0s, 120s, 30s, 0s -> paces 0.0, 2.0, 0.5, 0.0
    for cur in steps_times:
        await pace_step(prev, cur, 60, sleep=fake_sleep)
        prev = cur

    assert sleep_calls == [0.0, 2.0, 0.5, 0.0]

    # Run with speed='max': all paces are 0.0
    sleep_calls.clear()
    prev = None
    for cur in steps_times:
        await pace_step(prev, cur, "max", sleep=fake_sleep)
        prev = cur

    assert sleep_calls == [0.0, 0.0, 0.0, 0.0]
