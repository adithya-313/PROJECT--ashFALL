# File: tests/unit/test_replay_profiling.py
# Purpose: Unit tests for the replay phase timing profile (Level 18a1 measurement).
"""
WHAT: Unit tests for ReplayProfile: accumulation, timing, drain iteration buckets and JSON output.

WHY: The profile is the only measurement of where replay time goes; a silently broken accumulator
would make every later optimisation decision wrong.
"""

import pytest

from replay.profiling import ReplayProfile


def test_seconds_and_counts_add_up_across_repeated_calls() -> None:
    """Adding the same phase repeatedly accumulates instead of overwriting."""
    profile = ReplayProfile()
    profile.add_seconds("drain.consume", 0.5)
    profile.add_seconds("drain.consume", 1.25)
    profile.add_seconds("drain.sleep", 0.25)
    profile.add_count("drain.sleeps")
    profile.add_count("drain.sleeps")
    profile.add_count("drain.records_consumed", 7)

    assert profile.seconds == {"drain.consume": 1.75, "drain.sleep": 0.25}
    assert profile.counts == {"drain.sleeps": 2, "drain.records_consumed": 7}


def test_timed_records_time_and_reraises() -> None:
    """A raising body is still timed and its exception propagates unchanged."""
    profile = ReplayProfile()

    with pytest.raises(RuntimeError, match="phase failed"):
        with profile.timed("run.pace"):
            raise RuntimeError("phase failed")

    assert profile.seconds["run.pace"] > 0.0


def test_timed_records_time_on_success() -> None:
    """A normal body records a positive duration for its phase."""
    profile = ReplayProfile()

    with profile.timed("run.feed_send"):
        pass

    assert profile.seconds["run.feed_send"] >= 0.0


@pytest.mark.parametrize(
    ("iterations", "bucket"),
    [
        (1, "drain_iter_1"),
        (2, "drain_iter_2"),
        (3, "drain_iter_3"),
        (4, "drain_iter_4_to_10"),
        (10, "drain_iter_4_to_10"),
        (11, "drain_iter_over_10"),
    ],
)
def test_record_drain_iterations_buckets(iterations: int, bucket: str) -> None:
    """Each loop count lands in exactly one bucket and updates the totals."""
    profile = ReplayProfile()
    profile.record_drain_iterations(iterations)

    bucket_counts = {
        name: count
        for name, count in profile.counts.items()
        if name.startswith("drain_iter_") and not name.startswith("drain_iterations_")
    }
    assert bucket_counts == {bucket: 1}
    assert profile.counts["drain_iterations_total"] == iterations
    assert profile.counts["drain_iterations_max"] == iterations


def test_record_drain_iterations_keeps_max_across_calls() -> None:
    """The maximum survives smaller drains; the total sums them."""
    profile = ReplayProfile()
    profile.record_drain_iterations(12)
    profile.record_drain_iterations(1)

    assert profile.counts["drain_iterations_max"] == 12
    assert profile.counts["drain_iterations_total"] == 13
    assert profile.counts["drain_iter_over_10"] == 1
    assert profile.counts["drain_iter_1"] == 1


def test_to_dict_rounds_and_sorts() -> None:
    """Seconds are rounded to milliseconds and both key sets are sorted."""
    profile = ReplayProfile()
    profile.add_seconds("run.total", 12.3456)
    profile.add_seconds("drain.consume", 0.0004)
    profile.add_count("steps.decision", 3)
    profile.add_count("drain.sleeps", 2)

    result = profile.to_dict()

    assert result == {
        "seconds": {"drain.consume": 0.0, "run.total": 12.346},
        "counts": {"drain.sleeps": 2, "steps.decision": 3},
    }
    assert list(result["seconds"]) == ["drain.consume", "run.total"]
    assert list(result["counts"]) == ["drain.sleeps", "steps.decision"]
