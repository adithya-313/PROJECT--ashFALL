# File: replay/profiling.py
# Purpose: Phase timing profile for replay runs (Level 18a1 measurement).
"""
WHAT: Collects wall-clock seconds and event counts per named phase during a replay run.

WHY: Level 18a1 measures where replay time goes before anything is optimised. The profile is plain
data (dictionaries) so a dataset build can write it to JSON next to the replay result.
"""

import time
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any


class ReplayProfile:
    """
    WHAT: A named phase timer plus event counters for one replay run.

    WHY: Measurement needs both durations (where the seconds go) and counts (how often a phase ran),
    and every phase must be able to record while the default (unprofiled) path stays untouched.

    Side effects: holds only in-memory dictionaries; adding a phase accumulates into the same key.
    """

    def __init__(self) -> None:
        self.seconds: dict[str, float] = {}
        self.counts: dict[str, int] = {}

    def add_seconds(self, name: str, seconds: float) -> None:
        """
        WHAT: Adds elapsed wall-clock seconds to a named phase.

        WHY: A phase such as a drain runs many times per replay; its total is the sum of its calls.
        """
        self.seconds[name] = self.seconds.get(name, 0.0) + seconds

    def add_count(self, name: str, amount: int = 1) -> None:
        """
        WHAT: Adds to a named event counter.

        WHY: Counts show how often a phase ran (steps, drains, sleeps), which explains the timing.
        """
        self.counts[name] = self.counts.get(name, 0) + amount

    @contextmanager
    def timed(self, name: str) -> Iterator[None]:
        """
        WHAT: Times the body of a with-block with time.perf_counter and adds it to the named phase.

        WHY: The time is recorded in a finally block, so a raising body is still measured and the
        exception is re-raised unchanged (a profile must never hide a failure).
        """
        started = time.perf_counter()
        try:
            yield
        finally:
            self.add_seconds(name, time.perf_counter() - started)

    def record_drain_iterations(self, n: int) -> None:
        """
        WHAT: Records the loop count of one completed drain call.

        WHY: drain() polls until the pipeline converges; knowing the distribution of loop counts shows
        whether time goes to convergence waits or to the work inside each loop.
        """
        if n == 1:
            bucket = "drain_iter_1"
        elif n == 2:
            bucket = "drain_iter_2"
        elif n == 3:
            bucket = "drain_iter_3"
        elif 4 <= n <= 10:
            bucket = "drain_iter_4_to_10"
        else:
            bucket = "drain_iter_over_10"
        self.add_count(bucket)
        self.counts["drain_iterations_total"] = self.counts.get("drain_iterations_total", 0) + n
        self.counts["drain_iterations_max"] = max(self.counts.get("drain_iterations_max", 0), n)

    def to_dict(self) -> dict[str, Any]:
        """
        WHAT: Returns the profile as JSON-ready data: {"seconds": {...}, "counts": {...}}.

        WHY: Times are rounded to milliseconds and keys are sorted so two runs are easy to compare.
        """
        seconds = {name: round(self.seconds[name], 3) for name in sorted(self.seconds)}
        counts = {name: self.counts[name] for name in sorted(self.counts)}
        return {"seconds": seconds, "counts": counts}
