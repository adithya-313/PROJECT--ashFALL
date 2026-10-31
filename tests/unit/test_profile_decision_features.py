# File: tests/unit/test_profile_decision_features.py
# Purpose: Unit tests for the Redis round-trip counting proxy used by the decision latency profile.
# Issue: Level 15 decision latency profile.
"""
WHAT: Checks that CountingRedis counts a pipeline as ONE round trip holding all its commands, and plain
commands one round trip each.

WHY: The profile's "round trips per item" column is only meaningful if the proxy counts like the network does.
"""

import fakeredis

from benchmarks.profile_decision_features import CountingRedis


def test_pipeline_of_three_commands_is_one_round_trip() -> None:
    """Three buffered commands executed together are 1 round trip holding 3 commands."""
    counter = CountingRedis(fakeredis.FakeRedis())
    pipe = counter.pipeline(transaction=False)
    pipe.set("a", 1)
    pipe.set("b", 2)
    pipe.get("a")
    results = pipe.execute()

    assert results[-1] == b"1"
    assert counter.round_trips == 1
    assert counter.pipeline_executes == 1
    assert counter.pipeline_commands == 3


def test_two_plain_gets_are_two_round_trips() -> None:
    """Each plain command is its own round trip, and the real values still come back."""
    client = fakeredis.FakeRedis()
    client.set("k", "v")
    counter = CountingRedis(client)

    assert counter.get("k") == b"v"
    assert counter.get("missing") is None
    assert counter.round_trips == 2
    assert counter.pipeline_executes == 0
    assert counter.pipeline_commands == 0


def test_reset_zeroes_the_counters() -> None:
    """reset() starts a fresh count for the next measured item."""
    counter = CountingRedis(fakeredis.FakeRedis())
    counter.get("x")
    counter.reset()
    assert counter.round_trips == 0
