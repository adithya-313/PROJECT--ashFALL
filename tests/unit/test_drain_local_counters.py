# File: tests/unit/test_drain_local_counters.py
# Purpose: Unit tests for the local produced/consumed counters in replay drain (Level 18a1-ii). No Docker.
"""
WHAT: Verifies that with the two switches off the drain asks the broker for offsets as before, that with them on
it uses local counters instead, that the cross-check catches a mismatch, that the national relay counts per topic,
and that a replay hook forces both switches off.

WHY: The counters replace broker round trips. They must follow the real produced/consumed counts and never
hide a disagreement with the broker.
"""

import json
import time
from types import SimpleNamespace
from typing import Any

import pytest

import replay.runner as runner
from nationalsim.national_outbox import relay_national_outbox
from replay.runner import (
    DrainCounters,
    RegistryDrain,
    ReplayDrainTimeout,
    ReplayPipelineError,
    cross_check_counters,
    resolve_drain_switches,
)

pytestmark = pytest.mark.anyio

PAYMENTS_TOPIC = "ws.payments"
REGISTRY_TOPIC = "ws.registry.events"


class FakeConsumer:
    """Records broker offset calls; reports offsets that mean 'fully caught up' unless told otherwise."""

    def __init__(self, end: int = 0, committed: int = 0) -> None:
        self.end = end
        self.committed_value = committed
        self.calls: list[str] = []

    def partitions_for_topic(self, topic: str) -> set[int]:
        return {0}

    async def end_offsets(self, tps: Any) -> dict[Any, int]:
        self.calls.append("end_offsets")
        return {tp: self.end for tp in tps}

    async def committed(self, tp: Any) -> int:
        self.calls.append("committed")
        return self.committed_value


class FakeConn:
    """Workspace connection stand-in: no pollers run (banks is empty), so only count_waiting-like calls matter."""

    async def fetchval(self, *args: Any) -> int:
        return 0


class Harness:
    """Patches the drain's collaborators and records which ones were called."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch, publish: int = 0, consume: int = 0) -> None:
        self.calls: list[str] = []
        self.waiting = 1 if publish else 0
        self.publish = publish
        self.consume = consume
        self.registry_publish = 0
        self.registry_extra_topic: str | None = None
        self.registry_read = 0
        monkeypatch.setattr(runner, "count_waiting", self._count_waiting)
        monkeypatch.setattr(runner, "relay_one_batch", self._relay_one_batch)
        monkeypatch.setattr(runner, "consume_available", self._consume_available)
        monkeypatch.setattr(runner, "relay_national_outbox", self._relay_national_outbox)
        monkeypatch.setattr(runner, "read_registry_batch", self._read_registry_batch)
        monkeypatch.setattr(runner, "drain_registry", self._drain_registry)

    async def _count_waiting(self, conn: Any) -> int:
        return self.waiting

    async def _relay_one_batch(self, conn: Any, producer: Any, **kwargs: Any) -> Any:
        self.calls.append("relay_one_batch")
        self.waiting = 0
        return SimpleNamespace(published=self.publish)

    async def _consume_available(self, *args: Any, **kwargs: Any) -> int:
        taken, self.consume = self.consume, 0
        return taken

    async def _relay_national_outbox(self, conn: Any, producer: Any, **kwargs: Any) -> int:
        self.calls.append("relay_national_outbox")
        by_topic = kwargs["published_by_topic"]
        if self.registry_publish:
            by_topic[REGISTRY_TOPIC] = by_topic.get(REGISTRY_TOPIC, 0) + self.registry_publish
            self.registry_publish = 0
        if self.registry_extra_topic is not None:
            by_topic[self.registry_extra_topic] = 1
        return 0

    async def _read_registry_batch(self, registry: Any, conn: Any, timeout_ms: int = 20) -> int:
        self.calls.append("read_registry_batch")
        return self.registry_read

    async def _drain_registry(self, registry: Any, conn: Any, timeout_ms: int = 20) -> bool:
        self.calls.append("drain_registry")
        return True


async def run_drain(
    consumer: FakeConsumer,
    registry: RegistryDrain | None = None,
    counters: DrainCounters | None = None,
    deadline_seconds: float = 5.0,
    local_offset_tracking: bool = False,
    registry_poll_on_change: bool = False,
) -> float:
    """Runs drain() with empty banks and the fakes above."""
    workspace = SimpleNamespace(kafka_topic=PAYMENTS_TOPIC, registry_topic=REGISTRY_TOPIC)
    return await runner.drain(
        step_index=0,
        workspace_conn=FakeConn(),  # type: ignore[arg-type]
        producer=object(),
        consumer=consumer,  # type: ignore[arg-type]
        workspace=workspace,  # type: ignore[arg-type]
        redis_client=object(),
        acct_settings=object(),  # type: ignore[arg-type]
        banks=[],
        world_clock=object(),  # type: ignore[arg-type]
        run_id="r",
        poller_cursors={},
        deadline=time.monotonic() + deadline_seconds,
        registry=registry,
        counters=counters,
        local_offset_tracking=local_offset_tracking,
        registry_poll_on_change=registry_poll_on_change,
    )


def make_registry() -> RegistryDrain:
    return RegistryDrain(consumer=FakeConsumer(), topic=REGISTRY_TOPIC)  # type: ignore[arg-type]


async def test_switches_off_ask_the_broker_for_offsets(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that with both switches off the drain calls end_offsets and committed, as before."""
    Harness(monkeypatch)
    consumer = FakeConsumer()
    result = await run_drain(consumer)
    assert isinstance(result, float)
    assert "end_offsets" in consumer.calls
    assert "committed" in consumer.calls


async def test_local_offset_tracking_skips_the_broker_and_follows_counters(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that with local_offset_tracking on the drain makes no broker calls and ends at consumed >= produced."""
    harness = Harness(monkeypatch, publish=3, consume=3)
    consumer = FakeConsumer()
    counters = DrainCounters()
    await run_drain(consumer, counters=counters, local_offset_tracking=True)
    assert consumer.calls == []
    assert (counters.payments_produced, counters.payments_consumed) == (3, 3)
    assert harness.calls == ["relay_one_batch"]


async def test_local_offset_tracking_not_caught_up_until_consumed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that produced > consumed keeps the drain waiting, so it hits the deadline."""
    Harness(monkeypatch, publish=3, consume=0)
    with pytest.raises(ReplayDrainTimeout):
        await run_drain(FakeConsumer(), counters=DrainCounters(), deadline_seconds=0.2, local_offset_tracking=True)


async def test_registry_poll_skipped_when_produced_equals_consumed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that equal registry counts skip the registry read, while the national relay still runs."""
    harness = Harness(monkeypatch)
    counters = DrainCounters(registry_produced=2, registry_consumed=2)
    await run_drain(FakeConsumer(), registry=make_registry(), counters=counters, registry_poll_on_change=True)
    assert "relay_national_outbox" in harness.calls
    assert "read_registry_batch" not in harness.calls
    assert "drain_registry" not in harness.calls


async def test_registry_poll_runs_when_produced_exceeds_consumed(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that rows relayed this drain are read, and the consumed count catches up to produced."""
    harness = Harness(monkeypatch)
    harness.registry_publish = 2
    harness.registry_read = 2
    counters = DrainCounters()
    await run_drain(FakeConsumer(), registry=make_registry(), counters=counters, registry_poll_on_change=True)
    assert harness.calls.count("relay_national_outbox") >= 1
    assert "read_registry_batch" in harness.calls
    assert (counters.registry_produced, counters.registry_consumed) == (2, 2)


async def test_registry_switch_off_uses_drain_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that with the switch off the existing drain_registry runs every drain."""
    harness = Harness(monkeypatch)
    await run_drain(FakeConsumer(), registry=make_registry())
    assert "drain_registry" in harness.calls


async def test_unexpected_registry_publisher_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that a publish to any topic other than the registry topic raises while the switch is on."""
    harness = Harness(monkeypatch)
    harness.registry_extra_topic = "hold.requests"
    with pytest.raises(ReplayPipelineError, match="Unexpected publisher"):
        await run_drain(
            FakeConsumer(), registry=make_registry(), counters=DrainCounters(), registry_poll_on_change=True
        )


async def test_switch_on_without_counters_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that a switch with no DrainCounters raises ValueError instead of guessing."""
    Harness(monkeypatch)
    with pytest.raises(ValueError, match="DrainCounters"):
        await run_drain(FakeConsumer(), local_offset_tracking=True)


async def test_cross_check_passes_on_equality_and_raises_on_mismatch() -> None:
    """Verifies that end == produced and committed == consumed passes, and any difference names the topic."""
    workspace: Any = SimpleNamespace(kafka_topic=PAYMENTS_TOPIC)
    counters = DrainCounters(payments_produced=5, payments_consumed=5)
    await cross_check_counters(FakeConsumer(end=5, committed=5), workspace, None, counters)
    with pytest.raises(ReplayPipelineError, match=PAYMENTS_TOPIC):
        await cross_check_counters(FakeConsumer(end=6, committed=5), workspace, None, counters)
    with pytest.raises(ReplayPipelineError, match="committed=4"):
        await cross_check_counters(FakeConsumer(end=5, committed=4), workspace, None, counters)


async def test_cross_check_covers_the_registry_topic() -> None:
    """Verifies that a registry mismatch is reported with the registry topic name."""
    workspace: Any = SimpleNamespace(kafka_topic=PAYMENTS_TOPIC)
    registry = RegistryDrain(consumer=FakeConsumer(end=1, committed=1), topic=REGISTRY_TOPIC)  # type: ignore[arg-type]
    counters = DrainCounters(payments_produced=2, payments_consumed=2, registry_produced=2, registry_consumed=1)
    with pytest.raises(ReplayPipelineError, match=REGISTRY_TOPIC):
        await cross_check_counters(FakeConsumer(end=2, committed=2), workspace, registry, counters)


def test_hook_forces_both_switches_off() -> None:
    """Verifies that a supplied hook turns both switches off, and no hook leaves them as configured."""
    assert resolve_drain_switches(True, True, hooks_present=True) == (False, False)
    assert resolve_drain_switches(True, True, hooks_present=False) == (True, True)
    assert resolve_drain_switches(False, True, hooks_present=False) == (False, True)


async def test_hook_forced_off_means_broker_calls_are_made(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies that the switch values resolved for a hooked run make the drain call the broker."""
    Harness(monkeypatch)
    local, on_change = resolve_drain_switches(True, True, hooks_present=True)
    consumer = FakeConsumer()
    await run_drain(consumer, counters=DrainCounters(), local_offset_tracking=local, registry_poll_on_change=on_change)
    assert "end_offsets" in consumer.calls


class FakeTxn:
    async def __aenter__(self) -> "FakeTxn":
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None


class FakeOutboxConn:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def transaction(self) -> FakeTxn:
        return FakeTxn()

    async def fetch(self, *args: Any) -> list[dict[str, Any]]:
        return self.rows

    async def execute(self, *args: Any) -> None:
        return None


class FakeProducer:
    def __init__(self) -> None:
        self.sent: list[str] = []

    async def send_and_wait(self, topic: str, key: bytes, value: bytes) -> None:
        self.sent.append(topic)


def outbox_rows() -> list[dict[str, Any]]:
    return [
        {"id": 1, "event_id": "e1", "run_id": "r", "topic": "registry.events", "bank_key": "r:b0",
         "schema_version": "v1", "payload": json.dumps({"a": 1})},
        {"id": 2, "event_id": "e2", "run_id": "r", "topic": "hold.requests", "bank_key": "r:b1",
         "schema_version": "v1", "payload": json.dumps({"b": 2})},
    ]


async def test_relay_counts_per_topic_actually_used() -> None:
    """Verifies that published_by_topic holds one count per acknowledged row under the mapped topic."""
    by_topic: dict[str, int] = {}
    total = await relay_national_outbox(
        FakeOutboxConn(outbox_rows()), FakeProducer(),  # type: ignore[arg-type]
        topic_map={"registry.events": REGISTRY_TOPIC}, published_by_topic=by_topic,
    )
    assert total == 2
    assert by_topic == {REGISTRY_TOPIC: 1, "hold.requests": 1}


async def test_relay_without_published_by_topic_is_unchanged() -> None:
    """Verifies that the default (None) records nothing and returns the same total."""
    total = await relay_national_outbox(FakeOutboxConn(outbox_rows()), FakeProducer())  # type: ignore[arg-type]
    assert total == 2
