# File: tests/unit/test_payment_consumer_blocking.py
# Purpose: Unit tests proving run_payment_updater blocks on getone and consume_available uses getmany.
# Issue: Level 16b round 3b consumer blocking fix.
"""
WHAT: Unit tests verifying that run_payment_updater with idle_timeout_seconds=None calls getone()
and never getmany(), that it cleanly terminates when idle_timeout_seconds is provided, and that
consume_available calls getmany() once and processes messages in strict partition order.

WHY: Architect decision 16b: looping on getmany(timeout_ms=0) caused a tight busy loop when idle.
Blocking on getone() restores proper backpressure and CPU efficiency for streaming payment updater,
while consume_available serves batch draining for replay runners.
"""

import asyncio
from typing import Any, cast
from unittest.mock import patch

import fakeredis
import orjson
import pytest
from aiokafka import AIOKafkaConsumer, TopicPartition

from features.account_keys import get_default_settings
from features.account_stream import consume_available, run_payment_updater

pytestmark = pytest.mark.anyio


class SentinelStop(Exception):
    """Sentinel exception raised by mock consumer to stop the infinite updater loop."""


class FakeConsumerMessage:
    def __init__(self, topic: str, partition: int, offset: int, value: bytes) -> None:
        self.topic = topic
        self.partition = partition
        self.offset = offset
        self.value = value


RUN_ID = "0123456789abcdef"


def make_upi_payload(payment_id: str, partition: int) -> bytes:
    full_payment_id = f"{RUN_ID}-{payment_id}"
    return orjson.dumps(
        {
            "schema_version": "2",
            "event_type": "upi.payment.recorded",
            "payment_id": full_payment_id,
            "side": "debit",
            "event_time": "2026-10-28T10:00:00Z",
            "recorded_at": "2026-10-28T10:00:05Z",
            "arrival_time": "2026-10-28T10:00:05Z",
            "data": {
                "bank_id": "bank_0",
                "account_id": f"acc_{partition}",
                "counterparty_bank_id": "bank_1",
                "counterparty_vpa": "acc_cp",
                "amount_paise": 15000,
                "status": "success",
                "failure_domain": None,
                "payee_type": "person",
                "channel": "upi_app",
                "sender_device_fingerprint": None,
                "sender_os_generation": None,
                "balance_after_paise": 500000,
            },
        }
    )


class SpyBlockingConsumer:
    """Spy consumer that asserts getmany is never called and yields messages via getone."""

    def __init__(self, messages: list[FakeConsumerMessage]) -> None:
        self.messages = list(messages)
        self.getone_calls: int = 0
        self.getmany_calls: int = 0
        self.commits: list[dict[TopicPartition, int]] = []
        self.started: bool = False
        self.stopped: bool = False

    async def start(self) -> None:
        self.started = True

    async def stop(self) -> None:
        self.stopped = True

    async def getone(self) -> FakeConsumerMessage:
        self.getone_calls += 1
        if self.messages:
            return self.messages.pop(0)
        raise SentinelStop("No more messages; stopping infinite updater loop")

    async def getmany(self, *args: Any, **kwargs: Any) -> Any:
        self.getmany_calls += 1
        raise AssertionError("run_payment_updater must not call getmany; it must call getone")

    async def commit(self, offsets: dict[TopicPartition, int]) -> None:
        self.commits.append(offsets)


class SpyTimeoutConsumer:
    """Consumer that delivers one message then hangs to trigger asyncio.TimeoutError."""

    def __init__(self, first_msg: FakeConsumerMessage) -> None:
        self.first_msg: FakeConsumerMessage | None = first_msg
        self.stopped: bool = False
        self.commits: list[dict[TopicPartition, int]] = []

    async def start(self) -> None:
        pass

    async def stop(self) -> None:
        self.stopped = True

    async def getone(self) -> FakeConsumerMessage:
        if self.first_msg is not None:
            msg = self.first_msg
            self.first_msg = None
            return msg
        # Hang longer than test timeout
        await asyncio.sleep(5.0)
        raise RuntimeError("Should not be reached")

    async def commit(self, offsets: dict[TopicPartition, int]) -> None:
        self.commits.append(offsets)


class SpyBatchConsumer:
    """Consumer that tracks getmany calls and delivers partition-keyed batches."""

    def __init__(self, batches: dict[TopicPartition, list[FakeConsumerMessage]]) -> None:
        self.batches = batches
        self.getmany_calls: list[int] = []
        self.commits: list[dict[TopicPartition, int]] = []

    async def getmany(
        self,
        *partitions: Any,
        timeout_ms: int = 0,
        max_records: int | None = None,
    ) -> dict[TopicPartition, list[FakeConsumerMessage]]:
        self.getmany_calls.append(timeout_ms)
        delivered = self.batches
        self.batches = {}
        return delivered

    async def commit(self, offsets: dict[TopicPartition, int]) -> None:
        self.commits.append(offsets)


async def test_run_payment_updater_idle_none_calls_getone_and_never_getmany() -> None:
    """
    Verifies that with idle_timeout_seconds=None, run_payment_updater blocks on getone(),
    never invokes getmany(), processes each message, and stops cleanly when interrupted.
    """
    m1 = FakeConsumerMessage("upi.payments", 0, 1, make_upi_payload("p-01", 0))
    m2 = FakeConsumerMessage("upi.payments", 0, 2, make_upi_payload("p-02", 0))
    spy = SpyBlockingConsumer([m1, m2])
    redis_client = fakeredis.FakeRedis()

    with patch("features.account_stream.AIOKafkaConsumer", return_value=spy):
        with pytest.raises(SentinelStop, match="No more messages"):
            await run_payment_updater(redis_client, idle_timeout_seconds=None)

    assert spy.started is True
    assert spy.stopped is True
    assert spy.getone_calls == 3  # 2 messages + 1 sentinel stop
    assert spy.getmany_calls == 0
    assert len(spy.commits) == 2
    assert spy.commits[0] == {TopicPartition("upi.payments", 0): 2}
    assert spy.commits[1] == {TopicPartition("upi.payments", 0): 3}


async def test_run_payment_updater_with_idle_timeout_stops_after_timeout() -> None:
    """
    Verifies that with idle_timeout_seconds specified, run_payment_updater stops
    when no new message arrives within the timeout interval.
    """
    m1 = FakeConsumerMessage("upi.payments", 0, 1, make_upi_payload("p-01", 0))
    spy = SpyTimeoutConsumer(m1)
    redis_client = fakeredis.FakeRedis()

    with patch("features.account_stream.AIOKafkaConsumer", return_value=spy):
        processed = await run_payment_updater(redis_client, idle_timeout_seconds=0.05)

    assert processed == 1
    assert spy.stopped is True
    assert len(spy.commits) == 1
    assert spy.commits[0] == {TopicPartition("upi.payments", 0): 2}


async def test_consume_available_calls_getmany_once_and_processes_in_partition_order() -> None:
    """
    Verifies that consume_available invokes getmany() exactly once with timeout_ms,
    and iterates over partitions in sorted order (partition 0 before partition 1).
    """
    tp0 = TopicPartition("upi.payments", 0)
    tp1 = TopicPartition("upi.payments", 1)
    m_p0 = FakeConsumerMessage("upi.payments", 0, 5, make_upi_payload("p-p0", 0))
    m_p1 = FakeConsumerMessage("upi.payments", 1, 9, make_upi_payload("p-p1", 1))

    # Intentionally insert partition 1 first in the dictionary to verify sorting by partition
    batches: dict[TopicPartition, list[FakeConsumerMessage]] = {
        tp1: [m_p1],
        tp0: [m_p0],
    }
    spy = SpyBatchConsumer(batches)
    redis_client = fakeredis.FakeRedis()
    cfg = get_default_settings()

    count = await consume_available(cast(AIOKafkaConsumer, spy), redis_client, cfg, timeout_ms=75)

    assert count == 2
    assert spy.getmany_calls == [75]
    # Commits must happen in partition 0 first, then partition 1
    assert len(spy.commits) == 2
    assert spy.commits[0] == {tp0: 6}
    assert spy.commits[1] == {tp1: 10}
