# File: tests/unit/test_outbox_relay_topic.py
# Purpose: Proves the optional topic parameter of the outbox relay: with no topic the configured topic is used, and
#          with a topic the event goes there with the same key and value bytes. No Kafka or database is needed.
import asyncio
from typing import Any

from api.services.outbox_relay import publish_event, relay_one_batch
from core.config import config


class FakeProducer:
    """Stands in for the Kafka producer and remembers what it was asked to send."""

    def __init__(self) -> None:
        self.sent: list[tuple[str, bytes, bytes]] = []

    async def send_and_wait(self, topic: str, value: bytes, key: bytes) -> None:
        """Record the call instead of sending anything."""
        self.sent.append((topic, value, key))


class FakeTransaction:
    """A do-nothing 'async with conn.transaction()' block."""

    async def __aenter__(self) -> None:
        return None

    async def __aexit__(self, *error_info: Any) -> None:
        return None


class FakeConnection:
    """Stands in for the database: hands out one unpublished row and records the 'processed' updates."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        self.marked_ids: list[int] = []

    def transaction(self) -> FakeTransaction:
        """Return the do-nothing transaction block."""
        return FakeTransaction()

    async def fetch(self, query: str, *args: Any) -> list[dict[str, Any]]:
        """Pretend to read the unpublished rows."""
        return self.rows

    async def execute(self, query: str, *args: Any) -> None:
        """Record which row id was marked as processed."""
        self.marked_ids.append(args[0])

    async def fetchval(self, query: str, *args: Any) -> int:
        """Pretend nothing is left waiting."""
        return 0


def make_row() -> dict[str, Any]:
    """One fake outbox row with the three fields the relay reads."""
    return {"id": 1, "event_id": "6d9a3461-75d3-5bae-9a26-263072b6ef2a", "payload": '{"hello": "world"}'}


def test_publish_event_without_topic_uses_the_configured_topic():
    """With no topic given, the event goes to config.kafka_outbox_topic."""
    producer = FakeProducer()
    asyncio.run(publish_event(producer, make_row()))  # type: ignore[arg-type]
    assert producer.sent[0][0] == config.kafka_outbox_topic


def test_publish_event_with_topic_sends_there_with_same_key_and_value():
    """With a topic given, the event goes to that topic, and the key and value bytes are unchanged."""
    producer = FakeProducer()
    asyncio.run(publish_event(producer, make_row(), topic="custom.topic"))  # type: ignore[arg-type]
    topic, value, key = producer.sent[0]
    assert topic == "custom.topic"
    assert value == b'{"hello": "world"}'
    assert key == b"6d9a3461-75d3-5bae-9a26-263072b6ef2a"


def test_relay_one_batch_passes_the_topic_through():
    """relay_one_batch sends to the topic it was given and marks the row as processed."""
    producer = FakeProducer()
    connection = FakeConnection([make_row()])
    result = asyncio.run(relay_one_batch(connection, producer, 10, topic="custom.topic"))  # type: ignore[arg-type]
    assert producer.sent[0][0] == "custom.topic"
    assert connection.marked_ids == [1]
    assert result.published == 1 and result.error is None
