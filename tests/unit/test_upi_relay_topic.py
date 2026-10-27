# File: tests/unit/test_upi_relay_topic.py
# Purpose: Proves that the outbox relay routes schema "2" events to the UPI topic,
#          while schema "1", unversioned, or malformed payloads route to the default outbox topic.
import pytest

from api.services.outbox_relay import choose_topic, publish_event
from core.config import config

pytestmark = pytest.mark.anyio


class FakeProducer:
    """Stands in for the Kafka producer and records what it was asked to send."""

    def __init__(self) -> None:
        self.sent: list[tuple[str, bytes, bytes]] = []

    async def send_and_wait(self, topic: str, value: bytes, key: bytes) -> None:
        """Record the call instead of sending anything."""
        self.sent.append((topic, value, key))


async def test_v2_payload_routes_to_kafka_upi_topic() -> None:
    """A row with schema_version '2' routes to config.kafka_upi_topic ('upi.payments')."""
    row = {
        "id": 1,
        "event_id": "3b8d234a-e4ef-4c6e-8a21-9e8c459f2e71",
        "payload": '{"schema_version": "2", "event_type": "upi.payment.recorded"}',
    }
    producer = FakeProducer()
    await publish_event(producer, row)  # type: ignore[arg-type]

    assert len(producer.sent) == 1
    topic, _, _ = producer.sent[0]
    assert topic == config.kafka_upi_topic
    assert choose_topic(row) == config.kafka_upi_topic


async def test_v1_payload_routes_to_kafka_outbox_topic() -> None:
    """A row with schema_version '1' routes to config.kafka_outbox_topic ('transactions.created')."""
    row = {
        "id": 2,
        "event_id": "6d9a3461-75d3-5bae-9a26-263072b6ef2a",
        "payload": '{"schema_version": "1", "event_type": "transaction.created"}',
    }
    producer = FakeProducer()
    await publish_event(producer, row)  # type: ignore[arg-type]

    assert len(producer.sent) == 1
    topic, _, _ = producer.sent[0]
    assert topic == config.kafka_outbox_topic
    assert choose_topic(row) == config.kafka_outbox_topic


async def test_payload_without_schema_version_routes_to_default_topic() -> None:
    """A payload missing the schema_version field routes to config.kafka_outbox_topic."""
    row = {
        "id": 3,
        "event_id": "8f9a3461-75d3-5bae-9a26-263072b6ef2c",
        "payload": '{"event_type": "legacy.event", "data": {"foo": "bar"}}',
    }
    producer = FakeProducer()
    await publish_event(producer, row)  # type: ignore[arg-type]

    assert len(producer.sent) == 1
    topic, _, _ = producer.sent[0]
    assert topic == config.kafka_outbox_topic
    assert choose_topic(row) == config.kafka_outbox_topic


async def test_explicit_topic_wins_for_v2() -> None:
    """When an explicit topic is given (e.g. throwaway showcase topic), it overrides v2 routing."""
    row = {
        "id": 4,
        "event_id": "3b8d234a-e4ef-4c6e-8a21-9e8c459f2e71",
        "payload": '{"schema_version": "2", "event_type": "upi.payment.recorded"}',
    }
    producer = FakeProducer()
    custom_topic = "showcase.throwaway.topic"
    await publish_event(producer, row, topic=custom_topic)  # type: ignore[arg-type]

    assert len(producer.sent) == 1
    topic, _, _ = producer.sent[0]
    assert topic == custom_topic
    assert choose_topic(row, explicit_topic=custom_topic) == custom_topic


async def test_invalid_json_payload_routes_to_default_topic() -> None:
    """Documented fallback: a payload that fails to parse keeps the default outbox topic without raising."""
    row = {
        "id": 5,
        "event_id": "9a8b7c6d-1234-5678-9abc-def012345678",
        "payload": "{invalid json: definitely not valid}",
    }
    producer = FakeProducer()
    await publish_event(producer, row)  # type: ignore[arg-type]

    assert len(producer.sent) == 1
    topic, _, _ = producer.sent[0]
    assert topic == config.kafka_outbox_topic
    assert choose_topic(row) == config.kafka_outbox_topic
