# File: tests/integration/test_showcase_kafka_tools.py
# Purpose: Proves the showcase Kafka helpers work against the real broker: create a throwaway topic, read back what
#          was produced in order, delete the topic, and refuse to delete a topic that is not a showcase one.
#
# If Kafka is not reachable the test is SKIPPED with a printed reason. The broker is never stopped here.
import asyncio

import pytest

from core.config import config
from showcase.kafka_tools import create_topic, delete_topic, make_producer, make_topic_name, read_events
from showcase.preconditions import check_tcp_port


def skip_if_kafka_missing() -> None:
    """Skip the test with a clear reason when nothing listens on the Kafka address."""
    host, _, port = config.kafka_bootstrap_servers.partition(":")
    problem = check_tcp_port(host, int(port), "Kafka", "docker compose up -d redpanda")
    if problem is not None:
        pytest.skip(problem)


def test_topic_roundtrip_in_order_and_delete_safety():
    """Three messages come back in the order sent, the topic can be deleted, and a real topic name is refused."""
    skip_if_kafka_missing()

    async def scenario() -> None:
        topic = make_topic_name()
        assert topic.startswith("showcase.l07.")
        await create_topic(topic)
        try:
            # Send three messages with a real producer, then read them back.
            producer = await make_producer()
            try:
                for number in range(3):
                    await producer.send_and_wait(topic, value=f"value-{number}".encode(), key=f"key-{number}".encode())
            finally:
                await producer.stop()
            events = await read_events(topic, expected_count=3, idle_timeout_seconds=10)
            assert [key for _, key, _ in events] == ["key-0", "key-1", "key-2"]
            assert [value for _, _, value in events] == [b"value-0", b"value-1", b"value-2"]
        finally:
            assert await delete_topic(topic) is None
        # A name that is not a showcase topic must be refused, and nothing is touched.
        refused = await delete_topic(config.kafka_outbox_topic)
        assert refused is not None and "refused" in refused

    asyncio.run(scenario())
