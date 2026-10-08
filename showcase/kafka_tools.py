# File: showcase/kafka_tools.py
# Purpose: Small Kafka helpers for showcases: make and delete a throwaway topic, read the messages back from it, and
#          get a producer. Showcases publish to their own topic so they never mix with the real "transactions.created".
import asyncio
import time
import uuid

from aiokafka import AIOKafkaConsumer, AIOKafkaProducer
from aiokafka.admin import AIOKafkaAdminClient, NewTopic

from api.services.outbox_relay import start_producer
from core.config import config

# Every throwaway topic starts with this. delete_topic refuses any other name.
TOPIC_PREFIX = "showcase."
# How long one read waits for messages before looking at the clock again.
POLL_MILLISECONDS = 500


def make_topic_name(prefix: str = "l07") -> str:
    """
    Build a unique topic name such as showcase.l07.3f9a1c0b7d2e.

    The name comes ONLY from our own code (a fixed start plus random hex), never from user input.
    """
    return f"{TOPIC_PREFIX}{prefix}." + uuid.uuid4().hex[:12]


async def create_topic(topic: str) -> None:
    """Create a topic with 1 partition (one partition is what keeps messages in order) and 1 replica."""
    admin = AIOKafkaAdminClient(bootstrap_servers=config.kafka_bootstrap_servers)
    await admin.start()
    try:
        await admin.create_topics([NewTopic(name=topic, num_partitions=1, replication_factor=1)])
    finally:
        await admin.close()


async def delete_topic(topic: str) -> str | None:
    """
    Delete a throwaway topic. NEVER raises: returns None on success, or a short problem text.

    Safety rule: only names starting with "showcase." are deleted, so a bug can never delete a real topic such as
    "transactions.created". Clean-up must not hide the real result of a check, which is why errors are returned
    instead of raised.
    """
    if not topic.startswith(TOPIC_PREFIX):
        return f"refused: '{topic}' does not start with '{TOPIC_PREFIX}'"
    admin = AIOKafkaAdminClient(bootstrap_servers=config.kafka_bootstrap_servers)
    try:
        await admin.start()
        await admin.delete_topics([topic])
        return None
    except Exception as error:  # noqa: BLE001 - clean-up must never raise
        return f"could not delete topic: {type(error).__name__}: {error}"
    finally:
        try:
            await admin.close()
        except Exception:  # noqa: BLE001 - the client may never have connected
            pass


async def make_producer() -> AIOKafkaProducer:
    """The same producer the real relay uses (acks="all"). Reused, not copied."""
    return await start_producer()


async def read_events(
    topic: str,
    expected_count: int,
    idle_timeout_seconds: float,
    group_suffix: str = "",
) -> list[tuple[int, str, bytes]]:
    """
    Read messages from a topic, as (offset, key text, value bytes) in the order Kafka holds them.

    Stops when expected_count messages were read, or when nothing arrived for idle_timeout_seconds. Uses a FRESH
    consumer group every time, so it always starts at the beginning and never disturbs any real group.
    """
    consumer = AIOKafkaConsumer(
        topic,
        bootstrap_servers=config.kafka_bootstrap_servers,
        group_id="showcase-" + uuid.uuid4().hex[:12] + group_suffix,
        auto_offset_reset="earliest",
        enable_auto_commit=False,
    )
    await consumer.start()
    try:
        return await collect_messages(consumer, expected_count, idle_timeout_seconds)
    finally:
        await consumer.stop()


async def collect_messages(
    consumer: AIOKafkaConsumer, expected_count: int, idle_timeout_seconds: float,
) -> list[tuple[int, str, bytes]]:
    """Keep asking the consumer for messages until we have enough or it has been quiet for too long."""
    messages: list[tuple[int, str, bytes]] = []
    last_message_time = time.monotonic()
    while len(messages) < expected_count:
        batches = await consumer.getmany(timeout_ms=POLL_MILLISECONDS, max_records=100)
        for records in batches.values():
            for record in records:
                key_text = record.key.decode("utf-8") if record.key else ""
                messages.append((record.offset, key_text, record.value))
                last_message_time = time.monotonic()
        if time.monotonic() - last_message_time >= idle_timeout_seconds:
            break
        await asyncio.sleep(0)
    return messages
