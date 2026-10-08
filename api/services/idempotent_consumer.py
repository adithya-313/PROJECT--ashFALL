"""
File: api/services/idempotent_consumer.py
Purpose: A reusable helper for Kafka consumers that must handle each event only once ("idempotent consumer").
         Later levels (feature updater, label handling, tracing) reuse it instead of writing their own dedupe logic.

The story in plain English:
  Our outbox relay delivers events AT LEAST ONCE, so the same event can show up twice. A consumer would then do its
  work twice (count twice, update twice). This module prevents that. Before doing any work for an event, the consumer
  writes "consumer X handled event Y" into the processed_events table, and does its real work in the SAME database
  transaction. The table refuses a second note for the same (consumer, event), so a duplicate is detected and skipped.
  If the work fails, the transaction is rolled back, the note disappears with it, and the event can be retried.

What the helper covers: handle_once (the dedupe step), parse_event (check the message is a valid v1 event) and
run_consumer (the loop that reads Kafka, calls the two above, and commits the Kafka offset only afterwards).

Important limit: the "once" guarantee holds for work done through the SAME database connection inside handler(conn).
Side effects outside that transaction (an HTTP call, a message to another system) are not rolled back with it.
"""

import asyncio
import signal
import uuid
from collections.abc import Awaitable, Callable
from typing import Any

import asyncpg
import orjson  # the project uses orjson, never the standard json module (see tests/unit/test_no_stdlib_json.py)
from aiokafka import AIOKafkaConsumer, ConsumerRecord, TopicPartition

from core.config import config
from core.logging import get_logger

logger = get_logger(__name__)

# The only payload version this helper understands (see api/services/outbox.py).
SUPPORTED_SCHEMA_VERSION = "1"

# A handler receives the open database connection (already inside the dedupe transaction) and does its writes on it.
Handler = Callable[[asyncpg.Connection], Awaitable[None]]


class InvalidEventError(ValueError):
    """The Kafka message is not a valid event of a supported version."""


def parse_event(message_value: bytes) -> dict[str, Any]:
    """
    Decode a message's JSON and check it is an event we can safely process. Returns the payload as a dict.

    We refuse anything unexpected loudly, rather than guessing: a wrong schema_version means the shape may have
    changed, and a missing or malformed event_id would make deduplication impossible.
    """
    try:
        event = orjson.loads(message_value)
    except orjson.JSONDecodeError as error:
        raise InvalidEventError(f"message is not valid JSON: {error}") from error
    if not isinstance(event, dict):
        raise InvalidEventError("message JSON must be an object")
    if event.get("schema_version") != SUPPORTED_SCHEMA_VERSION:
        raise InvalidEventError(
            f"unsupported schema_version {event.get('schema_version')!r}, expected {SUPPORTED_SCHEMA_VERSION!r}")
    if not event.get("event_id"):
        raise InvalidEventError("event has no event_id")
    try:
        uuid.UUID(str(event["event_id"]))
    except ValueError as error:
        raise InvalidEventError(f"event_id is not a valid UUID: {event['event_id']!r}") from error
    return event


async def handle_once(conn: asyncpg.Connection, consumer_name: str, event_id: uuid.UUID, handler: Handler) -> bool:
    """
    Run handler(conn) for this event only if this consumer has not handled it before.

    Returns True if the handler ran, False if the event was a duplicate and was skipped.

    How "twice = once" works: inside ONE database transaction we try to insert (consumer_name, event_id) into
    processed_events. ON CONFLICT DO NOTHING makes a second insert return nothing instead of failing, which tells us
    it is a duplicate. For a new event the handler runs inside that same transaction, so its writes and the "handled"
    note commit together or not at all. If the handler raises, the transaction rolls back, the note is undone, and the
    event is not marked processed, so a retry will run the handler again.
    """
    async with conn.transaction():
        inserted = await conn.fetchval(
            """
            INSERT INTO processed_events (consumer_name, event_id)
            VALUES ($1, $2)
            ON CONFLICT DO NOTHING
            RETURNING event_id
            """,
            consumer_name, event_id,
        )
        if inserted is None:
            return False  # we have already handled this event: skip it
        await handler(conn)
        return True


async def process_message(conn: asyncpg.Connection, consumer_name: str, message: ConsumerRecord,
                          handler: Handler) -> bool:
    """Validate one Kafka message, handle it once, and log what happened. Returns True if it was newly handled."""
    event = parse_event(message.value)
    event_id = uuid.UUID(event["event_id"])
    handled = await handle_once(conn, consumer_name, event_id, handler)
    logger.info("event processed", consumer=consumer_name, event_id=str(event_id),
                outcome="handled" if handled else "duplicate-skipped", partition=message.partition,
                offset=message.offset)
    return handled


async def run_consumer(consumer_name: str, topic: str, group_id: str, handler: Handler,
                       idle_timeout_seconds: float | None = None) -> None:
    """
    Read events from Kafka and handle each one once.

    The Kafka offset is committed only AFTER the event was handled (or recognised as a duplicate). If the program dies
    in between, Kafka re-delivers the event and handle_once skips it, so nothing is lost and nothing is done twice.

    idle_timeout_seconds: stop after this many seconds without a new message (handy for demos and tests). None means
    run until SIGTERM or Ctrl-C.
    """
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signal_number in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(signal_number, stop_event.set)
        except NotImplementedError:  # Windows event loops: Ctrl-C still raises KeyboardInterrupt
            pass

    conn = await asyncpg.connect(dsn=config.database_url)
    consumer = AIOKafkaConsumer(
        topic,
        bootstrap_servers=config.kafka_bootstrap_servers,
        group_id=group_id,
        enable_auto_commit=False,  # we commit by hand, after the work is safely recorded
        auto_offset_reset="earliest",  # a brand-new group starts from the beginning of the topic
    )
    await consumer.start()
    logger.info("consumer started", consumer=consumer_name, topic=topic, group_id=group_id)
    idle_seconds = 0.0
    try:
        while not stop_event.is_set():
            batches = await consumer.getmany(timeout_ms=1000, max_records=100)
            if not batches:
                idle_seconds += 1.0
                if idle_timeout_seconds is not None and idle_seconds >= idle_timeout_seconds:
                    break
                continue
            idle_seconds = 0.0
            for partition, messages in batches.items():
                for message in messages:
                    await process_message(conn, consumer_name, message, handler)
                    # Commit "next offset to read" for this partition, only now that this message is done.
                    await consumer.commit({TopicPartition(partition.topic, partition.partition): message.offset + 1})
    finally:
        await consumer.stop()
        await conn.close()
        logger.info("consumer stopped", consumer=consumer_name)
