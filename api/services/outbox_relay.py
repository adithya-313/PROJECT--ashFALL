"""
File: api/services/outbox_relay.py
Purpose: The outbox relay. A small background program that reads outbox rows nobody has published yet and sends them
         to Kafka, marking each row as processed only after Kafka has confirmed it received the message.

Why it lives here and not in a top-level folder: the Docker image only contains api/, core/ and infra/, so a file in
this package runs inside the existing image with no Dockerfile change, and it sits next to api/services/outbox.py,
whose payload shape it forwards.

The story in plain English:
  /score stores a transaction and an "event to publish" row in one database transaction (api/services/outbox.py). This
  relay is the second half. Over and over it: (1) looks for unpublished events, oldest first; (2) sends each one to
  Kafka and waits for the broker's confirmation; (3) only then marks that row as processed. If Kafka is down, nothing is
  marked, the events simply wait in the table, and the relay retries with growing pauses (1s, 2s, 4s ... up to a cap).
  The API never needs Kafka to be up.

Delivery guarantee: AT LEAST ONCE. If the relay crashes after Kafka confirmed a message but before the row was marked
processed, the event is sent again on the next round. Every message carries its event_id as the Kafka key and inside
the payload, so a consumer can recognise and skip a repeat.

Run it with:  python -m api.services.outbox_relay
"""

import asyncio
import signal
from typing import NamedTuple

import asyncpg
from aiokafka import AIOKafkaProducer

from core.config import config
from core.logging import configure_logging, get_logger

logger = get_logger(__name__)

# The wait after the first failure. It doubles on every further failure, up to config.relay_max_backoff_seconds.
INITIAL_BACKOFF_SECONDS = 1.0

# How long one Kafka request may take before it counts as failed. The default (40 s) would keep the database rows
# locked for a very long time while the broker is down, so we use a shorter limit.
KAFKA_REQUEST_TIMEOUT_MS = 10_000


class BatchResult(NamedTuple):
    """What happened in one round: how many events were sent, how many are still waiting, and any failure."""

    published: int
    waiting: int
    error: Exception | None


async def fetch_unprocessed_batch(conn: asyncpg.Connection, batch_size: int) -> list[asyncpg.Record]:
    """
    Read (and lock) the oldest unpublished events. Must be called inside a database transaction.

    FOR UPDATE SKIP LOCKED locks the rows we read and silently skips rows another relay already holds, so two relays
    running at once never take the same event.
    """
    return await conn.fetch(
        """
        SELECT id, event_id, payload
        FROM outbox
        WHERE processed_at IS NULL
        ORDER BY created_at, id
        LIMIT $1
        FOR UPDATE SKIP LOCKED
        """,
        batch_size,
    )


async def publish_event(producer: AIOKafkaProducer, row: asyncpg.Record, topic: str | None = None) -> None:
    """
    Send one event to Kafka and wait for the broker's confirmation.

    topic is optional: by default the event goes to config.kafka_outbox_topic. A different topic can be given, for
    example by the showcases, which publish to a throwaway topic.

    The message key is the event_id (so a consumer can spot repeats), and the value is the stored JSON payload exactly
    as the API wrote it. send_and_wait returns only after the broker acknowledged the message (acks="all"), and raises
    an error if it could not.
    """
    await producer.send_and_wait(
        topic or config.kafka_outbox_topic,
        value=row["payload"].encode("utf-8"),  # asyncpg hands jsonb back as JSON text
        key=str(row["event_id"]).encode("utf-8"),
    )


async def mark_processed(conn: asyncpg.Connection, row_id: int) -> None:
    """Record that this event was delivered. Called only AFTER Kafka confirmed this row."""
    await conn.execute("UPDATE outbox SET processed_at = now() WHERE id = $1", row_id)


async def count_waiting(conn: asyncpg.Connection) -> int:
    """How many events are still unpublished (cheap: the partial index only holds unprocessed rows)."""
    return await conn.fetchval("SELECT count(*) FROM outbox WHERE processed_at IS NULL")


async def relay_one_batch(conn: asyncpg.Connection, producer: AIOKafkaProducer, batch_size: int,
                          topic: str | None = None) -> BatchResult:
    """
    Publish one batch in order and mark each row right after its own confirmation.

    topic is optional and is passed to publish_event (None means the configured topic).

    If a publish fails partway, we STOP the batch: the rows already confirmed stay marked, the failed row and every row
    after it stay unprocessed, and the database transaction still commits the marks for the confirmed rows (we do not
    raise inside the transaction block, so nothing is rolled back).
    """
    published = 0
    error: Exception | None = None
    async with conn.transaction():
        rows = await fetch_unprocessed_batch(conn, batch_size)
        for row in rows:
            try:
                await publish_event(producer, row, topic)
            except Exception as publish_error:  # noqa: BLE001 - any Kafka problem stops the batch, never skips a row
                error = publish_error
                break
            await mark_processed(conn, row["id"])
            published += 1
    return BatchResult(published, await count_waiting(conn), error)


async def start_producer() -> AIOKafkaProducer:
    """Create a Kafka producer that waits for all in-sync replicas to confirm each message (acks="all")."""
    producer = AIOKafkaProducer(
        bootstrap_servers=config.kafka_bootstrap_servers,
        acks="all",
        request_timeout_ms=KAFKA_REQUEST_TIMEOUT_MS,
    )
    await producer.start()
    return producer


async def close_quietly(producer: AIOKafkaProducer | None, conn: asyncpg.Connection | None) -> None:
    """Close the producer and the database connection, ignoring errors (they may already be broken)."""
    for closer in (producer.stop() if producer else None, conn.close() if conn else None):
        if closer is None:
            continue
        try:
            await closer
        except Exception:  # noqa: BLE001 - we are cleaning up after a failure; there is nothing more to do
            pass


async def sleep_or_stop(stop_event: asyncio.Event, seconds: float) -> None:
    """Sleep, but wake up immediately if a shutdown was requested."""
    try:
        await asyncio.wait_for(stop_event.wait(), timeout=seconds)
    except asyncio.TimeoutError:
        pass


def next_backoff(current: float) -> float:
    """Double the wait after a failure, but never go above the configured cap."""
    return min(current * 2, config.relay_max_backoff_seconds)


async def run_relay(stop_event: asyncio.Event) -> None:
    """
    The main loop. Each round: make sure we are connected, relay one batch, then wait.

    After a success we wait the normal poll interval. After a failure we wait the current backoff, which doubles each
    time up to the cap, and we throw away the producer and database connection so the next round starts with fresh,
    healthy ones. The first success after failures resets the backoff to its starting value.
    """
    conn: asyncpg.Connection | None = None
    producer: AIOKafkaProducer | None = None
    backoff = INITIAL_BACKOFF_SECONDS
    logger.info("outbox relay started", topic=config.kafka_outbox_topic, kafka=config.kafka_bootstrap_servers,
                batch_size=config.relay_batch_size)

    while not stop_event.is_set():
        failure: Exception | None = None
        published = 0
        try:
            conn = conn or await asyncpg.connect(dsn=config.database_url)
            producer = producer or await start_producer()
            result = await relay_one_batch(conn, producer, config.relay_batch_size)
            published, failure = result.published, result.error
            if published or failure:
                logger.info("outbox batch", published=published, still_waiting=result.waiting,
                            failed=failure is not None)
        except Exception as error:  # noqa: BLE001 - Kafka down, database restarting, etc.: back off and retry
            failure = error

        if failure is None:
            backoff = INITIAL_BACKOFF_SECONDS  # healthy again: reset the backoff
            await sleep_or_stop(stop_event, config.relay_poll_interval_seconds)
            continue

        logger.warning("outbox relay failure, will retry", error_type=type(failure).__name__, error=str(failure),
                       retry_in_seconds=backoff)
        await close_quietly(producer, conn)  # discard possibly broken connections; they are recreated next round
        producer, conn = None, None
        await sleep_or_stop(stop_event, backoff)
        backoff = next_backoff(backoff)

    await close_quietly(producer, conn)
    logger.info("outbox relay stopped")


async def main() -> None:
    """Set up logging and signal handling, then run until asked to stop (SIGTERM from Docker, or Ctrl-C)."""
    configure_logging(config.log_level)
    stop_event = asyncio.Event()
    loop = asyncio.get_running_loop()
    for signal_number in (signal.SIGTERM, signal.SIGINT):
        try:
            loop.add_signal_handler(signal_number, stop_event.set)
        except NotImplementedError:  # Windows event loops do not support this; Ctrl-C still raises KeyboardInterrupt
            pass
    await run_relay(stop_event)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
