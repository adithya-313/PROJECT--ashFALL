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
from typing import Any, Mapping, NamedTuple

import asyncpg
import orjson
from aiokafka import AIOKafkaProducer
from aiokafka.errors import (
    CorruptRecordException,
    InvalidTopicError,
    MessageSizeTooLargeError,
    RecordTooLargeError,
)
from prometheus_client import Counter, start_http_server

from core.config import config
from core.logging import configure_logging, get_logger

logger = get_logger(__name__)

# The wait after the first failure. It doubles on every further failure, up to config.relay_max_backoff_seconds.
INITIAL_BACKOFF_SECONDS = 1.0

# How long one Kafka request may take before it counts as failed. The default (40 s) would keep the database rows
# locked for a very long time while the broker is down, so we use a shorter limit.
KAFKA_REQUEST_TIMEOUT_MS = 10_000

# Errors that mean "Kafka rejects THIS message, whatever the broker's health" (a poison message). Only these count as
# attempts towards the dead-letter limit. Everything else (broker down, timeouts, leader changes, unknown errors) is
# treated as "try again later" and never counts, so an outage can never park a healthy message.
POISON_ERRORS = (MessageSizeTooLargeError, RecordTooLargeError, InvalidTopicError, CorruptRecordException)

# A counter that only ever goes up: how many messages this relay process has parked. Prometheus reads it from the
# metrics port (see main), and a rate or increase over time shows when poison messages appear.
DEAD_LETTERED_TOTAL = Counter("outbox_dead_lettered_total", "Messages parked in outbox_dead_letter")

# Longest error text stored in the database, so one huge error message cannot bloat a row.
MAX_ERROR_TEXT_LENGTH = 1000


class BatchResult(NamedTuple):
    """What happened in one round: how many events were sent, how many are still waiting, and any failure."""

    published: int
    waiting: int
    error: Exception | None
    dead_lettered: int = 0  # how many poison messages were parked in this round
    poison_pending: bool = False  # True when the round stopped only because a poison message is still being retried


async def fetch_unprocessed_batch(conn: asyncpg.Connection, batch_size: int) -> list[asyncpg.Record]:
    """
    Read (and lock) the oldest unpublished events. Must be called inside a database transaction.

    FOR UPDATE SKIP LOCKED locks the rows we read and silently skips rows another relay already holds, so two relays
    running at once never take the same event.
    """
    return await conn.fetch(
        """
        SELECT id, event_id, transaction_id, payload, attempt_count
        FROM outbox
        WHERE processed_at IS NULL AND dead_lettered_at IS NULL
        ORDER BY created_at, id
        LIMIT $1
        FOR UPDATE SKIP LOCKED
        """,
        batch_size,
    )


def choose_topic(row: asyncpg.Record | Mapping[str, Any], explicit_topic: str | None = None) -> str:
    """
    Select the Kafka destination topic for an outbox event row.

    Rules:
      1. If explicit_topic is provided (such as in showcase runs using isolated throwaway topics),
         it is returned directly.
      2. Otherwise, row["payload"] is parsed with orjson. If the payload is a dictionary whose
         schema_version == "2", config.kafka_upi_topic ("upi.payments") is returned.
      3. For schema_version "1" or any other schema version, config.kafka_outbox_topic ("transactions.created")
         is returned.
      4. THE ONE ALLOWED FALLBACK: If row["payload"] fails to parse as valid JSON (for example due to corrupt
         or malformed stored data), it keeps the old default topic (config.kafka_outbox_topic), exactly as
         Level 7 behaved prior to schema versioning.
    """
    if explicit_topic is not None:
        return explicit_topic

    payload_raw = row["payload"]
    try:
        parsed = orjson.loads(payload_raw)
        if isinstance(parsed, dict) and parsed.get("schema_version") == "2":
            return config.kafka_upi_topic
    except (orjson.JSONDecodeError, TypeError):
        # Documented fallback: non-parseable payload keeps the old default topic (Level 7 behavior)
        return config.kafka_outbox_topic

    return config.kafka_outbox_topic


async def publish_event(producer: AIOKafkaProducer, row: asyncpg.Record, topic: str | None = None) -> None:
    """
    Send one event to Kafka and wait for the broker's confirmation.

    topic is optional: by default choose_topic selects config.kafka_upi_topic for schema "2" events
    and config.kafka_outbox_topic for schema "1" events. A different topic can be given, for
    example by the showcases, which publish to a throwaway topic.

    The message key is the event_id (so a consumer can spot repeats), and the value is the stored JSON payload exactly
    as the API wrote it. send_and_wait returns only after the broker acknowledged the message (acks="all"), and raises
    an error if it could not.
    """
    target_topic = choose_topic(row, topic)
    await producer.send_and_wait(
        target_topic,
        value=row["payload"].encode("utf-8"),  # asyncpg hands jsonb back as JSON text
        key=str(row["event_id"]).encode("utf-8"),
    )


async def mark_processed(conn: asyncpg.Connection, row_id: int) -> None:
    """Record that this event was delivered. Called only AFTER Kafka confirmed this row."""
    await conn.execute("UPDATE outbox SET processed_at = now() WHERE id = $1", row_id)


async def count_waiting(conn: asyncpg.Connection) -> int:
    """How many events are still waiting to be published (parked poison messages do not count as waiting)."""
    return await conn.fetchval("SELECT count(*) FROM outbox WHERE processed_at IS NULL AND dead_lettered_at IS NULL")


def is_poison_error(error: Exception) -> bool:
    """True only for errors that mean Kafka rejects this very message (see POISON_ERRORS)."""
    return isinstance(error, POISON_ERRORS)


def note_dead_lettered(count: int) -> None:
    """Add `count` to the metric that counts parked messages (call it only after the database commit)."""
    DEAD_LETTERED_TOTAL.inc(count)


async def record_failed_attempt(conn: asyncpg.Connection, row_id: int, error: Exception) -> int:
    """Count one more rejection for this row, remember why, and return the new attempt count."""
    error_text = f"{type(error).__name__}: {error}"[:MAX_ERROR_TEXT_LENGTH]
    return await conn.fetchval(
        """
        UPDATE outbox
        SET attempt_count = attempt_count + 1,
            last_error = $2,
            first_failed_at = COALESCE(first_failed_at, now())
        WHERE id = $1
        RETURNING attempt_count
        """,
        row_id, error_text,
    )


async def move_to_dead_letter(conn: asyncpg.Connection, row: asyncpg.Record, error: Exception, attempts: int) -> None:
    """
    Park a poison message: copy it into outbox_dead_letter and mark the outbox row as parked.

    The outbox row is never deleted: a repeated request with the same idempotency_key looks it up to find the
    original transaction. Both statements run inside the caller's database transaction, so the row is either fully
    parked or not at all.
    """
    error_text = str(error)[:MAX_ERROR_TEXT_LENGTH]
    await conn.execute(
        """
        INSERT INTO outbox_dead_letter
          (outbox_id, event_id, transaction_id, payload, error_type, error_message, attempt_count, first_failed_at)
        SELECT id, event_id, transaction_id, payload, $2::text, $3::text, $4::integer, first_failed_at
        FROM outbox WHERE id = $1
        """,
        row["id"], type(error).__name__, error_text, attempts,
    )
    await conn.execute("UPDATE outbox SET dead_lettered_at = now() WHERE id = $1", row["id"])


async def handle_publish_failure(
    conn: asyncpg.Connection, row: asyncpg.Record, error: Exception, max_attempts: int,
) -> str:
    """
    Decide what a failed publish means. Returns "outage", "retry" or "parked".

    outage: not a poison error (Kafka down, timeout ...). Nothing is counted; the caller stops and backs off.
    retry:  a poison error that has not reached the limit yet. The attempt is counted; the caller stops the batch so
            the order is kept while we retry.
    parked: a poison error at the limit. The message is moved to the dead-letter table; the caller carries on.
    """
    if not is_poison_error(error):
        return "outage"
    attempts = await record_failed_attempt(conn, row["id"], error)
    if attempts < max_attempts:
        logger.warning("outbox poison attempt", event_id=str(row["event_id"]), attempts=attempts,
                       max_attempts=max_attempts, error_type=type(error).__name__)
        return "retry"
    await move_to_dead_letter(conn, row, error, attempts)
    logger.error("outbox message dead-lettered", event_id=str(row["event_id"]), attempts=attempts,
                 error_type=type(error).__name__)
    return "parked"


async def relay_one_batch(
    conn: asyncpg.Connection,
    producer: AIOKafkaProducer,
    batch_size: int,
    topic: str | None = None,
    max_attempts: int | None = None,
) -> BatchResult:
    """
    Publish one batch in order and mark each row right after its own confirmation.

    topic is optional and is passed to publish_event (None means the configured topic). max_attempts is how many
    times a poison message is tried before it is parked (None means config.relay_max_attempts).

    A failure that is NOT a poison error (for example the broker is down) stops the batch: the rows already confirmed
    stay marked, the failed row and every row after it stay unprocessed, and the database transaction still commits
    the marks for the confirmed rows (we do not raise inside the transaction block, so nothing is rolled back).
    A poison error is counted; while the limit is not reached it also stops the batch (order is kept), and once the
    limit is reached the message is parked and the batch carries on with the next row.
    """
    limit = max_attempts or config.relay_max_attempts
    published = 0
    dead_lettered = 0
    error: Exception | None = None
    poison_pending = False
    async with conn.transaction():
        rows = await fetch_unprocessed_batch(conn, batch_size)
        for row in rows:
            try:
                await publish_event(producer, row, topic)
            except Exception as publish_error:  # noqa: BLE001 - classified below; never skips a row silently
                outcome = await handle_publish_failure(conn, row, publish_error, limit)
                if outcome == "parked":
                    dead_lettered += 1
                    continue
                error = publish_error
                poison_pending = outcome == "retry"
                break
            await mark_processed(conn, row["id"])
            published += 1
    # The transaction has committed here, so the metric only counts messages that really are parked.
    if dead_lettered:
        note_dead_lettered(dead_lettered)
    return BatchResult(published, await count_waiting(conn), error, dead_lettered, poison_pending)


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

    A poison message that is still being retried is NOT an outage: the broker answered, it just rejected that message.
    So that round uses the normal poll interval and does not grow the backoff.
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
            if result.poison_pending:
                failure = None  # a poison message is being retried: the broker is fine, so no outage backoff
            if published or result.error or result.dead_lettered:
                logger.info("outbox batch", published=published, still_waiting=result.waiting,
                            failed=result.error is not None, dead_lettered=result.dead_lettered)
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


def start_metrics_server() -> None:
    """
    Serve the metrics on config.relay_metrics_port. If the port cannot be used, only log a warning: metrics are
    nice to have, and they must never stop messages from being delivered.
    """
    try:
        start_http_server(config.relay_metrics_port)
        logger.info("relay metrics server started", port=config.relay_metrics_port)
    except OSError as error:
        logger.warning("relay metrics server could not start, continuing without metrics", error=str(error))


async def main() -> None:
    """Set up logging and signal handling, then run until asked to stop (SIGTERM from Docker, or Ctrl-C)."""
    configure_logging(config.log_level)
    start_metrics_server()
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
