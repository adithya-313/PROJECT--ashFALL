# File: features/account_stream.py
# Purpose: Level 15a streaming ingestion adapters for UPI payments (Kafka) and account events (Postgres).
# Issue: Level 15a account feature updater (plan Task 1, C4, K2).
"""
WHAT: Provides asynchronous stream consumers and polling adapters that feed incoming
commercial bank payment events and account lifecycle events into Redis account state.

WHY: Forms the operational bridge between messaging infrastructure (Kafka topic upi.payments)
and transactional database storage (Postgres table bank_account_events) into the Level 15
Redis sliding-window feature store.
"""

import asyncio
import logging
from typing import Any

from aiokafka import AIOKafkaConsumer, TopicPartition

from core.config import config
from core.event_clock import Clock
from features.account_keys import AccountFeaturesSettings, get_default_settings
from features.account_labels import apply_complaints_since, apply_feedback_since
from features.account_records import (
    PaymentSideRecord,
    account_event_from_row,
    parse_upi_event,
    withdrawal_from_row,
)
from features.account_summaries import to_unix_seconds
from features.account_updater import apply_account_event, apply_payment_side, apply_withdrawal
from features.ingest_watermark import record_payment_watermark, record_poller_watermark

logger = logging.getLogger(__name__)

__all__ = [
    "consume_available",
    "process_payment_message",
    "record_watermark_for_side",
    "run_payment_updater",
    "apply_account_events_since",
    "apply_withdrawals_since",
    "apply_feedback_since",
    "apply_complaints_since",
]


def record_watermark_for_side(
    redis_client: Any,
    partition: int,
    record: PaymentSideRecord,
    cfg: AccountFeaturesSettings,
) -> bool:
    """
    Advances the payment watermark partition offset in Redis.

    If the payment side record arrived via world ingestion clock, routes the watermark
    to the run-scoped partition key (acctwm:{run_id}:payments:{partition}) and registers the
    partition in the run's active partitions set.
    Otherwise (real-mode / legacy), routes to the shared partition key (acctwm:payments:{partition}).
    """
    arrival_sec = int(to_unix_seconds(record.arrival_time))
    watermark_ts = arrival_sec - cfg.commit_grace
    run_id = record.run_id if record.arrival_from_ingestion_clock else None
    return record_payment_watermark(redis_client, partition, watermark_ts, run_id=run_id)


async def process_payment_message(
    consumer: Any,
    redis_client: Any,
    msg: Any,
    cfg: AccountFeaturesSettings,
) -> None:
    """
    WHAT: Ingests a single schema-2 UPI payment event: parses, applies to Redis, commits offset, and records watermark.
    WHY: Factored out into a single helper shared by both run_payment_updater (blocking loop) and
    consume_available (batch draining) to guarantee identical processing semantics and error handling.
    """
    try:
        record = parse_upi_event(msg.value)
    except Exception as exc:
        logger.error(
            "Invalid account payment event on topic %s partition %s offset %s: %s",
            msg.topic,
            msg.partition,
            msg.offset,
            exc,
        )
        raise

    apply_payment_side(redis_client, record, cfg)

    # Commit offset synchronously per partition
    tp = TopicPartition(msg.topic, msg.partition)
    await consumer.commit({tp: msg.offset + 1})

    # Ingestion watermark advances only on applied messages:
    # record.arrival_time is the payload arrival_time (or recorded_at). Never msg.timestamp, never event_time.
    record_watermark_for_side(redis_client, msg.partition, record, cfg)


async def consume_available(
    consumer: AIOKafkaConsumer,
    redis_client: Any,
    cfg: AccountFeaturesSettings,
    timeout_ms: int = 0,
) -> int:
    """
    WHAT: Drains available UPI payment events from the consumer up to timeout_ms, applies each to Redis, and commits.
    WHY: Factored out of run_payment_updater so batch draining and replay consumers can process available messages
    without managing consumer lifecycle or topic subscriptions.

    Args:
        consumer: Active AIOKafkaConsumer instance.
        redis_client: Redis client for state updates and watermarks.
        cfg: Validated AccountFeaturesSettings.
        timeout_ms: Maximum milliseconds to wait for available records.

    Returns:
        Integer count of successfully processed and applied payment sides.

    Raises:
        InvalidAccountEventError / Exception: If any message fails validation or processing.
    """
    raw_batches = await consumer.getmany(timeout_ms=timeout_ms)
    applied_count = 0
    for tp in sorted(raw_batches.keys(), key=lambda p: (p.topic, p.partition)):
        for msg in raw_batches[tp]:
            await process_payment_message(consumer, redis_client, msg, cfg)
            applied_count += 1
    return applied_count


async def run_payment_updater(
    redis_client: Any,
    settings: AccountFeaturesSettings | None = None,
    group_id: str = "account-feature-updater",
    idle_timeout_seconds: float | None = None,
    topic: str = config.kafka_upi_topic,
) -> int:
    """
    WHAT: Consumes schema-2 UPI payment events from Kafka, applies each to Redis, and commits offsets.
Kafka gives no per-account ordering here; state is correct in any order because every write is idempotent
(see the shuffle test).

    WHY: Ingests commercial bank transaction sides in near-real-time. Manual commit guarantees
    at-least-once delivery: consumer offsets are advanced only after Redis state is updated.
    If an invalid message is received, it logs partition and offset details and re-raises immediately
    (no silent skipping).

    Args:
        redis_client: Redis client instance.
        settings: Optional AccountFeaturesSettings; defaults to repository configuration.
        group_id: Consumer group identifier for Kafka subscription.
        idle_timeout_seconds: Optional timeout float; stops consumption if no new message arrives.
        topic: Destination Kafka topic; defaults to config.kafka_upi_topic ('upi.payments').

    Returns:
        Integer count of successfully processed payment sides.

    Raises:
        InvalidAccountEventError / Exception: If any message fails validation or processing.
    """
    cfg = settings if settings is not None else get_default_settings()

    consumer = AIOKafkaConsumer(
        topic,
        bootstrap_servers=config.kafka_bootstrap_servers,
        group_id=group_id,
        enable_auto_commit=False,
        auto_offset_reset="earliest",
    )
    await consumer.start()
    processed_count = 0

    try:
        while True:
            try:
                if idle_timeout_seconds is not None:
                    msg = await asyncio.wait_for(consumer.getone(), timeout=idle_timeout_seconds)
                else:
                    msg = await consumer.getone()
            except asyncio.TimeoutError:
                break

            await process_payment_message(consumer, redis_client, msg, cfg)
            processed_count += 1
    finally:
        await consumer.stop()

    return processed_count


async def apply_account_events_since(
    connection: Any,
    redis_client: Any,
    settings: AccountFeaturesSettings | None = None,
    bank_id: str = "bank_0",
    after_id: int = 0,
    clock: Clock | None = None,
    run_id: str | None = None,
    *,
    overlap_rows: int | None = None,
) -> int:
    """
    WHAT: Queries bank_account_events table for events with id > after_id and applies each to Redis.

    WHY: Commercial bank account lifecycle events (device binding changes, SIM swaps, openings)
    reside in PostgreSQL and are ingested into Redis feature state through periodic polling.

    WHY ID ORDER = ARRIVAL ORDER:
    The bank_account_events table uses a BIGSERIAL primary key (id). Under Postgres sequence
    guarantees, each inserted row obtains a strictly increasing integer identifier. Ordering by
    `id` therefore preserves exact ingestion arrival order across all lifecycle events.

    WHY RE-RUNNING FROM AN OLD ID IS SAFE:
    All write operations in apply_account_event are strictly idempotent:
      - ZADD with score replacement is idempotent.
      - HSET with payload overwriting is idempotent.
      - Watermarks use LT / GT conditionals, so re-evaluating older events leaves scores unchanged.
      - Retention pruning is deterministic.
    If a consumer crashes before recording the checkpointed after_id, re-polling already processed
    events causes zero state corruption or duplicate counts in Redis.

    Args:
        connection: Active database connection or connection-like object (asyncpg.Connection).
        redis_client: Redis client instance.
        settings: Optional AccountFeaturesSettings; defaults to repository configuration.
        bank_id: Commercial bank identifier to query.
        after_id: Cursor identifier; only rows with id > after_id are selected.
        clock: Optional domain ingestion clock. When provided, poll_start uses clock.now_ist()
            without querying DB now(); when None, falls back to fetchval("SELECT now()").
        run_id: Optional simulation run identifier. When clock is provided, run_id must also be
            provided to write run-scoped watermarks ('acctwm:{run_id}:{bank_id}:events').
            Run-scoped keys belong only to world-clock runs; if clock is None and run_id is given,
            raises ValueError.
        overlap_rows: Optional override of how many trailing rows before after_id are re-read.
            None uses cfg.account_event_overlap_rows (the live and catch-up behaviour). An int >= 0 uses
            that value; 0 reads only rows after the cursor. The drain's watermark-only call passes 0
            because it has no reason to re-apply rows it already applied. A negative value raises ValueError.

    Returns:
        The highest row ID processed, or after_id if no new rows were found.
    """
    if overlap_rows is not None and overlap_rows < 0:
        raise ValueError(f"overlap_rows must be >= 0 or None, got {overlap_rows}")

    if clock is not None and run_id is None:
        raise ValueError("When clock is provided, run_id must also be provided to write run-scoped watermarks")
    if clock is None and run_id is not None:
        raise ValueError(
            "run_id cannot be provided when clock is None; run-scoped keys belong only to world-clock runs"
        )

    cfg = settings if settings is not None else get_default_settings()
    if clock is not None:
        poll_start_db_time = clock.now_ist()
    else:
        poll_start_db_time = await connection.fetchval("SELECT now()")

    batch_size = cfg.account_event_batch_size
    overlap = cfg.account_event_overlap_rows if overlap_rows is None else overlap_rows
    start_id = max(0, after_id - overlap)

    rows = await connection.fetch(
        """
        SELECT id, bank_id, event_id, account_id, kind, event_time, details, created_at, arrival_time
        FROM bank_account_events
        WHERE bank_id = $1 AND id > $2
        ORDER BY id
        LIMIT $3
        """,
        bank_id,
        start_id,
        batch_size,
    )

    last_id = after_id
    if rows:
        for row in rows:
            record = account_event_from_row(row)
            apply_account_event(redis_client, record, cfg)
            row_id = int(row["id"])
            if row_id > last_id:
                last_id = row_id

    watermark_ts = int(to_unix_seconds(poll_start_db_time)) - cfg.commit_grace
    record_poller_watermark(redis_client, bank_id, "events", watermark_ts, run_id=run_id)

    return last_id


async def apply_withdrawals_since(
    connection: Any,
    redis_client: Any,
    settings: AccountFeaturesSettings | None = None,
    bank_id: str = "bank_0",
    after_id: int = 0,
    clock: Clock | None = None,
    run_id: str | None = None,
    *,
    overlap_rows: int | None = None,
) -> int:
    """
    WHAT: Queries bank_withdrawals table for rows with id > after_id and applies each to Redis.

    WHY: Cash withdrawals (ATM and branch) reside in PostgreSQL and are ingested into Redis feature
    state through periodic polling alongside account lifecycle events.

    WHY ID ORDER = ARRIVAL ORDER:
    The bank_withdrawals table uses a BIGINT GENERATED ALWAYS AS IDENTITY primary key (id).
    Under Postgres sequence semantics, each inserted row obtains a monotonically increasing
    integer identifier. Ordering by `id` preserves exact ingestion arrival order across all withdrawals.

    WHY RE-RUNNING FROM AN OLD ID IS SAFE:
    Concurrent transactions may obtain sequence IDs out of commit order (transaction with id=1
    commits after transaction with id=2). Re-reading the trailing account_event_overlap_rows (200 rows)
    guarantees that late-committing lower-ID rows are never skipped. All operations in apply_withdrawal
    (ZADD with score replacement, HSET with payload overwriting, conditional watermarks, and atomic Lua trim)
    are strictly idempotent, so re-processing already-applied withdrawals produces identical Redis state.

    Args:
        connection: Active database connection or connection-like object (asyncpg.Connection).
        redis_client: Redis client instance.
        settings: Optional AccountFeaturesSettings; defaults to repository configuration.
        bank_id: Commercial bank identifier to query.
        after_id: Cursor identifier; only rows with id > after_id are selected.
        clock: Optional domain ingestion clock. When provided, poll_start uses clock.now_ist()
            without querying DB now(); when None, falls back to fetchval("SELECT now()").
        run_id: Optional simulation run identifier. When clock is provided, run_id must also be
            provided to write run-scoped watermarks ('acctwm:{run_id}:{bank_id}:withdrawals').
            Run-scoped keys belong only to world-clock runs; if clock is None and run_id is given,
            raises ValueError.
        overlap_rows: Optional override of how many trailing rows before after_id are re-read.
            None uses cfg.account_event_overlap_rows (the live and catch-up behaviour). An int >= 0 uses
            that value; 0 reads only rows after the cursor. The drain's watermark-only call passes 0
            because it has no reason to re-apply rows it already applied. A negative value raises ValueError.

    Returns:
        The highest row ID processed, or after_id if no new rows were found.
    """
    if overlap_rows is not None and overlap_rows < 0:
        raise ValueError(f"overlap_rows must be >= 0 or None, got {overlap_rows}")

    if clock is not None and run_id is None:
        raise ValueError("When clock is provided, run_id must also be provided to write run-scoped watermarks")
    if clock is None and run_id is not None:
        raise ValueError(
            "run_id cannot be provided when clock is None; run-scoped keys belong only to world-clock runs"
        )

    cfg = settings if settings is not None else get_default_settings()
    if clock is not None:
        poll_start_db_time = clock.now_ist()
    else:
        poll_start_db_time = await connection.fetchval("SELECT now()")

    batch_size = cfg.account_event_batch_size
    overlap = cfg.account_event_overlap_rows if overlap_rows is None else overlap_rows
    start_id = max(0, after_id - overlap)

    rows = await connection.fetch(
        """
        SELECT id, bank_id, withdrawal_id, account_id, channel, amount_paise,
               status, failure_domain, event_time, balance_after_paise, recorded_at, arrival_time
        FROM bank_withdrawals
        WHERE bank_id = $1 AND id > $2
        ORDER BY id
        LIMIT $3
        """,
        bank_id,
        start_id,
        batch_size,
    )

    last_id = after_id
    if rows:
        for row in rows:
            record = withdrawal_from_row(row)
            apply_withdrawal(redis_client, record, cfg)
            row_id = int(row["id"])
            if row_id > last_id:
                last_id = row_id

    watermark_ts = int(to_unix_seconds(poll_start_db_time)) - cfg.commit_grace
    record_poller_watermark(redis_client, bank_id, "withdrawals", watermark_ts, run_id=run_id)

    return last_id

