# File: showcase/level07_helpers.py
# Purpose: Small building blocks for the Level 7 showcase checks: fake requests, a second database connection,
#          clearing tables, draining the relay, and a throwaway topic that is always deleted afterwards.
import random
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import asyncpg

from api.routers.scoring import TransactionRequest
from api.services.outbox import StoredTransaction, record_transaction_with_outbox
from api.services.outbox_relay import relay_one_batch
from showcase.isolation import Workspace
from showcase.kafka_tools import create_topic, delete_topic, make_producer, make_topic_name

MODEL_VERSION = "showcase-model-1"
FRAUD_SCORE = 0.25
RELAY_BATCH_SIZE = 50
# Safety stop for draining: a healthy drain finishes long before this many rounds.
MAX_DRAIN_ROUNDS = 1000


def make_request(index: int, seed_rng: random.Random, key: str | None) -> TransactionRequest:
    """Build a fake transaction. The random generator is seeded by the caller, so the same seed gives the same data."""
    amount = round(seed_rng.uniform(1.0, 500.0), 2)
    return TransactionRequest(
        amount=amount,
        account_id=f"showcase-account-{index % 10}",
        merchant_id=f"showcase-merchant-{index % 7}",
        device_fingerprint=f"showcase-device-{index}",
        geo_coordinates="40.7128,-74.0060",
        idempotency_key=key,
    )


async def open_second_connection(workspace: Workspace) -> asyncpg.Connection:
    """
    Open another connection into the SAME throwaway schema.

    The relay and the writer need separate connections, because one connection can only do one thing at a time.
    """
    return await asyncpg.connect(dsn=workspace.dsn, timeout=5, server_settings={"search_path": workspace.schema_name})


async def clear_workspace_tables(conn: asyncpg.Connection) -> None:
    """Empty every table in the workspace in one statement, so each check starts from nothing."""
    await conn.execute("TRUNCATE transactions, outbox, processed_events, demo_event_counts")


async def write_transactions(
    conn: asyncpg.Connection, requests: list[TransactionRequest],
) -> list[StoredTransaction]:
    """Write each request with the real atomic writer, one after the other."""
    stored: list[StoredTransaction] = []
    for request in requests:
        stored.append(await record_transaction_with_outbox(conn, request, FRAUD_SCORE, MODEL_VERSION))
    return stored


async def drain_relay(
    conn: asyncpg.Connection, producer: object, topic: str, batch_size: int = RELAY_BATCH_SIZE,
) -> int:
    """
    Run the real relay batch by batch until nothing is waiting, or until a batch reports an error.

    Returns how many events were published in total.
    """
    total_published = 0
    for _ in range(MAX_DRAIN_ROUNDS):
        result = await relay_one_batch(conn, producer, batch_size, topic)  # type: ignore[arg-type]
        total_published += result.published
        if result.error is not None or result.waiting == 0:
            break
    return total_published


async def drain_with_new_producer(conn: asyncpg.Connection, topic: str) -> int:
    """Create a producer, drain the relay with it, and always stop the producer."""
    producer = await make_producer()
    try:
        return await drain_relay(conn, producer, topic)
    finally:
        await producer.stop()


async def fetch_outbox_event_ids(conn: asyncpg.Connection) -> list[uuid.UUID]:
    """All outbox event ids in the relay's own order (created_at, then id)."""
    rows = await conn.fetch("SELECT event_id FROM outbox ORDER BY created_at, id")
    return [row["event_id"] for row in rows]


async def check_order_rules_agree(conn: asyncpg.Connection) -> bool:
    """True when ordering by (created_at, id) gives the same sequence as ordering by id alone."""
    by_time = [row["id"] for row in await conn.fetch("SELECT id FROM outbox ORDER BY created_at, id")]
    by_id = [row["id"] for row in await conn.fetch("SELECT id FROM outbox ORDER BY id")]
    return by_time == by_id


@asynccontextmanager
async def throwaway_topic() -> AsyncIterator[str]:
    """Create a fresh topic, hand out its name, and ALWAYS delete it afterwards (even if the check failed)."""
    topic = make_topic_name()
    await create_topic(topic)
    try:
        yield topic
    finally:
        await delete_topic(topic)
