# File: showcase/level_07.py
# Purpose: The Level 7 showcase (Transactional Integrity, the outbox). Each check uses the REAL writer, relay and
#          consumer code, a throwaway database schema and a throwaway Kafka topic.
import random
import types
import uuid
from typing import Any

import asyncpg

from api.services.demo_counter_consumer import count_one_event
from api.services.idempotent_consumer import parse_event, process_message
from api.services.outbox import make_event_id, record_transaction_with_outbox
from core.config import config
from showcase.framework import FIXED_SEED, CheckDefinition, CheckOutcome
from showcase.isolation import Workspace, get_showcase_database_url
from showcase.kafka_tools import read_events
from showcase.level07_helpers import (
    FRAUD_SCORE,
    MODEL_VERSION,
    check_order_rules_agree,
    clear_workspace_tables,
    drain_with_new_producer,
    fetch_outbox_event_ids,
    make_request,
    throwaway_topic,
    write_transactions,
)
from showcase.preconditions import check_database, check_tcp_port

LEVEL_NUMBER = 7
LEVEL_TITLE = "Transactional Integrity (Outbox)"
NEEDS_DATABASE = True
WHAT_YOU_SAW = (
    "Every transaction was saved together with a small message about it (the outbox), in one all-or-nothing "
    "database step. A separate relay then delivered those messages to Kafka in the order they were written. "
    "When the Kafka broker was stopped, nothing was lost: the messages waited in the database and were delivered "
    "once the broker came back. When the same message arrived twice, the consumer counted it only once. When a "
    "write failed halfway, nothing was left behind, neither the transaction nor its message. And sending the same "
    "input again produced the same event IDs, so a retry can always be recognised."
)

CONSUMER_NAME = "showcase-l07"
EVENT_COUNT = 100
READ_IDLE_SECONDS = 10.0


async def find_missing_preconditions() -> list[str]:
    """List what is missing (Postgres, Kafka) in plain English. An empty list means we can run."""
    problems: list[str] = []
    database_problem = await check_database(get_showcase_database_url(), "docker compose up -d postgres")
    if database_problem is not None:
        problems.append(database_problem)
    host, _, port = config.kafka_bootstrap_servers.split(",")[0].partition(":")
    kafka_problem = check_tcp_port(host, int(port or 9092), "Kafka", "docker compose up -d redpanda")
    if kafka_problem is not None:
        problems.append(kafka_problem)
    return problems


def as_fake_record(offset: int, key_text: str, value: bytes) -> Any:
    """Wrap a message read from Kafka for the real process_message (it only needs value, partition, offset)."""
    return types.SimpleNamespace(value=value, key=key_text.encode("utf-8"), partition=0, offset=offset)


def parse_all_events(events: list[tuple[int, str, bytes]]) -> list[dict[str, Any]]:
    """Run the real parse_event on every message. It raises if any message is not a valid version-1 event."""
    return [parse_event(value) for _, _, value in events]


# ---------------------------------------------------------------------------
# Check 1: 100 transactions produce 100 ordered events
# ---------------------------------------------------------------------------

async def run_normal_check(workspace: Workspace) -> CheckOutcome:
    """Write 100 transactions, relay them, and prove Kafka holds 100 valid events in the database's order."""
    conn = workspace.connection
    await clear_workspace_tables(conn)
    async with throwaway_topic() as topic:
        # Write 100 transactions with the real atomic writer.
        rng = random.Random(FIXED_SEED)
        requests = [make_request(i, rng, f"showcase-l07-normal-{i}") for i in range(EVENT_COUNT)]
        await write_transactions(conn, requests)

        # Let the real relay deliver them, then read them back from Kafka.
        await drain_with_new_producer(conn, topic)
        events = await read_events(topic, EVENT_COUNT, READ_IDLE_SECONDS)
        parsed = parse_all_events(events)

        # Compare Kafka's order with the database's order.
        kafka_ids = [item["event_id"] for item in parsed]
        outbox_ids = [str(event_id) for event_id in await fetch_outbox_event_ids(conn)]
        waiting = await conn.fetchval("SELECT count(*) FROM outbox WHERE processed_at IS NULL")
        order_rules_agree = await check_order_rules_agree(conn)

    assert len(events) == EVENT_COUNT, f"expected {EVENT_COUNT} events in Kafka, read {len(events)}"
    assert kafka_ids == outbox_ids, "events in Kafka are not in the same order as the outbox rows"
    assert order_rules_agree, "ordering by (created_at, id) differs from ordering by id"
    assert waiting == 0, f"{waiting} outbox rows still have no processed_at"
    return CheckOutcome(
        key_number=f"{len(events)} of {EVENT_COUNT} events in order",
        evidence={"kafka_order": kafka_ids, "outbox_order": outbox_ids},
    )


# ---------------------------------------------------------------------------
# Check 3: duplicate delivery counted once
# ---------------------------------------------------------------------------

async def run_duplicate_check(workspace: Workspace) -> CheckOutcome:
    """Deliver one event twice (a simulated relay crash) and prove the consumer counts it once."""
    conn = workspace.connection
    await clear_workspace_tables(conn)
    async with throwaway_topic() as topic:
        # Write one transaction and let the relay publish it.
        request = make_request(0, random.Random(FIXED_SEED), "showcase-l07-duplicate-0")
        await write_transactions(conn, [request])
        first_publish = await drain_with_new_producer(conn, topic)

        # Simulate a relay that crashed after publishing but before saving "processed": forget the mark, publish again.
        await conn.execute("UPDATE outbox SET processed_at = NULL")
        second_publish = await drain_with_new_producer(conn, topic)
        events = await read_events(topic, 2, READ_IDLE_SECONDS)

        # Feed both deliveries to the real consumer code.
        outcomes = [
            await process_message(conn, CONSUMER_NAME, as_fake_record(offset, key, value), count_one_event)
            for offset, key, value in events
        ]
        processed_rows = await conn.fetchval("SELECT count(*) FROM processed_events")
        counter = await conn.fetchval("SELECT events_counted FROM demo_event_counts WHERE consumer_name = 'demo'")

    assert (first_publish, second_publish) == (1, 1), f"relay published {first_publish} then {second_publish}"
    assert len(events) == 2 and events[0][1] == events[1][1], "expected two messages with the same event_id"
    assert outcomes == [True, False], f"consumer results were {outcomes}, expected [True, False]"
    assert processed_rows == 1 and counter == 1, f"processed_events={processed_rows}, counter={counter}"
    return CheckOutcome(
        key_number="2 deliveries, counted 1 time",
        evidence={"event_id": events[0][1], "consumer_results": outcomes, "processed_events_rows": processed_rows,
                  "counter": counter},
    )


# ---------------------------------------------------------------------------
# Check 4: forced database error leaves nothing
# ---------------------------------------------------------------------------

async def force_failed_write(conn: asyncpg.Connection) -> str:
    """Make every outbox insert fail, try one write, and return the name of the database error class we saw."""
    await conn.execute("ALTER TABLE outbox ADD CONSTRAINT showcase_force_fail CHECK (false) NOT VALID")
    try:
        request = make_request(0, random.Random(FIXED_SEED), "showcase-l07-rollback-0")
        try:
            await record_transaction_with_outbox(conn, request, FRAUD_SCORE, MODEL_VERSION)
        except asyncpg.PostgresError as error:
            return type(error).__name__
        raise AssertionError("the write succeeded although the outbox insert was forced to fail")
    finally:
        # Always remove the trap, so later checks are not affected.
        await conn.execute("ALTER TABLE outbox DROP CONSTRAINT showcase_force_fail")


async def run_rollback_check(workspace: Workspace) -> CheckOutcome:
    """Force the outbox insert to fail and prove the transaction insert was rolled back too."""
    conn = workspace.connection
    await clear_workspace_tables(conn)
    async with throwaway_topic() as topic:
        # The transactions INSERT runs first and works; the outbox INSERT then fails; Postgres rolls back BOTH.
        error_class = await force_failed_write(conn)
        transaction_rows = await conn.fetchval("SELECT count(*) FROM transactions")
        outbox_rows = await conn.fetchval("SELECT count(*) FROM outbox")

        # Nothing was written, so the relay has nothing to send and the topic stays empty.
        await drain_with_new_producer(conn, topic)
        events = await read_events(topic, 1, 2.0)

    assert transaction_rows == 0 and outbox_rows == 0, f"transactions={transaction_rows}, outbox={outbox_rows}"
    assert len(events) == 0, f"{len(events)} events reached Kafka after a failed write"
    return CheckOutcome(
        key_number="0 rows, 0 events after a failed write",
        evidence={"database_error_class": error_class, "transactions_rows": transaction_rows,
                  "outbox_rows": outbox_rows, "kafka_events": len(events)},
    )


# ---------------------------------------------------------------------------
# Check 5: same input gives the same event ids
# ---------------------------------------------------------------------------

async def write_and_read_event_ids(conn: asyncpg.Connection, keys: list[str]) -> list[uuid.UUID]:
    """Write one request per key into empty tables and return the resulting event ids in outbox order."""
    await clear_workspace_tables(conn)
    rng = random.Random(FIXED_SEED)
    await write_transactions(conn, [make_request(i, rng, key) for i, key in enumerate(keys)])
    return await fetch_outbox_event_ids(conn)


async def run_determinism_check(workspace: Workspace) -> CheckOutcome:
    """Write the same 10 keyed requests twice and prove the event ids are identical, version 5, and key-derived."""
    conn = workspace.connection
    keys = [f"showcase-l07-same-{i}" for i in range(10)]
    first_ids = await write_and_read_event_ids(conn, keys)
    second_ids = await write_and_read_event_ids(conn, keys)

    # Two requests with NO key must get different random ids.
    await clear_workspace_tables(conn)
    rng = random.Random(FIXED_SEED)
    await write_transactions(conn, [make_request(0, rng, None), make_request(1, rng, None)])
    unkeyed_ids = await fetch_outbox_event_ids(conn)

    # Payload bytes are NOT compared: transaction_id (random) and occurred_at (database clock) legitimately differ.
    assert first_ids == second_ids, "the same keys produced different event ids"
    assert first_ids == [make_event_id(key) for key in keys], "event ids are not the uuid5 of the keys"
    assert all(event_id.version == 5 for event_id in first_ids), "event ids are not version 5"
    assert unkeyed_ids[0] != unkeyed_ids[1], "two requests without keys got the same event id"
    return CheckOutcome(
        key_number=f"{len(first_ids)} of {len(keys)} event ids identical",
        evidence={"first_run": [str(i) for i in first_ids], "second_run": [str(i) for i in second_ids],
                  "unkeyed": [str(i) for i in unkeyed_ids]},
    )


def get_checks(quick: bool) -> list[CheckDefinition]:
    """The Level 7 checks, in the order they are shown. Quick and full mode currently use the same checks."""
    return [
        CheckDefinition(
            "100 transactions produce 100 ordered events", "normal",
            "Every saved transaction is delivered to Kafka exactly once, in the order it was written.",
            run_normal_check, "The relay skipped, reordered or lost an event.",
            "api/services/outbox_relay.py (relay_one_batch)"),
        CheckDefinition(
            "Duplicate delivery counted once", "edge",
            "If the same event is delivered twice, the consumer counts it only once.",
            run_duplicate_check, "The consumer's processed_events guard is not working.",
            "api/services/idempotent_consumer.py (handle_once)"),
        CheckDefinition(
            "Forced database error leaves nothing", "negative",
            "If saving the message fails, the transaction is not saved either.",
            run_rollback_check, "The two inserts are no longer in one database transaction.",
            "api/services/outbox.py (insert_transaction_and_event)"),
        CheckDefinition(
            "Same input gives same event IDs", "consistency",
            "Sending the same idempotency key again always produces the same event ID.",
            run_determinism_check, "make_event_id or its namespace changed.",
            "api/services/outbox.py (make_event_id)"),
    ]
