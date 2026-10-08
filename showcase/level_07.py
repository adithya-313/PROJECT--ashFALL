# File: showcase/level_07.py
# Purpose: The Level 7 showcase (Transactional Integrity, the outbox). Each check uses the REAL writer, relay and
#          consumer code, a throwaway database schema and a throwaway Kafka topic.
import random
import types
import uuid
from typing import Any

import asyncpg
from prometheus_client import REGISTRY

from api.services.demo_counter_consumer import count_one_event
from api.services.idempotent_consumer import parse_event, process_message
from api.services.outbox import make_event_id, record_transaction_with_outbox
from core.config import config
from showcase.docker_control import docker_available
from showcase.framework import FIXED_SEED, CheckDefinition, CheckOutcome, NotRun
from showcase.isolation import Workspace, get_showcase_database_url
from showcase.kafka_tools import read_events
from showcase.level07_helpers import (
    FRAUD_SCORE,
    MODEL_VERSION,
    OutageResult,
    build_outage_chart,
    check_order_rules_agree,
    clear_workspace_tables,
    drain_with_new_producer,
    fetch_outbox_event_ids,
    make_request,
    run_outage_scenario,
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
    "once the broker came back. When one message was too large for Kafka to accept, the relay tried it a few "
    "times, then parked it in a separate table and kept delivering the messages behind it. When the same message "
    "arrived twice, the consumer counted it only once. When a "
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


# ---------------------------------------------------------------------------
# Checks 2 and 6: the broker is stopped (once, then repeatedly)
# ---------------------------------------------------------------------------

def assert_nothing_lost(result: OutageResult, total_writes: int) -> None:
    """Fail unless every committed transaction was delivered (as a set of event ids) and nothing is left waiting."""
    assert result.committed == total_writes, f"{result.committed} transactions committed, expected {total_writes}"
    assert result.outbox_rows == total_writes, f"{result.outbox_rows} outbox rows, expected {total_writes}"
    assert result.unprocessed == 0, f"{result.unprocessed} outbox rows were never delivered"
    # A broker outage must never park a message or count an attempt.
    assert result.dead_letter_rows == 0, f"{result.dead_letter_rows} messages were parked during an outage"
    assert result.attempts_total == 0, f"{result.attempts_total} attempts were counted during an outage"
    missing = result.outbox_ids - result.kafka_ids
    extra = result.kafka_ids - result.outbox_ids
    assert not missing and not extra, f"Kafka differs from the outbox: {len(missing)} missing, {len(extra)} unexpected"


def make_outage_evidence(result: OutageResult) -> dict[str, Any]:
    """Timing and count facts. They change on every run, so they live in the evidence file, never in the key number."""
    return {
        "seconds_broker_was_down_per_kill": [round(value, 2) for value in result.kills.down_seconds],
        "seconds_until_broker_ready_per_kill": [round(value, 2) for value in result.kills.ready_seconds],
        "raw_messages_in_kafka": result.raw_count,
        "distinct_event_ids_in_kafka": len(result.kafka_ids),
    }


async def run_broker_stopped_check(workspace: Workspace) -> CheckOutcome:
    """Stop the broker once while writing and prove every transaction is still delivered afterwards."""
    problem = docker_available()
    if problem is not None:
        raise NotRun(problem)
    print("NOTE: this check stops the shared Redpanda broker for a few seconds; "
          "the dev outbox-relay will back off and recover on its own.")
    async with throwaway_topic() as topic:
        result = await run_outage_scenario(workspace, topic, EVENT_COUNT, 1)
    assert_nothing_lost(result, EVENT_COUNT)
    return CheckOutcome(
        key_number=f"{EVENT_COUNT} of {EVENT_COUNT} events delivered after outage",
        evidence=make_outage_evidence(result),
        chart=build_outage_chart(result.timeline, result.kills.bands),
    )


def make_repeated_kills_check(quick: bool) -> Any:
    """Build the repeated-kills check function for quick (2 kills) or full (5 kills) mode."""
    kill_count = 2 if quick else 5

    async def run_repeated_kills_check(workspace: Workspace) -> CheckOutcome:
        """Kill the broker several times during continuous writes and prove nothing is lost."""
        problem = docker_available()
        if problem is not None:
            raise NotRun(problem)
        print(f"NOTE: this check stops the shared Redpanda broker {kill_count} times for a few seconds each; "
              "the dev outbox-relay will back off and recover on its own.")
        async with throwaway_topic() as topic:
            result = await run_outage_scenario(workspace, topic, EVENT_COUNT, kill_count)
        assert_nothing_lost(result, EVENT_COUNT)
        return CheckOutcome(
            key_number=f"{kill_count} kills: {result.committed} committed, {len(result.kafka_ids)} delivered",
            evidence=make_outage_evidence(result),
            chart=build_outage_chart(result.timeline, result.kills.bands),
        )

    return run_repeated_kills_check


# ---------------------------------------------------------------------------
# Check 5: a poison message is parked and the rest is still delivered
# ---------------------------------------------------------------------------

# About 2 MB of padding: more than the 1 MB the Kafka client accepts, so the real producer rejects the message.
POISON_PADDING_BYTES = 2_000_000
POISON_ROW_POSITION = 1  # the 2nd outbox row (counting from 0)
DEAD_LETTER_METRIC = "outbox_dead_lettered_total"


async def make_second_row_too_large(conn: asyncpg.Connection) -> str:
    """Pad the payload of the 2nd outbox row to about 2 MB and return its event_id as text."""
    event_id = await conn.fetchval(
        """
        UPDATE outbox
        SET payload = payload || jsonb_build_object('padding', repeat('x', $1::integer))
        WHERE id = (SELECT id FROM outbox ORDER BY created_at, id OFFSET $2 LIMIT 1)
        RETURNING event_id
        """,
        POISON_PADDING_BYTES, POISON_ROW_POSITION,
    )
    return str(event_id)


async def run_poison_check(workspace: Workspace) -> CheckOutcome:
    """Make one message too large for Kafka and prove it is parked while the other four are still delivered."""
    conn = workspace.connection
    await clear_workspace_tables(conn)
    max_attempts = config.relay_max_attempts
    async with throwaway_topic() as topic:
        # Write 5 transactions, then make the 2nd message too large for the real Kafka client to send.
        rng = random.Random(FIXED_SEED)
        await write_transactions(conn, [make_request(i, rng, f"showcase-l07-poison-{i}") for i in range(5)])
        all_ids = [str(event_id) for event_id in await fetch_outbox_event_ids(conn)]
        poison_id = await make_second_row_too_large(conn)
        counter_before = REGISTRY.get_sample_value(DEAD_LETTER_METRIC) or 0.0

        # Run the real relay until nothing waits. The poison message is retried, then parked.
        await drain_with_new_producer(conn, topic)
        counter_delta = (REGISTRY.get_sample_value(DEAD_LETTER_METRIC) or 0.0) - counter_before
        events = await read_events(topic, 4, READ_IDLE_SECONDS)
        parked = await conn.fetch("SELECT event_id, error_type, error_message, attempt_count FROM outbox_dead_letter")
        rows = await conn.fetch("SELECT event_id, processed_at, dead_lettered_at FROM outbox ORDER BY created_at, id")

    delivered_ids = [key for _, key, _ in events]
    expected_ids = [event_id for event_id in all_ids if event_id != poison_id]
    assert delivered_ids == expected_ids, "the 4 other messages were not delivered in order"
    assert len(parked) == 1 and str(parked[0]["event_id"]) == poison_id, f"{len(parked)} messages were parked"
    assert parked[0]["attempt_count"] == max_attempts, f"attempt count was {parked[0]['attempt_count']}"
    assert parked[0]["error_type"] == "MessageSizeTooLargeError", f"error type was {parked[0]['error_type']}"
    for row in rows:
        is_poison = str(row["event_id"]) == poison_id
        assert (row["dead_lettered_at"] is not None) == is_poison, "wrong row marked as parked"
        assert (row["processed_at"] is None) == is_poison, "wrong row marked as processed"
    assert counter_delta == 1, f"the dead-letter counter rose by {counter_delta}, expected 1"
    return CheckOutcome(
        key_number="1 parked, 4 of 4 others delivered",
        evidence={"poison_event_id": poison_id, "delivered_event_ids": delivered_ids,
                  "attempt_count": parked[0]["attempt_count"], "error_type": parked[0]["error_type"],
                  "error_text": parked[0]["error_message"][:200], "counter_delta": counter_delta},
    )



def get_checks(quick: bool) -> list[CheckDefinition]:
    """The Level 7 checks, in the order they are shown. Quick mode only uses fewer broker kills in the last check."""
    return [
        CheckDefinition(
            "100 transactions produce 100 ordered events", "normal",
            "Every saved transaction is delivered to Kafka exactly once, in the order it was written.",
            run_normal_check, "The relay skipped, reordered or lost an event.",
            "api/services/outbox_relay.py (relay_one_batch)"),
        CheckDefinition(
            "Broker stopped mid-run", "edge",
            "If Kafka is stopped while transactions keep arriving, every event is still delivered once it is back.",
            run_broker_stopped_check, "The relay gave up, lost rows, or marked rows done before Kafka confirmed them.",
            "api/services/outbox_relay.py (relay_one_batch, mark_processed)"),
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
            "Poison message is parked and the rest still delivered", "edge",
            "A message Kafka refuses for its own sake is parked after a few tries and does not block the others.",
            run_poison_check, "The relay does not park poison messages, or counts outages as failed attempts.",
            "api/services/outbox_relay.py (handle_publish_failure, move_to_dead_letter)"),
        CheckDefinition(
            "Same input gives same event IDs", "consistency",
            "Sending the same idempotency key again always produces the same event ID.",
            run_determinism_check, "make_event_id or its namespace changed.",
            "api/services/outbox.py (make_event_id)"),
        CheckDefinition(
            "Repeated broker kills lose nothing", "edge",
            "Even when Kafka is killed several times during continuous writes, no transaction is lost.",
            make_repeated_kills_check(quick), "The relay does not recover after repeated broker failures.",
            "api/services/outbox_relay.py (run_relay backoff and recovery)"),
    ]
