"""
nationalsim/national_outbox.py

Transactional outbox publisher and bank consumer helpers for national systems.
Stages national registry events and 1930 hold requests in national_outbox,
relays them to Redpanda/Kafka keyed by bank, and provides a privacy-preserving
consumer helper that rejects foreign bank messages at the client boundary.
"""

import asyncio
import json
import time
import uuid
from dataclasses import dataclass
from datetime import datetime
from typing import Any

import asyncpg
from aiokafka import AIOKafkaConsumer, AIOKafkaProducer

from core.config import config
from core.event_clock import IST
from core.logging import get_logger

logger = get_logger(__name__)

ALLOWED_TOPICS: tuple[str, str] = ("registry.events", "hold.requests")
NATIONAL_SCHEMA_VERSION: str = "national-1"


def create_national_envelope(
    event_type: str,
    event_id: uuid.UUID,
    run_id: str,
    occurred_at: str,
    data: dict[str, Any],
) -> dict[str, Any]:
    """Wraps payload data into the standardized national outbox envelope.

    What it is for:
        Ensures all messages published to national Kafka topics (registry.events and hold.requests)
        share a uniform schema envelope (schema_version 'national-1') with explicit timestamps
        and provenance tracking under Revision 2.

    Inputs:
        event_type: String topic/event descriptor ('registry.events' or 'hold.requests').
        event_id: UUID uniquely identifying this event for deduplication.
        run_id: String simulation run identifier.
        occurred_at: ISO 8601 formatted timestamp string in IST timezone (+05:30).
        data: Domain payload dictionary containing event specifics (ids, hashes, amounts in paise).

    What it returns:
        Dictionary adhering to the 'national-1' envelope specification.

    How it works:
        Packages the metadata fields and payload data into a top-level dictionary.

    Errors and side effects:
        Raises ValueError if event_type is not 'registry.events' or 'hold.requests'.
        Raises ValueError if occurred_at does not contain IST timezone offset '+05:30'.

    Limits:
        Does not validate schema of the inner data dictionary.
    """
    if event_type not in ALLOWED_TOPICS:
        raise ValueError(f"Invalid event_type '{event_type}'. Must be one of {ALLOWED_TOPICS}")
    if "+05:30" not in occurred_at:
        raise ValueError(f"occurred_at '{occurred_at}' must include IST timezone offset '+05:30'")

    return {
        "event_type": event_type,
        "schema_version": NATIONAL_SCHEMA_VERSION,
        "event_id": str(event_id),
        "run_id": str(run_id),
        "occurred_at": str(occurred_at),
        "data": data,
    }


async def write_event(
    conn: asyncpg.Connection,
    topic: str,
    run_id: str,
    bank_id: int | str,
    payload: dict[str, Any],
) -> uuid.UUID:
    """Inserts a single event into the national_outbox table.

    What it is for:
        Provides transactional staging for national registry events and administrative hold requests
        prior to broker delivery, guaranteeing at-least-once publishing without losing events.

    Inputs:
        conn: Open asyncpg Connection (typically inside an active database transaction).
        topic: Destination Kafka topic string ('registry.events' or 'hold.requests').
        run_id: String simulation run identifier.
        bank_id: Bank identifier (integer index or string key) representing the owning institution.
        payload: Event payload dictionary. May be the raw domain data or a pre-formed envelope.

    What it returns:
        UUID of the written (or deduplicated) outbox event.

    How it works:
        1. Validates that topic is allowed ('registry.events' or 'hold.requests').
        2. Extracts or deterministically derives event_id (supporting complaint_id deduplication).
        3. Wraps the payload in the 'national-1' envelope if not already wrapped.
        4. Inserts the row into national_outbox with ON CONFLICT (event_id) DO NOTHING.
           Why ON CONFLICT DO NOTHING: Deduplication at the database boundary guarantees at-most-once
           staging. When upstream replay passes, retried batch iterations, or parallel workers submit
           the same logical event (sharing event_id or derived complaint_id), the database safely ignores
           the duplicate rather than producing multiple Kafka messages or raising constraint errors.

    Errors and side effects:
        Raises ValueError if topic is invalid.
        Executes an INSERT statement on national_outbox.

    Limits:
        Relies on Postgres unique constraint on event_id for idempotency.
    """
    if topic not in ALLOWED_TOPICS:
        raise ValueError(f"Invalid topic '{topic}'. Must be one of {ALLOWED_TOPICS}")

    # Extract or derive event_id
    if "event_id" in payload:
        event_id = uuid.UUID(str(payload["event_id"]))
    elif "complaint_id" in payload:
        # Deterministic UUIDv5 from complaint_id guarantees repeated calls share event_id
        event_id = uuid.uuid5(uuid.NAMESPACE_DNS, f"{run_id}:{payload['complaint_id']}")
    elif isinstance(payload.get("data"), dict) and "complaint_id" in payload["data"]:
        event_id = uuid.uuid5(uuid.NAMESPACE_DNS, f"{run_id}:{payload['data']['complaint_id']}")
    else:
        event_id = uuid.uuid4()

    # Form envelope if not already enveloped
    if payload.get("schema_version") == NATIONAL_SCHEMA_VERSION:
        envelope = payload
    else:
        occurred_at = payload.get("occurred_at")
        if not occurred_at:
            occurred_at = datetime.now(IST).isoformat()
        envelope = create_national_envelope(
            event_type=topic,
            event_id=event_id,
            run_id=run_id,
            occurred_at=str(occurred_at),
            data=payload.get("data", payload),
        )

    bank_key = f"{run_id}:{bank_id}"
    payload_json = json.dumps(envelope)

    await conn.execute(
        """
        INSERT INTO national_outbox (event_id, run_id, topic, bank_key, schema_version, payload)
        VALUES ($1, $2, $3, $4, $5, $6::jsonb)
        ON CONFLICT (event_id) DO NOTHING
        """,
        event_id,
        run_id,
        topic,
        bank_key,
        NATIONAL_SCHEMA_VERSION,
        payload_json,
    )

    return event_id


async def relay_national_outbox(
    conn: asyncpg.Connection,
    producer: AIOKafkaProducer,
    batch_size: int = 100,
) -> int:
    """Relays unprocessed national outbox rows to Kafka, marking rows processed only after broker ACK.

    What it is for:
        Transfers staged national outbox events to Kafka (Redpanda) in FIFO order while
        preserving transactional safety: rows are marked processed strictly after the message
        broker acknowledges persistent receipt.

    Inputs:
        conn: Open asyncpg Connection to the application database.
        producer: Started AIOKafkaProducer instance with acks='all'.
        batch_size: Maximum number of unprocessed rows to publish in this batch.

    What it returns:
        Integer count of successfully published and acknowledged messages.

    How it works:
        1. Queries national_outbox for rows WHERE processed_at IS NULL, ordered by created_at, id.
        2. Uses FOR UPDATE SKIP LOCKED to prevent concurrent relay instances from contending.
        3. For each row, sends message to row['topic'] with message key = bank_key (<run_id>:<bank_id>).
        4. Awaits producer.send_and_wait confirmation from Kafka (acks='all').
        5. Updates national_outbox SET processed_at = now() for the acknowledged row.
        6. Commits the transaction.

    Errors and side effects:
        Raises Exception if Kafka or Postgres operations fail. Rows already acknowledged
        prior to a failure remain marked as processed; subsequent rows remain unprocessed.

    Limits:
        Does not perform poison message dead-letter routing (reserved for transaction outbox).
    """
    published = 0
    async with conn.transaction():
        rows = await conn.fetch(
            """
            SELECT id, event_id, run_id, topic, bank_key, schema_version, payload
            FROM national_outbox
            WHERE processed_at IS NULL
            ORDER BY created_at, id
            LIMIT $1
            FOR UPDATE SKIP LOCKED
            """,
            batch_size,
        )

        for row in rows:
            topic = row["topic"]
            key_bytes = row["bank_key"].encode("utf-8")
            # asyncpg returns JSONB as text or dict
            payload_str = row["payload"] if isinstance(row["payload"], str) else json.dumps(row["payload"])
            val_bytes = payload_str.encode("utf-8")

            # Publish to broker and wait for acknowledgement
            await producer.send_and_wait(topic, key=key_bytes, value=val_bytes)

            # Mark processed ONLY after acknowledgement
            await conn.execute("UPDATE national_outbox SET processed_at = now() WHERE id = $1", row["id"])
            published += 1

    return published


@dataclass(frozen=True)
class BankConsumerResult:
    """Audit outcome of a bank-scoped Kafka consumption session.

    What it is for:
        Encapsulates the messages accepted by an institution along with an audit count
        of foreign messages encountered on shared multi-tenant topics.

    Inputs:
        accepted_messages: List of decoded payload dictionaries whose message key matched the bank.
        foreign_keys_count: Count of messages whose key did not match and whose payload was ignored.

    What it returns:
        Immutable BankConsumerResult instance.

    How it works:
        Stores accepted messages and foreign message counters.

    Errors and side effects:
        None. Dataclass is frozen.

    Limits:
        In-memory representation of a single polling batch.
    """

    accepted_messages: list[dict[str, Any]]
    foreign_keys_count: int


async def consume_own_messages(
    bank_id: int | str,
    run_id: str,
    topic: str,
    bootstrap_servers: str | None = None,
    expected_count: int | None = None,
    idle_timeout_seconds: float = 3.0,
) -> BankConsumerResult:
    """Consumes Kafka messages for a specific bank, ignoring and counting foreign keys without reading payloads.

    What it is for:
        Enforces bank-level isolation on shared topics in development environments lacking broker ACLs.
        A bank consumer reads only its own messages, rejecting rival bank traffic at the boundary.

    Inputs:
        bank_id: Institution identifier (e.g. 0, 1, or string).
        run_id: Simulation run identifier.
        topic: Kafka topic to consume from ('registry.events' or 'hold.requests').
        bootstrap_servers: Optional broker address string; defaults to core config setting.
        expected_count: Optional target count of accepted messages to collect before returning.
        idle_timeout_seconds: Duration to wait with no incoming records before stopping.

    What it returns:
        BankConsumerResult containing accepted messages and count of unread foreign messages.

    How it works:
        1. Starts an AIOKafkaConsumer on the specified topic with a fresh throwaway group_id.
        2. Expected wire key is formatted as '<run_id>:<bank_id>'.
        3. Inspects record.key for each message received.
        4. If key matches expected_key, decodes and parses record.value JSON payload into accepted_messages.
        5. If key does not match, increments foreign_keys_count and logs a warning WITHOUT decoding
           or inspecting record.value. This enforces banking confidentiality across shared partitions.
        6. Stops when expected_count accepted or idle timeout expires.

    Errors and side effects:
        Logs warnings for foreign message keys. Does not commit offsets.

    Limits:
        Client-side privacy filter; production deployments require broker SASL/topic ACLs.
    """
    servers = bootstrap_servers or config.kafka_bootstrap_servers
    consumer = AIOKafkaConsumer(
        topic,
        bootstrap_servers=servers,
        group_id=f"bank-consumer-{bank_id}-{uuid.uuid4().hex[:12]}",
        auto_offset_reset="earliest",
        enable_auto_commit=False,
    )
    await consumer.start()

    accepted_messages: list[dict[str, Any]] = []
    foreign_keys_count = 0
    expected_key = f"{run_id}:{bank_id}"

    try:
        last_message_time = time.monotonic()
        while True:
            batches = await consumer.getmany(timeout_ms=500, max_records=50)
            if batches:
                for records in batches.values():
                    for record in records:
                        last_message_time = time.monotonic()
                        key_text = record.key.decode("utf-8") if record.key else ""
                        if key_text == expected_key:
                            payload_dict = json.loads(record.value.decode("utf-8"))
                            accepted_messages.append(payload_dict)
                        else:
                            foreign_keys_count += 1
                            logger.warning(
                                "Ignored message with foreign bank key without reading payload",
                                expected_key=expected_key,
                                foreign_key=key_text,
                            )
            if expected_count is not None and len(accepted_messages) >= expected_count:
                break
            if time.monotonic() - last_message_time >= idle_timeout_seconds:
                break
            await asyncio.sleep(0.05)
    finally:
        await consumer.stop()

    return BankConsumerResult(
        accepted_messages=accepted_messages,
        foreign_keys_count=foreign_keys_count,
    )
