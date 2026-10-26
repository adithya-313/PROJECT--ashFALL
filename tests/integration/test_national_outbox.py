# File: tests/integration/test_national_outbox.py
# Purpose: Integration tests proving national outbox publishing and bank message isolation against live stack:
#          - hold request written once is published once, idempotent over multiple relay passes
#          - duplicate complaint_id written twice produces exactly one outbox row and one message
#          - bank_1 consumer receives only bank_1's messages while bank_0's message is counted foreign and unread
#          - message key on the wire exactly equals "<run_id>:<bank_id>"
import uuid

import pytest
from aiokafka import AIOKafkaConsumer

from api.services.outbox_relay import start_producer
from core.config import config
from nationalsim.keyed_hash import compute_identifier_hash
from nationalsim.national_outbox import (
    consume_own_messages,
    relay_national_outbox,
    resolve_account_ref,
    write_event,
)

pytestmark = [pytest.mark.anyio, pytest.mark.needs_stack]


async def test_hold_request_published_once_even_if_relay_runs_twice(isolated_schema) -> None:
    """A hold request staged in national_outbox is published once; second relay run does not republish."""
    connection, _ = isolated_schema
    run_id = f"run-idemp-{uuid.uuid4().hex[:8]}"
    bank_id = 1
    topic = "hold.requests"
    i4c_key = "local-dev-i4c-key-001"
    account_ref = compute_identifier_hash(
        key=i4c_key,
        run_id=run_id,
        identifier_type="account",
        raw_value="acc_target_01",
        bank_index=bank_id,
    )

    hold_payload = {
        "complaint_id": f"comp-{uuid.uuid4().hex[:6]}",
        "payment_id": f"pay-{uuid.uuid4().hex[:6]}",
        "amount_paise": 750000,
        "account_ref": account_ref,
    }

    # 1. Stage the hold request
    event_id = await write_event(connection, topic, run_id, bank_id, hold_payload)
    assert isinstance(event_id, uuid.UUID)

    producer = await start_producer()
    try:
        # 2. First relay run publishes the message
        published_1 = await relay_national_outbox(connection, producer)
        assert published_1 == 1

        # 3. Second relay run sees no unprocessed rows and publishes nothing
        published_2 = await relay_national_outbox(connection, producer)
        assert published_2 == 0
    finally:
        await producer.stop()

    # 4. Consumer for bank 1 retrieves exactly the single published message
    consumer_res = await consume_own_messages(
        bank_id=bank_id,
        run_id=run_id,
        topic=topic,
        expected_count=1,
        idle_timeout_seconds=2.0,
    )
    assert len(consumer_res.accepted_messages) == 1
    msg = consumer_res.accepted_messages[0]
    assert msg["event_id"] == str(event_id)
    assert msg["data"]["complaint_id"] == hold_payload["complaint_id"]
    assert msg["data"]["amount_paise"] == 750000
    assert msg["data"]["account_ref"] == account_ref
    assert "account_id" not in msg["data"]

    # 5. Receiving bank resolves account_ref back to local account
    resolved = resolve_account_ref(
        account_ref=msg["data"]["account_ref"],
        bank_accounts=["acc_other_01", "acc_target_01"],
        run_id=run_id,
        bank_index=bank_id,
        i4c_key=i4c_key,
    )
    assert resolved == "acc_target_01"


async def test_same_complaint_id_written_twice_gives_one_message(isolated_schema) -> None:
    """Submitting the same complaint_id twice is deduplicated by event_id at the DB boundary."""
    connection, _ = isolated_schema
    run_id = f"run-dedup-{uuid.uuid4().hex[:8]}"
    bank_id = 0
    topic = "hold.requests"
    complaint_id = f"c_repeat_{uuid.uuid4().hex[:6]}"
    i4c_key = "local-dev-i4c-key-001"
    account_ref = compute_identifier_hash(
        key=i4c_key,
        run_id=run_id,
        identifier_type="account",
        raw_value="acc_victim_01",
        bank_index=bank_id,
    )

    payload_a = {
        "complaint_id": complaint_id,
        "amount_paise": 500000,
        "account_ref": account_ref,
    }
    payload_b = {
        "complaint_id": complaint_id,
        "amount_paise": 500000,
        "account_ref": account_ref,
    }

    # Write twice with identical complaint_id
    id_a = await write_event(connection, topic, run_id, bank_id, payload_a)
    id_b = await write_event(connection, topic, run_id, bank_id, payload_b)
    assert id_a == id_b

    # Verify only one row was inserted into national_outbox
    row_count = await connection.fetchval(
        "SELECT count(*) FROM national_outbox WHERE run_id = $1 AND event_id = $2",
        run_id,
        id_a,
    )
    assert row_count == 1

    producer = await start_producer()
    try:
        published = await relay_national_outbox(connection, producer)
        assert published == 1
    finally:
        await producer.stop()

    consumer_res = await consume_own_messages(
        bank_id=bank_id,
        run_id=run_id,
        topic=topic,
        expected_count=1,
        idle_timeout_seconds=2.0,
    )
    assert len(consumer_res.accepted_messages) == 1
    assert consumer_res.accepted_messages[0]["event_id"] == str(id_a)


async def test_write_event_rejects_raw_account_id(isolated_schema) -> None:
    """Proves national outbox rejects payloads containing raw account_id."""
    connection, _ = isolated_schema
    run_id = f"run-reject-{uuid.uuid4().hex[:8]}"
    with pytest.raises(ValueError, match="National outbox payloads must carry 'account_ref'"):
        await write_event(
            connection,
            topic="hold.requests",
            run_id=run_id,
            bank_id=1,
            payload={"complaint_id": "c1", "account_id": "acc_raw_01", "amount_paise": 100},
        )


async def test_bank_1_consumer_receives_only_bank_1_messages_and_bank_0_is_foreign(isolated_schema) -> None:
    """Bank 1 consumer accepts only its own key and counts bank 0 messages as foreign without reading payload."""
    connection, _ = isolated_schema
    run_id = f"run-isol-{uuid.uuid4().hex[:8]}"
    topic = "registry.events"

    # Stage an event for Bank 0 and an event for Bank 1
    await write_event(
        connection,
        topic,
        run_id,
        bank_id=0,
        payload={"report_id": f"rep-b0-{uuid.uuid4().hex[:6]}", "layer": 1},
    )
    await write_event(
        connection,
        topic,
        run_id,
        bank_id=1,
        payload={"report_id": f"rep-b1-{uuid.uuid4().hex[:6]}", "layer": 2},
    )

    producer = await start_producer()
    try:
        published = await relay_national_outbox(connection, producer)
        assert published == 2
    finally:
        await producer.stop()

    # Bank 1 consumes from the shared topic
    consumer_res = await consume_own_messages(
        bank_id=1,
        run_id=run_id,
        topic=topic,
        expected_count=1,
        idle_timeout_seconds=2.0,
    )

    # Bank 1 accepts only its own message
    assert len(consumer_res.accepted_messages) == 1
    accepted_data = consumer_res.accepted_messages[0]["data"]
    assert accepted_data["layer"] == 2
    assert "rep-b1-" in accepted_data["report_id"]

    # Bank 0 message was detected and logged as foreign, not ingested
    assert consumer_res.foreign_keys_count >= 1


async def test_key_on_wire_equals_run_id_colon_bank_id(isolated_schema) -> None:
    """Messages on Kafka carry key matching '<run_id>:<bank_id>' exactly."""
    connection, _ = isolated_schema
    run_id = f"run-wire-{uuid.uuid4().hex[:8]}"
    bank_id = "testbank99"
    topic = "registry.events"

    await write_event(
        connection,
        topic,
        run_id,
        bank_id=bank_id,
        payload={"event_id": str(uuid.uuid4()), "status": "active"},
    )

    producer = await start_producer()
    try:
        published = await relay_national_outbox(connection, producer)
        assert published == 1
    finally:
        await producer.stop()

    # Raw inspection of the record key on the Kafka wire
    raw_consumer = AIOKafkaConsumer(
        topic,
        bootstrap_servers=config.kafka_bootstrap_servers,
        group_id=f"raw-inspector-{uuid.uuid4().hex[:12]}",
        auto_offset_reset="earliest",
        enable_auto_commit=False,
    )
    await raw_consumer.start()
    found_wire_key = None
    try:
        batches = await raw_consumer.getmany(timeout_ms=3000, max_records=100)
        for records in batches.values():
            for record in records:
                if record.key:
                    k = record.key.decode("utf-8")
                    if k == f"{run_id}:{bank_id}":
                        found_wire_key = k
                        break
    finally:
        await raw_consumer.stop()

    assert found_wire_key == f"{run_id}:{bank_id}", (
        f"Wire message key must equal '{run_id}:{bank_id}', got {found_wire_key}"
    )
