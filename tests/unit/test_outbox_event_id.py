# File: tests/unit/test_outbox_event_id.py
# Purpose: Checks the event_id rule in api/services/outbox.py: the same idempotency key always gives the same event_id,
#          different keys give different ones, and no key gives a fresh random one. No database needed.
import uuid

from api.services.outbox import EVENT_ID_NAMESPACE, SCHEMA_VERSION, OutboxEventPayload, make_event_id


def test_same_key_gives_same_event_id():
    assert make_event_id("order-1001") == make_event_id("order-1001")


def test_event_id_is_the_uuid5_of_the_key():
    assert make_event_id("order-1001") == uuid.uuid5(EVENT_ID_NAMESPACE, "order-1001")


def test_different_keys_give_different_event_ids():
    assert make_event_id("order-1001") != make_event_id("order-1002")


def test_no_key_gives_a_new_random_id_each_time():
    first, second = make_event_id(None), make_event_id(None)
    assert first != second and first.version == 4


def test_empty_key_is_treated_as_no_key():
    assert make_event_id("") != make_event_id("")  # both random, so they differ


def test_payload_envelope_is_versioned_and_carries_the_event_id():
    payload = OutboxEventPayload(
        event_id=str(make_event_id("k")), transaction_id=str(uuid.uuid4()), occurred_at="2026-01-01T00:00:00+00:00",
        source="ecommerce",
        data={"account_id": "a", "merchant_id": None, "amount": "1.00", "fraud_score": None, "model_version": "v"},
    )
    body = payload.model_dump()
    assert body["schema_version"] == SCHEMA_VERSION == "1"
    assert set(body) == {"event_type", "schema_version", "event_id", "transaction_id", "occurred_at", "source", "data"}
