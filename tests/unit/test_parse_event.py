# File: tests/unit/test_parse_event.py
# Purpose: Checks parse_event in api/services/idempotent_consumer.py accepts a valid v1 event and rejects anything
#          else with a clear error. No database or Kafka needed.
import uuid

import pytest

from api.services.idempotent_consumer import InvalidEventError, parse_event


def test_valid_event_is_returned_as_a_dict():
    event_id = str(uuid.uuid4())
    event = parse_event(f'{{"schema_version": "1", "event_id": "{event_id}", "extra": 5}}'.encode())
    assert event["event_id"] == event_id and event["extra"] == 5


@pytest.mark.parametrize("bad_message", [
    b"not json at all",
    b"[1, 2, 3]",  # valid JSON but not an object
    b'{"schema_version": "2", "event_id": "6d9a3461-75d3-5bae-9a26-263072b6ef2a"}',  # unknown version
    b'{"event_id": "6d9a3461-75d3-5bae-9a26-263072b6ef2a"}',  # no version
    b'{"schema_version": "1"}',  # no event_id
    b'{"schema_version": "1", "event_id": "not-a-uuid"}',
    b"\xff\xfe",  # not even valid text
])
def test_invalid_messages_are_rejected_with_a_clear_error(bad_message):
    with pytest.raises(InvalidEventError):
        parse_event(bad_message)
