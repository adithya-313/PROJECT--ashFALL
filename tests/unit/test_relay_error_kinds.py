# File: tests/unit/test_relay_error_kinds.py
# Purpose: Proves the relay tells "Kafka rejects this message" (a poison message, which counts towards parking it)
#          apart from "Kafka is unavailable" (which never counts). No Kafka is needed.
import pytest
from aiokafka.errors import (
    CorruptRecordException,
    InvalidTopicError,
    KafkaConnectionError,
    KafkaTimeoutError,
    MessageSizeTooLargeError,
    NodeNotReadyError,
    RecordTooLargeError,
    UnknownError,
)
from prometheus_client import REGISTRY

from api.services.outbox_relay import is_poison_error, note_dead_lettered


@pytest.mark.parametrize("error", [
    MessageSizeTooLargeError("too big"),
    RecordTooLargeError("too big"),
    InvalidTopicError("bad topic"),
    CorruptRecordException("corrupt"),
])
def test_message_specific_errors_are_poison(error):
    """The four 'this message is the problem' errors are recognised as poison."""
    assert is_poison_error(error) is True


@pytest.mark.parametrize("error", [
    KafkaConnectionError("broker down"),
    KafkaTimeoutError("slow"),
    NodeNotReadyError("not ready"),
    UnknownError("unknown"),
    RuntimeError("something else"),
])
def test_outages_and_unknown_errors_are_never_poison(error):
    """Outages, timeouts, unknown errors and plain exceptions never count as poison, so they never park a message."""
    assert is_poison_error(error) is False


def test_note_dead_lettered_raises_the_counter():
    """note_dead_lettered(2) raises the Prometheus counter by exactly 2."""
    name = "outbox_dead_lettered_total"
    before = REGISTRY.get_sample_value(name) or 0.0
    note_dead_lettered(2)
    assert REGISTRY.get_sample_value(name) == before + 2
