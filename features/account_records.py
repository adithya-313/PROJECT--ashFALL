# File: features/account_records.py
# Purpose: Level 15a data contracts, parsing, validation, and serialization for account feature records.
# Issue: Level 15a account feature updater (plan Task 1, C4, K2).
"""
WHAT: Provides immutable data representations (PaymentSideRecord and AccountEventRecord)
for commercial bank transaction sides and account lifecycle events, with strict schema validation
and serialization routines.

WHY: Enforces zero-trust boundaries at feature ingestion. All Kafka schema-2 payment events
and Postgres bank account events are validated for timezone-awareness, run-ID identity integrity,
and required banking fields before admission into Redis sliding-window structures.
"""

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Any

import orjson

from features.account_keys import extract_run_id


class InvalidAccountEventError(ValueError):
    """
    WHAT: Dedicated error raised when an incoming event payload fails schema, type, or integrity checks.
    WHY: Distinguishes malformed or non-schema-2 events from general ValueError programming errors.
    """


@dataclass(frozen=True)
class PaymentSideRecord:
    """
    Immutable representation of a single commercial bank transaction side (debit or credit).

    Attributes:
        run_id: 16-hex simulation run identifier extracted from payment_id.
        bank_id: Tenant commercial bank identifier.
        account_id: Customer account identifier.
        payment_id: Global payment transaction identifier.
        side: Transaction side ('debit' or 'credit').
        counterparty_bank_id: Remote bank identifier of the counterparty.
        counterparty_vpa: Optional Virtual Payment Address of counterparty.
        amount_paise: Transaction amount in integer paise.
        status: Payment outcome status ('success' or 'failed').
        failure_domain: Optional failure classification string if failed.
        payee_type: Optional classification ('merchant' or 'person').
        channel: Optional delivery channel (e.g. 'upi_app').
        sender_device_fingerprint: Optional device fingerprint (debit-only).
        sender_os_generation: Optional OS generation number (debit-only).
        event_time: Timezone-aware timestamp of payment occurrence.
        arrival_time: Timezone-aware timestamp of ingestion/storage (recorded_at).
    """

    run_id: str
    bank_id: str
    account_id: str
    payment_id: str
    side: str
    counterparty_bank_id: str
    counterparty_vpa: str | None
    amount_paise: int
    status: str
    failure_domain: str | None
    payee_type: str | None
    channel: str | None
    sender_device_fingerprint: str | None
    sender_os_generation: int | None
    event_time: datetime
    arrival_time: datetime


@dataclass(frozen=True)
class AccountEventRecord:
    """
    Immutable representation of an account lifecycle event (device change, SIM swap, opening).

    Attributes:
        run_id: 16-hex simulation run identifier extracted from event_id.
        bank_id: Tenant commercial bank identifier.
        account_id: Customer account identifier.
        event_id: Unique event identifier.
        kind: Account lifecycle event classification.
        event_time: Timezone-aware timestamp of event occurrence.
        arrival_time: Timezone-aware database creation timestamp (created_at).
        details: Arbitrary event metadata dictionary.
    """

    run_id: str
    bank_id: str
    account_id: str
    event_id: str
    kind: str
    event_time: datetime
    arrival_time: datetime
    details: dict[str, Any]


def _ensure_timezone_aware(dt_val: datetime, field_name: str) -> datetime:
    """Validates that a datetime object has an explicit timezone attached."""
    if dt_val.tzinfo is None or dt_val.tzinfo.utcoffset(dt_val) is None:
        raise InvalidAccountEventError(f"Field {field_name!r} must be timezone-aware; got {dt_val!r}")
    return dt_val


def parse_upi_event(message_value: bytes) -> PaymentSideRecord:
    """
    WHAT: Parses and validates a raw Kafka message value representing a schema-2 UPI payment event.

    WHY: Serves as the primary ingress parser for the payment feature updater. Validates that
    the message adheres strictly to schema_version '2' and event_type 'upi.payment.recorded',
    verifies timezone-awareness of all timestamps, and extracts the simulation run ID.

    Args:
        message_value: Raw bytes received from the Kafka topic.

    Returns:
        Validated, immutable PaymentSideRecord.

    Raises:
        InvalidAccountEventError: If payload cannot be parsed, has incorrect schema version or
            event type, contains naive timestamps, or carries an invalid run prefix.
    """
    try:
        payload = orjson.loads(message_value)
    except Exception as exc:
        raise InvalidAccountEventError(f"Failed to decode JSON payload: {exc}") from exc

    if not isinstance(payload, dict):
        raise InvalidAccountEventError("UPI event payload must be a JSON object")

    # Schema version and event type enforcement
    schema_ver = payload.get("schema_version")
    if schema_ver != "2":
        raise InvalidAccountEventError(f"Unsupported schema_version {schema_ver!r}; expected '2'")

    event_type = payload.get("event_type")
    if event_type != "upi.payment.recorded":
        raise InvalidAccountEventError(f"Unexpected event_type {event_type!r}; expected 'upi.payment.recorded'")

    # Extract payment ID and validate run prefix
    payment_id = payload.get("payment_id")
    if not isinstance(payment_id, str) or not payment_id:
        raise InvalidAccountEventError("Missing or empty 'payment_id' in UPI event envelope")

    try:
        run_id = extract_run_id(payment_id)
    except ValueError as exc:
        raise InvalidAccountEventError(f"Invalid run ID in payment_id {payment_id!r}: {exc}") from exc

    # Parse and validate envelope timestamps
    event_time_raw = payload.get("event_time")
    if not event_time_raw:
        raise InvalidAccountEventError("Missing 'event_time' in UPI event envelope")
    try:
        event_time = datetime.fromisoformat(event_time_raw)
    except Exception as exc:
        raise InvalidAccountEventError(f"Invalid ISO event_time {event_time_raw!r}: {exc}") from exc
    _ensure_timezone_aware(event_time, "event_time")

    recorded_at_raw = payload.get("recorded_at")
    if not recorded_at_raw:
        raise InvalidAccountEventError("Missing 'recorded_at' in UPI event envelope")
    try:
        recorded_at = datetime.fromisoformat(recorded_at_raw)
    except Exception as exc:
        raise InvalidAccountEventError(f"Invalid ISO recorded_at {recorded_at_raw!r}: {exc}") from exc
    _ensure_timezone_aware(recorded_at, "recorded_at")

    # Transaction side
    side = payload.get("side")
    if side not in ("debit", "credit"):
        raise InvalidAccountEventError(f"Invalid transaction side {side!r}; must be 'debit' or 'credit'")

    # Extract nested business data
    data = payload.get("data")
    if not isinstance(data, dict):
        raise InvalidAccountEventError("Missing or invalid 'data' payload object")

    bank_id = data.get("bank_id")
    if not isinstance(bank_id, str) or not bank_id:
        raise InvalidAccountEventError("Missing 'bank_id' in data payload")

    account_id = data.get("account_id")
    if not isinstance(account_id, str) or not account_id:
        raise InvalidAccountEventError("Missing 'account_id' in data payload")

    counterparty_bank_id = data.get("counterparty_bank_id")
    if not isinstance(counterparty_bank_id, str) or not counterparty_bank_id:
        raise InvalidAccountEventError("Missing 'counterparty_bank_id' in data payload")

    amount_paise = data.get("amount_paise")
    if amount_paise is None or not isinstance(amount_paise, int) or amount_paise <= 0:
        raise InvalidAccountEventError(f"Invalid 'amount_paise' {amount_paise!r}; must be integer > 0")

    status = data.get("status")
    if status not in ("success", "failed"):
        raise InvalidAccountEventError(f"Invalid 'status' {status!r}; must be 'success' or 'failed'")

    return PaymentSideRecord(
        run_id=run_id,
        bank_id=bank_id,
        account_id=account_id,
        payment_id=payment_id,
        side=side,
        counterparty_bank_id=counterparty_bank_id,
        counterparty_vpa=data.get("counterparty_vpa"),
        amount_paise=amount_paise,
        status=status,
        failure_domain=data.get("failure_domain"),
        payee_type=data.get("payee_type"),
        channel=data.get("channel"),
        sender_device_fingerprint=data.get("sender_device_fingerprint"),
        sender_os_generation=data.get("sender_os_generation"),
        event_time=event_time,
        arrival_time=recorded_at,  # arrival_time mapped from ashFALL's storage timestamp
    )


def account_event_from_row(row: Mapping[str, Any]) -> AccountEventRecord:
    """
    WHAT: Converts a database row from bank_account_events into an AccountEventRecord.

    WHY: Maps database columns to immutable domain records. Ensures event_time and created_at
    are converted to timezone-aware UTC datetimes, assigns created_at as arrival_time,
    and extracts the simulation run ID from event_id.

    Args:
        row: Mapping of column names to values from bank_account_events.

    Returns:
        Validated AccountEventRecord.

    Raises:
        ValueError: If event_id has an invalid run prefix or required columns are missing.
    """
    event_id = str(row["event_id"])
    run_id = extract_run_id(event_id)

    bank_id = str(row["bank_id"])
    account_id = str(row["account_id"])
    kind = str(row["kind"])

    raw_event_time = row["event_time"]
    if isinstance(raw_event_time, str):
        event_time = datetime.fromisoformat(raw_event_time)
    else:
        event_time = raw_event_time
    if event_time.tzinfo is None:
        event_time = event_time.replace(tzinfo=timezone.utc)

    raw_created_at = row["created_at"]
    if isinstance(raw_created_at, str):
        created_at = datetime.fromisoformat(raw_created_at)
    else:
        created_at = raw_created_at
    if created_at.tzinfo is None:
        created_at = created_at.replace(tzinfo=timezone.utc)

    raw_details = row.get("details", {})
    if isinstance(raw_details, (bytes, str)):
        details = orjson.loads(raw_details)
    elif isinstance(raw_details, dict):
        details = dict(raw_details)
    else:
        details = {}

    return AccountEventRecord(
        run_id=run_id,
        bank_id=bank_id,
        account_id=account_id,
        event_id=event_id,
        kind=kind,
        event_time=event_time,
        arrival_time=created_at,  # arrival_time mapped from Postgres created_at
        details=details,
    )


def record_member(record: PaymentSideRecord | AccountEventRecord) -> str:
    """
    WHAT: Generates the canonical member string for Redis sorted sets and hash fields.

    WHY: Uniquely identifies a record within an account's time series. For payment sides,
    the format is 'payment_id:side' to allow debit and credit sides of the same payment to coexist.
    For account lifecycle events, the member is simply event_id.

    Args:
        record: PaymentSideRecord or AccountEventRecord instance.

    Returns:
        String member identifier.
    """
    if isinstance(record, PaymentSideRecord):
        return f"{record.payment_id}:{record.side}"
    if isinstance(record, AccountEventRecord):
        return record.event_id
    raise TypeError(f"Unsupported record type: {type(record)}")


def serialize_record(record: PaymentSideRecord | AccountEventRecord) -> bytes:
    """Serializes a record to JSON bytes for Redis HASH storage."""
    d = asdict(record)
    d["event_time"] = record.event_time.isoformat()
    d["arrival_time"] = record.arrival_time.isoformat()
    return orjson.dumps(d)


def deserialize_payment_side_record(data: bytes | str) -> PaymentSideRecord:
    """Deserializes a PaymentSideRecord from JSON bytes or string."""
    d = orjson.loads(data)
    d["event_time"] = datetime.fromisoformat(d["event_time"])
    d["arrival_time"] = datetime.fromisoformat(d["arrival_time"])
    return PaymentSideRecord(**d)


def deserialize_account_event_record(data: bytes | str) -> AccountEventRecord:
    """Deserializes an AccountEventRecord from JSON bytes or string."""
    d = orjson.loads(data)
    d["event_time"] = datetime.fromisoformat(d["event_time"])
    d["arrival_time"] = datetime.fromisoformat(d["arrival_time"])
    return AccountEventRecord(**d)

