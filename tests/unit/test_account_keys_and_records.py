# File: tests/unit/test_account_keys_and_records.py
# Purpose: Unit tests for Level 15a account keys, run-ID extraction, and record parsing.
# Issue: Level 15a account feature updater (plan Task 1, C4, K2).
"""
WHAT: Unit tests covering run-ID extraction, namespaced Redis key generation, schema-2 UPI
event parsing from UpiEventPayload, and account event conversion from database rows.

WHY: Verifies that identity, tenant isolation, and strict validation guards prevent malformed,
unsupported, or naive-timestamped events from entering the account feature pipeline.
"""

from datetime import datetime, timezone

import pytest

from api.v1.upi_outbox import UpiEventData, UpiEventPayload
from features.account_keys import (
    AccountFeaturesSettings,
    event_records_key,
    event_times_key,
    extract_run_id,
    load_account_features_settings,
    marks_key,
    namespace_pattern,
    side_records_key,
    side_times_key,
)
from features.account_records import (
    AccountEventRecord,
    InvalidAccountEventError,
    PaymentSideRecord,
    account_event_from_row,
    deserialize_account_event_record,
    deserialize_payment_side_record,
    parse_upi_event,
    record_member,
    serialize_record,
)

VALID_RUN_ID = "66db9aa439b092ee"
VALID_PAYMENT_ID = f"{VALID_RUN_ID}-000042"
VALID_EVENT_ID = f"{VALID_RUN_ID}-dev-000001"
NOW_UTC = datetime(2026, 10, 28, 12, 0, 0, tzinfo=timezone.utc)


def test_settings_load_from_yaml() -> None:
    """Verifies that load_account_features_settings correctly populates operational settings."""
    settings = load_account_features_settings()
    assert isinstance(settings, AccountFeaturesSettings)
    assert settings.windows_seconds == (3600, 86400, 604800)
    assert settings.retention_seconds == 604800
    assert settings.key_expiry_margin_seconds == 86400
    assert settings.expiry_seconds == 604800 + 86400
    assert settings.key_prefix == "acctf"
    assert settings.run_id_hex_length == 16
    assert settings.account_event_batch_size == 500
    assert settings.upi_key_by_account is True


def test_extract_run_id_valid() -> None:
    """Verifies that a valid 16-hex lowercase run ID is extracted from hyphenated identifier."""
    assert extract_run_id(VALID_PAYMENT_ID) == VALID_RUN_ID
    assert extract_run_id(VALID_EVENT_ID) == VALID_RUN_ID


def test_extract_run_id_wrong_length() -> None:
    """Verifies that identifiers with run IDs shorter or longer than 16 hex chars are rejected."""
    with pytest.raises(ValueError, match="invalid run ID length"):
        extract_run_id("66db9aa439b092e-000042")  # 15 chars
    with pytest.raises(ValueError, match="invalid run ID length"):
        extract_run_id("66db9aa439b092ee0-000042")  # 17 chars


def test_extract_run_id_uppercase() -> None:
    """Verifies that uppercase hexadecimal run IDs are strictly rejected."""
    with pytest.raises(ValueError, match="must contain only lowercase hexadecimal digits"):
        extract_run_id("66DB9AA439B092EE-000042")


def test_extract_run_id_no_dash() -> None:
    """Verifies that identifiers lacking a hyphen delimiter are rejected."""
    with pytest.raises(ValueError, match="missing '-' delimiter"):
        extract_run_id("66db9aa439b092ee000042")


def test_key_builders_format_and_colon_rejection() -> None:
    """Verifies key builder format and rejection of empty or colon-containing key components."""
    bank_id = "bank_1"
    account_id = "acc_person_0001"

    st_key = side_times_key(VALID_RUN_ID, bank_id, account_id)
    sr_key = side_records_key(VALID_RUN_ID, bank_id, account_id)
    et_key = event_times_key(VALID_RUN_ID, bank_id, account_id)
    er_key = event_records_key(VALID_RUN_ID, bank_id, account_id)
    m_key = marks_key(VALID_RUN_ID, bank_id, account_id)
    pattern = namespace_pattern(VALID_RUN_ID, bank_id)

    assert st_key == f"acctf:{VALID_RUN_ID}:{bank_id}:{account_id}:side_times"
    assert sr_key == f"acctf:{VALID_RUN_ID}:{bank_id}:{account_id}:side_records"
    assert et_key == f"acctf:{VALID_RUN_ID}:{bank_id}:{account_id}:event_times"
    assert er_key == f"acctf:{VALID_RUN_ID}:{bank_id}:{account_id}:event_records"
    assert m_key == f"acctf:{VALID_RUN_ID}:{bank_id}:{account_id}:marks"
    assert pattern == f"acctf:{VALID_RUN_ID}:{bank_id}:*"

    # Reject empty components
    with pytest.raises(ValueError, match="cannot be empty"):
        side_times_key("", bank_id, account_id)
    with pytest.raises(ValueError, match="cannot be empty"):
        side_times_key(VALID_RUN_ID, "", account_id)
    with pytest.raises(ValueError, match="cannot be empty"):
        side_times_key(VALID_RUN_ID, bank_id, "")

    # Reject colon separators in components (prevent namespace collisions)
    with pytest.raises(ValueError, match="cannot contain ':'"):
        side_times_key("run:id", bank_id, account_id)
    with pytest.raises(ValueError, match="cannot contain ':'"):
        side_times_key(VALID_RUN_ID, "bank:1", account_id)
    with pytest.raises(ValueError, match="cannot contain ':'"):
        side_times_key(VALID_RUN_ID, bank_id, "acc:0001")


def test_two_runs_give_different_keys_for_same_account() -> None:
    """Verifies that distinct simulation runs produce disjoint Redis keys for identical accounts."""
    run_1 = "1111222233334444"
    run_2 = "aaaabbbbccccdddd"
    bank = "bank_0"
    acc = "acc_person_0001"

    key_1 = side_times_key(run_1, bank, acc)
    key_2 = side_times_key(run_2, bank, acc)
    assert key_1 != key_2
    assert key_1.split(":")[1] == run_1
    assert key_2.split(":")[1] == run_2


def _make_sample_payload() -> UpiEventPayload:
    """Helper constructing a valid UpiEventPayload instance via api.v1.upi_outbox."""
    return UpiEventPayload(
        event_id="3b8d234a-e4ef-4c6e-8a21-9e8c459f2e71",
        transaction_id="d1d2d3d4-e5e6-4a7b-8c9d-0e1f2a3b4c5d",
        payment_id=VALID_PAYMENT_ID,
        side="debit",
        event_time=NOW_UTC.isoformat(),
        recorded_at=NOW_UTC.isoformat(),
        source="world",
        data=UpiEventData(
            bank_id="bank_1",
            account_id="acc_person_0001",
            counterparty_vpa="",
            counterparty_bank_id="bank_2",
            sender_bank_id="bank_1",
            receiver_bank_id="bank_2",
            amount_paise=50000,
            status="success",
            failure_domain=None,
            payee_type="person",
            channel="upi_app",
            sender_device_fingerprint="fp_debit_001",
            sender_os_generation=2,
        ),
    )


def test_parse_upi_event_happy_path() -> None:
    """Verifies parsing a valid schema-2 UPI payment event produced by UpiEventPayload."""
    payload = _make_sample_payload()
    raw_bytes = payload.model_dump_json().encode("utf-8")

    record = parse_upi_event(raw_bytes)
    assert isinstance(record, PaymentSideRecord)
    assert record.run_id == VALID_RUN_ID
    assert record.bank_id == "bank_1"
    assert record.account_id == "acc_person_0001"
    assert record.payment_id == VALID_PAYMENT_ID
    assert record.side == "debit"
    assert record.counterparty_bank_id == "bank_2"
    assert record.amount_paise == 50000
    assert record.status == "success"
    assert record.failure_domain is None
    assert record.payee_type == "person"
    assert record.channel == "upi_app"
    assert record.sender_device_fingerprint == "fp_debit_001"
    assert record.sender_os_generation == 2
    assert record.event_time == NOW_UTC
    assert record.arrival_time == NOW_UTC


def test_parse_upi_event_rejects_schema_1() -> None:
    """Verifies that events with schema_version other than '2' are rejected."""
    payload = _make_sample_payload().model_dump()
    payload["schema_version"] = "1"
    import orjson

    with pytest.raises(InvalidAccountEventError, match="Unsupported schema_version '1'"):
        parse_upi_event(orjson.dumps(payload))


def test_parse_upi_event_rejects_naive_datetimes() -> None:
    """Verifies that naive timestamps lacking timezone offsets are rejected."""
    import orjson

    payload_naive_event = _make_sample_payload().model_dump()
    payload_naive_event["event_time"] = "2026-10-28T12:00:00"  # No UTC offset
    with pytest.raises(InvalidAccountEventError, match="must be timezone-aware"):
        parse_upi_event(orjson.dumps(payload_naive_event))

    payload_naive_recorded = _make_sample_payload().model_dump()
    payload_naive_recorded["recorded_at"] = "2026-10-28T12:00:00"
    with pytest.raises(InvalidAccountEventError, match="must be timezone-aware"):
        parse_upi_event(orjson.dumps(payload_naive_recorded))


def test_parse_upi_event_rejects_bad_run_prefix() -> None:
    """Verifies that invalid run prefixes in payment_id raise InvalidAccountEventError."""
    import orjson

    payload = _make_sample_payload().model_dump()
    payload["payment_id"] = "short-000042"
    with pytest.raises(InvalidAccountEventError, match="Invalid run ID in payment_id"):
        parse_upi_event(orjson.dumps(payload))


def test_account_event_from_row_maps_created_at_to_arrival_time() -> None:
    """Verifies that account_event_from_row maps created_at to arrival_time and extracts run_id."""
    row = {
        "event_id": VALID_EVENT_ID,
        "bank_id": "bank_2",
        "account_id": "acc_shop_0002",
        "kind": "device_binding_change",
        "event_time": NOW_UTC,
        "created_at": NOW_UTC,
        "details": {"platform": "android", "fingerprint": "dev_01"},
    }

    record = account_event_from_row(row)
    assert isinstance(record, AccountEventRecord)
    assert record.run_id == VALID_RUN_ID
    assert record.bank_id == "bank_2"
    assert record.account_id == "acc_shop_0002"
    assert record.event_id == VALID_EVENT_ID
    assert record.kind == "device_binding_change"
    assert record.event_time == NOW_UTC
    assert record.arrival_time == NOW_UTC
    assert record.details == {"platform": "android", "fingerprint": "dev_01"}


def test_record_member_and_serialization() -> None:
    """Verifies record_member keys and JSON serialization/deserialization fidelity."""
    side = PaymentSideRecord(
        run_id=VALID_RUN_ID,
        bank_id="bank_0",
        account_id="acc_0",
        payment_id=VALID_PAYMENT_ID,
        side="debit",
        counterparty_bank_id="bank_1",
        counterparty_vpa="",
        amount_paise=1000,
        status="success",
        failure_domain=None,
        payee_type="person",
        channel="upi_app",
        sender_device_fingerprint=None,
        sender_os_generation=None,
        event_time=NOW_UTC,
        arrival_time=NOW_UTC,
    )
    assert record_member(side) == f"{VALID_PAYMENT_ID}:debit"

    serialized_side = serialize_record(side)
    deserialized_side = deserialize_payment_side_record(serialized_side)
    assert deserialized_side == side

    ev = AccountEventRecord(
        run_id=VALID_RUN_ID,
        bank_id="bank_0",
        account_id="acc_0",
        event_id=VALID_EVENT_ID,
        kind="account_opened",
        event_time=NOW_UTC,
        arrival_time=NOW_UTC,
        details={"status": "active"},
    )
    assert record_member(ev) == VALID_EVENT_ID

    serialized_ev = serialize_record(ev)
    deserialized_ev = deserialize_account_event_record(serialized_ev)
    assert deserialized_ev == ev

