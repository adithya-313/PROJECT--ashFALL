# File: tests/unit/test_feature_replay.py
# Purpose: Unit tests for historical feature replay and row conversion functions.
# Issue: Level 15c Round 2a (recompute a logged feature row from history).
"""
WHAT: Unit tests for historical feature replay and schema-2 payment side parsing.

WHY: Verifies that payment_side_from_transaction_row converts database rows from
the transactions table into PaymentSideRecord instances identical to those produced
by the live Kafka ingress path (parse_upi_event with UpiEventPayload), and enforces
strict KeyError behavior on missing required columns.
"""

from datetime import datetime, timezone
from typing import Any

import fakeredis
import pytest

from api.v1.upi_outbox import (
    EVENT_TYPE_UPI_PAYMENT_RECORDED,
    UPI_SCHEMA_VERSION,
    UpiEventData,
    UpiEventPayload,
)
from features.account_keys import side_times_key
from features.account_records import (
    InvalidAccountEventError,
    parse_upi_event,
    parse_upi_payload,
    payment_side_from_transaction_row,
)
from features.feature_replay import (
    LoggedAnswerScreener,
    replay_history_into_redis,
)
from features.registry_screening import ScreenOutcome

pytestmark = pytest.mark.anyio


def test_payment_side_from_transaction_row_matches_live_upi_event_payload() -> None:
    """
    D1a: Verifies that payment_side_from_transaction_row produces a PaymentSideRecord
    identical to parse_upi_event decoding a live UpiEventPayload for the same transaction side.
    """
    event_time = datetime(2026, 10, 28, 10, 0, 0, tzinfo=timezone.utc)
    created_at = datetime(2026, 10, 28, 10, 0, 5, tzinfo=timezone.utc)
    payment_id = "0123456789abcdef-pay-001"
    bank_id = "bank_0"
    account_id = "acc_payer_001"
    side = "debit"
    counterparty_bank_id = "bank_1"
    counterparty_vpa = "acc_payee@bank1"
    amount_paise = 250000
    status = "success"
    failure_domain = None
    payee_type = "merchant"
    channel = "in_store"
    sender_device_fingerprint = "fp_abc123"
    sender_os_generation = 14
    balance_after_paise = 500000

    row = {
        "payment_id": payment_id,
        "bank_id": bank_id,
        "account_id": account_id,
        "side": side,
        "counterparty_bank_id": counterparty_bank_id,
        "counterparty_vpa": counterparty_vpa,
        "amount_paise": amount_paise,
        "status": status,
        "failure_domain": failure_domain,
        "payee_type": payee_type,
        "channel": channel,
        "sender_device_fingerprint": sender_device_fingerprint,
        "sender_os_generation": sender_os_generation,
        "balance_after_paise": balance_after_paise,
        "event_time": event_time,
        "created_at": created_at,
    }

    # Record from transaction row
    record_from_row = payment_side_from_transaction_row(row)

    # Real live UpiEventPayload model as built by api/v1/upi_outbox.py
    payload_model = UpiEventPayload(
        event_type=EVENT_TYPE_UPI_PAYMENT_RECORDED,
        schema_version=UPI_SCHEMA_VERSION,
        event_id="00000000-0000-0000-0000-000000000001",
        transaction_id="00000000-0000-0000-0000-000000000002",
        payment_id=payment_id,
        side=side,
        event_time=event_time.isoformat(),
        recorded_at=created_at.isoformat(),
        source="world",
        data=UpiEventData(
            bank_id=bank_id,
            account_id=account_id,
            counterparty_vpa=counterparty_vpa,
            counterparty_bank_id=counterparty_bank_id,
            sender_bank_id=bank_id,
            receiver_bank_id=counterparty_bank_id,
            amount_paise=amount_paise,
            status=status,
            failure_domain=failure_domain,
            payee_type=payee_type,
            channel=channel,
            sender_device_fingerprint=sender_device_fingerprint,
            sender_os_generation=sender_os_generation,
            balance_after_paise=balance_after_paise,
        ),
    )
    raw_bytes = payload_model.model_dump_json().encode("utf-8")
    record_from_event = parse_upi_event(raw_bytes)
    record_from_payload = parse_upi_payload(payload_model.model_dump())

    # Assert byte-for-byte / field-for-field equality
    assert record_from_row == record_from_event
    assert record_from_row == record_from_payload
    assert record_from_row.arrival_time == created_at
    assert record_from_row.event_time == event_time
    assert record_from_row.run_id == "0123456789abcdef"


def test_payment_side_from_transaction_row_missing_created_at_raises_key_error() -> None:
    """
    D1a: Verifies that a row missing 'created_at' raises KeyError without silent fallback.
    """
    event_time = datetime(2026, 10, 28, 10, 0, 0, tzinfo=timezone.utc)
    row = {
        "payment_id": "0123456789abcdef-pay-001",
        "bank_id": "bank_0",
        "account_id": "acc_001",
        "side": "debit",
        "counterparty_bank_id": "bank_1",
        "counterparty_vpa": "acc_002",
        "amount_paise": 10000,
        "status": "success",
        "failure_domain": None,
        "payee_type": "person",
        "channel": "app",
        "sender_device_fingerprint": None,
        "sender_os_generation": None,
        "balance_after_paise": None,
        "event_time": event_time,
        # 'created_at' intentionally omitted
    }
    with pytest.raises(KeyError) as exc_info:
        payment_side_from_transaction_row(row)
    assert "created_at" in str(exc_info.value)


def test_payment_side_from_transaction_row_non_datetime_raises_invalid_account_event_error() -> None:
    """
    T3: Verifies that passing a string timestamp instead of a datetime object
    raises InvalidAccountEventError directly without fallback.
    """
    row = {
        "payment_id": "0123456789abcdef-pay-001",
        "bank_id": "bank_0",
        "account_id": "acc_001",
        "side": "debit",
        "counterparty_bank_id": "bank_1",
        "counterparty_vpa": "acc_002",
        "amount_paise": 10000,
        "status": "success",
        "failure_domain": None,
        "payee_type": "person",
        "channel": "app",
        "sender_device_fingerprint": None,
        "sender_os_generation": None,
        "balance_after_paise": None,
        "event_time": "2026-10-28T10:00:00+00:00",  # String instead of datetime
        "created_at": datetime(2026, 10, 28, 10, 0, 5, tzinfo=timezone.utc),
    }
    with pytest.raises(InvalidAccountEventError) as exc_info:
        payment_side_from_transaction_row(row)
    assert "event_time" in str(exc_info.value)


class FakeReplayConnection:
    """Fake asyncpg connection returning pre-configured rows for replay unit tests."""

    def __init__(
        self,
        transactions: list[dict[str, Any]] | None = None,
        account_events: list[dict[str, Any]] | None = None,
        withdrawals: list[dict[str, Any]] | None = None,
        feedback: list[dict[str, Any]] | None = None,
        complaints: list[dict[str, Any]] | None = None,
    ) -> None:
        self.transactions = list(transactions) if transactions else []
        self.account_events = list(account_events) if account_events else []
        self.withdrawals = list(withdrawals) if withdrawals else []
        self.feedback = list(feedback) if feedback else []
        self.complaints = list(complaints) if complaints else []

    async def fetch(self, query: str, *args: Any) -> list[dict[str, Any]]:
        bank_id = args[0]
        cutoff = args[1]
        if "FROM transactions" in query and "payment_id = $2" in query:
            # Feedback transaction lookup
            return [r for r in self.transactions if r["bank_id"] == bank_id and r["payment_id"] == args[1]]
        elif "FROM transactions" in query:
            if "created_at <= $2" in query:
                return [r for r in self.transactions if r["bank_id"] == bank_id and r["created_at"] <= cutoff]
            return [r for r in self.transactions if r["bank_id"] == bank_id]
        elif "FROM bank_account_events" in query:
            if "created_at <= $2" in query:
                return [r for r in self.account_events if r["bank_id"] == bank_id and r["created_at"] <= cutoff]
            return [r for r in self.account_events if r["bank_id"] == bank_id]
        elif "FROM bank_withdrawals" in query:
            if "recorded_at <= $2" in query:
                return [r for r in self.withdrawals if r["bank_id"] == bank_id and r["recorded_at"] <= cutoff]
            return [r for r in self.withdrawals if r["bank_id"] == bank_id]
        elif "FROM bank_feedback" in query:
            if "created_at <= $2" in query:
                return [r for r in self.feedback if r["bank_id"] == bank_id and r["created_at"] <= cutoff]
            return [r for r in self.feedback if r["bank_id"] == bank_id]
        elif "FROM bank_complaints" in query:
            if "created_at <= $2" in query:
                return [r for r in self.complaints if r["bank_id"] == bank_id and r["created_at"] <= cutoff]
            return [r for r in self.complaints if r["bank_id"] == bank_id]
        return []


async def test_replay_applies_only_rows_with_arrival_lte_cutoff() -> None:
    """
    D3 unit test: Verifies that replay applies only rows whose arrival <= cutoff.
    Late-arriving rows (arrival > cutoff) are strictly excluded.
    """
    redis_client = fakeredis.FakeRedis()
    run_id = "0123456789abcdef"
    bank_id = "bank_0"
    cutoff = datetime(2026, 10, 28, 12, 0, 0, tzinfo=timezone.utc)

    row_early = {
        "payment_id": f"{run_id}-pay-001",
        "bank_id": bank_id,
        "account_id": "acc_001",
        "side": "debit",
        "counterparty_bank_id": "bank_1",
        "counterparty_vpa": "acc_002",
        "amount_paise": 10000,
        "status": "success",
        "failure_domain": None,
        "payee_type": "person",
        "channel": "app",
        "sender_device_fingerprint": None,
        "sender_os_generation": None,
        "balance_after_paise": None,
        "event_time": datetime(2026, 10, 28, 11, 0, 0, tzinfo=timezone.utc),
        "created_at": datetime(2026, 10, 28, 11, 30, 0, tzinfo=timezone.utc),
    }
    row_late = {
        "payment_id": f"{run_id}-pay-002",
        "bank_id": bank_id,
        "account_id": "acc_001",
        "side": "debit",
        "counterparty_bank_id": "bank_1",
        "counterparty_vpa": "acc_002",
        "amount_paise": 20000,
        "status": "success",
        "failure_domain": None,
        "payee_type": "person",
        "channel": "app",
        "sender_device_fingerprint": None,
        "sender_os_generation": None,
        "balance_after_paise": None,
        "event_time": datetime(2026, 10, 28, 11, 0, 0, tzinfo=timezone.utc),
        "created_at": datetime(2026, 10, 28, 12, 30, 0, tzinfo=timezone.utc),  # after cutoff
    }

    conn = FakeReplayConnection(transactions=[row_early, row_late])
    counts = await replay_history_into_redis(
        pg_connection=conn,
        scratch_redis=redis_client,
        run_id=run_id,
        bank_id=bank_id,
        arrival_cutoff=cutoff,
    )

    assert counts.transactions == 1
    # Verify the early record is present in Redis and late record is absent
    st_key = side_times_key(run_id, bank_id, "acc_001")
    members = redis_client.zrange(st_key, 0, -1)
    assert len(members) == 1
    assert b"pay-001:debit" in members[0]


async def test_replay_order_by_arrival_not_id(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    D3 unit test & Sabotage S2 target: Verifies that replay orders by arrival time,
    not lexicographical ID order, when two rows have divergent ID and arrival sequences.
    """
    redis_client = fakeredis.FakeRedis()
    run_id = "0123456789abcdef"
    bank_id = "bank_0"
    cutoff = datetime(2026, 10, 28, 12, 0, 0, tzinfo=timezone.utc)

    # Row A: id 001, but arrived LATER (11:00)
    row_a = {
        "payment_id": f"{run_id}-pay-001",
        "bank_id": bank_id,
        "account_id": "acc_001",
        "side": "debit",
        "counterparty_bank_id": "bank_1",
        "counterparty_vpa": "acc_002",
        "amount_paise": 10000,
        "status": "success",
        "failure_domain": None,
        "payee_type": "person",
        "channel": "app",
        "sender_device_fingerprint": None,
        "sender_os_generation": None,
        "balance_after_paise": None,
        "event_time": datetime(2026, 10, 28, 10, 0, 0, tzinfo=timezone.utc),
        "created_at": datetime(2026, 10, 28, 11, 0, 0, tzinfo=timezone.utc),
    }
    # Row B: id 002, but arrived EARLIER (10:30)
    row_b = {
        "payment_id": f"{run_id}-pay-002",
        "bank_id": bank_id,
        "account_id": "acc_001",
        "side": "debit",
        "counterparty_bank_id": "bank_1",
        "counterparty_vpa": "acc_002",
        "amount_paise": 20000,
        "status": "success",
        "failure_domain": None,
        "payee_type": "person",
        "channel": "app",
        "sender_device_fingerprint": None,
        "sender_os_generation": None,
        "balance_after_paise": None,
        "event_time": datetime(2026, 10, 28, 10, 0, 0, tzinfo=timezone.utc),
        "created_at": datetime(2026, 10, 28, 10, 30, 0, tzinfo=timezone.utc),
    }

    conn = FakeReplayConnection(transactions=[row_a, row_b])

    applied_ids: list[str] = []
    from features.account_records import PaymentSideRecord
    from features.account_updater import apply_payment_side as real_apply

    def spy_apply(client: Any, rec: PaymentSideRecord, settings: Any = None) -> str:
        applied_ids.append(rec.payment_id)
        return real_apply(client, rec, settings)

    monkeypatch.setattr("features.feature_replay.apply_payment_side", spy_apply)

    counts = await replay_history_into_redis(
        pg_connection=conn,
        scratch_redis=redis_client,
        run_id=run_id,
        bank_id=bank_id,
        arrival_cutoff=cutoff,
    )
    assert counts.transactions == 2
    # S2 Sabotage target: if sorted by id, order would be [pay-001, pay-002].
    # By arrival, row_b (10:30) MUST be applied before row_a (11:00).
    assert applied_ids == [f"{run_id}-pay-002", f"{run_id}-pay-001"]


async def test_replay_skips_other_run_ids() -> None:
    """
    D3 unit test: Verifies that rows belonging to other simulation runs are ignored.
    """
    redis_client = fakeredis.FakeRedis()
    run_id = "0123456789abcdef"
    other_run_id = "fedcba9876543210"
    bank_id = "bank_0"
    cutoff = datetime(2026, 10, 28, 12, 0, 0, tzinfo=timezone.utc)

    row_other = {
        "payment_id": f"{other_run_id}-pay-001",
        "bank_id": bank_id,
        "account_id": "acc_001",
        "side": "debit",
        "counterparty_bank_id": "bank_1",
        "counterparty_vpa": "acc_002",
        "amount_paise": 10000,
        "status": "success",
        "failure_domain": None,
        "payee_type": "person",
        "channel": "app",
        "sender_device_fingerprint": None,
        "sender_os_generation": None,
        "balance_after_paise": None,
        "event_time": datetime(2026, 10, 28, 10, 0, 0, tzinfo=timezone.utc),
        "created_at": datetime(2026, 10, 28, 10, 30, 0, tzinfo=timezone.utc),
    }

    conn = FakeReplayConnection(transactions=[row_other])
    counts = await replay_history_into_redis(
        pg_connection=conn,
        scratch_redis=redis_client,
        run_id=run_id,
        bank_id=bank_id,
        arrival_cutoff=cutoff,
    )
    assert counts.transactions == 0
    st_key = side_times_key(run_id, bank_id, "acc_001")
    assert redis_client.zcard(st_key) == 0


def test_logged_answer_screener_raises_on_unlogged_question() -> None:
    """
    D3 unit test & Sabotage S3 target: Verifies that LoggedAnswerScreener answers logged
    questions faithfully and raises LookupError on any question that was not logged.
    """
    logged_answers = [
        {
            "identifier_hash": "hash_logged_acc",
            "identifier_type": "account",
            "outcome": {"status": "match", "first_visible_at": "2026-10-28T10:00:00+00:00"},
            "as_of": "2026-10-28T10:00:00+00:00",
            "received_at": "2026-10-28T10:00:01+00:00",
        }
    ]
    screener = LoggedAnswerScreener(logged_answers)

    # Logged question returns recorded outcome
    outcome = screener.screen("hash_logged_acc", "account")
    assert isinstance(outcome, ScreenOutcome)
    assert outcome.status == "match"
    assert outcome.first_visible_at == datetime(2026, 10, 28, 10, 0, 0, tzinfo=timezone.utc)

    # S3 target: Unlogged question must raise LookupError, not return 'no_match'
    with pytest.raises(LookupError) as exc_info:
        screener.screen("hash_unlogged_acc", "account")
    assert "was not logged" in str(exc_info.value)
