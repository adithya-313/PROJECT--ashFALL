# File: tests/unit/test_decision_features.py
# Purpose: Unit tests for Level 15c decision-time feature computation (D4).
# Issue: Level 15c watermark and feature log (plan Task 1, D4).
"""
WHAT: Tests decision feature extraction, watermark validation guards,
incomplete request handling, and canonical vector serialization.

WHY: Verifies Section 22 correctness: decisions are only scored when
ingestion watermark is strictly ahead of or equal to decision_time.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import fakeredis
import pytest

from features.decision_features import (
    SystemClock,
    compute_decision_row,
)
from features.feature_log import canonical_feature_names
from features.ingest_watermark import (
    POLLER_SOURCES,
    record_payment_watermark,
    record_poller_watermark,
)

RUN_ID = "66db9aa439b092ee"
BANK_0 = "bank_0"
NOW_UTC = datetime(2026, 10, 28, 12, 0, 0, tzinfo=timezone.utc)


class DummyClock:
    def __init__(self, dt: datetime) -> None:
        self.dt = dt

    def now_utc(self) -> datetime:
        return self.dt


class DummyRequest:
    def __init__(
        self,
        payment_id: str = f"{RUN_ID}-0001",
        sender_account_id: str = "acc_sender_1",
        receiver_bank_id: str = "bank_1",
        receiver_vpa: str | None = "acc_recv_1",
        amount_paise: int = 50000,
        event_time: datetime | None = NOW_UTC,
        channel: str | None = "upi",
        payee_type: str | None = "person",
        bank_id: str = BANK_0,
    ) -> None:
        self.payment_id = payment_id
        self.sender_account_id = sender_account_id
        self.receiver_bank_id = receiver_bank_id
        self.receiver_vpa = receiver_vpa
        self.amount_paise = amount_paise
        self.event_time = event_time
        self.channel = channel
        self.payee_type = payee_type
        self.bank_id = bank_id


def _set_all_watermarks(client: fakeredis.FakeRedis, bank_id: str, ts: datetime) -> None:
    record_payment_watermark(client, 0, ts)
    for source in POLLER_SOURCES:
        record_poller_watermark(client, bank_id, source, ts)


def test_system_clock() -> None:
    clock = SystemClock()
    now_utc = clock.now_utc()
    assert now_utc.tzinfo == timezone.utc
    now_ist = clock.now_ist()
    assert now_ist.tzinfo is not None


def test_compute_decision_row_no_watermark_when_none() -> None:
    client = fakeredis.FakeRedis()
    req = DummyRequest()
    clock = DummyClock(NOW_UTC)

    row = compute_decision_row(client, req, clock, bank_id=BANK_0)
    assert row.status == "no_watermark"
    assert row.arrival_watermark is None
    assert row.payment_id == req.payment_id
    assert row.decision_id == f"{req.payment_id}-dec"
    assert row.feature_values_ordered is None

    canonical = canonical_feature_names()
    for name in canonical:
        assert row.feature_values[name] is None


def test_compute_decision_row_no_watermark_when_watermark_behind() -> None:
    client = fakeredis.FakeRedis()
    watermark_ts = NOW_UTC - timedelta(seconds=10)
    _set_all_watermarks(client, BANK_0, watermark_ts)
    req = DummyRequest()
    clock = DummyClock(NOW_UTC)

    row = compute_decision_row(client, req, clock, bank_id=BANK_0)
    assert row.status == "no_watermark"
    assert row.arrival_watermark == watermark_ts
    assert row.feature_values_ordered is None


@pytest.mark.parametrize(
    "corrupt_field,value",
    [
        ("payment_id", ""),
        ("sender_account_id", ""),
        ("amount_paise", 0),
        ("amount_paise", -100),
        ("event_time", None),
    ],
)
def test_compute_decision_row_incomplete_request(corrupt_field: str, value: object) -> None:
    client = fakeredis.FakeRedis()
    watermark_ts = NOW_UTC + timedelta(seconds=10)
    _set_all_watermarks(client, BANK_0, watermark_ts)

    kwargs = {
        "payment_id": f"{RUN_ID}-0001",
        "sender_account_id": "acc_sender_1",
        "amount_paise": 50000,
        "event_time": NOW_UTC,
    }
    kwargs[corrupt_field] = value
    req = DummyRequest(**kwargs)  # type: ignore[arg-type]
    clock = DummyClock(NOW_UTC)

    row = compute_decision_row(client, req, clock, bank_id=BANK_0)
    assert row.status == "incomplete_request"
    assert row.arrival_watermark == watermark_ts
    assert row.feature_values_ordered is None


def test_compute_decision_row_ok_with_watermark_ahead() -> None:
    client = fakeredis.FakeRedis()
    watermark_ts = NOW_UTC + timedelta(seconds=60)
    _set_all_watermarks(client, BANK_0, watermark_ts)
    req = DummyRequest()
    clock = DummyClock(NOW_UTC)

    row = compute_decision_row(client, req, clock, bank_id=BANK_0)
    assert row.status == "ok"
    assert row.arrival_watermark == watermark_ts
    assert row.feature_values_ordered is not None
    assert len(row.feature_values_ordered) == len(canonical_feature_names())

    # Payment features should be populated
    assert row.feature_values["payee_type"] == "person"
    assert "hour_ist" in row.feature_values


def test_compute_decision_row_error_handling() -> None:
    client = fakeredis.FakeRedis()
    watermark_ts = NOW_UTC + timedelta(seconds=60)
    _set_all_watermarks(client, BANK_0, watermark_ts)
    req = DummyRequest()
    clock = DummyClock(NOW_UTC)

    with patch("features.decision_features.compute_account_features", side_effect=RuntimeError("Redis boom")):
        row = compute_decision_row(client, req, clock, bank_id=BANK_0)
        assert row.status == "error"
        assert "Redis boom" in row.feature_values["error"]
        assert row.feature_values_ordered is None
