# File: tests/unit/test_decision_features.py
# Purpose: Unit tests for Level 15c decision-time feature computation (D4).
# Issue: Level 15c watermark and feature log (plan Task 1, D4).
"""
WHAT: Tests decision feature extraction, watermark validation guards,
incomplete request handling, and canonical vector serialization.

WHY: Verifies Section 22 correctness: decision_time is request.event_time, decided_at is clock.now_ist(),
watermark_lag_ms tracks ingestion latency, and incomplete requests fall back to payer-only features.
"""

from datetime import datetime, timedelta, timezone
from typing import Any

import fakeredis

from api.v1.models import DecisionRequest
from core.event_clock import IST, ManualClock
from features.account_keys import get_default_settings
from features.account_records import PaymentSideRecord
from features.account_updater import apply_payment_side
from features.decision_features import compute_decision_row
from features.feature_log import canonical_feature_names
from features.ingest_watermark import (
    POLLER_SOURCES,
    record_payment_watermark,
    record_poller_watermark,
)
from features.m1_features import compute_account_features
from features.m1_payment import ProposedPayment, compute_payment_features
from features.registry_screening import FakeRegistryScreener, ScreenOutcome

RUN_ID = "66db9aa439b092ee"
BANK_0 = "bank_0"
DPIP_KEY = "test_dpip_key_32_bytes_long_exact"
NOW_UTC = datetime(2026, 10, 28, 12, 0, 0, tzinfo=timezone.utc)
NOW_IST = NOW_UTC.astimezone(IST)


def _make_request(
    payment_id: str = f"{RUN_ID}-0001",
    sender_account_id: str = "acc_sender_1",
    receiver_bank_id: str = "bank_1",
    receiver_vpa: str | None = "acc_recv_1",
    amount_paise: int = 50000,
    event_time: datetime = NOW_UTC,
    channel: str | None = "upi_app",
    payee_type: str | None = "person",
) -> DecisionRequest:
    return DecisionRequest(
        payment_id=payment_id,
        sender_account_id=sender_account_id,
        receiver_bank_id=receiver_bank_id,
        receiver_vpa=receiver_vpa,
        amount_paise=amount_paise,
        event_time=event_time,
        channel=channel,
        payee_type=payee_type,
    )


def _seed_payer_debit(client: Any, bank_id: str, account_id: str, ts: datetime) -> None:
    """Seeds historical settled debit side so ratio features are defined."""
    rec = PaymentSideRecord(
        run_id=RUN_ID,
        bank_id=bank_id,
        account_id=account_id,
        payment_id=f"{RUN_ID}-prior-01",
        side="debit",
        counterparty_bank_id="bank_1",
        counterparty_vpa="acc_counter_1",
        amount_paise=25000,
        status="success",
        failure_domain=None,
        payee_type="person",
        channel="upi_app",
        sender_device_fingerprint="fp1",
        sender_os_generation=1,
        event_time=ts - timedelta(hours=2),
        arrival_time=ts - timedelta(hours=2),
    )
    apply_payment_side(client, rec)


def _set_all_watermarks(client: Any, bank_id: str, ts: datetime) -> None:
    record_payment_watermark(client, 0, ts)
    for source in POLLER_SOURCES:
        record_poller_watermark(client, bank_id, source, ts)


def test_compute_decision_row_no_watermark_when_none() -> None:
    """F3: Verifies that status is 'no_watermark' strictly when watermark is None."""
    client = fakeredis.FakeRedis()
    req = _make_request()
    clock = ManualClock(NOW_IST)

    row = compute_decision_row(client, BANK_0, req, clock)
    assert row.status == "no_watermark"
    assert row.watermark_lag_ms is None
    assert row.payment_id == req.payment_id
    assert row.decision_time == req.event_time
    assert row.decided_at == NOW_IST

    canonical = canonical_feature_names()
    assert len(row.feature_values) == len(canonical)
    for val in row.feature_values:
        assert val is None


def test_compute_decision_row_watermark_behind_computes_with_lag() -> None:
    """
    F3 & SABOTAGE S4 TARGET:
    Verifies that watermark a few seconds BEHIND decided_at and event_time computes with status 'ok'
    and positive watermark_lag_ms (this was impossible before).
    """
    client = fakeredis.FakeRedis()
    watermark_ts = NOW_UTC - timedelta(seconds=10)
    _set_all_watermarks(client, BANK_0, watermark_ts)
    req = _make_request(event_time=NOW_UTC)
    clock = ManualClock(NOW_IST)

    row = compute_decision_row(client, BANK_0, req, clock)
    # Re-adding `watermark < decision_time -> no_watermark` will break this assertion!
    assert row.status == "ok"
    assert row.watermark_lag_ms == 10000


def test_compute_decision_row_payee_type_none_incomplete_request() -> None:
    """
    F4 & SABOTAGE S3 TARGET:
    Verifies that payee_type None -> incomplete_request, missing_request_fields == ["payee_type"],
    payee_type None, hour_ist and both amount ratios NOT None, every account feature equal
    to a direct compute_account_features call.
    """
    client = fakeredis.FakeRedis()
    watermark_ts = NOW_UTC + timedelta(seconds=10)
    _set_all_watermarks(client, BANK_0, watermark_ts)

    req = _make_request(payee_type=None)
    _seed_payer_debit(client, BANK_0, req.sender_account_id, NOW_UTC)
    clock = ManualClock(NOW_IST)

    row = compute_decision_row(client, BANK_0, req, clock)
    assert row.status == "incomplete_request"
    assert row.missing_request_fields == ["payee_type"]

    names = canonical_feature_names()
    assert isinstance(row.feature_values, list)
    assert len(row.feature_values) == len(names)

    # payee_type must be None
    idx_payee = names.index("payee_type")
    assert row.feature_values[idx_payee] is None

    # hour_ist and both amount ratios must NOT be None (S3 target)
    idx_hour = names.index("hour_ist")
    assert row.feature_values[idx_hour] is not None
    assert row.feature_values[idx_hour] == NOW_UTC.astimezone(IST).hour

    idx_median = names.index("amount_vs_payer_median_ratio")
    assert row.feature_values[idx_median] is not None
    assert row.feature_values[idx_median] == 50000 / 25000  # 2.0

    idx_max = names.index("amount_vs_payer_max_ratio")
    assert row.feature_values[idx_max] is not None
    assert row.feature_values[idx_max] == 50000 / 25000  # 2.0

    # Every account feature equal to direct compute_account_features call
    account_groups = [
        "inflow",
        "outflow",
        "balance_pattern",
        "routine_life",
        "lifecycle",
        "control",
        "region",
        "network",
        "intelligence",
    ]
    direct_acc = compute_account_features(
        client=client,
        run_id=RUN_ID,
        bank_id=BANK_0,
        account_id=req.sender_account_id,
        decision_time=req.event_time,
        arrival_cutoff=watermark_ts,
        settings=get_default_settings(),
        groups=account_groups,
    )
    for name, val in direct_acc.items():
        idx = names.index(name)
        assert row.feature_values[idx] == val


def test_compute_decision_row_receiver_vpa_none_exactly_three_vpa_features_none() -> None:
    """
    F4: Verifies receiver_vpa None -> exactly the 3 VPA features None, while other payment features
    (payee_type, hour_ist, amount ratios, failed attempts) are NOT None.
    """
    client = fakeredis.FakeRedis()
    watermark_ts = NOW_UTC + timedelta(seconds=10)
    _set_all_watermarks(client, BANK_0, watermark_ts)

    req = _make_request(receiver_vpa=None, payee_type="person")
    _seed_payer_debit(client, BANK_0, req.sender_account_id, NOW_UTC)
    clock = ManualClock(NOW_IST)

    row = compute_decision_row(client, BANK_0, req, clock)
    assert row.status == "incomplete_request"
    assert row.missing_request_fields == ["receiver_vpa"]

    names = canonical_feature_names()
    vpa_features = {"payee_registry_match", "first_payment_to_payee", "payee_same_bank_risk"}
    other_payment_features = {
        "amount_vs_payer_median_ratio",
        "amount_vs_payer_max_ratio",
        "customer_failed_attempts_24h",
        "hour_ist",
        "payee_type",
    }

    values = row.feature_values
    assert isinstance(values, list)

    # Exactly the 3 VPA features must be None
    for f in vpa_features:
        idx = names.index(f)
        assert values[idx] is None, f"Expected {f} to be None"

    # Other payment features must NOT be None
    for f in other_payment_features:
        idx = names.index(f)
        assert values[idx] is not None, f"Expected {f} to not be None"

    # payee_type is preserved as 'person'
    assert values[names.index("payee_type")] == "person"


def test_compute_decision_row_logged_values_equal_direct_calls_types_preserved() -> None:
    """
    F7 & SABOTAGE S6 TARGET:
    Verifies logged values equal direct compute_account_features + compute_payment_features output,
    with types strictly preserved (an int is an int, a string is a string, bool is bool).
    """
    client = fakeredis.FakeRedis()
    watermark_ts = NOW_UTC + timedelta(seconds=60)
    _set_all_watermarks(client, BANK_0, watermark_ts)

    req = _make_request()
    clock = ManualClock(NOW_IST)

    row = compute_decision_row(client, BANK_0, req, clock)
    assert row.status == "ok"

    names = canonical_feature_names()
    values = row.feature_values
    assert isinstance(values, list)
    assert len(values) == len(names)

    # Verify direct calls match exactly
    account_groups = [
        "inflow",
        "outflow",
        "balance_pattern",
        "routine_life",
        "lifecycle",
        "control",
        "region",
        "network",
        "intelligence",
    ]
    direct_acc = compute_account_features(
        client=client,
        run_id=RUN_ID,
        bank_id=BANK_0,
        account_id=req.sender_account_id,
        decision_time=req.event_time,
        arrival_cutoff=watermark_ts,
        settings=get_default_settings(),
        groups=account_groups,
    )
    assert req.receiver_vpa is not None
    assert req.payee_type is not None
    direct_pay = compute_payment_features(
        client=client,
        proposed=ProposedPayment(
            run_id=RUN_ID,
            bank_id=BANK_0,
            payer_account_id=req.sender_account_id,
            payee_bank_id=req.receiver_bank_id,
            payee_vpa=req.receiver_vpa,
            payee_type=req.payee_type,
            amount_paise=req.amount_paise,
            event_time=req.event_time,
        ),
        arrival_cutoff=watermark_ts,
        settings=get_default_settings(),
    )

    for name in names:
        idx = names.index(name)
        expected = direct_acc.get(name, direct_pay.get(name))
        assert values[idx] == expected, f"Mismatch on {name}: {values[idx]} vs {expected}"

    # Verify type preservation (Sabotage S6: casting logged ints to float will break these)
    idx_failed = names.index("customer_failed_attempts_24h")
    assert isinstance(values[idx_failed], int)
    assert not isinstance(values[idx_failed], float)

    idx_hour = names.index("hour_ist")
    assert isinstance(values[idx_hour], int)
    assert not isinstance(values[idx_hour], float)

    idx_payers = names.index("distinct_payers_1h")
    assert isinstance(values[idx_payers], int)
    assert not isinstance(values[idx_payers], float)

    idx_payee = names.index("payee_type")
    assert isinstance(values[idx_payee], str)


def test_compute_decision_row_screener_unknown() -> None:
    """
    F6: FakeRegistryScreener returning 'unknown' -> feature None, registry_answers status 'unknown',
    and the inner screener received as_of=None.
    """
    client = fakeredis.FakeRedis()
    watermark_ts = NOW_UTC + timedelta(seconds=60)
    _set_all_watermarks(client, BANK_0, watermark_ts)

    req = _make_request()
    clock = ManualClock(NOW_IST)
    screener = FakeRegistryScreener(default_outcome=ScreenOutcome(status="unknown", first_visible_at=None))

    row = compute_decision_row(client, BANK_0, req, clock, screener=screener, dpip_key=DPIP_KEY)
    assert row.status == "ok"

    names = canonical_feature_names()
    assert isinstance(row.feature_values, list)
    idx_match = names.index("payee_registry_match")
    assert row.feature_values[idx_match] is None

    assert row.registry_answers is not None
    assert len(row.registry_answers) >= 1
    assert row.registry_answers[0]["outcome"]["status"] == "unknown"

    # Inner screener received as_of=None via LiveRegistryScreener
    assert len(screener.calls) >= 1
    assert screener.calls[0]["as_of"] is None


def test_compute_decision_row_screener_recording_wrapper() -> None:
    """
    F6 & SABOTAGE S5 TARGET:
    Verifies that when screener is provided, RecordingRegistryScreener wraps LiveRegistryScreener
    and populates registry_answers.
    """
    client = fakeredis.FakeRedis()
    watermark_ts = NOW_UTC + timedelta(seconds=60)
    _set_all_watermarks(client, BANK_0, watermark_ts)

    req = _make_request()
    clock = ManualClock(NOW_IST)
    screener = FakeRegistryScreener(default_outcome=ScreenOutcome(status="no_match", first_visible_at=None))

    row = compute_decision_row(client, BANK_0, req, clock, screener=screener, dpip_key=DPIP_KEY)
    assert row.status == "ok"
    assert row.registry_answers is not None
    # S5 Target: passing original screener leaves registry_answers empty!
    assert len(row.registry_answers) >= 1
    assert row.registry_answers[0]["outcome"]["status"] == "no_match"
    assert len(screener.calls) >= 1
    assert all(call["as_of"] is None for call in screener.calls)


def test_compute_decision_row_future_leak_cutoff_at_watermark() -> None:
    """
    G1 & SABOTAGE S10 TARGET:
    Verifies Section 22 arrival_cutoff correctness:
    A payment side record with event_time < decision_time but arrival_time > watermark
    is NOT counted in computed features.
    When watermark advances past arrival_time, it IS counted.
    """
    client = fakeredis.FakeRedis()
    t_watermark = NOW_UTC - timedelta(minutes=5)
    _set_all_watermarks(client, BANK_0, t_watermark)

    payer = "acc_payer_leak"
    req = _make_request(
        sender_account_id=payer,
        amount_paise=50000,
        event_time=NOW_UTC,
    )
    clock = ManualClock(NOW_IST)

    # Insert a prior failure side record: event_time < decision_time, but arrival_time > watermark
    leak_rec = PaymentSideRecord(
        run_id=RUN_ID,
        bank_id=BANK_0,
        account_id=payer,
        payment_id=f"{RUN_ID}-future-01",
        side="debit",
        counterparty_bank_id="bank_1",
        counterparty_vpa="acc_counter_1",
        amount_paise=25000,
        status="failed",
        failure_domain="customer",
        payee_type="person",
        channel="upi_app",
        sender_device_fingerprint="fp1",
        sender_os_generation=1,
        event_time=NOW_UTC - timedelta(minutes=10),
        arrival_time=NOW_UTC - timedelta(minutes=2),  # > t_watermark!
    )
    apply_payment_side(client, leak_rec)

    # 1. With arrival_cutoff = watermark (t_watermark), leak_rec must NOT be counted
    row1 = compute_decision_row(client, BANK_0, req, clock)
    assert row1.status == "ok"
    assert row1.arrival_cutoff == t_watermark
    names = canonical_feature_names()
    idx_failed = names.index("customer_failed_attempts_24h")
    values1 = row1.feature_values
    assert isinstance(values1, list)
    # S10 Target: if arrival_cutoff=decision_time is used, this will be 1!
    assert values1[idx_failed] == 0

    # 2. Advance watermark past arrival_time
    t_advanced = NOW_UTC + timedelta(minutes=1)
    _set_all_watermarks(client, BANK_0, t_advanced)
    row2 = compute_decision_row(client, BANK_0, req, clock)
    assert row2.status == "ok"
    assert row2.arrival_cutoff == t_advanced
    values2 = row2.feature_values
    assert isinstance(values2, list)
    assert values2[idx_failed] == 1


def test_compute_decision_row_watermark_lag_against_clock() -> None:
    """
    G4 & SABOTAGE S11 TARGET:
    Verifies that watermark_lag_ms is measured against clock.now_ist(), NOT decision_time.
    """
    client = fakeredis.FakeRedis()
    watermark_ts = datetime(2026, 10, 28, 12, 0, 0, tzinfo=timezone.utc)
    _set_all_watermarks(client, BANK_0, watermark_ts)

    # Historical transaction from 2 hours ago
    historical_event_time = datetime(2026, 10, 28, 10, 0, 0, tzinfo=timezone.utc)
    req = _make_request(event_time=historical_event_time)

    # Clock is 30 seconds after watermark
    clock_dt = (watermark_ts + timedelta(seconds=30)).astimezone(IST)
    clock = ManualClock(clock_dt)

    row = compute_decision_row(client, BANK_0, req, clock)
    assert row.status == "ok"
    # S11 Target: if measured against decision_time, lag would be -7,200,000 ms (-2 hours)
    assert row.watermark_lag_ms == 30_000


def test_compute_decision_row_only_payee_type_missing_vpa_features_computed() -> None:
    """
    G5 & SABOTAGE S12 TARGET:
    Verifies that when only payee_type is missing/invalid, the 3 payee-VPA features
    are computed (NOT None) and payee_type is None.
    """
    client = fakeredis.FakeRedis()
    watermark_ts = NOW_UTC + timedelta(seconds=60)
    _set_all_watermarks(client, BANK_0, watermark_ts)

    req = _make_request(
        payee_type=None,
        receiver_bank_id=BANK_0,
        receiver_vpa="acc_payee_target",
    )
    _seed_payer_debit(client, BANK_0, req.sender_account_id, NOW_UTC)
    clock = ManualClock(NOW_IST)

    screener = FakeRegistryScreener(default_outcome=ScreenOutcome(status="no_match", first_visible_at=None))

    row = compute_decision_row(client, BANK_0, req, clock, screener=screener, dpip_key=DPIP_KEY)
    assert row.status == "incomplete_request"
    assert row.missing_request_fields == ["payee_type"]

    names = canonical_feature_names()
    values = row.feature_values
    assert isinstance(values, list)

    # S12 Target: blanking VPA features when payee_type is missing causes these to be None!
    idx_first = names.index("first_payment_to_payee")
    assert values[idx_first] is not None
    assert values[idx_first] is True

    idx_risk = names.index("payee_same_bank_risk")
    assert values[idx_risk] is not None
    assert values[idx_risk] == 0.0

    idx_match = names.index("payee_registry_match")
    assert values[idx_match] is not None
    assert values[idx_match] is False

    idx_payee = names.index("payee_type")
    assert values[idx_payee] is None

    # Payer-only features are also not None
    assert values[names.index("amount_vs_payer_median_ratio")] is not None
    assert values[names.index("customer_failed_attempts_24h")] is not None
    assert values[names.index("hour_ist")] is not None


def test_compute_decision_row_only_receiver_vpa_missing_vpa_features_none() -> None:
    """
    G5: Verifies that when only receiver_vpa is missing/empty, the 3 payee-VPA features
    are None and payee_type is 'person'.
    """
    client = fakeredis.FakeRedis()
    watermark_ts = NOW_UTC + timedelta(seconds=60)
    _set_all_watermarks(client, BANK_0, watermark_ts)

    req = _make_request(
        receiver_vpa=None,
        payee_type="person",
    )
    _seed_payer_debit(client, BANK_0, req.sender_account_id, NOW_UTC)
    clock = ManualClock(NOW_IST)

    row = compute_decision_row(client, BANK_0, req, clock)
    assert row.status == "incomplete_request"
    assert row.missing_request_fields == ["receiver_vpa"]

    names = canonical_feature_names()
    values = row.feature_values
    assert isinstance(values, list)

    idx_first = names.index("first_payment_to_payee")
    assert values[idx_first] is None

    idx_risk = names.index("payee_same_bank_risk")
    assert values[idx_risk] is None

    idx_match = names.index("payee_registry_match")
    assert values[idx_match] is None

    idx_payee = names.index("payee_type")
    assert values[idx_payee] == "person"
