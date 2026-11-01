# File: tests/unit/test_world_mode_decision_watermark.py
# Purpose: Unit tests for world-mode decision feature extraction and fail-closed watermark checks.
"""
WHAT: Tests compute_decision_row with run_scoped_watermarks=True vs False.

WHY: Verifies Section 22 and Ground Rule 30 fail-closed semantics: world-mode decisions
query run-scoped watermarks ('acctwm:{run_id}:...'), fail closed with status='no_watermark'
and zero account-state reads when any source watermark is missing, while real-clock decisions
(run_scoped_watermarks=False) remain completely unaffected using legacy shared watermark keys.
"""

from datetime import datetime, timedelta, timezone
from typing import Any

import fakeredis
import pytest

from api.v1.models import DecisionRequest
from core.event_clock import IST, ManualClock
from features.account_keys import get_default_settings
from features.account_labels import apply_complaints_since, apply_feedback_since
from features.account_records import PaymentSideRecord
from features.account_stream import (
    apply_account_events_since,
    apply_withdrawals_since,
    record_watermark_for_side,
)
from features.account_summaries import to_unix_seconds
from features.account_updater import apply_payment_side
from features.decision_features import compute_decision_row
from features.ingest_watermark import (
    POLLER_SOURCES,
    record_payment_watermark,
    record_poller_watermark,
)

pytestmark = pytest.mark.anyio

RUN_ID = "66db9aa439b092ee"
BANK_0 = "bank_0"
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


def _setup_shared_and_partial_run_watermarks(
    client: Any,
    bank_id: str,
    t_nov: datetime,
    t_world: datetime,
) -> None:
    # 1. Real-clock old-key watermarks for ALL sources at Nov 2026
    record_payment_watermark(client, 0, t_nov, run_id=None)
    for source in POLLER_SOURCES:
        record_poller_watermark(client, bank_id, source, t_nov, run_id=None)

    # 2. Run-scoped complaint and feedback watermarks for run R (as commit 2 writes them)
    record_poller_watermark(client, bank_id, "complaints", t_world, run_id=RUN_ID)
    record_poller_watermark(client, bank_id, "feedback", t_world, run_id=RUN_ID)


class FakePollerConnection:
    """Minimal fake database connection returning empty rows for watermark polling tests."""

    async def fetch(self, *args: Any, **kwargs: Any) -> list[dict[str, Any]]:
        return []


async def test_world_mode_decision_uses_real_run_watermarks() -> None:
    """
    World-mode decision (run_scoped_watermarks=True) uses run-scoped watermarks written
    through the real writers for all five sources:
      1. complaints poller (apply_complaints_since with ManualClock + run_id)
      2. feedback poller (apply_feedback_since with ManualClock + run_id)
      3. events poller (apply_account_events_since with ManualClock + run_id)
      4. withdrawals poller (apply_withdrawals_since with ManualClock + run_id)
      5. payments watermark writer (record_watermark_for_side for world-clock payment record)
    Verifies that status != 'no_watermark' and arrival_cutoff equals the MIN of the five.
    """
    client = fakeredis.FakeRedis()
    cfg = get_default_settings()
    t_world = datetime(2026, 9, 15, 12, 0, 0, tzinfo=timezone.utc)
    conn = FakePollerConnection()

    # 1. World-clock payment record advances payments watermark (+100s)
    pmt_rec = PaymentSideRecord(
        run_id=RUN_ID,
        bank_id=BANK_0,
        account_id="acc_sender_1",
        payment_id=f"{RUN_ID}-pmt-watermark-01",
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
        event_time=t_world,
        arrival_time=t_world + timedelta(seconds=100),
        arrival_from_ingestion_clock=True,
    )
    record_watermark_for_side(client, 0, pmt_rec, cfg)

    # 2. Events poller with ManualClock (+200s)
    clock_evt = ManualClock((t_world + timedelta(seconds=200)).astimezone(IST))
    await apply_account_events_since(conn, client, cfg, bank_id=BANK_0, clock=clock_evt, run_id=RUN_ID)

    # 3. Withdrawals poller with ManualClock (+50s) -> this is the MIN
    clock_wth = ManualClock((t_world + timedelta(seconds=50)).astimezone(IST))
    await apply_withdrawals_since(conn, client, cfg, bank_id=BANK_0, clock=clock_wth, run_id=RUN_ID)

    # 4. Feedback poller with ManualClock (+400s)
    clock_fb = ManualClock((t_world + timedelta(seconds=400)).astimezone(IST))
    await apply_feedback_since(conn, client, cfg, bank_id=BANK_0, clock=clock_fb, run_id=RUN_ID)

    # 5. Complaints poller with ManualClock (+500s)
    clock_cmp = ManualClock((t_world + timedelta(seconds=500)).astimezone(IST))
    await apply_complaints_since(conn, client, cfg, bank_id=BANK_0, clock=clock_cmp, run_id=RUN_ID)

    _seed_payer_debit(client, BANK_0, "acc_sender_1", t_world)

    req = _make_request(payment_id=f"{RUN_ID}-0001", event_time=t_world)
    clock = ManualClock(t_world.astimezone(IST))

    row = compute_decision_row(client, BANK_0, req, clock, run_scoped_watermarks=True)

    # Withdrawals was the minimum watermark (+50s minus commit grace)
    min_watermark_sec = to_unix_seconds(t_world + timedelta(seconds=50)) - cfg.commit_grace
    expected_cutoff = datetime.fromtimestamp(min_watermark_sec, tz=timezone.utc)

    assert row.status != "no_watermark"
    assert row.arrival_cutoff == expected_cutoff


def test_world_mode_decision_fails_closed_when_source_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    World-mode decision (run_scoped_watermarks=True) with incomplete run-scoped sources
    returns status='no_watermark', all features None, and executes NO account-state reads.
    """
    client = fakeredis.FakeRedis()
    t_nov = datetime(2026, 11, 15, 12, 0, 0, tzinfo=timezone.utc)
    t_world = datetime(2026, 9, 15, 12, 0, 0, tzinfo=timezone.utc)

    _setup_shared_and_partial_run_watermarks(client, BANK_0, t_nov, t_world)

    # Monkeypatch read_decision_reads to ensure no account state reading occurred
    def fail_if_read(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("Account-state read_decision_reads was invoked despite missing watermark!")

    monkeypatch.setattr("features.decision_features.read_decision_reads", fail_if_read)

    req = _make_request(payment_id=f"{RUN_ID}-0001", event_time=t_world)
    clock = ManualClock(t_world.astimezone(IST))

    row = compute_decision_row(client, BANK_0, req, clock, run_scoped_watermarks=True)

    assert row.status == "no_watermark"
    assert row.arrival_cutoff is None
    assert row.watermark_lag_ms is None
    values = (
        list(row.feature_values.values())
        if isinstance(row.feature_values, dict)
        else list(row.feature_values)
    )
    assert len(values) > 0
    assert all(val is None for val in values)


def test_real_mode_decision_unchanged() -> None:
    """
    Real-mode decision (run_scoped_watermarks=False) on same Redis uses legacy old keys
    and does NOT return status='no_watermark'.
    """
    client = fakeredis.FakeRedis()
    t_nov = datetime(2026, 11, 15, 12, 0, 0, tzinfo=timezone.utc)
    t_world = datetime(2026, 9, 15, 12, 0, 0, tzinfo=timezone.utc)

    _setup_shared_and_partial_run_watermarks(client, BANK_0, t_nov, t_world)
    _seed_payer_debit(client, BANK_0, "acc_sender_1", t_nov)

    req = _make_request(payment_id=f"{RUN_ID}-0001", event_time=t_nov)
    clock = ManualClock(t_nov.astimezone(IST))

    row = compute_decision_row(client, BANK_0, req, clock, run_scoped_watermarks=False)

    assert row.status != "no_watermark"
    assert row.status == "ok"
    assert row.arrival_cutoff == t_nov


def test_world_mode_decision_uses_run_watermark_when_all_present() -> None:
    """
    World-mode decision uses run-scoped watermark when all 5 run-scoped sources are present,
    setting arrival_cutoff equal to their MIN.
    """
    client = fakeredis.FakeRedis()
    t_world = datetime(2026, 9, 15, 12, 0, 0, tzinfo=timezone.utc)

    # Record all 5 run-scoped sources
    record_payment_watermark(client, 0, 100, run_id=RUN_ID)
    record_poller_watermark(client, BANK_0, "events", 200, run_id=RUN_ID)
    record_poller_watermark(client, BANK_0, "withdrawals", 300, run_id=RUN_ID)
    record_poller_watermark(client, BANK_0, "feedback", 400, run_id=RUN_ID)
    record_poller_watermark(client, BANK_0, "complaints", 500, run_id=RUN_ID)

    _seed_payer_debit(client, BANK_0, "acc_sender_1", t_world)

    req = _make_request(payment_id=f"{RUN_ID}-0001", event_time=t_world)
    clock = ManualClock(t_world.astimezone(IST))

    row = compute_decision_row(client, BANK_0, req, clock, run_scoped_watermarks=True)

    expected_cutoff = datetime.fromtimestamp(100, tz=timezone.utc)
    assert row.status != "no_watermark"
    assert row.arrival_cutoff == expected_cutoff
