# File: tests/unit/test_ingestion_clock_labels.py
# Purpose: Level 16a Round 2 - unit tests for ingestion clock arrival stamping, visibility, and watermarks.
import uuid
from datetime import datetime, timedelta
from typing import Any

import fakeredis
import pytest

from core.event_clock import IST, ManualClock
from features.account_keys import poller_watermark_key, watermark_key_for_poller
from features.account_labels import (
    apply_complaint_row,
    apply_complaints_since,
    apply_feedback_row,
    apply_feedback_since,
    arrival_of,
    complained_accounts_visible,
    labelled_accounts_visible,
    to_unix_seconds,
)


class FakePollerConnection:
    """Mock database connection simulating bank_feedback, bank_complaints, and transactions tables."""

    def __init__(
        self,
        db_now: datetime,
        complaint_rows: list[dict[str, Any]] | None = None,
        feedback_rows: list[dict[str, Any]] | None = None,
        transaction_rows: list[dict[str, Any]] | None = None,
    ) -> None:
        self.db_now = db_now
        self.complaint_rows = complaint_rows or []
        self.feedback_rows = feedback_rows or []
        self.transaction_rows = transaction_rows or []
        self.queries: list[tuple[str, tuple[Any, ...]]] = []

    async def fetch(self, query: str, *args: Any) -> list[dict[str, Any]]:
        self.queries.append((query, args))
        if "FROM bank_complaints" in query:
            bank_id, after_id, limit = args
            return [r for r in self.complaint_rows if r["bank_id"] == bank_id and r["id"] > after_id]
        if "FROM bank_feedback" in query:
            bank_id, after_id, limit = args
            return [r for r in self.feedback_rows if r["bank_id"] == bank_id and r["id"] > after_id]
        if "FROM transactions" in query:
            bank_id, payment_id = args
            return [r for r in self.transaction_rows if r["bank_id"] == bank_id and r["payment_id"] == payment_id]
        raise ValueError(f"Unexpected query in fetch: {query}")

    async def fetchval(self, query: str, *args: Any) -> Any:
        self.queries.append((query, args))
        if query == "SELECT now()":
            return self.db_now
        raise ValueError(f"Unexpected query in fetchval: {query}")


@pytest.fixture
def redis_client() -> fakeredis.FakeRedis:
    return fakeredis.FakeRedis()


def test_w1_complaint_world_mode_visibility(redis_client: fakeredis.FakeRedis) -> None:
    """
    W1: Complaint in world mode with received_at = arrival_time (2026-09-05) and created_at = 2026-11-01
    is visible on 2026-09-05 and 2026-09-06.
    """
    run_id = uuid.uuid4().hex[:16]
    bank_id = "bank_0"
    mule_acc = "acc_mule_w1"

    t_world = datetime(2026, 9, 5, 14, 0, 0, tzinfo=IST)
    t_real_now = datetime(2026, 11, 1, 12, 0, 0, tzinfo=IST)

    row = {
        "id": 1,
        "bank_id": bank_id,
        "complaint_id": f"{run_id}-cmp-001",
        "complainant_account": "acc_vic_01",
        "reported_account": mule_acc,
        "reported_bank_id": bank_id,
        "payment_ids": ["pay_01"],
        "claimed_type": "phishing",
        "received_at": t_world,
        "arrival_time": t_world,
        "created_at": t_real_now,
    }

    apply_complaint_row(redis_client, row)

    # Visible at exactly 2026-09-05 14:00
    vis_exact = complained_accounts_visible(
        redis_client, run_id, bank_id, decision_time=t_world, arrival_cutoff=t_world
    )
    assert mule_acc in vis_exact

    # Visible on 2026-09-06
    t_next_day = datetime(2026, 9, 6, 14, 0, 0, tzinfo=IST)
    vis_next_day = complained_accounts_visible(
        redis_client, run_id, bank_id, decision_time=t_next_day, arrival_cutoff=t_next_day
    )
    assert mule_acc in vis_next_day


@pytest.mark.anyio
async def test_w2_feedback_world_mode_visibility(redis_client: fakeredis.FakeRedis) -> None:
    """
    W2: Feedback in world mode with event_time = arrival_time (2026-09-05) and created_at = 2026-11-01
    labels credit-side mule.
    """
    run_id = uuid.uuid4().hex[:16]
    bank_id = "bank_0"
    mule_acc = "acc_mule_w2"
    payment_id = f"{run_id}-pmt-w2"

    t_world = datetime(2026, 9, 5, 14, 0, 0, tzinfo=IST)
    t_real_now = datetime(2026, 11, 1, 12, 0, 0, tzinfo=IST)

    txns = [{"bank_id": bank_id, "payment_id": payment_id, "side": "credit", "account_id": mule_acc}]
    conn = FakePollerConnection(db_now=t_real_now, transaction_rows=txns)

    row = {
        "id": 1,
        "bank_id": bank_id,
        "feedback_id": f"{run_id}-fb-001",
        "payment_id": payment_id,
        "outcome": "confirmed_fraud",
        "event_time": t_world,
        "arrival_time": t_world,
        "created_at": t_real_now,
    }

    await apply_feedback_row(conn, redis_client, row)

    # Visible at exactly 2026-09-05 14:00
    vis_exact = labelled_accounts_visible(
        redis_client, run_id, bank_id, decision_time=t_world, arrival_cutoff=t_world
    )
    assert mule_acc in vis_exact

    # Visible on 2026-09-06
    t_next_day = datetime(2026, 9, 6, 14, 0, 0, tzinfo=IST)
    vis_next_day = labelled_accounts_visible(
        redis_client, run_id, bank_id, decision_time=t_next_day, arrival_cutoff=t_next_day
    )
    assert mule_acc in vis_next_day


@pytest.mark.anyio
async def test_w3_boundary_visibility(redis_client: fakeredis.FakeRedis) -> None:
    """
    W3: Complaint and feedback arriving at 2026-09-10 are invisible at 2026-09-09 23:59:59,
    visible at 2026-09-10 00:00:00.
    """
    run_id = uuid.uuid4().hex[:16]
    bank_id = "bank_0"
    mule_cmp = "acc_mule_cmp_w3"
    mule_fb = "acc_mule_fb_w3"
    pmt_fb = f"{run_id}-pmt-fb-w3"

    t_boundary = datetime(2026, 9, 10, 0, 0, 0, tzinfo=IST)
    t_before = t_boundary - timedelta(seconds=1)

    # 1. Complaint arriving at boundary
    cmp_row = {
        "id": 10,
        "bank_id": bank_id,
        "complaint_id": f"{run_id}-cmp-003",
        "complainant_account": "acc_vic_03",
        "reported_account": mule_cmp,
        "reported_bank_id": bank_id,
        "payment_ids": ["pay_03"],
        "claimed_type": "phishing",
        "received_at": t_boundary,
        "arrival_time": t_boundary,
        "created_at": datetime(2026, 11, 1, 0, 0, 0, tzinfo=IST),
    }
    apply_complaint_row(redis_client, cmp_row)

    # 2. Feedback arriving at boundary
    txns = [{"bank_id": bank_id, "payment_id": pmt_fb, "side": "credit", "account_id": mule_fb}]
    conn = FakePollerConnection(db_now=datetime(2026, 11, 1, 0, 0, 0, tzinfo=IST), transaction_rows=txns)
    fb_row = {
        "id": 20,
        "bank_id": bank_id,
        "feedback_id": f"{run_id}-fb-003",
        "payment_id": pmt_fb,
        "outcome": "confirmed_fraud",
        "event_time": t_boundary,
        "arrival_time": t_boundary,
        "created_at": datetime(2026, 11, 1, 0, 0, 0, tzinfo=IST),
    }
    await apply_feedback_row(conn, redis_client, fb_row)

    # At t_before: neither is visible
    assert mule_cmp not in complained_accounts_visible(
        redis_client, run_id, bank_id, decision_time=t_before, arrival_cutoff=t_before
    )
    assert mule_fb not in labelled_accounts_visible(
        redis_client, run_id, bank_id, decision_time=t_before, arrival_cutoff=t_before
    )

    # At t_boundary: both are visible
    assert mule_cmp in complained_accounts_visible(
        redis_client, run_id, bank_id, decision_time=t_boundary, arrival_cutoff=t_boundary
    )
    assert mule_fb in labelled_accounts_visible(
        redis_client, run_id, bank_id, decision_time=t_boundary, arrival_cutoff=t_boundary
    )


@pytest.mark.anyio
async def test_w4_poller_injected_clock_watermark_no_select_now(redis_client: fakeredis.FakeRedis) -> None:
    """W4: Pollers with an injected ManualClock record watermark = time - commit_grace, and never call SELECT now()."""
    run_id = "66db9aa439b092ee"
    t_poll = datetime(2026, 9, 5, 14, 0, 0, tzinfo=IST)
    clock = ManualClock(t_poll)
    conn = FakePollerConnection(db_now=datetime(2026, 11, 1, 0, 0, 0, tzinfo=IST))

    # Test complaints poller
    await apply_complaints_since(conn, redis_client, bank_id="bank_0", clock=clock, run_id=run_id)

    expected_watermark = to_unix_seconds(t_poll) - 5
    cmp_wm = redis_client.get(watermark_key_for_poller("bank_0", "complaints", run_id=run_id))
    assert cmp_wm is not None
    assert int(cmp_wm) == expected_watermark

    # Shared legacy key must NOT be set
    assert redis_client.get(poller_watermark_key("bank_0", "complaints")) is None

    # Verify "SELECT now()" was NEVER called
    queries_text = [q[0] for q in conn.queries]
    assert not any("SELECT now()" in q for q in queries_text)

    # Test feedback poller
    conn.queries.clear()
    await apply_feedback_since(conn, redis_client, bank_id="bank_0", clock=clock, run_id=run_id)

    fb_wm = redis_client.get(watermark_key_for_poller("bank_0", "feedback", run_id=run_id))
    assert fb_wm is not None
    assert int(fb_wm) == expected_watermark

    # Shared legacy key must NOT be set
    assert redis_client.get(poller_watermark_key("bank_0", "feedback")) is None

    queries_text = [q[0] for q in conn.queries]
    assert not any("SELECT now()" in q for q in queries_text)


@pytest.mark.anyio
async def test_n1_arrival_time_null_falls_back_to_created_at(redis_client: fakeredis.FakeRedis) -> None:
    """N1: When arrival_time is NULL, arrival_of falls back to created_at for both complaints and feedback."""
    run_id = uuid.uuid4().hex[:16]
    bank_id = "bank_0"
    mule_cmp = "acc_mule_cmp_n1"
    mule_fb = "acc_mule_fb_n1"
    pmt_fb = f"{run_id}-pmt-fb-n1"

    t_event = datetime(2026, 9, 1, 10, 0, 0, tzinfo=IST)
    t_created = datetime(2026, 9, 5, 10, 0, 0, tzinfo=IST)

    # 1. Complaint with arrival_time = None
    cmp_row = {
        "id": 1,
        "bank_id": bank_id,
        "complaint_id": f"{run_id}-cmp-001",
        "complainant_account": "acc_vic_01",
        "reported_account": mule_cmp,
        "reported_bank_id": bank_id,
        "payment_ids": ["pay_01"],
        "claimed_type": "phishing",
        "received_at": t_event,
        "arrival_time": None,
        "created_at": t_created,
    }
    assert arrival_of(cmp_row) == t_created
    apply_complaint_row(redis_client, cmp_row)

    # Invisible before created_at
    assert mule_cmp not in complained_accounts_visible(
        redis_client, run_id, bank_id, decision_time=t_event, arrival_cutoff=t_event
    )
    # Visible at created_at
    assert mule_cmp in complained_accounts_visible(
        redis_client, run_id, bank_id, decision_time=t_created, arrival_cutoff=t_created
    )

    # 2. Feedback with arrival_time omitted
    txns = [{"bank_id": bank_id, "payment_id": pmt_fb, "side": "credit", "account_id": mule_fb}]
    conn = FakePollerConnection(db_now=t_created, transaction_rows=txns)
    fb_row = {
        "id": 2,
        "bank_id": bank_id,
        "feedback_id": f"{run_id}-fb-001",
        "payment_id": pmt_fb,
        "outcome": "confirmed_fraud",
        "event_time": t_event,
        "created_at": t_created,
    }
    assert arrival_of(fb_row) == t_created
    await apply_feedback_row(conn, redis_client, fb_row)

    # Invisible before created_at
    assert mule_fb not in labelled_accounts_visible(
        redis_client, run_id, bank_id, decision_time=t_event, arrival_cutoff=t_event
    )
    # Visible at created_at
    assert mule_fb in labelled_accounts_visible(
        redis_client, run_id, bank_id, decision_time=t_created, arrival_cutoff=t_created
    )


def test_n2_naive_arrival_time_raises_value_error() -> None:
    """N2: A naive datetime in arrival_time raises ValueError naming the row id."""
    row = {
        "complaint_id": "0123456789abcdef-cmp-bad",
        "created_at": datetime(2026, 9, 5, 14, 0, 0, tzinfo=IST),
        "arrival_time": datetime(2026, 9, 5, 14, 0, 0),  # naive
    }
    with pytest.raises(ValueError, match="0123456789abcdef-cmp-bad"):
        arrival_of(row)


@pytest.mark.anyio
async def test_r1_real_clock_mode_without_clock_calls_select_now(redis_client: fakeredis.FakeRedis) -> None:
    """R1: In real-clock mode (clock=None), pollers query SELECT now() exactly as before."""
    db_now = datetime(2026, 11, 1, 12, 0, 0, tzinfo=IST)
    conn = FakePollerConnection(db_now=db_now)

    await apply_complaints_since(conn, redis_client, bank_id="bank_0", clock=None)
    assert any(q[0] == "SELECT now()" for q in conn.queries)

    conn.queries.clear()
    await apply_feedback_since(conn, redis_client, bank_id="bank_0", clock=None)
    assert any(q[0] == "SELECT now()" for q in conn.queries)
