# File: tests/unit/test_account_labels.py
# Purpose: Unit tests for confirmed fraud labels, complaints polling, and same-bank counterparty edge extraction.
# Issue: Level 15b M1 network features (plan Task 1, G1, R8, architect decision 9b).
"""
WHAT: Unit tests covering:
  1. apply_feedback_since: confirmed_fraud labels credit-side account (mule) only, ignoring debit side (victim).
  2. Debit-only feedback labels nothing and logs an informational notice.
  3. not_fraud and customer_dispute outcomes are ignored.
  4. Complaints for other banks (reported_bank_id != this bank) are ignored.
  5. Repeat-safety: polling twice leaves Redis ZSET state identical.
  6. Point-in-time visibility: labels are invisible before available_from = max(event_time, created_at).
  7. Overlap re-read catches late-committing rows.
  8. Lifetime label and complaint keys have no expiry (TTL == -1).
  9. same_bank_counterparties extracts edge sets exclusively for this bank's accounts.
"""

from datetime import datetime, timedelta, timezone
from typing import Any

import fakeredis
import pytest

from features.account_keys import (
    complained_accounts_key,
    labelled_accounts_key,
    load_account_features_settings,
)
from features.account_labels import (
    apply_complaints_since,
    apply_feedback_since,
    complained_accounts_visible,
    labelled_accounts_visible,
    same_bank_counterparties,
)
from features.account_records import PaymentSideRecord
from features.account_summaries import update_account_summaries

pytestmark = pytest.mark.anyio

RUN_ID = "66db9aa439b092ee"
BANK_0 = "bank_0"
BANK_1 = "bank_1"
NOW_UTC = datetime(2026, 10, 28, 12, 0, 0, tzinfo=timezone.utc)
IST = timezone(timedelta(hours=5, minutes=30))


class FakeDBConnection:
    """In-memory mock database connection simulating bank_feedback, bank_complaints, and transactions tables."""

    def __init__(
        self,
        feedback_rows: list[dict[str, Any]] | None = None,
        complaint_rows: list[dict[str, Any]] | None = None,
        transaction_rows: list[dict[str, Any]] | None = None,
    ) -> None:
        self.feedback_rows = feedback_rows or []
        self.complaint_rows = complaint_rows or []
        self.transaction_rows = transaction_rows or []
        self.queries: list[tuple[str, tuple[Any, ...]]] = []

    async def fetch(self, query: str, *args: Any) -> list[dict[str, Any]]:
        self.queries.append((query, args))
        if "FROM bank_feedback" in query:
            bank_id, after_id, limit = args
            matching = [
                r for r in self.feedback_rows
                if r["bank_id"] == bank_id and r["id"] > after_id
            ]
            matching.sort(key=lambda r: r["id"])
            return matching[:limit]
        elif "FROM bank_complaints" in query:
            bank_id, after_id, limit = args
            matching = [
                r for r in self.complaint_rows
                if r["bank_id"] == bank_id and r["id"] > after_id
            ]
            matching.sort(key=lambda r: r["id"])
            return matching[:limit]
        elif "FROM transactions" in query:
            bank_id, payment_id = args
            matching = [
                r for r in self.transaction_rows
                if r["bank_id"] == bank_id and r["payment_id"] == payment_id
            ]
            return matching
        raise ValueError(f"Unexpected query: {query}")

    async def fetchval(self, query: str, *args: Any) -> Any:
        return NOW_UTC


@pytest.fixture
def redis_client() -> fakeredis.FakeRedis:
    return fakeredis.FakeRedis()


async def test_confirmed_fraud_labels_credit_side_and_not_sender(redis_client: fakeredis.FakeRedis) -> None:
    """
    CRITICAL: Confirmed fraud labels the receiving credit-side account (mule), never the sender (victim).
    (Sabotage proof: labelling the debit-side account instead fails this test).
    """
    payment_id = f"{RUN_ID}-pay-0001"
    # transactions: debit on acc_victim (sender), credit on acc_mule (receiver)
    txns = [
        {"bank_id": BANK_0, "payment_id": payment_id, "side": "debit", "account_id": "acc_victim_01"},
        {"bank_id": BANK_0, "payment_id": payment_id, "side": "credit", "account_id": "acc_mule_01"},
    ]
    feedbacks = [
        {
            "id": 1,
            "bank_id": BANK_0,
            "feedback_id": f"{RUN_ID}-fb-01",
            "payment_id": payment_id,
            "outcome": "confirmed_fraud",
            "event_time": NOW_UTC,
            "created_at": NOW_UTC,
        }
    ]
    conn = FakeDBConnection(feedback_rows=feedbacks, transaction_rows=txns)
    last_id = await apply_feedback_since(conn, redis_client, bank_id=BANK_0, after_id=0)
    assert last_id == 1

    # Hand-worked expectation: exactly 1 labelled account, 'acc_mule_01'; 'acc_victim_01' is not labelled
    # count(labelled) == 1, 'acc_mule_01' in set, 'acc_victim_01' not in set
    decision_time = NOW_UTC + timedelta(seconds=10)
    labelled = labelled_accounts_visible(redis_client, RUN_ID, BANK_0, decision_time, decision_time)
    assert labelled == frozenset({"acc_mule_01"})
    assert "acc_victim_01" not in labelled


async def test_only_debit_side_labels_nothing(redis_client: fakeredis.FakeRedis) -> None:
    """Verifies that if only a debit side exists in transactions, nothing is labelled."""
    payment_id = f"{RUN_ID}-pay-0002"
    txns = [
        {"bank_id": BANK_0, "payment_id": payment_id, "side": "debit", "account_id": "acc_victim_02"},
    ]
    feedbacks = [
        {
            "id": 2,
            "bank_id": BANK_0,
            "feedback_id": f"{RUN_ID}-fb-02",
            "payment_id": payment_id,
            "outcome": "confirmed_fraud",
            "event_time": NOW_UTC,
            "created_at": NOW_UTC,
        }
    ]
    conn = FakeDBConnection(feedback_rows=feedbacks, transaction_rows=txns)
    await apply_feedback_since(conn, redis_client, bank_id=BANK_0, after_id=0)

    decision_time = NOW_UTC + timedelta(seconds=10)
    labelled = labelled_accounts_visible(redis_client, RUN_ID, BANK_0, decision_time, decision_time)
    # Only debit side observed: no mule account identified
    assert labelled == frozenset()


async def test_not_fraud_and_customer_dispute_ignored(redis_client: fakeredis.FakeRedis) -> None:
    """Verifies that not_fraud and customer_dispute outcomes are ignored by the confirmed fraud poller."""
    p1 = f"{RUN_ID}-pay-0003"
    p2 = f"{RUN_ID}-pay-0004"
    txns = [
        {"bank_id": BANK_0, "payment_id": p1, "side": "credit", "account_id": "acc_legit_01"},
        {"bank_id": BANK_0, "payment_id": p2, "side": "credit", "account_id": "acc_dispute_01"},
    ]
    feedbacks = [
        {
            "id": 10,
            "bank_id": BANK_0,
            "feedback_id": f"{RUN_ID}-fb-10",
            "payment_id": p1,
            "outcome": "not_fraud",
            "event_time": NOW_UTC,
            "created_at": NOW_UTC,
        },
        {
            "id": 11,
            "bank_id": BANK_0,
            "feedback_id": f"{RUN_ID}-fb-11",
            "payment_id": p2,
            "outcome": "customer_dispute",
            "event_time": NOW_UTC,
            "created_at": NOW_UTC,
        },
    ]
    conn = FakeDBConnection(feedback_rows=feedbacks, transaction_rows=txns)
    last_id = await apply_feedback_since(conn, redis_client, bank_id=BANK_0, after_id=0)
    assert last_id == 11

    decision_time = NOW_UTC + timedelta(seconds=10)
    labelled = labelled_accounts_visible(redis_client, RUN_ID, BANK_0, decision_time, decision_time)
    assert labelled == frozenset()


async def test_complaint_about_other_bank_account_ignored(redis_client: fakeredis.FakeRedis) -> None:
    """Verifies that complaints reporting accounts at another bank are ignored."""
    complaints = [
        {
            "id": 1,
            "bank_id": BANK_0,
            "complaint_id": f"{RUN_ID}-cmp-01",
            "complainant_account": "acc_victim_01",
            "reported_account": "acc_mule_bank1",
            "reported_bank_id": BANK_1,  # Other bank
            "payment_ids": [f"{RUN_ID}-pay-10"],
            "claimed_type": "phishing",
            "received_at": NOW_UTC,
            "created_at": NOW_UTC,
        },
        {
            "id": 2,
            "bank_id": BANK_0,
            "complaint_id": f"{RUN_ID}-cmp-02",
            "complainant_account": "acc_victim_02",
            "reported_account": "acc_mule_bank0",
            "reported_bank_id": BANK_0,  # This bank
            "payment_ids": [f"{RUN_ID}-pay-11"],
            "claimed_type": "identity_theft",
            "received_at": NOW_UTC,
            "created_at": NOW_UTC,
        },
    ]
    conn = FakeDBConnection(complaint_rows=complaints)
    await apply_complaints_since(conn, redis_client, bank_id=BANK_0, after_id=0)

    decision_time = NOW_UTC + timedelta(seconds=10)
    complained = complained_accounts_visible(redis_client, RUN_ID, BANK_0, decision_time, decision_time)
    # Only the account at this bank (acc_mule_bank0) is recorded
    assert complained == frozenset({"acc_mule_bank0"})


async def test_labels_repeat_safe(redis_client: fakeredis.FakeRedis) -> None:
    """Verifies that re-running pollers over identical rows produces identical Redis state."""
    payment_id = f"{RUN_ID}-pay-0005"
    txns = [{"bank_id": BANK_0, "payment_id": payment_id, "side": "credit", "account_id": "acc_mule_05"}]
    feedbacks = [
        {
            "id": 1,
            "bank_id": BANK_0,
            "feedback_id": f"{RUN_ID}-fb-05",
            "payment_id": payment_id,
            "outcome": "confirmed_fraud",
            "event_time": NOW_UTC,
            "created_at": NOW_UTC,
        }
    ]
    conn = FakeDBConnection(feedback_rows=feedbacks, transaction_rows=txns)
    await apply_feedback_since(conn, redis_client, bank_id=BANK_0, after_id=0)
    key = labelled_accounts_key(RUN_ID, BANK_0)
    dump_1 = redis_client.zrange(key, 0, -1, withscores=True)

    # Second run
    await apply_feedback_since(conn, redis_client, bank_id=BANK_0, after_id=0)
    dump_2 = redis_client.zrange(key, 0, -1, withscores=True)
    assert dump_1 == dump_2


async def test_invisible_before_available_from(redis_client: fakeredis.FakeRedis) -> None:
    """Verifies that labelled accounts are invisible when decision_time < available_from."""
    t_event = datetime(2026, 10, 28, 10, 0, 0, tzinfo=timezone.utc)
    t_created = datetime(2026, 10, 28, 10, 5, 0, tzinfo=timezone.utc)
    # available_from = max(t_event, t_created) = 10:05:00 UTC
    payment_id = f"{RUN_ID}-pay-0006"
    txns = [{"bank_id": BANK_0, "payment_id": payment_id, "side": "credit", "account_id": "acc_mule_06"}]
    feedbacks = [
        {
            "id": 1,
            "bank_id": BANK_0,
            "feedback_id": f"{RUN_ID}-fb-06",
            "payment_id": payment_id,
            "outcome": "confirmed_fraud",
            "event_time": t_event,
            "created_at": t_created,
        }
    ]
    conn = FakeDBConnection(feedback_rows=feedbacks, transaction_rows=txns)
    await apply_feedback_since(conn, redis_client, bank_id=BANK_0, after_id=0)

    # 1. 1 second before available_from: invisible
    dec_early = t_created - timedelta(seconds=1)
    assert labelled_accounts_visible(redis_client, RUN_ID, BANK_0, dec_early, dec_early) == frozenset()

    # 2. Exactly at available_from: visible
    assert labelled_accounts_visible(redis_client, RUN_ID, BANK_0, t_created, t_created) == frozenset({"acc_mule_06"})


async def test_overlap_poller_catches_late_committing_row(redis_client: fakeredis.FakeRedis) -> None:
    """Verifies that the overlap re-read window catches late-committing rows with lower IDs."""
    p1 = f"{RUN_ID}-pay-0007"
    p2 = f"{RUN_ID}-pay-0008"
    txns = [
        {"bank_id": BANK_0, "payment_id": p1, "side": "credit", "account_id": "acc_mule_07"},
        {"bank_id": BANK_0, "payment_id": p2, "side": "credit", "account_id": "acc_mule_08"},
    ]
    # First batch: row 10 committed first
    conn = FakeDBConnection(
        feedback_rows=[
            {
                "id": 10,
                "bank_id": BANK_0,
                "feedback_id": f"{RUN_ID}-fb-10",
                "payment_id": p1,
                "outcome": "confirmed_fraud",
                "event_time": NOW_UTC,
                "created_at": NOW_UTC,
            }
        ],
        transaction_rows=txns,
    )
    last_id = await apply_feedback_since(conn, redis_client, bank_id=BANK_0, after_id=0)
    assert last_id == 10

    # Row 7 commits late (id 7 < 10)
    conn.feedback_rows.append(
        {
            "id": 7,
            "bank_id": BANK_0,
            "feedback_id": f"{RUN_ID}-fb-07",
            "payment_id": p2,
            "outcome": "confirmed_fraud",
            "event_time": NOW_UTC,
            "created_at": NOW_UTC,
        }
    )

    # Second poll starts with after_id=10; overlap re-read (10 - 200 -> start_id=0) catches row 7
    await apply_feedback_since(conn, redis_client, bank_id=BANK_0, after_id=10)

    # Both accounts must now be labelled
    decision_time = NOW_UTC + timedelta(seconds=10)
    labelled = labelled_accounts_visible(redis_client, RUN_ID, BANK_0, decision_time, decision_time)
    # Hand-worked expectation: {"acc_mule_07", "acc_mule_08"}
    assert labelled == frozenset({"acc_mule_07", "acc_mule_08"})


async def test_no_ttl_on_label_keys(redis_client: fakeredis.FakeRedis) -> None:
    """Verifies that labelled_accounts and complained_accounts lifetime keys have no TTL (expiry = -1)."""
    p1 = f"{RUN_ID}-pay-0009"
    c1 = f"{RUN_ID}-cmp-0009"
    conn = FakeDBConnection(
        feedback_rows=[
            {
                "id": 1,
                "bank_id": BANK_0,
                "feedback_id": f"{RUN_ID}-fb-09",
                "payment_id": p1,
                "outcome": "confirmed_fraud",
                "event_time": NOW_UTC,
                "created_at": NOW_UTC,
            }
        ],
        complaint_rows=[
            {
                "id": 1,
                "bank_id": BANK_0,
                "complaint_id": c1,
                "complainant_account": "acc_vic_09",
                "reported_account": "acc_cmp_09",
                "reported_bank_id": BANK_0,
                "payment_ids": [p1],
                "claimed_type": "fraud",
                "received_at": NOW_UTC,
                "created_at": NOW_UTC,
            }
        ],
        transaction_rows=[{"bank_id": BANK_0, "payment_id": p1, "side": "credit", "account_id": "acc_lbl_09"}],
    )
    await apply_feedback_since(conn, redis_client, bank_id=BANK_0, after_id=0)
    await apply_complaints_since(conn, redis_client, bank_id=BANK_0, after_id=0)

    lbl_key = labelled_accounts_key(RUN_ID, BANK_0)
    cmp_key = complained_accounts_key(RUN_ID, BANK_0)

    assert redis_client.ttl(lbl_key) == -1
    assert redis_client.ttl(cmp_key) == -1


def test_same_bank_counterparties_from_closed_days_and_raw_window(redis_client: fakeredis.FakeRedis) -> None:
    """
    CRITICAL: Verifies same_bank_counterparties helper extracts edges exclusively for this bank's accounts.
    Proves closed counterparty days (both directions) and 7d raw window are included, other banks excluded.
    """
    settings = load_account_features_settings()
    acc_main = "acc_main_01"

    # Closed day: Day 10 (Oct 11 2026 IST)
    # Day 10 end IST is Oct 11 23:59:59 IST.
    # Allowed lateness is 1 day. Closed threshold is Oct 12 23:59:59 IST.
    t_d10 = datetime(2026, 10, 11, 12, 0, 0, tzinfo=IST)

    # 1. Closed debit to same bank: bank_0:acc_peer_01 -> INCLUDED
    p_closed_debit = PaymentSideRecord(
        run_id=RUN_ID,
        bank_id=BANK_0,
        payment_id=f"{RUN_ID}-p-01",
        account_id=acc_main,
        side="debit",
        counterparty_vpa="acc_peer_01",
        counterparty_bank_id=BANK_0,
        amount_paise=100000,
        status="success",
        failure_domain=None,
        payee_type="person",
        channel="upi_app",
        sender_device_fingerprint=None,
        sender_os_generation=None,
        event_time=t_d10,
        arrival_time=t_d10,
        balance_after_paise=900000,
    )
    update_account_summaries(redis_client, p_closed_debit, settings)

    # 2. Closed credit from same bank: bank_0:acc_peer_02 -> INCLUDED
    p_closed_credit = PaymentSideRecord(
        run_id=RUN_ID,
        bank_id=BANK_0,
        payment_id=f"{RUN_ID}-p-02",
        account_id=acc_main,
        side="credit",
        counterparty_vpa="acc_peer_02",
        counterparty_bank_id=BANK_0,
        amount_paise=50000,
        status="success",
        failure_domain=None,
        payee_type="person",
        channel="upi_app",
        sender_device_fingerprint=None,
        sender_os_generation=None,
        event_time=t_d10,
        arrival_time=t_d10,
        balance_after_paise=950000,
    )
    update_account_summaries(redis_client, p_closed_credit, settings)

    # 3. Closed debit to OTHER bank: bank_1:acc_other_01 -> EXCLUDED (other bank)
    p_closed_other = PaymentSideRecord(
        run_id=RUN_ID,
        bank_id=BANK_0,
        payment_id=f"{RUN_ID}-p-03",
        account_id=acc_main,
        side="debit",
        counterparty_vpa="acc_other_01",
        counterparty_bank_id=BANK_1,
        amount_paise=20000,
        status="success",
        failure_domain=None,
        payee_type="person",
        channel="upi_app",
        sender_device_fingerprint=None,
        sender_os_generation=None,
        event_time=t_d10,
        arrival_time=t_d10,
        balance_after_paise=930000,
    )
    update_account_summaries(redis_client, p_closed_other, settings)

    # 4. Visible 7-day raw window: Oct 20 2026 IST
    t_raw = datetime(2026, 10, 20, 14, 0, 0, tzinfo=IST)
    from features.account_updater import apply_payment_side

    # Raw window debit to same bank: bank_0:acc_peer_03 -> INCLUDED
    p_raw_same = PaymentSideRecord(
        run_id=RUN_ID,
        bank_id=BANK_0,
        payment_id=f"{RUN_ID}-p-04",
        account_id=acc_main,
        side="debit",
        counterparty_vpa="acc_peer_03",
        counterparty_bank_id=BANK_0,
        amount_paise=30000,
        status="success",
        failure_domain=None,
        payee_type="merchant",
        channel="upi_app",
        sender_device_fingerprint=None,
        sender_os_generation=None,
        event_time=t_raw,
        arrival_time=t_raw,
        balance_after_paise=900000,
    )
    apply_payment_side(redis_client, p_raw_same, settings)

    # Raw window credit from other bank: bank_1:acc_other_02 -> EXCLUDED
    p_raw_other = PaymentSideRecord(
        run_id=RUN_ID,
        bank_id=BANK_0,
        payment_id=f"{RUN_ID}-p-05",
        account_id=acc_main,
        side="credit",
        counterparty_vpa="acc_other_02",
        counterparty_bank_id=BANK_1,
        amount_paise=15000,
        status="success",
        failure_domain=None,
        payee_type="person",
        channel="upi_app",
        sender_device_fingerprint=None,
        sender_os_generation=None,
        event_time=t_raw,
        arrival_time=t_raw,
        balance_after_paise=915000,
    )
    apply_payment_side(redis_client, p_raw_other, settings)

    decision_time = datetime(2026, 10, 20, 16, 0, 0, tzinfo=IST)
    cps = same_bank_counterparties(redis_client, RUN_ID, BANK_0, acc_main, decision_time, decision_time, settings)

    # Arithmetic comment:
    # Included: acc_peer_01 (closed debit), acc_peer_02 (closed credit), acc_peer_03 (raw debit)
    # Excluded: acc_other_01 (bank_1), acc_other_02 (bank_1)
    # Total expected: exactly 3 accounts: {"acc_peer_01", "acc_peer_02", "acc_peer_03"}
    assert cps == frozenset({"acc_peer_01", "acc_peer_02", "acc_peer_03"})
