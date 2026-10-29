# File: tests/unit/test_account_summaries.py
# Purpose: Unit tests validating long-lived account summaries, closed-day reading, and lifetime metrics.
# Issue: Level 15b long-history summaries (plan Task 1, G1, G2, R5, R7).
"""
WHAT: Unit tests covering long-lived Redis account summaries in features/account_summaries.py:
day totals idempotency, on-time and late arrivals within lateness, closed-day reading boundaries,
retention trimming of day totals and seen sets, counterparty tracking, opened_at origin,
device fingerprint tracking with conservative availability and warning logging,
shuffle order invariance across mixed records, and history_days_available calculation.

WHY: Verifies the Section 22 future-leak guarantees and idempotency required by the Level 15
architecture, ensuring summaries never leak unclosed days or future information.
"""

import logging
import random
from datetime import datetime, timedelta, timezone
from typing import Any

import fakeredis
import pytest

from features.account_keys import (
    active_days_key,
    day_seen_key,
    day_totals_key,
    days_index_key,
    fingerprints_key,
    load_account_features_settings,
    marks_key,
    opened_at_key,
    side_records_key,
    side_times_key,
)
from features.account_reader import read_account_snapshot
from features.account_records import (
    AccountEventRecord,
    PaymentSideRecord,
    WithdrawalRecord,
)
from features.account_summaries import (
    IST,
    history_days_available,
    read_counterparty_days,
    read_day_totals,
    read_lifetime,
    to_day_number,
    update_account_summaries,
)
from features.account_updater import apply_account_event, apply_payment_side

RUN_ID = "66db9aa439b092ee"
BANK_0 = "bank_0"
ACC_1 = "acc_person_0001"
BASE_TIME = datetime(2026, 10, 1, 12, 0, 0, tzinfo=IST)


@pytest.fixture
def redis_client() -> fakeredis.FakeRedis:
    return fakeredis.FakeRedis()


def _make_payment(
    payment_id: str,
    event_time: datetime,
    arrival_time: datetime,
    side: str = "credit",
    amount_paise: int = 50000,
    status: str = "success",
    counterparty_vpa: str = "cp_acc_002",
    sender_device_fingerprint: str | None = None,
    account_id: str = ACC_1,
) -> PaymentSideRecord:
    """Helper constructing a valid PaymentSideRecord."""
    return PaymentSideRecord(
        run_id=RUN_ID,
        bank_id=BANK_0,
        account_id=account_id,
        payment_id=payment_id,
        side=side,
        counterparty_bank_id="bank_1",
        counterparty_vpa=counterparty_vpa,
        amount_paise=amount_paise,
        status=status,
        failure_domain=None if status == "success" else "receiver_bank",
        payee_type="person",
        channel="upi_app",
        sender_device_fingerprint=sender_device_fingerprint,
        sender_os_generation=1,
        event_time=event_time,
        arrival_time=arrival_time,
        balance_after_paise=100000,
    )


def _make_withdrawal(
    withdrawal_id: str,
    event_time: datetime,
    arrival_time: datetime,
    amount_paise: int = 20000,
    status: str = "success",
    account_id: str = ACC_1,
) -> WithdrawalRecord:
    """Helper constructing a valid WithdrawalRecord."""
    return WithdrawalRecord(
        run_id=RUN_ID,
        bank_id=BANK_0,
        account_id=account_id,
        withdrawal_id=withdrawal_id,
        channel="atm",
        amount_paise=amount_paise,
        status=status,
        failure_domain=None if status == "success" else "atm_hardware",
        balance_after_paise=80000,
        event_time=event_time,
        arrival_time=arrival_time,
    )


def _make_event(
    event_id: str,
    kind: str,
    event_time: datetime,
    arrival_time: datetime,
    details: dict[str, Any] | None = None,
    account_id: str = ACC_1,
) -> AccountEventRecord:
    """Helper constructing a valid AccountEventRecord."""
    return AccountEventRecord(
        run_id=RUN_ID,
        bank_id=BANK_0,
        account_id=account_id,
        event_id=event_id,
        kind=kind,
        event_time=event_time,
        arrival_time=arrival_time,
        details=details if details is not None else {},
    )


def _dump_all_keys(client: fakeredis.FakeRedis, pattern: str) -> dict[bytes, Any]:
    """Helper dumping sorted keys and structures for byte-identical comparison."""
    dump: dict[bytes, Any] = {}
    for k in sorted(client.scan_iter(match=pattern)):
        k_type = client.type(k)
        if k_type == b"zset":
            dump[k] = (k_type, client.zrange(k, 0, -1, withscores=True))
        elif k_type == b"hash":
            raw_hash: Any = client.hgetall(k)
            dump[k] = (k_type, sorted(raw_hash.items()))
        elif k_type == b"set":
            raw_set: Any = client.smembers(k)
            dump[k] = (k_type, sorted(raw_set))
        else:
            dump[k] = (k_type, client.get(k))
    return dump


# 1. Same event twice -> day totals exact
def test_1_same_event_twice_day_totals_exact(redis_client: fakeredis.FakeRedis) -> None:
    """Verifies that applying the exact same event multiple times produces exact, uninflated totals."""
    settings = load_account_features_settings()
    rec = _make_payment(f"{RUN_ID}-p-001", BASE_TIME, BASE_TIME, side="credit", amount_paise=50000)

    # Apply twice
    res1 = update_account_summaries(redis_client, rec, settings)
    assert res1 == "stored"
    res2 = update_account_summaries(redis_client, rec, settings)
    assert res2 == "stored"

    # Evaluate at decision time well after day closed (3 days later)
    decision_time = BASE_TIME + timedelta(days=3)
    totals = read_day_totals(redis_client, RUN_ID, BANK_0, ACC_1, decision_time, decision_time, settings)

    assert len(totals) == 1
    t = totals[0]
    assert t.credit_count == 1
    assert t.credit_paise == 50000
    assert t.debit_count == 0
    assert t.failed_attempt_count == 0


# 2. Same event once on time and once again late (within lateness) -> day totals exact
def test_2_same_event_on_time_and_late_within_lateness_day_totals_exact(
    redis_client: fakeredis.FakeRedis,
) -> None:
    """Verifies that an event redelivered later within allowed lateness leaves day totals exact."""
    settings = load_account_features_settings()
    event_time = BASE_TIME
    arrival_on_time = BASE_TIME + timedelta(hours=1)
    arrival_late_valid = BASE_TIME + timedelta(hours=20)  # Next day, within 86400s

    rec_first = _make_payment(f"{RUN_ID}-p-002", event_time, arrival_on_time, side="debit", amount_paise=25000)
    rec_second = _make_payment(f"{RUN_ID}-p-002", event_time, arrival_late_valid, side="debit", amount_paise=25000)

    update_account_summaries(redis_client, rec_first, settings)
    update_account_summaries(redis_client, rec_second, settings)

    decision_time = BASE_TIME + timedelta(days=3)
    totals = read_day_totals(redis_client, RUN_ID, BANK_0, ACC_1, decision_time, decision_time, settings)

    assert len(totals) == 1
    assert totals[0].debit_count == 1
    assert totals[0].debit_paise == 25000


# 3. An event arriving after day end + lateness -> "too_late_for_summary"; totals unchanged; still present in raw window
def test_3_event_arriving_after_day_end_plus_lateness_too_late_for_summary(
    redis_client: fakeredis.FakeRedis,
) -> None:
    """
    Verifies that events arriving after day end + allowed_lateness are excluded
    from summary but stored in raw window.
    """
    settings = load_account_features_settings()
    event_time = BASE_TIME
    # Allowed lateness is 86400s from day end (midnight of next day). Lateness threshold = Oct 3 00:00:00 IST.
    arrival_too_late = datetime(2026, 10, 3, 2, 0, 0, tzinfo=IST)

    rec = _make_payment(f"{RUN_ID}-p-003", event_time, arrival_too_late, side="credit", amount_paise=75000)

    res = update_account_summaries(redis_client, rec, settings)
    assert res == "too_late_for_summary"

    # Summaries must remain empty
    decision_time = BASE_TIME + timedelta(days=5)
    totals = read_day_totals(redis_client, RUN_ID, BANK_0, ACC_1, decision_time, decision_time, settings)
    assert len(totals) == 0

    # Still present in the raw window when applied via apply_payment_side
    raw_status = apply_payment_side(redis_client, rec, settings)
    assert raw_status == "stored"
    snap = read_account_snapshot(redis_client, RUN_ID, BANK_0, ACC_1, decision_time, arrival_too_late, settings)
    assert len(snap.side_records) == 1
    assert snap.side_records[0].payment_id == f"{RUN_ID}-p-003"


# 4. Reader excludes today and any day not yet closed by lateness; includes a day that has closed
def test_4_reader_excludes_unclosed_and_includes_closed(redis_client: fakeredis.FakeRedis) -> None:
    """Verifies that summary readers strictly exclude unclosed calendar days until day_end + lateness."""
    settings = load_account_features_settings()

    # Day 1 event (Oct 1). Closed at Oct 3 00:00:00 IST.
    p_day1 = _make_payment(
        f"{RUN_ID}-p-d1",
        datetime(2026, 10, 1, 10, 0, tzinfo=IST),
        datetime(2026, 10, 1, 11, 0, tzinfo=IST),
    )

    # Day 2 event (Oct 2). Closed at Oct 4 00:00:00 IST.
    p_day2 = _make_payment(
        f"{RUN_ID}-p-d2",
        datetime(2026, 10, 2, 10, 0, tzinfo=IST),
        datetime(2026, 10, 2, 11, 0, tzinfo=IST),
    )

    update_account_summaries(redis_client, p_day1, settings)
    update_account_summaries(redis_client, p_day2, settings)

    # Checkpoint 1: During Day 2 (Oct 2 15:00:00 IST) -> Neither day closed yet
    snap1 = read_day_totals(
        redis_client, RUN_ID, BANK_0, ACC_1,
        datetime(2026, 10, 2, 15, 0, tzinfo=IST), datetime(2026, 10, 2, 15, 0, tzinfo=IST),
        settings,
    )
    assert snap1 == ()

    # Checkpoint 2: Oct 3 12:00:00 IST -> Day 1 is closed; Day 2 is not
    snap2 = read_day_totals(
        redis_client, RUN_ID, BANK_0, ACC_1,
        datetime(2026, 10, 3, 12, 0, tzinfo=IST), datetime(2026, 10, 3, 12, 0, tzinfo=IST),
        settings,
    )
    assert len(snap2) == 1
    assert snap2[0].day == to_day_number(datetime(2026, 10, 1, 10, 0, tzinfo=IST))

    # Checkpoint 3: Oct 4 12:00:00 IST -> Both Day 1 and Day 2 are closed
    snap3 = read_day_totals(
        redis_client, RUN_ID, BANK_0, ACC_1,
        datetime(2026, 10, 4, 12, 0, tzinfo=IST), datetime(2026, 10, 4, 12, 0, tzinfo=IST),
        settings,
    )
    assert len(snap3) == 2


# 5. Trim removes days older than the horizon together with their seen sets (no orphans); the days index matches
def test_5_trim_removes_expired_days_and_seen_sets_no_orphans(redis_client: fakeredis.FakeRedis) -> None:
    """Verifies that trimming removes days older than summary_horizon_days and purges seen sets atomically."""
    settings = load_account_features_settings()

    dt_old = datetime(2026, 1, 1, 10, 0, tzinfo=IST)
    dt_new = dt_old + timedelta(days=100)  # 100 days later (> 62d horizon)

    day_old = to_day_number(dt_old)
    day_new = to_day_number(dt_new)

    p_old = _make_payment(f"{RUN_ID}-p-old", dt_old, dt_old, amount_paise=10000)
    p_new = _make_payment(f"{RUN_ID}-p-new", dt_new, dt_new, amount_paise=20000)

    update_account_summaries(redis_client, p_old, settings)
    assert redis_client.exists(day_totals_key(RUN_ID, BANK_0, ACC_1, day_old)) == 1
    assert redis_client.exists(day_seen_key(RUN_ID, BANK_0, ACC_1, day_old)) == 1

    # Applying the new payment triggers trimming
    update_account_summaries(redis_client, p_new, settings)

    # Day old keys must be completely purged (no orphans)
    assert redis_client.exists(day_totals_key(RUN_ID, BANK_0, ACC_1, day_old)) == 0
    assert redis_client.exists(day_seen_key(RUN_ID, BANK_0, ACC_1, day_old)) == 0
    assert redis_client.zscore(days_index_key(RUN_ID, BANK_0, ACC_1), str(day_old)) is None

    # Day new keys must exist
    assert redis_client.exists(day_totals_key(RUN_ID, BANK_0, ACC_1, day_new)) == 1
    assert redis_client.exists(day_seen_key(RUN_ID, BANK_0, ACC_1, day_new)) == 1


# 6. Counterparty days: the same payer on 3 different days -> 3 days; the same day twice -> 1
def test_6_counterparty_days_distinct_and_idempotent(redis_client: fakeredis.FakeRedis) -> None:
    """Verifies counterparty days ZSET tracks distinct active days and deduplicates repeat sightings."""
    settings = load_account_features_settings()
    cp = "bank_1:vpa_person_099"

    d1 = datetime(2026, 10, 1, 10, 0, tzinfo=IST)
    d2 = datetime(2026, 10, 2, 10, 0, tzinfo=IST)
    d3 = datetime(2026, 10, 3, 10, 0, tzinfo=IST)

    # 3 distinct days
    p1 = _make_payment(f"{RUN_ID}-p-01", d1, d1, side="credit", counterparty_vpa="vpa_person_099")
    p2 = _make_payment(f"{RUN_ID}-p-02", d2, d2, side="credit", counterparty_vpa="vpa_person_099")
    p3 = _make_payment(f"{RUN_ID}-p-03", d3, d3, side="credit", counterparty_vpa="vpa_person_099")
    # Same day 2 again (duplicate)
    p2_dup = _make_payment(f"{RUN_ID}-p-02b", d2, d2, side="credit", counterparty_vpa="vpa_person_099")

    for p in (p1, p2, p3, p2_dup):
        update_account_summaries(redis_client, p, settings)

    # Query at decision time after all 3 days have closed (Oct 6)
    decision = datetime(2026, 10, 6, 0, 0, tzinfo=IST)
    cp_days = read_counterparty_days(redis_client, RUN_ID, BANK_0, ACC_1, "credit", decision, decision, settings)

    assert cp in cp_days
    assert len(cp_days[cp]) == 3
    assert cp_days[cp] == [to_day_number(d1), to_day_number(d2), to_day_number(d3)]


# 7. opened_at comes from account_opened, not from the first payment seen
def test_7_opened_at_comes_from_account_opened_not_first_payment(
    redis_client: fakeredis.FakeRedis,
) -> None:
    """Verifies that opened_at is recorded strictly from account_opened and not from payment sightings."""
    settings = load_account_features_settings()

    # 1. Payment seen first
    p = _make_payment(
        f"{RUN_ID}-p-01",
        datetime(2026, 10, 1, 10, 0, tzinfo=IST),
        datetime(2026, 10, 1, 10, 0, tzinfo=IST),
    )
    update_account_summaries(redis_client, p, settings)

    decision = datetime(2026, 10, 5, 0, 0, tzinfo=IST)
    life_initial = read_lifetime(redis_client, RUN_ID, BANK_0, ACC_1, decision, decision, settings)
    assert life_initial.opened_at is None
    assert life_initial.last_active_day == to_day_number(datetime(2026, 10, 1, 10, 0, tzinfo=IST))

    # 2. account_opened event arrives with genuine historical opening date
    opening_date = datetime(2025, 4, 15, 9, 30, tzinfo=timezone.utc)
    ev_opened = _make_event(
        f"{RUN_ID}-open-01",
        "account_opened",
        opening_date,
        datetime(2026, 10, 1, 11, 0, tzinfo=IST),
    )
    update_account_summaries(redis_client, ev_opened, settings)

    life_after = read_lifetime(redis_client, RUN_ID, BANK_0, ACC_1, decision, decision, settings)
    assert life_after.opened_at == opening_date


# 8. Fingerprint: OS update same fingerprint adds nothing new; different phone adds one;
# sighting arrived after cutoff not visible; warning logged
def test_8_fingerprint_updates_visibility_and_warning_logging(
    redis_client: fakeredis.FakeRedis,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Verifies fingerprint extraction, conservative cutoff visibility, and warning on threshold exceedance."""
    settings = load_account_features_settings()
    t_seen = datetime(2026, 10, 1, 10, 0, tzinfo=IST)

    # Sighting 1: Initial phone debit
    p1 = _make_payment(f"{RUN_ID}-p-01", t_seen, t_seen, side="debit", sender_device_fingerprint="fp_phone_001")
    update_account_summaries(redis_client, p1, settings)

    # Sighting 2: OS update on same phone -> same fingerprint
    ev_os = _make_event(
        f"{RUN_ID}-dev-01", "os_update", t_seen + timedelta(hours=1), t_seen + timedelta(hours=1),
        details={"fingerprint": "fp_phone_001", "os_generation": 2},
    )
    update_account_summaries(redis_client, ev_os, settings)

    life_1 = read_lifetime(
        redis_client, RUN_ID, BANK_0, ACC_1,
        t_seen + timedelta(hours=2), t_seen + timedelta(hours=2), settings,
    )
    assert len(life_1.fingerprints) == 1
    assert "fp_phone_001" in life_1.fingerprints

    # Sighting 3: Device change to different phone
    ev_dev = _make_event(
        f"{RUN_ID}-dev-02", "device_binding_change", t_seen + timedelta(hours=3), t_seen + timedelta(hours=3),
        details={"fingerprint_before": "fp_phone_001", "fingerprint_after": "fp_phone_002"},
    )
    update_account_summaries(redis_client, ev_dev, settings)

    life_2 = read_lifetime(
        redis_client, RUN_ID, BANK_0, ACC_1,
        t_seen + timedelta(hours=4), t_seen + timedelta(hours=4), settings,
    )
    assert len(life_2.fingerprints) == 2
    assert set(life_2.fingerprints) == {"fp_phone_001", "fp_phone_002"}

    # Sighting 4: Arrival after cutoff -> not visible
    p_future = _make_payment(
        f"{RUN_ID}-p-fut", t_seen + timedelta(hours=5), t_seen + timedelta(hours=10),
        side="debit", sender_device_fingerprint="fp_phone_003",
    )
    update_account_summaries(redis_client, p_future, settings)

    # Query with arrival_cutoff = t_seen + 6 hours (before p_future arrived at +10 hours)
    life_cutoff = read_lifetime(
        redis_client, RUN_ID, BANK_0, ACC_1,
        t_seen + timedelta(hours=12), t_seen + timedelta(hours=6), settings,
    )
    assert "fp_phone_003" not in life_cutoff.fingerprints

    # Warning on count > fingerprint_warning_count (10)
    with caplog.at_level(logging.WARNING):
        for idx in range(4, 15):  # Adds 11 more fingerprints -> total > 10
            ev_warn = _make_event(
                f"{RUN_ID}-fp-warn-{idx}", "device_binding_change", t_seen, t_seen,
                details={"fingerprint_after": f"fp_overflow_{idx}"},
            )
            update_account_summaries(redis_client, ev_warn, settings)

    assert any(ACC_1 in rec.message and "fingerprint_warning_count" in rec.message for rec in caplog.records)

    # Assert nothing was dropped or capped
    life_all = read_lifetime(
        redis_client, RUN_ID, BANK_0, ACC_1,
        t_seen + timedelta(days=1), t_seen + timedelta(days=1), settings,
    )
    assert len(life_all.fingerprints) >= 13


# 9. Shuffle: a fixed list of >= 30 mixed records applied in 3 fixed shuffled orders gives byte-identical summary state
def test_9_shuffle_order_independence() -> None:
    """Verifies that applying >= 30 mixed records across 3 shuffled orders produces byte-identical summary state."""
    settings = load_account_features_settings()
    t_start = datetime(2026, 10, 1, 10, 0, tzinfo=IST)

    records: list[Any] = []
    # 12 payment sides
    for i in range(12):
        side = "credit" if i % 2 == 0 else "debit"
        amt = 10000 * (i + 1)
        st = "success" if i % 3 != 0 else "failed"
        records.append(
            _make_payment(
                f"{RUN_ID}-pmt-{i:03d}",
                t_start + timedelta(hours=i * 2),
                t_start + timedelta(hours=i * 2, minutes=5),
                side=side,
                amount_paise=amt,
                status=st,
                counterparty_vpa=f"cp_{i % 4}",
                sender_device_fingerprint=f"fp_{i % 3}" if side == "debit" else None,
            )
        )

    # 10 withdrawals
    for i in range(10):
        st = "success" if i % 4 != 0 else "failed"
        records.append(
            _make_withdrawal(
                f"{RUN_ID}-wth-{i:03d}",
                t_start + timedelta(hours=i * 3 + 1),
                t_start + timedelta(hours=i * 3 + 2),
                amount_paise=15000 * (i + 1),
                status=st,
            )
        )

    # 10 account events
    for i in range(10):
        kind = "device_binding_change" if i % 2 == 0 else "sim_swap"
        records.append(
            _make_event(
                f"{RUN_ID}-evt-{i:03d}",
                kind,
                t_start + timedelta(hours=i * 2 + 1),
                t_start + timedelta(hours=i * 2 + 2),
                details={"fingerprint_after": f"fp_evt_{i % 3}"},
            )
        )

    assert len(records) >= 30

    # 3 deterministic seeds
    seeds = (42, 1337, 2026)
    dumps: list[dict[bytes, Any]] = []

    for seed in seeds:
        client = fakeredis.FakeRedis()
        shuffled = list(records)
        random.Random(seed).shuffle(shuffled)
        for r in shuffled:
            update_account_summaries(client, r, settings)
        dumps.append(_dump_all_keys(client, f"acctf:{RUN_ID}:*"))

    assert dumps[0] == dumps[1]
    assert dumps[1] == dumps[2]


# 10. history_days_available reads real state: first payment 10 days before decision -> 10; no data -> 0
def test_10_history_days_available(redis_client: fakeredis.FakeRedis) -> None:
    """Verifies history_days_available bounds, caps, and empty history handling on real state."""
    settings = load_account_features_settings()
    acc_id = "acc_test_10"
    decision = datetime(2026, 10, 25, 12, 0, tzinfo=IST)
    arrival_cutoff = decision

    # Case A: No data -> 0
    assert history_days_available(redis_client, RUN_ID, BANK_0, acc_id, decision, arrival_cutoff, settings) == 0

    # Case B: First payment was 10 days before decision on closed day -> history_days_available == 10
    # Day is Oct 15 (start of day is Oct 15 00:00 IST). Decision is Oct 25 12:00 IST.
    # Whole days elapsed from start of Oct 15 to Oct 25 12:00 is (10 days + 12 hours) // 86400 = 10 days.
    t_10d = datetime(2026, 10, 15, 10, 0, tzinfo=IST)
    p = _make_payment(f"{RUN_ID}-p-10d", t_10d, t_10d, account_id=acc_id)
    update_account_summaries(redis_client, p, settings)

    assert history_days_available(redis_client, RUN_ID, BANK_0, acc_id, decision, arrival_cutoff, settings) == 10


# 11. A starting account whose account_opened is dated 5 years back and whose first payment was 10 days before decision
def test_account_opened_five_years_back_first_payment_ten_days_back(redis_client: fakeredis.FakeRedis) -> None:
    """
    Verifies that an account opened 5 years ago whose first payment was 10 days before decision
    yields history_days_available == 10, proving history measures transaction activity held by bank,
    not account opening age.
    """
    settings = load_account_features_settings()
    acc_id = "acc_aged_01"
    decision = datetime(2026, 10, 25, 12, 0, tzinfo=IST)
    arrival_cutoff = decision

    # Account opened 5 years back
    t_open = datetime(2021, 10, 15, 10, 0, tzinfo=timezone.utc)
    ev_open = _make_event(f"{RUN_ID}-ev-open-5y", "account_opened", t_open, t_open, account_id=acc_id)
    update_account_summaries(redis_client, ev_open, settings)

    # First payment 10 days before decision (Oct 15 10:00 IST)
    t_pmt = datetime(2026, 10, 15, 10, 0, tzinfo=IST)
    p = _make_payment(f"{RUN_ID}-p-10d-aged", t_pmt, t_pmt, account_id=acc_id)
    update_account_summaries(redis_client, p, settings)

    days = history_days_available(redis_client, RUN_ID, BANK_0, acc_id, decision, arrival_cutoff, settings)
    assert days == 10


# 12. An account with 70 days of payments -> history_days_available == 62
def test_account_with_seventy_days_of_payments_capped_at_horizon(redis_client: fakeredis.FakeRedis) -> None:
    """Verifies that 70 days of payments caps history_days_available at summary_horizon_days (62)."""
    settings = load_account_features_settings()
    acc_id = "acc_70d_01"
    decision = datetime(2026, 10, 25, 12, 0, tzinfo=IST)
    arrival_cutoff = decision

    # Earliest payment 70 days before decision
    t_70d = decision - timedelta(days=70)
    p_early = _make_payment(f"{RUN_ID}-p-70d", t_70d, t_70d, account_id=acc_id)
    update_account_summaries(redis_client, p_early, settings)

    # Recent payment to confirm multiple payments
    t_recent = decision - timedelta(days=5)
    p_recent = _make_payment(f"{RUN_ID}-p-recent", t_recent, t_recent, account_id=acc_id)
    update_account_summaries(redis_client, p_recent, settings)

    days = history_days_available(redis_client, RUN_ID, BANK_0, acc_id, decision, arrival_cutoff, settings)
    assert days == 62


# 13. A payment whose day has not closed does not count in last_active_day or history_days_available
def test_unclosed_payment_not_in_last_active_day_or_history_days(redis_client: fakeredis.FakeRedis) -> None:
    """Verifies an unclosed payment is excluded from last_active_day and history_days_available."""
    settings = load_account_features_settings()
    acc_id = "acc_unclosed_01"
    # Payment on Oct 10 10:00 IST
    t_pmt = datetime(2026, 10, 10, 10, 0, tzinfo=IST)
    p = _make_payment(f"{RUN_ID}-p-unclosed", t_pmt, t_pmt, account_id=acc_id)
    update_account_summaries(redis_client, p, settings)

    # Decision on same day (Oct 10 15:00 IST) -> day is unclosed (closes at Oct 12 00:00:00 IST)
    decision_same_day = datetime(2026, 10, 10, 15, 0, tzinfo=IST)
    life = read_lifetime(redis_client, RUN_ID, BANK_0, acc_id, decision_same_day, decision_same_day, settings)
    assert life.last_active_day is None

    days = history_days_available(redis_client, RUN_ID, BANK_0, acc_id, decision_same_day, decision_same_day, settings)
    assert days == 0


# 14. A device event on a later day does not change last_active_day
def test_device_event_on_later_day_does_not_change_last_active_day(redis_client: fakeredis.FakeRedis) -> None:
    """Verifies account lifecycle events do not write active_days and do not alter last_active_day."""
    settings = load_account_features_settings()
    acc_id = "acc_dev_later_01"

    # Payment on Oct 1
    t_pmt = datetime(2026, 10, 1, 10, 0, tzinfo=IST)
    p = _make_payment(f"{RUN_ID}-p-d1", t_pmt, t_pmt, account_id=acc_id)
    update_account_summaries(redis_client, p, settings)

    # Device binding change on Oct 10 (later day)
    t_dev = datetime(2026, 10, 10, 10, 0, tzinfo=IST)
    ev_dev = _make_event(
        f"{RUN_ID}-dev-later", "device_binding_change", t_dev, t_dev,
        details={"fingerprint_after": "fp_dev_later"},
        account_id=acc_id,
    )
    update_account_summaries(redis_client, ev_dev, settings)

    # Decision on Oct 15 (both days closed)
    decision = datetime(2026, 10, 15, 12, 0, tzinfo=IST)
    life = read_lifetime(redis_client, RUN_ID, BANK_0, acc_id, decision, decision, settings)
    # Must be Oct 1 payment day, NOT Oct 10 device event day!
    assert life.last_active_day == to_day_number(t_pmt)


# 15. After applying records, client.ttl() on opened_at, fingerprints and active_days returns -1
def test_lifetime_keys_have_no_expiry_raw_keys_keep_positive_ttl(redis_client: fakeredis.FakeRedis) -> None:
    """Verifies lifetime keys (opened_at, fingerprints, active_days) have no TTL (-1), while raw keys have TTL."""
    settings = load_account_features_settings()
    acc_id = "acc_ttl_01"
    t_now = datetime(2026, 10, 1, 10, 0, tzinfo=IST)

    # 1. Apply payment side via raw updater (which calls update_account_summaries)
    p = _make_payment(
        f"{RUN_ID}-p-ttl", t_now, t_now,
        side="debit", sender_device_fingerprint="fp_ttl_1", account_id=acc_id,
    )
    apply_payment_side(redis_client, p, settings)

    # 2. Apply account_opened via raw updater (which calls update_account_summaries)
    ev = _make_event(f"{RUN_ID}-open-ttl", "account_opened", t_now - timedelta(days=30), t_now, account_id=acc_id)
    apply_account_event(redis_client, ev, settings)

    # Lifetime keys: NO EXPIRE (ttl == -1)
    act_k = active_days_key(RUN_ID, BANK_0, acc_id)
    op_k = opened_at_key(RUN_ID, BANK_0, acc_id)
    fp_k = fingerprints_key(RUN_ID, BANK_0, acc_id)

    assert redis_client.ttl(act_k) == -1
    assert redis_client.ttl(op_k) == -1
    assert redis_client.ttl(fp_k) == -1

    # Raw-window keys: positive TTL
    st_k = side_times_key(RUN_ID, BANK_0, acc_id)
    sr_k = side_records_key(RUN_ID, BANK_0, acc_id)
    m_k = marks_key(RUN_ID, BANK_0, acc_id)

    ttl_st: Any = redis_client.ttl(st_k)
    ttl_sr: Any = redis_client.ttl(sr_k)
    ttl_m: Any = redis_client.ttl(m_k)

    assert ttl_st > 0
    assert ttl_sr > 0
    assert ttl_m > 0


# 16. A debit side arriving too late for summaries still adds its fingerprint (visible once available_from <= decision)
def test_late_debit_side_excluded_from_summaries_still_records_fingerprint(
    redis_client: fakeredis.FakeRedis,
) -> None:
    """
    Verifies debit payment arriving after closed-day threshold returns
    too_late_for_summary but records fingerprint.
    """
    settings = load_account_features_settings()
    acc_id = "acc_late_fp_01"
    # Event occurred on Oct 1
    t_event = datetime(2026, 10, 1, 10, 0, tzinfo=IST)
    # Arrived on Oct 5 (past day end + allowed lateness of Oct 3 00:00:00 IST)
    t_arrival = datetime(2026, 10, 5, 10, 0, tzinfo=IST)

    p_late = _make_payment(
        f"{RUN_ID}-p-late-fp", t_event, t_arrival,
        side="debit", sender_device_fingerprint="fp_late_survivor",
        account_id=acc_id,
    )
    result = update_account_summaries(redis_client, p_late, settings)
    assert result == "too_late_for_summary"

    # Summary day totals must be empty
    decision = datetime(2026, 10, 10, 12, 0, tzinfo=IST)
    totals = read_day_totals(redis_client, RUN_ID, BANK_0, acc_id, decision, decision, settings)
    assert len(totals) == 0

    # Fingerprint available_from = max(t_event, t_arrival) = t_arrival (Oct 5 10:00 IST)
    # Query before arrival: not visible
    life_before = read_lifetime(
        redis_client, RUN_ID, BANK_0, acc_id,
        decision_time=decision, arrival_cutoff=datetime(2026, 10, 4, 12, 0, tzinfo=IST),
        settings=settings,
    )
    assert "fp_late_survivor" not in life_before.fingerprints

    # Query after arrival: visible!
    life_after = read_lifetime(
        redis_client, RUN_ID, BANK_0, acc_id,
        decision_time=decision, arrival_cutoff=datetime(2026, 10, 6, 12, 0, tzinfo=IST),
        settings=settings,
    )
    assert "fp_late_survivor" in life_after.fingerprints
