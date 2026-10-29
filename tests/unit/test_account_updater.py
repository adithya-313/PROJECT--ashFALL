# File: tests/unit/test_account_updater.py
# Purpose: Unit tests for Level 15a account updater and point-in-time reader using fakeredis.
# Issue: Level 15a account feature updater (plan Task 1, C4, K2).
"""
WHAT: Unit tests covering atomic state updating in Redis, idempotency, retention horizon trimming,
no-orphan hash deletion, multi-run tenant isolation, arrival-watermark future-leak prevention,
and sliding window boundary counts.

WHY: Asserts that account state mutations are strictly idempotent, bounded in memory, leak-free,
and respect point-in-time causality constraints.
"""

import random
from datetime import datetime, timedelta, timezone
from typing import Any

import fakeredis
import pytest

from features.account_keys import (
    load_account_features_settings,
    marks_key,
    side_records_key,
    side_times_key,
)
from features.account_reader import (
    count_sides_in_window,
    empty_snapshot,
    read_account_snapshot,
)
from features.account_records import (
    AccountEventRecord,
    PaymentSideRecord,
    record_member,
)
from features.account_updater import (
    apply_account_event,
    apply_payment_side,
)

RUN_A = "aaaaaaaaaaaaaaaa"
RUN_B = "bbbbbbbbbbbbbbbb"
BANK_0 = "bank_0"
ACC_1 = "acc_person_0001"
BASE_TIME = datetime(2026, 10, 28, 12, 0, 0, tzinfo=timezone.utc)


def _make_payment_side(
    run_id: str,
    bank_id: str,
    account_id: str,
    payment_id: str,
    side: str,
    event_time: datetime,
    arrival_time: datetime,
    amount_paise: int = 10000,
) -> PaymentSideRecord:
    """Helper constructing a valid PaymentSideRecord."""
    return PaymentSideRecord(
        run_id=run_id,
        bank_id=bank_id,
        account_id=account_id,
        payment_id=payment_id,
        side=side,
        counterparty_bank_id="bank_1",
        counterparty_vpa="",
        amount_paise=amount_paise,
        status="success",
        failure_domain=None,
        payee_type="person",
        channel="upi_app",
        sender_device_fingerprint="dev_fp_1",
        sender_os_generation=1,
        event_time=event_time,
        arrival_time=arrival_time,
    )


def _make_account_event(
    run_id: str,
    bank_id: str,
    account_id: str,
    event_id: str,
    kind: str,
    event_time: datetime,
    arrival_time: datetime,
) -> AccountEventRecord:
    """Helper constructing a valid AccountEventRecord."""
    return AccountEventRecord(
        run_id=run_id,
        bank_id=bank_id,
        account_id=account_id,
        event_id=event_id,
        kind=kind,
        event_time=event_time,
        arrival_time=arrival_time,
        details={"reason": "test"},
    )


@pytest.fixture
def redis_client() -> fakeredis.FakeRedis:
    """Provides a clean in-memory FakeRedis instance for each test."""
    return fakeredis.FakeRedis()


def test_empty_history_returns_defaults(redis_client: fakeredis.FakeRedis) -> None:
    """Verifies that querying an account with no keys returns empty_snapshot with 0 counts."""
    snapshot = read_account_snapshot(
        redis_client,
        RUN_A,
        BANK_0,
        "acc_unknown",
        decision_time=BASE_TIME,
        arrival_cutoff=BASE_TIME,
    )
    assert snapshot == empty_snapshot()
    assert snapshot.has_history is False
    assert snapshot.side_records == ()
    assert snapshot.account_events == ()
    assert snapshot.first_event_time is None
    assert snapshot.last_event_time is None
    assert snapshot.arrival_watermark is None

    # Count for unknown account in valid 1-hour window returns 0
    count = count_sides_in_window(snapshot, "debit", BASE_TIME, 3600)
    assert count == 0


def test_same_record_applied_twice_gives_identical_state(redis_client: fakeredis.FakeRedis) -> None:
    """Verifies that applying the same record multiple times leaves identical Redis state."""
    rec = _make_payment_side(
        RUN_A,
        BANK_0,
        ACC_1,
        f"{RUN_A}-000001",
        "debit",
        BASE_TIME,
        BASE_TIME,
    )

    # 1. First apply
    outcome_1 = apply_payment_side(redis_client, rec)
    assert outcome_1 == "stored"

    # Dump all keys and values
    dump_1: dict[bytes, Any] = {}
    keys_obj_1: Any = redis_client.keys()
    assert isinstance(keys_obj_1, list)
    for k in sorted(keys_obj_1):
        k_type = redis_client.type(k)
        if k_type == b"zset":
            dump_1[k] = redis_client.zrange(k, 0, -1, withscores=True)
        elif k_type == b"hash":
            dump_1[k] = redis_client.hgetall(k)

    # 2. Re-apply the same record
    outcome_2 = apply_payment_side(redis_client, rec)
    assert outcome_2 == "stored"

    dump_2: dict[bytes, Any] = {}
    keys_obj_2: Any = redis_client.keys()
    assert isinstance(keys_obj_2, list)
    for k in sorted(keys_obj_2):
        k_type = redis_client.type(k)
        if k_type == b"zset":
            dump_2[k] = redis_client.zrange(k, 0, -1, withscores=True)
        elif k_type == b"hash":
            dump_2[k] = redis_client.hgetall(k)

    # State must be completely identical
    assert dump_1 == dump_2


def test_two_runs_with_identical_account_ids_isolated(redis_client: fakeredis.FakeRedis) -> None:
    """Verifies that two runs using the same bank and account ID never see each other's records."""
    rec_a = _make_payment_side(
        RUN_A,
        BANK_0,
        ACC_1,
        f"{RUN_A}-000001",
        "debit",
        BASE_TIME,
        BASE_TIME,
    )
    rec_b = _make_payment_side(
        RUN_B,
        BANK_0,
        ACC_1,
        f"{RUN_B}-000001",
        "debit",
        BASE_TIME,
        BASE_TIME,
    )

    apply_payment_side(redis_client, rec_a)
    apply_payment_side(redis_client, rec_b)

    snap_a = read_account_snapshot(redis_client, RUN_A, BANK_0, ACC_1, BASE_TIME, BASE_TIME)
    snap_b = read_account_snapshot(redis_client, RUN_B, BANK_0, ACC_1, BASE_TIME, BASE_TIME)

    assert len(snap_a.side_records) == 1
    assert snap_a.side_records[0].payment_id == f"{RUN_A}-000001"
    assert snap_a.side_records[0].run_id == RUN_A

    assert len(snap_b.side_records) == 1
    assert snap_b.side_records[0].payment_id == f"{RUN_B}-000001"
    assert snap_b.side_records[0].run_id == RUN_B


def test_late_arrival_handling(redis_client: fakeredis.FakeRedis) -> None:
    """
    Verifies that a late event (event t1, t3 then t2 arriving last) leaves windows ending at t1
    unchanged, and windows ending at t3 includes t2 once arrival cutoff permits.
    """
    t1 = BASE_TIME
    t2 = BASE_TIME + timedelta(minutes=10)
    t3 = BASE_TIME + timedelta(minutes=20)

    # Ingestion order: t1 (arr 12:01), t3 (arr 12:21), t2 (arr 12:25)
    rec_1 = _make_payment_side(RUN_A, BANK_0, ACC_1, f"{RUN_A}-000001", "debit", t1, t1 + timedelta(minutes=1))
    rec_3 = _make_payment_side(RUN_A, BANK_0, ACC_1, f"{RUN_A}-000003", "debit", t3, t3 + timedelta(minutes=1))
    rec_2 = _make_payment_side(RUN_A, BANK_0, ACC_1, f"{RUN_A}-000002", "debit", t2, t2 + timedelta(minutes=15))

    apply_payment_side(redis_client, rec_1)
    apply_payment_side(redis_client, rec_3)
    apply_payment_side(redis_client, rec_2)

    # 1. Snapshot evaluated at t1 with arrival cutoff 12:05 -> only t1 is present
    snap_t1 = read_account_snapshot(redis_client, RUN_A, BANK_0, ACC_1, t1, t1 + timedelta(minutes=5))
    assert len(snap_t1.side_records) == 1
    assert snap_t1.side_records[0].payment_id == f"{RUN_A}-000001"

    # 2. Snapshot evaluated at t3 with arrival cutoff 12:22 (before t2 arrived at 12:25) -> excludes t2
    snap_t3_early = read_account_snapshot(redis_client, RUN_A, BANK_0, ACC_1, t3, t3 + timedelta(minutes=2))
    assert [r.payment_id for r in snap_t3_early.side_records] == [f"{RUN_A}-000001", f"{RUN_A}-000003"]

    # 3. Snapshot evaluated at t3 with arrival cutoff 12:30 (after t2 arrived) -> includes all three
    snap_t3_late = read_account_snapshot(redis_client, RUN_A, BANK_0, ACC_1, t3, t3 + timedelta(minutes=10))
    assert [r.payment_id for r in snap_t3_late.side_records] == [
        f"{RUN_A}-000001",
        f"{RUN_A}-000002",
        f"{RUN_A}-000003",
    ]


def test_arrival_cutoff_prevents_future_leak(redis_client: fakeredis.FakeRedis) -> None:
    """
    Verifies that a record whose event_time is before decision_time is excluded if its
    arrival_time is after the arrival cutoff (future-leak protection).
    """
    event_time = BASE_TIME + timedelta(minutes=5)
    future_arrival = BASE_TIME + timedelta(hours=1)

    rec = _make_payment_side(RUN_A, BANK_0, ACC_1, f"{RUN_A}-000001", "debit", event_time, future_arrival)
    apply_payment_side(redis_client, rec)

    # Decision at 12:10, but arrival cutoff is 12:10 (before the record reached storage at 13:00)
    snapshot = read_account_snapshot(
        redis_client,
        RUN_A,
        BANK_0,
        ACC_1,
        decision_time=BASE_TIME + timedelta(minutes=10),
        arrival_cutoff=BASE_TIME + timedelta(minutes=10),
    )
    assert snapshot == empty_snapshot()


def test_older_than_retention_returns_status_and_is_pruned(redis_client: fakeredis.FakeRedis) -> None:
    """
    Verifies that an event older than retention horizon returns 'older_than_retention'
    and is trimmed from the snapshot.
    """
    settings = load_account_features_settings()
    retention_days = settings.retention_seconds // 86400  # 7 days

    # 1. Anchor account state with recent event at BASE_TIME
    rec_recent = _make_payment_side(RUN_A, BANK_0, ACC_1, f"{RUN_A}-000001", "debit", BASE_TIME, BASE_TIME)
    status_recent = apply_payment_side(redis_client, rec_recent, settings)
    assert status_recent == "stored"

    # 2. Apply old event 10 days before BASE_TIME (older than 7d retention)
    t_old = BASE_TIME - timedelta(days=retention_days + 3)
    rec_old = _make_payment_side(RUN_A, BANK_0, ACC_1, f"{RUN_A}-000002", "debit", t_old, t_old)
    status_old = apply_payment_side(redis_client, rec_old, settings)
    assert status_old == "older_than_retention"

    # 3. Read snapshot at BASE_TIME -> old event must NOT be present
    snapshot = read_account_snapshot(redis_client, RUN_A, BANK_0, ACC_1, BASE_TIME, BASE_TIME, settings)
    assert len(snapshot.side_records) == 1
    assert snapshot.side_records[0].payment_id == f"{RUN_A}-000001"


def test_trim_removes_hash_entry_no_orphans(redis_client: fakeredis.FakeRedis) -> None:
    """Verifies that trimming an expired record deletes both the ZSET score and the HASH record."""
    settings = load_account_features_settings()
    st_key = side_times_key(RUN_A, BANK_0, ACC_1)
    sr_key = side_records_key(RUN_A, BANK_0, ACC_1)

    t1 = BASE_TIME
    rec_1 = _make_payment_side(RUN_A, BANK_0, ACC_1, f"{RUN_A}-000001", "debit", t1, t1)
    apply_payment_side(redis_client, rec_1, settings)
    mem_1 = record_member(rec_1)

    # Member 1 exists in both structures
    assert redis_client.zscore(st_key, mem_1) is not None
    assert redis_client.hexists(sr_key, mem_1) is True

    # Apply event 9 days later -> advances last_event beyond (retention + lateness = 8d) and trims rec_1
    t2 = BASE_TIME + timedelta(days=9)
    rec_2 = _make_payment_side(RUN_A, BANK_0, ACC_1, f"{RUN_A}-000002", "debit", t2, t2)
    apply_payment_side(redis_client, rec_2, settings)

    # Member 1 must be pruned from both ZSET and HASH
    assert redis_client.zscore(st_key, mem_1) is None
    assert redis_client.hexists(sr_key, mem_1) is False


def test_count_window_boundaries_inclusion_exclusion(redis_client: fakeredis.FakeRedis) -> None:
    """
    Verifies that window count evaluates (decision_time - window, decision_time]:
    events exactly window seconds old are excluded, events exactly at decision_time are included.
    """
    decision_time = BASE_TIME
    window_1h = 3600

    # Event 1: exactly 3600s old (window boundary) -> EXCLUDED
    t_start = decision_time - timedelta(seconds=window_1h)
    rec_start = _make_payment_side(RUN_A, BANK_0, ACC_1, f"{RUN_A}-000001", "debit", t_start, t_start)

    # Event 2: 1800s old (inside window) -> INCLUDED
    t_mid = decision_time - timedelta(seconds=1800)
    rec_mid = _make_payment_side(RUN_A, BANK_0, ACC_1, f"{RUN_A}-000002", "debit", t_mid, t_mid)

    # Event 3: exactly decision_time -> INCLUDED
    rec_end = _make_payment_side(RUN_A, BANK_0, ACC_1, f"{RUN_A}-000003", "debit", decision_time, decision_time)

    # Event 4: credit side at decision_time -> NOT counted in debit count
    rec_credit = _make_payment_side(RUN_A, BANK_0, ACC_1, f"{RUN_A}-000004", "credit", decision_time, decision_time)

    apply_payment_side(redis_client, rec_start)
    apply_payment_side(redis_client, rec_mid)
    apply_payment_side(redis_client, rec_end)
    apply_payment_side(redis_client, rec_credit)

    snapshot = read_account_snapshot(redis_client, RUN_A, BANK_0, ACC_1, decision_time, decision_time)
    debit_count = count_sides_in_window(snapshot, "debit", decision_time, window_1h)
    credit_count = count_sides_in_window(snapshot, "credit", decision_time, window_1h)

    # Exactly 2 debits (rec_mid and rec_end; rec_start is excluded)
    assert debit_count == 2
    assert credit_count == 1


def test_count_sides_in_invalid_window_raises(redis_client: fakeredis.FakeRedis) -> None:
    """Verifies that requesting an unregistered window interval raises ValueError."""
    snapshot = empty_snapshot()
    with pytest.raises(ValueError, match="Invalid window_seconds 7200"):
        count_sides_in_window(snapshot, "debit", BASE_TIME, 7200)


def test_apply_account_event_integration(redis_client: fakeredis.FakeRedis) -> None:
    """Verifies that apply_account_event writes lifecycle events and updates marks."""
    ev = _make_account_event(
        RUN_A,
        BANK_0,
        ACC_1,
        f"{RUN_A}-dev-000001",
        "device_binding_change",
        BASE_TIME,
        BASE_TIME,
    )
    outcome = apply_account_event(redis_client, ev)
    assert outcome == "stored"

    snapshot = read_account_snapshot(redis_client, RUN_A, BANK_0, ACC_1, BASE_TIME, BASE_TIME)
    assert len(snapshot.account_events) == 1
    assert snapshot.account_events[0].kind == "device_binding_change"
    assert snapshot.first_event_time == BASE_TIME
    assert snapshot.last_event_time == BASE_TIME


def test_late_events_within_allowed_lateness_not_pruned(redis_client: fakeredis.FakeRedis) -> None:
    """
    Verifies that an event older than retention (7d) but within allowed_lateness (86400s)
    is NOT pruned and returns 'stored'.
    """
    settings = load_account_features_settings()
    st_key = side_times_key(RUN_A, BANK_0, ACC_1)
    sr_key = side_records_key(RUN_A, BANK_0, ACC_1)

    # 1. Base event at BASE_TIME
    rec_base = _make_payment_side(RUN_A, BANK_0, ACC_1, f"{RUN_A}-000001", "debit", BASE_TIME, BASE_TIME)
    apply_payment_side(redis_client, rec_base, settings)

    # 2. Latest event at BASE_TIME + 7 days + 12 hours (older than 7d retention, but within 7d + 1d = 8d)
    t_latest = BASE_TIME + timedelta(days=7, hours=12)
    rec_latest = _make_payment_side(RUN_A, BANK_0, ACC_1, f"{RUN_A}-000002", "debit", t_latest, t_latest)
    apply_payment_side(redis_client, rec_latest, settings)

    # 3. Base event must still exist in both structures (not pruned)
    mem_base = record_member(rec_base)
    assert redis_client.zscore(st_key, mem_base) is not None
    assert redis_client.hexists(sr_key, mem_base) is True


def test_events_older_than_cutoff_pruned_atomically(redis_client: fakeredis.FakeRedis) -> None:
    """
    Verifies that events older than cutoff = last_event - (retention + allowed_lateness)
    are pruned from BOTH the zset and the hash.
    """
    settings = load_account_features_settings()
    st_key = side_times_key(RUN_A, BANK_0, ACC_1)
    sr_key = side_records_key(RUN_A, BANK_0, ACC_1)

    t1 = BASE_TIME
    rec_1 = _make_payment_side(RUN_A, BANK_0, ACC_1, f"{RUN_A}-000001", "debit", t1, t1)
    apply_payment_side(redis_client, rec_1, settings)
    mem_1 = record_member(rec_1)

    # Apply event 9 days later (> 8 days = retention + allowed_lateness)
    t2 = BASE_TIME + timedelta(days=9)
    rec_2 = _make_payment_side(RUN_A, BANK_0, ACC_1, f"{RUN_A}-000002", "debit", t2, t2)
    apply_payment_side(redis_client, rec_2, settings)

    # rec_1 must be pruned from BOTH zset and hash
    assert redis_client.zscore(st_key, mem_1) is None
    assert redis_client.hexists(sr_key, mem_1) is False


def test_key_ttl_equals_retention_plus_lateness_plus_margin(redis_client: fakeredis.FakeRedis) -> None:
    """Verifies that key TTL equals retention + allowed_lateness + key_expiry_margin."""
    settings = load_account_features_settings()
    expected_ttl = (
        settings.retention_seconds
        + settings.allowed_lateness_seconds
        + settings.key_expiry_margin_seconds
    )
    assert settings.expiry_seconds == expected_ttl

    rec = _make_payment_side(RUN_A, BANK_0, ACC_1, f"{RUN_A}-000001", "debit", BASE_TIME, BASE_TIME)
    apply_payment_side(redis_client, rec, settings)

    for k in (
        side_times_key(RUN_A, BANK_0, ACC_1),
        side_records_key(RUN_A, BANK_0, ACC_1),
        marks_key(RUN_A, BANK_0, ACC_1),
    ):
        ttl = redis_client.ttl(k)
        assert ttl == expected_ttl


def test_shuffle_order_independence() -> None:
    """
    Fix 3 / Architect Decision 7: Shuffling >=20 mixed records produces byte-identical
    Redis state across 3 fixed seeds (42, 1337, 2026) and matches chronological order.
    """
    settings = load_account_features_settings()

    # Build a fixed batch of 24 mixed records
    records: list[PaymentSideRecord | AccountEventRecord] = []
    for i in range(16):
        t = BASE_TIME + timedelta(hours=i * 5)
        side = "debit" if i % 2 == 0 else "credit"
        records.append(
            _make_payment_side(
                RUN_A,
                BANK_0,
                ACC_1,
                f"{RUN_A}-pmt-{i:04d}",
                side,
                event_time=t,
                arrival_time=t + timedelta(minutes=(i % 3) * 10),
                amount_paise=10000 * (i + 1),
            )
        )
    for j in range(8):
        t = BASE_TIME + timedelta(hours=j * 11)
        kind = "device_binding_change" if j % 2 == 0 else "sim_swap"
        records.append(
            _make_account_event(
                RUN_A,
                BANK_0,
                ACC_1,
                f"{RUN_A}-dev-{j:04d}",
                kind=kind,
                event_time=t,
                arrival_time=t + timedelta(minutes=(j % 2) * 15),
            )
        )

    def _canonical_dump(client: fakeredis.FakeRedis) -> bytes:
        dump_parts: list[bytes] = []
        keys_obj: Any = client.keys()
        assert isinstance(keys_obj, list)
        for k in sorted(keys_obj):
            k_type = client.type(k)
            dump_parts.append(k + b":" + k_type)
            if k_type == b"zset":
                items: Any = client.zrange(k, 0, -1, withscores=True)
                assert isinstance(items, list)
                for member, score in items:
                    dump_parts.append(b" " + member + b"=" + str(score).encode("utf-8"))
            elif k_type == b"hash":
                entries: Any = client.hgetall(k)
                assert isinstance(entries, dict)
                for field in sorted(entries):
                    dump_parts.append(b" " + field + b"=" + entries[field])
        return b"\n".join(dump_parts)

    def _apply_batch(client: fakeredis.FakeRedis, batch: list[Any]) -> None:
        for item in batch:
            if isinstance(item, PaymentSideRecord):
                apply_payment_side(client, item, settings)
            else:
                apply_account_event(client, item, settings)

    # Chronological baseline
    r_chrono = fakeredis.FakeRedis()
    _apply_batch(r_chrono, records)
    dump_chrono = _canonical_dump(r_chrono)
    assert len(dump_chrono) > 0

    # Test 3 fixed seeds
    for seed in (42, 1337, 2026):
        shuffled = list(records)
        random.Random(seed).shuffle(shuffled)
        r_shuffled = fakeredis.FakeRedis()
        _apply_batch(r_shuffled, shuffled)
        dump_shuffled = _canonical_dump(r_shuffled)
        assert dump_shuffled == dump_chrono, f"State diverged under seed {seed}"
