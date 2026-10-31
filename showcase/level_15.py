# File: showcase/level_15.py
# Purpose: Level 15 showcase: Live mule features and account state (Part 15a).
"""
WHAT: Verification checks for Level 15 part a (Account Feature Updater).
Proves window counts over normal accounts, default empty snapshot handling,
late event arrival isolation, multi-run tenant isolation, and idempotent replay consistency.

WHY: Satisfies Level 15 showcase requirements ensuring streaming feature storage
is resilient, leak-free, and isolated across simulation runs and banks.
"""

import importlib
import os
import random
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from core.config import config
from features.account_keys import (
    get_default_settings,
    namespace_pattern,
)
from features.account_reader import (
    count_sides_in_window,
    empty_snapshot,
    read_account_snapshot,
)
from features.account_records import AccountEventRecord, PaymentSideRecord
from features.account_updater import apply_account_event, apply_payment_side
from showcase.framework import CheckDefinition, CheckOutcome
from showcase.isolation import get_showcase_database_url
from showcase.preconditions import check_database, check_redis, check_tcp_port

LEVEL_NUMBER = 15
LEVEL_TITLE = "Live mule features and account state (built: parts 15a, 15b, 15c)"
NEEDS_DATABASE = True

WHAT_YOU_SAW = (
    "Level 15 introduces near-real-time account feature updating over Redis sliding windows. "
    "Part 15a establishes namespaced key structures with TTL expiration, point-in-time snapshot "
    "reading enforcing the Section 22 future-leak rule, idempotent updates supporting replay "
    "without data corruption or set explosion, strict tenant isolation across commercial banks, "
    "and streaming ingestion from Kafka and PostgreSQL. Part 15b computes the full M1 behavioral "
    "feature catalog, cleanly differentiating mules from everyday citizens, enforcing bank ledger "
    "isolation, respecting Issue R5 OS update safety, and surfacing rented mule dynamics amidst "
    "normal routine life. Part 15c logs the exact feature vector behind each decision: a late event never "
    "rewrites it, replaying history reproduces it exactly, a feature failure never changes the bank's "
    "answer, and the decision path is timed against the pre-registered latency budget."
)

REPO_ROOT = Path(__file__).resolve().parent.parent
REQUIRED_FILES = [
    REPO_ROOT / "config" / "account_features.yaml",
    REPO_ROOT / "features" / "account_keys.py",
    REPO_ROOT / "features" / "account_records.py",
    REPO_ROOT / "features" / "account_updater.py",
    REPO_ROOT / "features" / "account_reader.py",
    REPO_ROOT / "features" / "account_stream.py",
    REPO_ROOT / "benchmarks" / "decision_latency.py",
    REPO_ROOT / "showcase" / "level_15_support_c.py",
]


def host_side_redis_url() -> str:
    """Resolve Redis connection URL on the host machine."""
    raw = os.environ.get("TEST_REDIS_URL") or config.redis_url
    return raw.replace("://redis:", "://localhost:").replace("@redis:", "@localhost:")


async def find_missing_preconditions() -> list[str]:
    """Check that Postgres, Redpanda and Redis are reachable and all mandatory Level 15 files exist."""
    problems: list[str] = []
    database_problem = await check_database(get_showcase_database_url(), "docker compose up -d postgres redpanda")
    if database_problem is not None:
        problems.append(database_problem)
    kafka_host, _, kafka_port = config.kafka_bootstrap_servers.split(",")[0].partition(":")
    kafka_problem = check_tcp_port(
        kafka_host, int(kafka_port or 9092), "Redpanda (Kafka)", "docker compose up -d postgres redpanda"
    )
    if kafka_problem is not None:
        problems.append(kafka_problem)
    redis_problem = check_redis(host_side_redis_url(), "docker compose up -d redis")
    if redis_problem is not None:
        problems.append(redis_problem)

    for f_path in REQUIRED_FILES:
        if not f_path.is_file():
            problems.append(f"{f_path.name} is missing - it is part of Level 15 files")

    return problems


def _get_redis() -> Any:
    """Dynamically load redis package without adding a static module dependency in showcase."""
    return importlib.import_module("redis")


def _dump_account_keys(client: Any, run_id: str) -> dict[bytes, Any]:
    """Helper dumping all Redis keys, types, and values under the run namespace."""
    dump: dict[bytes, Any] = {}
    pattern = f"acctf:{run_id}:*"
    for k in sorted(client.scan_iter(match=pattern)):
        k_type = client.type(k)
        if k_type == b"zset":
            dump[k] = (k_type, client.zrange(k, 0, -1, withscores=True))
        elif k_type == b"hash":
            dump[k] = (k_type, client.hgetall(k))
        else:
            dump[k] = (k_type, client.get(k))
    return dump


def make_showcase_record(
    run_id: str,
    bank_id: str,
    account_id: str,
    payment_id: str,
    side: str,
    event_time: datetime,
    arrival_time: datetime,
    amount_paise: int = 10000,
) -> PaymentSideRecord:
    """Helper constructing a valid PaymentSideRecord with all required fields."""
    return PaymentSideRecord(
        run_id=run_id,
        bank_id=bank_id,
        account_id=account_id,
        payment_id=payment_id,
        side=side,
        counterparty_bank_id="bank_1",
        counterparty_vpa="user@bank1",
        amount_paise=amount_paise,
        status="success",
        failure_domain=None,
        payee_type="person",
        channel="upi_app",
        sender_device_fingerprint="fp_showcase_1",
        sender_os_generation=1,
        event_time=event_time,
        arrival_time=arrival_time,
    )


async def check_15a_1_normal_window_counts(workspace: Any) -> CheckOutcome:
    """
    Check 15a-1 (normal): normal account gets correct window counts: apply a fixed set of credits
    and debits spread over 8 days for one account. At fixed decision_time, count_sides_in_window
    for 1h, 24h and 7d must equal expected counts worked out from the fixed timestamps.
    """
    client = _get_redis().Redis.from_url(host_side_redis_url())
    settings = get_default_settings()
    run_id = uuid.uuid4().hex[:16]
    bank_id = "bank_0"
    account_id = "acc_showcase_normal"

    decision_time = datetime(2026, 10, 29, 12, 0, 0, tzinfo=timezone.utc)
    arrival_time = decision_time

    # Fixed events spread over 8 days relative to decision_time
    events_spec: list[tuple[timedelta, str, int]] = [
        (timedelta(minutes=15), "debit", 10000),  # inside 1h, 24h, 7d
        (timedelta(minutes=30), "credit", 15000),  # inside 1h, 24h, 7d
        (timedelta(minutes=45), "debit", 20000),  # inside 1h, 24h, 7d
        (timedelta(hours=1), "debit", 25000),  # lower boundary of 1h (excluded from 1h, inside 24h, 7d)
        (timedelta(hours=6), "debit", 30000),  # outside 1h, inside 24h, 7d
        (timedelta(hours=12), "credit", 35000),  # outside 1h, inside 24h, 7d
        (timedelta(hours=18), "debit", 40000),  # outside 1h, inside 24h, 7d
        (timedelta(hours=24), "debit", 45000),  # lower boundary of 24h (excluded from 24h, inside 7d)
        (timedelta(days=3), "debit", 50000),  # outside 1h, 24h, inside 7d
        (timedelta(days=5), "credit", 55000),  # outside 1h, 24h, inside 7d
        (timedelta(days=7), "debit", 60000),  # lower boundary of 7d (excluded from 7d)
        (timedelta(days=8), "debit", 65000),  # outside 7d (> 7d retention)
    ]

    try:
        for idx, (offset, side, amt) in enumerate(events_spec):
            t = decision_time - offset
            rec = make_showcase_record(
                run_id=run_id,
                bank_id=bank_id,
                account_id=account_id,
                payment_id=f"{run_id}-spec-{idx:03d}",
                side=side,
                event_time=t,
                arrival_time=arrival_time,
                amount_paise=amt,
            )
            apply_payment_side(client, rec, settings)

        snap = read_account_snapshot(client, run_id, bank_id, account_id, decision_time, arrival_time, settings)

        cnt_1h_debit = count_sides_in_window(snap, "debit", decision_time, 3600, settings)
        cnt_1h_credit = count_sides_in_window(snap, "credit", decision_time, 3600, settings)
        cnt_24h_debit = count_sides_in_window(snap, "debit", decision_time, 86400, settings)
        cnt_24h_credit = count_sides_in_window(snap, "credit", decision_time, 86400, settings)
        cnt_7d_debit = count_sides_in_window(snap, "debit", decision_time, 604800, settings)
        cnt_7d_credit = count_sides_in_window(snap, "credit", decision_time, 604800, settings)

        # 1h look-back: (decision_time - 3600s, decision_time]
        # Debits inside: T-15m, T-45m (2 events). T-1h is on boundary (excluded). Expected = 2.
        # Credits inside: T-30m (1 event). Expected = 1.
        assert cnt_1h_debit == 2, f"Expected 2 1h debits, got {cnt_1h_debit}"
        assert cnt_1h_credit == 1, f"Expected 1 1h credit, got {cnt_1h_credit}"

        # 24h look-back: (decision_time - 86400s, decision_time]
        # Debits inside: T-15m, T-45m, T-1h, T-6h, T-18h (5 events). T-24h is on boundary (excluded). Expected = 5.
        # Credits inside: T-30m, T-12h (2 events). Expected = 2.
        assert cnt_24h_debit == 5, f"Expected 5 24h debits, got {cnt_24h_debit}"
        assert cnt_24h_credit == 2, f"Expected 2 24h credits, got {cnt_24h_credit}"

        # 7d look-back: (decision_time - 604800s, decision_time]
        # Debits inside: T-15m, T-45m, T-1h, T-6h, T-18h, T-24h, T-3d (7 events). T-7d is on boundary (excluded).
        # T-8d is older than 7d. Expected = 7.
        # Credits inside: T-30m, T-12h, T-5d (3 events). Expected = 3.
        assert cnt_7d_debit == 7, f"Expected 7 7d debits, got {cnt_7d_debit}"
        assert cnt_7d_credit == 3, f"Expected 3 7d credits, got {cnt_7d_credit}"

        windows_checked = 3
        return CheckOutcome(
            key_number=str(windows_checked),
            evidence={
                "windows_checked": windows_checked,
                "counts": {
                    "1h": {"debit": cnt_1h_debit, "credit": cnt_1h_credit},
                    "24h": {"debit": cnt_24h_debit, "credit": cnt_24h_credit},
                    "7d": {"debit": cnt_7d_debit, "credit": cnt_7d_credit},
                },
            },
        )
    finally:
        for pat in (namespace_pattern(run_id, bank_id),):
            for k in client.scan_iter(match=pat):
                client.delete(k)
        client.close()


async def check_15a_2_empty_history(workspace: Any) -> CheckOutcome:
    """
    Check 15a-2 (edge): read an account that was never written -> equals empty_snapshot();
    count_sides_in_window is 0 for every configured window. Key number: fields checked.
    """
    client = _get_redis().Redis.from_url(host_side_redis_url())
    settings = get_default_settings()
    run_id = uuid.uuid4().hex[:16]
    bank_id = "bank_0"
    account_id = "acc_showcase_never_written"
    now = datetime.now(timezone.utc)

    try:
        snap = read_account_snapshot(client, run_id, bank_id, account_id, now, now, settings)
        assert snap == empty_snapshot()
        assert snap.has_history is False
        assert snap.side_records == ()
        assert snap.account_events == ()
        assert snap.first_event_time is None
        assert snap.last_event_time is None
        assert snap.arrival_watermark is None
        fields_checked = 6

        for w_sec in settings.windows_seconds:
            cnt_debit = count_sides_in_window(snap, "debit", now, w_sec, settings)
            assert cnt_debit == 0
            cnt_credit = count_sides_in_window(snap, "credit", now, w_sec, settings)
            assert cnt_credit == 0
            fields_checked += 2

        return CheckOutcome(
            key_number=str(fields_checked),
            evidence={
                "fields_checked": fields_checked,
                "empty_snapshot_verified": True,
                "windows_checked": list(settings.windows_seconds),
            },
        )
    finally:
        client.close()


async def check_15a_3_late_event_lands_in_right_window(workspace: Any) -> CheckOutcome:
    """
    Check 15a-3 (edge): late event lands in the right window: apply events at t1 and t3,
    record 1h count at decision times just after t1 and t3; apply late event at t2 (arrival after t3).
    Count at t1 must be unchanged; count at t3 must include t2.
    """
    client = _get_redis().Redis.from_url(host_side_redis_url())
    settings = get_default_settings()
    run_id = uuid.uuid4().hex[:16]
    bank_id = "bank_0"
    account_id = "acc_showcase_late"

    base_time = datetime(2026, 10, 29, 10, 0, 0, tzinfo=timezone.utc)
    t1 = base_time
    t3 = base_time + timedelta(hours=2)
    a1 = t1
    a3 = t3

    # Apply t1 and t3
    rec_t1 = make_showcase_record(run_id, bank_id, account_id, f"{run_id}-t1", "debit", t1, a1, 10000)
    rec_t3 = make_showcase_record(run_id, bank_id, account_id, f"{run_id}-t3", "debit", t3, a3, 20000)

    dec_1 = t1 + timedelta(minutes=5)
    dec_3 = t3 + timedelta(minutes=5)

    try:
        apply_payment_side(client, rec_t1, settings)
        apply_payment_side(client, rec_t3, settings)

        # Baseline count at dec_1 (1h window: t1 is inside)
        snap_1_before = read_account_snapshot(client, run_id, bank_id, account_id, dec_1, a3, settings)
        cnt_1_before = count_sides_in_window(snap_1_before, "debit", dec_1, 3600, settings)
        assert cnt_1_before == 1

        # Baseline count at dec_3 (1h window: t3 is inside, t1 is 2h old so outside)
        snap_3_before = read_account_snapshot(client, run_id, bank_id, account_id, dec_3, a3, settings)
        cnt_3_before = count_sides_in_window(snap_3_before, "debit", dec_3, 3600, settings)
        assert cnt_3_before == 1

        # Apply late event at t2 with arrival_time after t3's arrival
        t2 = base_time + timedelta(hours=1, minutes=45)
        a2 = t3 + timedelta(minutes=30)
        rec_t2_late = make_showcase_record(run_id, bank_id, account_id, f"{run_id}-t2-late", "debit", t2, a2, 30000)
        apply_payment_side(client, rec_t2_late, settings)

        # Re-query with arrival_cutoff after a2
        query_cutoff = a2 + timedelta(minutes=1)

        # Count at t1 must be unchanged (earlier window)
        snap_1_after = read_account_snapshot(client, run_id, bank_id, account_id, dec_1, query_cutoff, settings)
        cnt_1_after = count_sides_in_window(snap_1_after, "debit", dec_1, 3600, settings)
        earlier_changed = abs(cnt_1_after - cnt_1_before)
        assert earlier_changed == 0
        assert cnt_1_after == 1

        # Count at t3 must include t2 (1h window now has t3 and t2)
        snap_3_after = read_account_snapshot(client, run_id, bank_id, account_id, dec_3, query_cutoff, settings)
        cnt_3_after = count_sides_in_window(snap_3_after, "debit", dec_3, 3600, settings)
        assert cnt_3_after == 2

        return CheckOutcome(
            key_number=str(earlier_changed),
            evidence={
                "cnt_1_before": cnt_1_before,
                "cnt_1_after": cnt_1_after,
                "cnt_3_before": cnt_3_before,
                "cnt_3_after": cnt_3_after,
                "earlier_window_counts_changed": earlier_changed,
            },
        )
    finally:
        for pat in (namespace_pattern(run_id, bank_id),):
            for k in client.scan_iter(match=pat):
                client.delete(k)
        client.close()


async def check_15a_4_two_runs_separate_state(workspace: Any) -> CheckOutcome:
    """
    Check 15a-4 (negative): two FIXED run ids, same bank_id and account_id, different records ->
    each run's snapshot contains only its own records. Key number: cross-run records seen (must be 0).
    """
    client = _get_redis().Redis.from_url(host_side_redis_url())
    settings = get_default_settings()
    run_1 = "1111111111111111"
    run_2 = "2222222222222222"
    bank_id = "bank_0"
    account_id = "acc_showcase_fixed"
    now = datetime.now(timezone.utc)

    rec_1 = make_showcase_record(
        run_id=run_1,
        bank_id=bank_id,
        account_id=account_id,
        payment_id=f"{run_1}-000001",
        side="debit",
        event_time=now,
        arrival_time=now,
        amount_paise=10000,
    )
    rec_2 = make_showcase_record(
        run_id=run_2,
        bank_id=bank_id,
        account_id=account_id,
        payment_id=f"{run_2}-000002",
        side="credit",
        event_time=now,
        arrival_time=now,
        amount_paise=20000,
    )

    try:
        apply_payment_side(client, rec_1, settings)
        apply_payment_side(client, rec_2, settings)

        snap_1 = read_account_snapshot(client, run_1, bank_id, account_id, now, now, settings)
        snap_2 = read_account_snapshot(client, run_2, bank_id, account_id, now, now, settings)

        assert snap_1.has_history is True
        assert len(snap_1.side_records) == 1
        assert snap_1.side_records[0].payment_id == f"{run_1}-000001"

        assert snap_2.has_history is True
        assert len(snap_2.side_records) == 1
        assert snap_2.side_records[0].payment_id == f"{run_2}-000002"

        # Count cross-run records seen
        cross_run_in_1 = [r for r in snap_1.side_records if r.run_id != run_1]
        cross_run_in_2 = [r for r in snap_2.side_records if r.run_id != run_2]
        cross_run_count = len(cross_run_in_1) + len(cross_run_in_2)
        assert cross_run_count == 0

        return CheckOutcome(
            key_number=str(cross_run_count),
            evidence={
                "run_1": run_1,
                "run_2": run_2,
                "run_1_records": len(snap_1.side_records),
                "run_2_records": len(snap_2.side_records),
                "cross_run_records_seen": cross_run_count,
            },
        )
    finally:
        for r_id in (run_1, run_2):
            for pat in (namespace_pattern(r_id, bank_id),):
                for k in client.scan_iter(match=pat):
                    client.delete(k)
        client.close()


async def check_15a_5_replay_twice_identical(workspace: Any) -> CheckOutcome:
    """
    Check 15a-5 (consistency): apply a fixed list of >=20 mixed records (payment sides and
    account events, including one duplicate and one late record), dump, apply the whole list again,
    dump -> identical; then for 3 FIXED seeds apply random.Random(seed).shuffle order to a fresh
    namespace and compare -> identical. Key number: keys compared.
    """
    client = _get_redis().Redis.from_url(host_side_redis_url())
    settings = get_default_settings()
    run_main = "3333333333333333"
    bank_id = "bank_0"
    account_id = "acc_showcase_replay"
    fixed_time = datetime(2026, 10, 29, 12, 0, 0, tzinfo=timezone.utc)

    def _build_batch(r_id: str) -> list[PaymentSideRecord | AccountEventRecord]:
        b: list[PaymentSideRecord | AccountEventRecord] = []
        for i in range(14):
            t = fixed_time - timedelta(hours=i * 4)
            s = "debit" if i % 2 == 0 else "credit"
            b.append(
                make_showcase_record(
                    run_id=r_id,
                    bank_id=bank_id,
                    account_id=account_id,
                    payment_id=f"{r_id}-pmt-{i:04d}",
                    side=s,
                    event_time=t,
                    arrival_time=t + timedelta(minutes=10 * (i % 3)),
                    amount_paise=10000 * (i + 1),
                )
            )
        for j in range(6):
            t = fixed_time - timedelta(hours=j * 8)
            kind = "device_binding_change" if j % 2 == 0 else "sim_swap"
            b.append(
                AccountEventRecord(
                    run_id=r_id,
                    bank_id=bank_id,
                    account_id=account_id,
                    event_id=f"{r_id}-dev-{j:04d}",
                    kind=kind,
                    event_time=t,
                    arrival_time=t + timedelta(minutes=15 * (j % 2)),
                    details={"step": j},
                )
            )
        # Include one duplicate record
        b.append(b[0])
        # Include one late record (event_time 3 days ago, arriving at fixed_time)
        late_t = fixed_time - timedelta(days=3)
        b.append(
            make_showcase_record(
                run_id=r_id,
                bank_id=bank_id,
                account_id=account_id,
                payment_id=f"{r_id}-pmt-late-001",
                side="debit",
                event_time=late_t,
                arrival_time=fixed_time,
                amount_paise=99900,
            )
        )
        return b

    def _apply_all(r_id: str, items: list[PaymentSideRecord | AccountEventRecord]) -> None:
        for item in items:
            if isinstance(item, PaymentSideRecord):
                apply_payment_side(client, item, settings)
            else:
                apply_account_event(client, item, settings)

    def _normalize_dump(raw_dump: dict[bytes, Any], r_id: str) -> dict[bytes, Any]:
        norm: dict[bytes, Any] = {}
        r_bytes = r_id.encode("utf-8")
        for k, (t, v) in raw_dump.items():
            norm_k = k.replace(r_bytes, b"<RUN>")
            norm_v: Any
            if isinstance(v, dict):
                norm_v = {fk.replace(r_bytes, b"<RUN>"): fv.replace(r_bytes, b"<RUN>") for fk, fv in v.items()}
            elif isinstance(v, list):
                norm_v = [(m.replace(r_bytes, b"<RUN>"), s) for m, s in v]
            else:
                norm_v = v
            norm[norm_k] = (t, norm_v)
        return norm

    batch_main = _build_batch(run_main)
    runs_to_clean = [run_main]
    try:
        # 1. First apply
        _apply_all(run_main, batch_main)
        dump_initial = _dump_account_keys(client, run_main)
        assert len(dump_initial) > 0

        # 2. Replay whole list again
        _apply_all(run_main, batch_main)
        dump_replayed = _dump_account_keys(client, run_main)
        assert dump_initial == dump_replayed, "Redis state mutated during replay"

        # 3. Test 3 FIXED seeds on fresh namespaces
        norm_baseline = _normalize_dump(dump_initial, run_main)
        seeds = (42, 1337, 2026)
        for seed in seeds:
            shuffled_run = f"shuf{seed:012d}"
            runs_to_clean.append(shuffled_run)
            shuffled_batch = _build_batch(shuffled_run)
            random.Random(seed).shuffle(shuffled_batch)
            _apply_all(shuffled_run, shuffled_batch)
            dump_shuffled = _dump_account_keys(client, shuffled_run)
            norm_shuffled = _normalize_dump(dump_shuffled, shuffled_run)
            assert norm_shuffled == norm_baseline, f"Normalized state diverged under seed {seed}"

        keys_compared = len(dump_initial)
        return CheckOutcome(
            key_number=str(keys_compared),
            evidence={
                "records_applied": len(batch_main),
                "keys_compared": keys_compared,
                "replay_identical": True,
                "seeds_tested": list(seeds),
                "seeds_identical": True,
            },
        )
    finally:
        for r_id in runs_to_clean:
            for pat in (namespace_pattern(r_id, bank_id),):
                for k in client.scan_iter(match=pat):
                    client.delete(k)
        client.close()


def get_checks(quick: bool = False) -> list[CheckDefinition]:
    """
    Return the Level 15 verification checks.
    Tagged with part='a', 'b' and 'c' for selective execution.
    """
    from showcase.level_15_support_b import (
        check_15b_1_three_profiles_normal,
        check_15b_2_cross_bank_isolation_edge,
        check_15b_3_os_update_not_new_device_edge,
        check_15b_4_rented_mule_vs_salaried_edge,
    )
    from showcase.level_15_support_c import (
        check_15c_1_late_event_leaves_logged_unchanged,
        check_15c_2_recompute_exact_match,
        check_15c_4_feature_failure_unscored,
        check_15c_5_no_unexpected_error_rows,
        make_check_15c_3,
    )

    return [
        CheckDefinition(
            name="15a-1 normal account gets correct window counts",
            kind="normal",
            proves="Window counts for 1h, 24h, and 7d match exact manual arithmetic over an 8-day transaction history.",
            run=check_15a_1_normal_window_counts,
            likely_cause="Window duration calculation or boundary inequality in count_sides_in_window was incorrect.",
            where_to_look="features/account_reader.py",
            part="a",
        ),
        CheckDefinition(
            name="15a-2 empty history gets defined defaults",
            kind="edge",
            proves=(
                "Reading an unwritten account returns empty_snapshot() and "
                "count_sides_in_window is 0 for every configured window."
            ),
            run=check_15a_2_empty_history,
            likely_cause="Reader returned populated snapshot or non-zero count for unwritten account.",
            where_to_look="features/account_reader.py",
            part="a",
        ),
        CheckDefinition(
            name="15a-3 late event lands in the right window",
            kind="edge",
            proves="A late-arriving event updates the containing window without mutating earlier window counts.",
            run=check_15a_3_late_event_lands_in_right_window,
            likely_cause=(
                "Decision time filtering or arrival cutoff logic leaked future events into historical queries."
            ),
            where_to_look="features/account_reader.py, features/account_updater.py",
            part="a",
        ),
        CheckDefinition(
            name="15a-4 two runs keep separate state",
            kind="negative",
            proves="Two fixed run IDs writing to identical bank and account IDs remain strictly isolated in Redis.",
            run=check_15a_4_two_runs_separate_state,
            likely_cause="Key namespacing omitted run_id or reader leaked records across runs.",
            where_to_look="features/account_keys.py, features/account_reader.py",
            part="a",
        ),
        CheckDefinition(
            name="15a-5 replay twice gives identical state",
            kind="consistency",
            proves="Applying mixed records twice and across shuffled seeds produces byte-identical Redis state.",
            run=check_15a_5_replay_twice_identical,
            likely_cause="Updates appended duplicates or diverged under arrival order.",
            where_to_look="features/account_updater.py",
            part="a",
        ),
        CheckDefinition(
            name="15b-1 a mule, a tea stall and a salaried person side by side",
            kind="normal",
            proves=(
                "M1 feature profiles cleanly separate high pass-through mules, small-credit tea stalls, "
                "and long-tenure salaried persons."
            ),
            run=check_15b_1_three_profiles_normal,
            likely_cause=(
                "Window aggregation, cash-out channel classification, or history days calculation returned "
                "incorrect feature values."
            ),
            where_to_look="features/m1_features.py, features/m1_inflow_outflow.py, features/m1_routine_lifecycle.py",
            part="b",
        ),
        CheckDefinition(
            name="15b-2 a cross-bank payment shows only each bank's own side",
            kind="edge",
            proves=(
                "Cross-bank payments reflect strictly sender-debit in bank 0 and receiver-credit in bank 1 "
                "without ledger bleed."
            ),
            run=check_15b_2_cross_bank_isolation_edge,
            likely_cause="Feature computation crossed tenant boundaries or leaked remote bank ledger entries.",
            where_to_look="features/account_reader.py, features/m1_inflow_outflow.py",
            part="b",
        ),
        CheckDefinition(
            name="15b-3 an OS update is not a new device",
            kind="edge",
            proves=(
                "Issue R5 constraint: operating system updates do not trigger new device flags; "
                "physical hardware switches do."
            ),
            run=check_15b_3_os_update_not_new_device_edge,
            likely_cause=(
                "Control group treated os_update as device binding change or failed to detect genuine hardware "
                "replacement."
            ),
            where_to_look="features/m1_control_region.py",
            part="b",
        ),
        CheckDefinition(
            name="15b-4 a rented mule compared with a genuine salaried person",
            kind="edge",
            proves=(
                "A rented mule keeps its owner's routine life; "
                "recent pass-through, first-time payers and other-bank outflow "
                "separate it from a genuine salaried person."
            ),
            run=check_15b_4_rented_mule_vs_salaried_edge,
            likely_cause="Routine life rhythm, pass-through ratio, or first-time payer calculation failed.",
            where_to_look="features/m1_routine_lifecycle.py, features/m1_inflow_outflow.py",
            part="b",
        ),
        CheckDefinition(
            name="15c-1 an event that arrives after a decision leaves its logged features unchanged",
            kind="edge",
            proves=(
                "A payment side recorded after a decision never rewrites that decision's feature_log row, "
                "while a fresh computation at the new watermark does change."
            ),
            run=check_15c_1_late_event_leaves_logged_unchanged,
            likely_cause="The feature log row was updated after insert, or the late event never reached Redis.",
            where_to_look="features/feature_log.py, features/decision_features.py, features/account_stream.py",
            part="c",
        ),
        CheckDefinition(
            name="15c-2 a logged vector recomputed from history matches exactly",
            kind="consistency",
            proves="Replaying Postgres history into a scratch Redis reproduces all 52 logged values exactly.",
            run=check_15c_2_recompute_exact_match,
            likely_cause="Replay applied a different set of events than live ingestion or ignored the arrival cutoff.",
            where_to_look="features/feature_replay.py",
            part="c",
        ),
        CheckDefinition(
            name="15c-3 decision latency against the budget",
            kind="normal",
            proves=(
                "In-process record_decision with feature logging on, and compute_decision_row alone, stay within "
                "the p99 budgets in config/evaluation.yaml (judged in full mode; recorded in quick mode)."
            ),
            run=make_check_15c_3(quick),
            likely_cause=(
                "Feature computation or the log insert became slower, or this machine is slower than the budget "
                "machine."
            ),
            where_to_look="benchmarks/decision_latency.py, features/decision_features.py, api/v1/bank_service.py",
            part="c",
        ),
        CheckDefinition(
            name="15c-4 a feature failure never changes the bank's answer",
            kind="negative",
            proves=(
                "When feature computation raises, the decision is still 'unscored' and the log row records "
                "status error, the error type and no arrival cutoff."
            ),
            run=check_15c_4_feature_failure_unscored,
            likely_cause="record_decision let the feature error escape, or logged an ok row for a failed computation.",
            where_to_look="api/v1/bank_service.py",
            part="c",
        ),
        CheckDefinition(
            name="15c-5 no unexpected error rows",
            kind="consistency",
            proves="Apart from the deliberate failure of 15c-4, no feature_log row has status error.",
            run=check_15c_5_no_unexpected_error_rows,
            likely_cause="A normal decision hit a feature computation error that was logged instead of raised.",
            where_to_look="features/decision_features.py, api/v1/bank_service.py",
            part="c",
        ),
    ]
