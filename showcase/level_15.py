# File: showcase/level_15.py
# Purpose: Level 15 showcase: Live mule features and account state (Part 15a).
"""
WHAT: Verification checks for Level 15 part a (Account Feature Updater).
Proves default empty snapshot handling, multi-run tenant isolation,
and idempotent replay consistency across duplicate and shuffled events.

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
from showcase.preconditions import check_redis

LEVEL_NUMBER = 15
LEVEL_TITLE = "Live mule features and account state (built: part 15a)"
NEEDS_DATABASE = False

WHAT_YOU_SAW = (
    "Level 15 introduces near-real-time account feature updating over Redis sliding windows. "
    "Part 15a establishes namespaced key structures with TTL expiration, point-in-time snapshot "
    "reading enforcing the Section 22 future-leak rule, idempotent updates supporting replay "
    "without data corruption or set explosion, strict tenant isolation across commercial banks, "
    "and streaming ingestion from Kafka and PostgreSQL."
)

REPO_ROOT = Path(__file__).resolve().parent.parent
REQUIRED_FILES = [
    REPO_ROOT / "config" / "account_features.yaml",
    REPO_ROOT / "features" / "account_keys.py",
    REPO_ROOT / "features" / "account_records.py",
    REPO_ROOT / "features" / "account_updater.py",
    REPO_ROOT / "features" / "account_reader.py",
    REPO_ROOT / "features" / "account_stream.py",
]


def host_side_redis_url() -> str:
    """Resolve Redis connection URL on the host machine."""
    raw = os.environ.get("TEST_REDIS_URL") or config.redis_url
    return raw.replace("://redis:", "://localhost:").replace("@redis:", "@localhost:")


async def find_missing_preconditions() -> list[str]:
    """Check that Redis is reachable and all mandatory Level 15 implementation files exist."""
    problems: list[str] = []
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


async def check_15a_1_empty_history(workspace: Any) -> CheckOutcome:
    """
    Check 15a-1 (edge): read an account that was never written -> equals empty_snapshot();
    count_sides_in_window is 0 for every configured window. Key number: fields checked.
    """
    client = _get_redis().Redis.from_url(host_side_redis_url())
    settings = get_default_settings()
    run_id = uuid.uuid4().hex[:16]
    bank_id = "bank_0"
    account_id = "acc_showcase_never_written"
    now = datetime.now(timezone.utc)

    try:
        snap = read_account_snapshot(
            client, run_id, bank_id, account_id, now, now, settings
        )
        assert snap == empty_snapshot()
        assert snap.has_history is False
        assert snap.side_records == ()
        assert snap.account_events == ()
        assert snap.first_event_time is None
        assert snap.last_event_time is None
        assert snap.arrival_watermark is None
        fields_checked = 6

        for w_sec in settings.windows_seconds:
            cnt_debit = count_sides_in_window(
                snap, "debit", now, w_sec, settings
            )
            assert cnt_debit == 0
            cnt_credit = count_sides_in_window(
                snap, "credit", now, w_sec, settings
            )
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


async def check_15a_2_two_runs_separate_state(workspace: Any) -> CheckOutcome:
    """
    Check 15a-2 (negative): two FIXED run ids, same bank_id and account_id, different records ->
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

        snap_1 = read_account_snapshot(
            client, run_1, bank_id, account_id, now, now, settings
        )
        snap_2 = read_account_snapshot(
            client, run_2, bank_id, account_id, now, now, settings
        )

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


async def check_15a_3_replay_twice_identical(workspace: Any) -> CheckOutcome:
    """
    Check 15a-3 (consistency): apply a fixed list of >=20 mixed records (payment sides and
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
            if isinstance(v, dict):
                norm_v = {
                    fk.replace(r_bytes, b"<RUN>"): fv.replace(r_bytes, b"<RUN>")
                    for fk, fv in v.items()
                }
            elif isinstance(v, list):
                norm_v = [
                    (m.replace(r_bytes, b"<RUN>"), s)
                    for m, s in v
                ]
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
    Tagged with part='a' for selective execution.
    """
    return [
        CheckDefinition(
            name="15a-1 empty history gets defined defaults",
            kind="edge",
            proves=(
                "Reading an unwritten account returns empty_snapshot() and "
                "count_sides_in_window is 0 for every configured window."
            ),
            run=check_15a_1_empty_history,
            likely_cause="Reader returned populated snapshot or non-zero count for unwritten account.",
            where_to_look="features/account_reader.py",
            part="a",
        ),
        CheckDefinition(
            name="15a-2 two runs keep separate state",
            kind="negative",
            proves=(
                "Two fixed run IDs writing to identical bank and account IDs "
                "remain strictly isolated in Redis."
            ),
            run=check_15a_2_two_runs_separate_state,
            likely_cause="Key namespacing omitted run_id or reader leaked records across runs.",
            where_to_look="features/account_keys.py, features/account_reader.py",
            part="a",
        ),
        CheckDefinition(
            name="15a-3 replay twice gives identical state",
            kind="consistency",
            proves="Applying mixed records twice and across shuffled seeds produces byte-identical Redis state.",
            run=check_15a_3_replay_twice_identical,
            likely_cause="Updates appended duplicates or diverged under arrival order.",
            where_to_look="features/account_updater.py",
            part="a",
        ),
    ]
