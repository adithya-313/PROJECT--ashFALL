# File: showcase/level_15.py
# Purpose: Level 15 showcase: Live mule features and account state (Part 15a).
"""
WHAT: Verification checks for Level 15 part a (Account Feature Updater).
Proves Redis key structure and TTL enforcement, point-in-time snapshot filtering,
idempotent replay without memory growth, and tenant isolation between commercial banks.

WHY: Satisfies Level 15 showcase requirements ensuring streaming feature storage
is resilient, leak-free, and isolated across simulation runs and banks.
"""

import os
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import redis

from core.config import config
from features.account_keys import (
    get_default_settings,
    marks_key,
    namespace_pattern,
    side_records_key,
    side_times_key,
)
from features.account_reader import empty_snapshot, read_account_snapshot
from features.account_records import PaymentSideRecord
from features.account_updater import apply_payment_side
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


def _dump_account_keys(client: redis.Redis, run_id: str) -> dict[bytes, Any]:
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


async def check_15a_1_key_structure_and_ttl(workspace: Any) -> CheckOutcome:
    """
    Check 15a-1 (normal): Redis key structure and TTL: write one payment side, assert all 3 keys
    exist with the right prefix, TTL > 0 and <= expiry_seconds.
    """
    client = redis.Redis.from_url(host_side_redis_url())
    settings = get_default_settings()
    run_id = uuid.uuid4().hex[:16]
    bank_id = "bank_0"
    account_id = "acc_showcase_01"
    now = datetime.now(timezone.utc)

    st_key = side_times_key(run_id, bank_id, account_id)
    sr_key = side_records_key(run_id, bank_id, account_id)
    m_key = marks_key(run_id, bank_id, account_id)

    rec = make_showcase_record(
        run_id=run_id,
        bank_id=bank_id,
        account_id=account_id,
        payment_id=f"{run_id}-000001",
        side="debit",
        event_time=now,
        arrival_time=now,
        amount_paise=250000,
    )

    try:
        outcome = apply_payment_side(client, rec, settings)
        assert outcome == "stored"

        # Assert all 3 keys exist
        assert client.exists(st_key) == 1, f"Missing sorted set key {st_key}"
        assert client.exists(sr_key) == 1, f"Missing hash records key {sr_key}"
        assert client.exists(m_key) == 1, f"Missing marks key {m_key}"

        # Assert TTL > 0 and <= expiry_seconds
        ttl_st = client.ttl(st_key)
        ttl_sr = client.ttl(sr_key)
        ttl_m = client.ttl(m_key)

        assert 0 < ttl_st <= settings.expiry_seconds, f"Unexpected TTL for {st_key}: {ttl_st}"
        assert 0 < ttl_sr <= settings.expiry_seconds, f"Unexpected TTL for {sr_key}: {ttl_sr}"
        assert 0 < ttl_m <= settings.expiry_seconds, f"Unexpected TTL for {m_key}: {ttl_m}"

        return CheckOutcome(
            key_number="15a-1",
            evidence={
                "st_key": st_key,
                "sr_key": sr_key,
                "m_key": m_key,
                "ttls": {"side_times": ttl_st, "side_records": ttl_sr, "marks": ttl_m},
                "max_expiry_seconds": settings.expiry_seconds,
            },
        )
    finally:
        for pat in (namespace_pattern(run_id, bank_id),):
            for k in client.scan_iter(match=pat):
                client.delete(k)
        client.close()


async def check_15a_2_point_in_time_reader(workspace: Any) -> CheckOutcome:
    """
    Check 15a-2 (edge): point-in-time reader: read snapshot with decision_time at event_time
    and event_time - 1s; assert event only appears in the first.
    """
    client = redis.Redis.from_url(host_side_redis_url())
    settings = get_default_settings()
    run_id = uuid.uuid4().hex[:16]
    bank_id = "bank_0"
    account_id = "acc_showcase_02"
    event_time = datetime(2026, 10, 29, 10, 0, 0, tzinfo=timezone.utc)
    arrival_time = event_time

    rec = make_showcase_record(
        run_id=run_id,
        bank_id=bank_id,
        account_id=account_id,
        payment_id=f"{run_id}-000002",
        side="debit",
        event_time=event_time,
        arrival_time=arrival_time,
        amount_paise=100000,
    )

    try:
        apply_payment_side(client, rec, settings)

        # Snapshot at exact event_time -> event must be visible
        snap_at = read_account_snapshot(
            client, run_id, bank_id, account_id, event_time, arrival_time, settings
        )
        assert snap_at.has_history is True
        assert len(snap_at.side_records) == 1
        assert snap_at.side_records[0].payment_id == f"{run_id}-000002"

        # Snapshot 1 second before event_time -> event must NOT be visible
        one_sec_before = event_time - timedelta(seconds=1)
        snap_before = read_account_snapshot(
            client, run_id, bank_id, account_id, one_sec_before, arrival_time, settings
        )
        assert snap_before.has_history is False
        assert len(snap_before.side_records) == 0

        return CheckOutcome(
            key_number="15a-2",
            evidence={
                "event_time": event_time.isoformat(),
                "visible_at_event_time": len(snap_at.side_records),
                "visible_before_event_time": len(snap_before.side_records),
            },
        )
    finally:
        for pat in (namespace_pattern(run_id, bank_id),):
            for k in client.scan_iter(match=pat):
                client.delete(k)
        client.close()


async def check_15a_3_idempotency_and_replay(workspace: Any) -> CheckOutcome:
    """
    Check 15a-3 (edge): idempotency / replay: replay the same side 3 times, verify Redis hash/zset
    sizes do not grow, byte dump before == byte dump after.
    """
    client = redis.Redis.from_url(host_side_redis_url())
    settings = get_default_settings()
    run_id = uuid.uuid4().hex[:16]
    bank_id = "bank_0"
    account_id = "acc_showcase_03"
    now = datetime.now(timezone.utc)

    rec = make_showcase_record(
        run_id=run_id,
        bank_id=bank_id,
        account_id=account_id,
        payment_id=f"{run_id}-000003",
        side="credit",
        event_time=now,
        arrival_time=now,
        amount_paise=50000,
    )

    try:
        # First write
        outcome_1 = apply_payment_side(client, rec, settings)
        assert outcome_1 == "stored"
        dump_initial = _dump_account_keys(client, run_id)
        assert len(dump_initial) > 0

        # Replay the same side 3 times
        for _ in range(3):
            outcome_replay = apply_payment_side(client, rec, settings)
            assert outcome_replay == "stored"

        dump_replayed = _dump_account_keys(client, run_id)
        assert dump_initial == dump_replayed, "Redis state mutated during replay"

        return CheckOutcome(
            key_number="15a-3",
            evidence={
                "replays_executed": 3,
                "keys_checked": len(dump_initial),
                "dumps_identical": True,
            },
        )
    finally:
        for pat in (namespace_pattern(run_id, bank_id),):
            for k in client.scan_iter(match=pat):
                client.delete(k)
        client.close()


async def check_15a_4_bank_tenant_isolation(workspace: Any) -> CheckOutcome:
    """
    Check 15a-4 (negative): isolation: writes under bank_0 do not appear under bank_1 queries.
    """
    client = redis.Redis.from_url(host_side_redis_url())
    settings = get_default_settings()
    run_id = uuid.uuid4().hex[:16]
    bank_0 = "bank_0"
    bank_1 = "bank_1"
    account_id = "acc_showcase_04"
    now = datetime.now(timezone.utc)

    rec = make_showcase_record(
        run_id=run_id,
        bank_id=bank_0,
        account_id=account_id,
        payment_id=f"{run_id}-000004",
        side="debit",
        event_time=now,
        arrival_time=now,
        amount_paise=75000,
    )

    try:
        apply_payment_side(client, rec, settings)

        # bank_0 sees the record
        snap_bank_0 = read_account_snapshot(client, run_id, bank_0, account_id, now, now, settings)
        assert snap_bank_0.has_history is True
        assert len(snap_bank_0.side_records) == 1

        # bank_1 querying the exact same account_id sees nothing
        snap_bank_1 = read_account_snapshot(client, run_id, bank_1, account_id, now, now, settings)
        assert snap_bank_1 == empty_snapshot()
        assert snap_bank_1.has_history is False
        assert len(snap_bank_1.side_records) == 0

        return CheckOutcome(
            key_number="15a-4",
            evidence={
                "bank_0_visible_count": len(snap_bank_0.side_records),
                "bank_1_visible_count": len(snap_bank_1.side_records),
                "isolation_maintained": True,
            },
        )
    finally:
        for pat in (namespace_pattern(run_id, bank_0), namespace_pattern(run_id, bank_1)):
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
            name="1. Redis key structure and TTL enforcement",
            kind="normal",
            proves="All 3 account feature keys exist with correct namespacing and positive TTL <= expiry_seconds.",
            run=check_15a_1_key_structure_and_ttl,
            likely_cause="Redis key builder incorrect or TTL expiry margin not applied in pipeline.",
            where_to_look="features/account_keys.py, features/account_updater.py",
            part="a",
        ),
        CheckDefinition(
            name="2. Point-in-time reader respects decision_time boundary",
            kind="edge",
            proves="Records at decision_time are visible while records past decision_time are strictly excluded.",
            run=check_15a_2_point_in_time_reader,
            likely_cause="Score filtering in zrangebyscore or inequality check in read_account_snapshot.",
            where_to_look="features/account_reader.py",
            part="a",
        ),
        CheckDefinition(
            name="3. Replay idempotency leaves Redis byte dump identical",
            kind="edge",
            proves="Repeatedly applying the same payment side leaves Redis key sizes and values byte-identical.",
            run=check_15a_3_idempotency_and_replay,
            likely_cause="apply_payment_side appending duplicate entries or mutating watermarks on replay.",
            where_to_look="features/account_updater.py",
            part="a",
        ),
        CheckDefinition(
            name="4. Commercial bank tenant isolation refused cross-bank visibility",
            kind="negative",
            proves="Payment sides stored under bank_0 are completely invisible to queries under bank_1.",
            run=check_15a_4_bank_tenant_isolation,
            likely_cause="Key namespacing omitting bank_id or reader querying wrong namespace.",
            where_to_look="features/account_keys.py, features/account_reader.py",
            part="a",
        ),
    ]
