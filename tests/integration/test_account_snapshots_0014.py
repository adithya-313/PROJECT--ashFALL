# File: tests/integration/test_account_snapshots_0014.py
# Purpose: Integration tests for account snapshots schema migration 0014, cadence, and RLS.
import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

import orjson
import pytest
import redis

from core.config import config
from features.account_keys import namespace_pattern
from features.account_records import PaymentSideRecord
from features.account_snapshots import (
    account_feature_names,
    select_all_accounts,
    select_daily_snapshot_accounts,
    write_account_snapshots,
)
from features.account_updater import apply_payment_side
from features.ingest_watermark import record_payment_watermark, record_poller_watermark
from features.m1_features import SUPPORTED_GROUPS, compute_account_features

pytestmark = [pytest.mark.anyio, pytest.mark.needs_stack]


def host_side_redis_url() -> str:
    """Resolve Redis connection string for test runner on host machine."""
    raw = os.environ.get("TEST_REDIS_URL") or config.redis_url
    return raw.replace("://redis:", "://localhost:").replace("@redis:", "@localhost:")


@pytest.fixture
def redis_client() -> redis.Redis:
    """Provides real Redis client connected to host endpoint."""
    return redis.Redis.from_url(host_side_redis_url())


async def test_account_snapshots_quiet_vs_active_cadence(
    isolated_schema: Any,
    redis_client: redis.Redis,
) -> None:
    """
    Verifies that:
    1. Active account appears in daily and full snapshot selections.
    2. Quiet account (activity outside window) appears only in full snapshot.
    3. Daily snapshot inserts only active account.
    4. Full snapshot inserts both active and quiet accounts.
    5. Re-executing snapshots is idempotent (inserts 0 rows on conflict).
    """
    conn, _ = isolated_schema
    run_id = uuid.uuid4().hex[:16]
    bank_id = "bank_0"

    day_start = datetime(2026, 10, 20, 0, 0, 0, tzinfo=timezone.utc)
    day_end = day_start + timedelta(days=1)

    quiet_acc = "acc_quiet_001"
    active_acc = "acc_active_002"

    def _clean_redis() -> None:
        for k in redis_client.scan_iter(match=namespace_pattern(run_id, bank_id)):
            redis_client.delete(k)
        for k in redis_client.scan_iter(match="acctwm:*"):
            redis_client.delete(k)

    _clean_redis()
    try:
        # 1. Quiet account: opened 5 days ago, last transaction 2 days ago (strictly before day_start)
        t_opened_quiet = day_start - timedelta(days=5)
        t_txn_quiet = day_start - timedelta(days=2)

        await conn.execute(
            """
            INSERT INTO bank_account_events (bank_id, event_id, account_id, kind, event_time, details, created_at)
            VALUES ($1, $2, $3, 'account_opened', $4, '{}'::jsonb, $5)
            """,
            bank_id,
            f"{run_id}-evt-quiet-open",
            quiet_acc,
            t_opened_quiet,
            t_opened_quiet,
        )
        await conn.execute(
            """
            INSERT INTO transactions (
                source, account_id, amount, payment_id, bank_id, side,
                counterparty_vpa, counterparty_bank_id, amount_paise, event_time,
                status, created_at
            ) VALUES (
                'world', $1, 10.00, $2, $3, 'debit',
                'peer@bank1', 'bank_1', 1000, $4,
                'success', $5
            )
            """,
            quiet_acc,
            f"{run_id}-tx-quiet-old",
            bank_id,
            t_txn_quiet,
            t_txn_quiet,
        )

        # 2. Active account: opened 5 days ago, transaction today (inside [day_start, day_end))
        t_opened_active = day_start - timedelta(days=5)
        t_txn_active = day_start + timedelta(hours=10)

        await conn.execute(
            """
            INSERT INTO bank_account_events (bank_id, event_id, account_id, kind, event_time, details, created_at)
            VALUES ($1, $2, $3, 'account_opened', $4, '{}'::jsonb, $5)
            """,
            bank_id,
            f"{run_id}-evt-act-open",
            active_acc,
            t_opened_active,
            t_opened_active,
        )
        await conn.execute(
            """
            INSERT INTO transactions (
                source, account_id, amount, payment_id, bank_id, side,
                counterparty_vpa, counterparty_bank_id, amount_paise, event_time,
                status, created_at
            ) VALUES (
                'world', $1, 25.00, $2, $3, 'credit',
                'buyer@bank1', 'bank_1', 2500, $4,
                'success', $5
            )
            """,
            active_acc,
            f"{run_id}-tx-act-today",
            bank_id,
            t_txn_active,
            t_txn_active,
        )

        # 3. Test selection logic
        daily_accs = await select_daily_snapshot_accounts(conn, run_id, bank_id, day_start, day_end)
        assert daily_accs == [active_acc]
        assert quiet_acc not in daily_accs
        assert active_acc in daily_accs

        all_accs = await select_all_accounts(conn, run_id, bank_id, day_end)
        assert all_accs == sorted([active_acc, quiet_acc])
        assert quiet_acc in all_accs
        assert active_acc in all_accs

        # 4. Set watermark in Redis at day_end
        record_payment_watermark(redis_client, 0, day_end)
        for src in ("events", "withdrawals", "feedback", "complaints"):
            record_poller_watermark(redis_client, bank_id, src, day_end)

        # Pre-create day partition for snapshot_time
        await conn.execute("SELECT create_account_snapshots_partition($1::date)", day_end.date())

        # 5. Write daily snapshot
        count_daily = await write_account_snapshots(
            pg=conn,
            redis=redis_client,
            run_id=run_id,
            bank_id=bank_id,
            snapshot_time=day_end,
            snapshot_kind="daily",
            account_ids=daily_accs,
        )
        assert count_daily == 1

        # 6. Write full snapshot
        count_full = await write_account_snapshots(
            pg=conn,
            redis=redis_client,
            run_id=run_id,
            bank_id=bank_id,
            snapshot_time=day_end,
            snapshot_kind="full",
            account_ids=all_accs,
        )
        assert count_full == 2

        # 7. Check database rows: quiet account appears in full but not daily
        daily_rows = await conn.fetch(
            """
            SELECT account_id FROM account_snapshots
            WHERE bank_id = $1 AND run_id = $2 AND snapshot_kind = 'daily'
            ORDER BY account_id
            """,
            bank_id,
            run_id,
        )
        assert [r["account_id"] for r in daily_rows] == [active_acc]

        full_rows = await conn.fetch(
            """
            SELECT account_id FROM account_snapshots
            WHERE bank_id = $1 AND run_id = $2 AND snapshot_kind = 'full'
            ORDER BY account_id
            """,
            bank_id,
            run_id,
        )
        assert [r["account_id"] for r in full_rows] == [active_acc, quiet_acc]

        # 8. Idempotency check: inserting same snapshot again inserts 0 rows
        idemp_daily = await write_account_snapshots(
            pg=conn,
            redis=redis_client,
            run_id=run_id,
            bank_id=bank_id,
            snapshot_time=day_end,
            snapshot_kind="daily",
            account_ids=daily_accs,
        )
        assert idemp_daily == 0

        idemp_full = await write_account_snapshots(
            pg=conn,
            redis=redis_client,
            run_id=run_id,
            bank_id=bank_id,
            snapshot_time=day_end,
            snapshot_kind="full",
            account_ids=all_accs,
        )
        assert idemp_full == 0

    finally:
        _clean_redis()


async def test_account_snapshots_watermark_cutoff_parity(
    isolated_schema: Any,
    redis_client: redis.Redis,
) -> None:
    """
    Verifies that write_account_snapshots enforces arrival_cutoff from ingestion watermark:
    1. A transaction arriving before watermark is included in computed features.
    2. A transaction arriving after watermark (even if event_time < snapshot_time) is excluded.
    3. Stored snapshot feature_values match direct compute_account_features parity at watermark cutoff.
    """
    conn, _ = isolated_schema
    run_id = uuid.uuid4().hex[:16]
    bank_id = "bank_0"
    account_id = "acc_parity_001"

    t_base = datetime(2026, 10, 20, 10, 0, 0, tzinfo=timezone.utc)
    t_wm = t_base + timedelta(hours=2)  # 12:00:00 UTC (arrival watermark)
    t_snap = t_base + timedelta(hours=4)  # 14:00:00 UTC (snapshot evaluation time)

    def _clean_redis() -> None:
        for k in redis_client.scan_iter(match=namespace_pattern(run_id, bank_id)):
            redis_client.delete(k)
        for k in redis_client.scan_iter(match="acctwm:*"):
            redis_client.delete(k)

    _clean_redis()
    try:
        # On-time credit: event_time 10:00, arrived 11:00 (< t_wm 12:00)
        rec_ontime = PaymentSideRecord(
            run_id=run_id,
            bank_id=bank_id,
            account_id=account_id,
            payment_id=f"{run_id}-tx-ontime",
            side="credit",
            counterparty_bank_id="bank_1",
            counterparty_vpa="payer1@bank1",
            amount_paise=1000,
            event_time=t_base,
            arrival_time=t_base + timedelta(hours=1),
            status="success",
            failure_domain=None,
            payee_type="person",
            channel="app",
            sender_device_fingerprint=None,
            sender_os_generation=None,
            balance_after_paise=10000,
        )
        apply_payment_side(redis_client, rec_ontime)

        # Late-arriving credit: event_time 11:30 (< t_snap), arrived 13:00 (> t_wm 12:00!)
        rec_late = PaymentSideRecord(
            run_id=run_id,
            bank_id=bank_id,
            account_id=account_id,
            payment_id=f"{run_id}-tx-late",
            side="credit",
            counterparty_bank_id="bank_1",
            counterparty_vpa="payer2@bank1",
            amount_paise=2000,
            event_time=t_base + timedelta(hours=1, minutes=30),
            arrival_time=t_base + timedelta(hours=3),
            status="success",
            failure_domain=None,
            payee_type="person",
            channel="app",
            sender_device_fingerprint=None,
            sender_os_generation=None,
            balance_after_paise=12000,
        )
        apply_payment_side(redis_client, rec_late)

        # Set watermark at t_wm (12:00 UTC)
        record_payment_watermark(redis_client, 0, t_wm)
        for src in ("events", "withdrawals", "feedback", "complaints"):
            record_poller_watermark(redis_client, bank_id, src, t_wm)

        # Direct computation with arrival_cutoff = t_wm (excludes rec_late, distinct_payers_24h == 1)
        direct_feats = compute_account_features(
            client=redis_client,
            run_id=run_id,
            bank_id=bank_id,
            account_id=account_id,
            decision_time=t_snap,
            arrival_cutoff=t_wm,
            groups=tuple(SUPPORTED_GROUPS),
            screener=None,
        )
        assert direct_feats["distinct_payers_24h"] == 1

        # Pre-create day partition for snapshot_time
        await conn.execute("SELECT create_account_snapshots_partition($1::date)", t_snap.date())

        # Write snapshot
        count = await write_account_snapshots(
            pg=conn,
            redis=redis_client,
            run_id=run_id,
            bank_id=bank_id,
            snapshot_time=t_snap,
            snapshot_kind="daily",
            account_ids=[account_id],
        )
        assert count == 1

        # Read persisted row
        row = await conn.fetchrow(
            """
            SELECT arrival_cutoff, feature_values
            FROM account_snapshots
            WHERE bank_id = $1 AND run_id = $2 AND account_id = $3 AND snapshot_kind = 'daily'
            """,
            bank_id,
            run_id,
            account_id,
        )
        assert row is not None
        assert row["arrival_cutoff"] == t_wm

        persisted_values = orjson.loads(row["feature_values"])
        names = account_feature_names()
        expected_ordered = [direct_feats[name] for name in names]
        assert persisted_values == expected_ordered

    finally:
        _clean_redis()


async def test_account_snapshots_rls_isolation(isolated_schema: Any) -> None:
    """
    Verifies PostgreSQL Row Level Security enforces tenant isolation on account_snapshots
    for role ashfall_bank_api using app.bank_id setting.
    """
    conn, _ = isolated_schema
    run_id = uuid.uuid4().hex[:16]
    snap_time = datetime(2026, 10, 20, 12, 0, 0, tzinfo=timezone.utc)

    # Insert two snapshot rows directly as superuser/owner: one for bank_0, one for bank_1
    await conn.execute("SELECT create_account_snapshots_partition($1::date)", snap_time.date())
    insert_sql = """
    INSERT INTO account_snapshots (
        bank_id, run_id, account_id, snapshot_time, snapshot_kind,
        arrival_cutoff, feature_set_version, feature_names_sha256, feature_values
    ) VALUES ($1, $2, $3, $4, 'daily', $4, 'm1', 'sha_dummy', '[]'::jsonb)
    """
    await conn.execute(insert_sql, "bank_0", run_id, "acc_b0", snap_time)
    await conn.execute(insert_sql, "bank_1", run_id, "acc_b1", snap_time)

    # 1. Connected as ashfall_bank_api with bank_0: sees only bank_0
    async with conn.transaction():
        await conn.execute("SET LOCAL ROLE ashfall_bank_api")
        await conn.execute("SELECT set_config('app.bank_id', 'bank_0', true)")
        rows_b0 = await conn.fetch("SELECT bank_id, account_id FROM account_snapshots ORDER BY bank_id")
        assert len(rows_b0) == 1
        assert rows_b0[0]["bank_id"] == "bank_0"
        assert rows_b0[0]["account_id"] == "acc_b0"

    # 2. Connected as ashfall_bank_api with bank_1: sees only bank_1
    async with conn.transaction():
        await conn.execute("SET LOCAL ROLE ashfall_bank_api")
        await conn.execute("SELECT set_config('app.bank_id', 'bank_1', true)")
        rows_b1 = await conn.fetch("SELECT bank_id, account_id FROM account_snapshots ORDER BY bank_id")
        assert len(rows_b1) == 1
        assert rows_b1[0]["bank_id"] == "bank_1"
        assert rows_b1[0]["account_id"] == "acc_b1"

    # 3. Connected as ashfall_bank_api with unset bank_id: fails closed (0 rows)
    async with conn.transaction():
        await conn.execute("SET LOCAL ROLE ashfall_bank_api")
        await conn.execute("SELECT set_config('app.bank_id', '', true)")
        rows_empty = await conn.fetch("SELECT bank_id, account_id FROM account_snapshots")
        assert len(rows_empty) == 0
