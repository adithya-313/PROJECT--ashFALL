# File: tests/integration/test_snapshots_world_clock.py
# Purpose: Integration tests for account snapshot selection by arrival time and run-scoped watermarks.
import os
import uuid
from datetime import datetime, timezone
from typing import Any

import pytest
import redis

from core.config import config
from features.account_keys import namespace_pattern
from features.account_snapshots import (
    NoWatermarkError,
    select_all_accounts,
    select_daily_snapshot_accounts,
    write_account_snapshots,
)
from features.ingest_watermark import record_payment_watermark, record_poller_watermark

pytestmark = [pytest.mark.anyio, pytest.mark.needs_stack]


def host_side_redis_url() -> str:
    """Resolve Redis connection string for test runner on host machine."""
    raw = os.environ.get("TEST_REDIS_URL") or config.redis_url
    return raw.replace("://redis:", "://localhost:").replace("@redis:", "@localhost:")


@pytest.fixture
def redis_client() -> redis.Redis:
    """Provides real Redis client connected to host endpoint."""
    return redis.Redis.from_url(host_side_redis_url())


async def test_daily_snapshot_selection_by_arrival_time(
    isolated_schema: Any,
) -> None:
    """
    Rows with arrival_time inside the world day but created_at = real now
    are selected by the daily window for that world day, and NOT selected
    for the day before.
    """
    conn, _ = isolated_schema
    run_id = uuid.uuid4().hex[:16]
    bank_id = "bank_0"

    day_start = datetime(2026, 9, 5, 0, 0, 0, tzinfo=timezone.utc)
    day_end = datetime(2026, 9, 6, 0, 0, 0, tzinfo=timezone.utc)
    prev_day_start = datetime(2026, 9, 4, 0, 0, 0, tzinfo=timezone.utc)
    prev_day_end = datetime(2026, 9, 5, 0, 0, 0, tzinfo=timezone.utc)

    real_now = datetime(2026, 11, 1, 10, 0, 0, tzinfo=timezone.utc)
    world_arrival = datetime(2026, 9, 5, 14, 0, 0, tzinfo=timezone.utc)

    acc_tx = "acc_world_tx"
    acc_evt = "acc_world_evt"
    acc_wth = "acc_world_wth"
    acc_fb = "acc_world_fb"
    acc_cmp = "acc_world_cmp"

    # 1. Transactions row
    await conn.execute(
        """
        INSERT INTO transactions (
            source, account_id, amount, payment_id, bank_id, side,
            counterparty_vpa, counterparty_bank_id, amount_paise, event_time,
            status, created_at, arrival_time
        ) VALUES (
            'world', $1, 100.00, $2, $3, 'debit',
            'peer@bank1', 'bank_1', 10000, $4,
            'success', $5, $6
        )
        """,
        acc_tx,
        f"{run_id}-tx-01",
        bank_id,
        world_arrival,
        real_now,
        world_arrival,
    )

    # 2. Bank account events row
    await conn.execute(
        """
        INSERT INTO bank_account_events (
            bank_id, event_id, account_id, kind, event_time, details, created_at, arrival_time
        ) VALUES ($1, $2, $3, 'sim_swap', $4, '{}'::jsonb, $5, $6)
        """,
        bank_id,
        f"{run_id}-evt-01",
        acc_evt,
        world_arrival,
        real_now,
        world_arrival,
    )

    # 3. Bank withdrawals row
    await conn.execute(
        """
        INSERT INTO bank_withdrawals (
            bank_id, withdrawal_id, account_id, channel, amount_paise,
            status, event_time, recorded_at, arrival_time
        ) VALUES ($1, $2, $3, 'atm', 5000, 'success', $4, $5, $6)
        """,
        bank_id,
        f"{run_id}-wth-01",
        acc_wth,
        world_arrival,
        real_now,
        world_arrival,
    )

    # 4. Bank feedback row (confirmed_fraud -> credit mule)
    await conn.execute(
        """
        INSERT INTO transactions (
            source, account_id, amount, payment_id, bank_id, side,
            counterparty_vpa, counterparty_bank_id, amount_paise, event_time,
            status, created_at, arrival_time
        ) VALUES (
            'world', $1, 200.00, $2, $3, 'credit',
            'victim@bank1', 'bank_1', 20000, $4,
            'success', $5, $6
        )
        """,
        acc_fb,
        f"{run_id}-fb-pmt-01",
        bank_id,
        world_arrival,
        real_now,
        world_arrival,
    )
    await conn.execute(
        """
        INSERT INTO bank_feedback (
            bank_id, feedback_id, payment_id, outcome, event_time, created_at, arrival_time
        ) VALUES ($1, $2, $3, 'confirmed_fraud', $4, $5, $6)
        """,
        bank_id,
        f"{run_id}-fb-01",
        f"{run_id}-fb-pmt-01",
        world_arrival,
        real_now,
        world_arrival,
    )

    # 5. Bank complaints row
    await conn.execute(
        """
        INSERT INTO bank_complaints (
            bank_id, complaint_id, complainant_account, reported_account, reported_bank_id,
            payment_ids, claimed_type, received_at, created_at, arrival_time
        ) VALUES ($1, $2, 'complainant_001', $3, $4, ARRAY['pmt_001'], 'person', $5, $6, $7)
        """,
        bank_id,
        f"{run_id}-cmp-01",
        acc_cmp,
        bank_id,
        world_arrival,
        real_now,
        world_arrival,
    )

    # Selected for the world day [day_start, day_end)
    selected = await select_daily_snapshot_accounts(conn, run_id, bank_id, day_start, day_end)
    assert acc_tx in selected
    assert acc_evt in selected
    assert acc_wth in selected
    assert acc_fb in selected
    assert acc_cmp in selected

    # NOT selected for the previous day [prev_day_start, prev_day_end)
    prev_selected = await select_daily_snapshot_accounts(conn, run_id, bank_id, prev_day_start, prev_day_end)
    assert acc_tx not in prev_selected
    assert acc_evt not in prev_selected
    assert acc_wth not in prev_selected
    assert acc_fb not in prev_selected
    assert acc_cmp not in prev_selected


async def test_account_opened_with_arrival_time_selects_at_world_cutoff(
    isolated_schema: Any,
) -> None:
    """
    account_opened with arrival_time selects into select_all_accounts at a world cutoff,
    and is not selected before arrival.
    """
    conn, _ = isolated_schema
    run_id = uuid.uuid4().hex[:16]
    bank_id = "bank_0"
    acc_opened = "acc_opened_world"

    real_now = datetime(2026, 11, 1, 10, 0, 0, tzinfo=timezone.utc)
    arrival = datetime(2026, 9, 5, 10, 0, 0, tzinfo=timezone.utc)

    await conn.execute(
        """
        INSERT INTO bank_account_events (
            bank_id, event_id, account_id, kind, event_time, details, created_at, arrival_time
        ) VALUES ($1, $2, $3, 'account_opened', $4, '{}'::jsonb, $5, $6)
        """,
        bank_id,
        f"{run_id}-evt-open-01",
        acc_opened,
        arrival,
        real_now,
        arrival,
    )

    # Cutoff at 12:00 (after arrival 10:00) -> selected
    cutoff_after = datetime(2026, 9, 5, 12, 0, 0, tzinfo=timezone.utc)
    all_after = await select_all_accounts(conn, run_id, bank_id, cutoff_after)
    assert acc_opened in all_after

    # Cutoff at 09:00 (before arrival 10:00) -> not selected
    cutoff_before = datetime(2026, 9, 5, 9, 0, 0, tzinfo=timezone.utc)
    all_before = await select_all_accounts(conn, run_id, bank_id, cutoff_before)
    assert acc_opened not in all_before


async def test_null_arrival_time_rows_behave_as_before(
    isolated_schema: Any,
) -> None:
    """NULL arrival_time rows fall back to created_at/recorded_at exactly as before."""
    conn, _ = isolated_schema
    run_id = uuid.uuid4().hex[:16]
    bank_id = "bank_0"
    acc_legacy = "acc_legacy_null_arrival"

    day_start = datetime(2026, 9, 5, 0, 0, 0, tzinfo=timezone.utc)
    day_end = datetime(2026, 9, 6, 0, 0, 0, tzinfo=timezone.utc)
    event_time = datetime(2026, 9, 5, 14, 0, 0, tzinfo=timezone.utc)

    # Insert with arrival_time NULL
    await conn.execute(
        """
        INSERT INTO bank_account_events (
            bank_id, event_id, account_id, kind, event_time, details, created_at, arrival_time
        ) VALUES ($1, $2, $3, 'account_opened', $4, '{}'::jsonb, $5, NULL)
        """,
        bank_id,
        f"{run_id}-evt-legacy-01",
        acc_legacy,
        event_time,
        event_time,
    )

    # Selected by daily window via created_at
    daily = await select_daily_snapshot_accounts(conn, run_id, bank_id, day_start, day_end)
    assert acc_legacy in daily

    # Selected by select_all_accounts via created_at <= cutoff
    all_accs = await select_all_accounts(conn, run_id, bank_id, day_end)
    assert acc_legacy in all_accs


async def test_write_account_snapshots_run_scoped_watermarks(
    isolated_schema: Any,
    redis_client: redis.Redis,
) -> None:
    """
    write_account_snapshots(run_scoped_watermarks=True) with all five run watermarks
    writes rows with arrival_cutoff equal to their MIN.
    """
    conn, _ = isolated_schema
    run_id = uuid.uuid4().hex[:16]
    bank_id = "bank_0"
    acc_id = "acc_snap_001"

    # Clean redis for this run
    for k in redis_client.scan_iter(match=namespace_pattern(run_id, bank_id)):
        redis_client.delete(k)
    for k in redis_client.scan_iter(match=f"acctwm:{run_id}:*"):
        redis_client.delete(k)

    # Set five run-scoped watermarks with withdrawals as the MIN (1788605000)
    record_payment_watermark(redis_client, 0, 1788610000, run_id=run_id)
    record_poller_watermark(redis_client, bank_id, "events", 1788620000, run_id=run_id)
    record_poller_watermark(redis_client, bank_id, "withdrawals", 1788605000, run_id=run_id)
    record_poller_watermark(redis_client, bank_id, "feedback", 1788630000, run_id=run_id)
    record_poller_watermark(redis_client, bank_id, "complaints", 1788640000, run_id=run_id)

    expected_cutoff = datetime.fromtimestamp(1788605000, tz=timezone.utc)
    snap_time = datetime(2026, 9, 5, 23, 59, 59, tzinfo=timezone.utc)

    # Pre-create day partition for snapshot_time
    await conn.execute("SELECT create_account_snapshots_partition($1::date)", snap_time.date())

    inserted = await write_account_snapshots(
        pg=conn,
        redis=redis_client,
        run_id=run_id,
        bank_id=bank_id,
        snapshot_time=snap_time,
        snapshot_kind="daily",
        account_ids=[acc_id],
        run_scoped_watermarks=True,
    )
    assert inserted == 1

    row = await conn.fetchrow(
        """
        SELECT arrival_cutoff
        FROM account_snapshots
        WHERE run_id = $1 AND account_id = $2 AND snapshot_kind = 'daily'
        """,
        run_id,
        acc_id,
    )
    assert row is not None
    assert row["arrival_cutoff"] == expected_cutoff


async def test_write_account_snapshots_run_scoped_watermarks_missing_raises_no_watermark(
    isolated_schema: Any,
    redis_client: redis.Redis,
) -> None:
    """With one source's run watermark missing, write_account_snapshots raises NoWatermarkError."""
    conn, _ = isolated_schema
    run_id = uuid.uuid4().hex[:16]
    bank_id = "bank_0"
    acc_id = "acc_snap_missing"

    for k in redis_client.scan_iter(match=f"acctwm:{run_id}:*"):
        redis_client.delete(k)

    # Only 4 sources recorded; complaints is missing
    record_payment_watermark(redis_client, 0, 1788610000, run_id=run_id)
    record_poller_watermark(redis_client, bank_id, "events", 1788620000, run_id=run_id)
    record_poller_watermark(redis_client, bank_id, "withdrawals", 1788605000, run_id=run_id)
    record_poller_watermark(redis_client, bank_id, "feedback", 1788630000, run_id=run_id)

    snap_time = datetime(2026, 9, 5, 23, 59, 59, tzinfo=timezone.utc)

    with pytest.raises(NoWatermarkError, match="No watermark available for bank"):
        await write_account_snapshots(
            pg=conn,
            redis=redis_client,
            run_id=run_id,
            bank_id=bank_id,
            snapshot_time=snap_time,
            snapshot_kind="daily",
            account_ids=[acc_id],
            run_scoped_watermarks=True,
        )
