# File: tests/integration/test_labels_0016.py
# Purpose: Level 16a Round 2 - integration tests for migration 0016 (police_bulk rename and via column).
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

import asyncpg
import pytest

from features.label_store import LabelRow, insert_labels, read_labels_known_by
from scripts.run_migrations import apply_one_file, ensure_tracking_table, find_migration_files

pytestmark = [pytest.mark.anyio, pytest.mark.needs_stack]


async def test_migration_0016_migrates_bulk_confirmation_to_police_bulk(empty_schema: Any) -> None:
    """Verifies that an existing bulk_confirmation row under 0015 is migrated to police_bulk by 0016."""
    conn, _ = empty_schema
    await ensure_tracking_table(conn)
    all_files = find_migration_files()

    # 1. Apply migrations up to 0015
    for file_path in all_files:
        if file_path.name <= "0015_create_labels.sql":
            await apply_one_file(conn, file_path)

    # 2. Insert a 'bulk_confirmation' row under 0015
    run_id = uuid.uuid4().hex[:16]
    label_id = f"{run_id}-lbl-bulk"
    now_utc = datetime(2026, 10, 20, 12, 0, 0, tzinfo=timezone.utc)
    await conn.execute(
        """
        INSERT INTO labels (
            run_id, bank_id, label_id, source, account_id, payment_ids,
            verdict, arrival_time
        ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        """,
        run_id,
        "bank_0",
        label_id,
        "bulk_confirmation",
        "acc_001",
        ["pay_001"],
        "fraud",
        now_utc,
    )

    # Verify row was inserted with source 'bulk_confirmation'
    pre_source = await conn.fetchval(
        "SELECT source FROM labels WHERE label_id = $1", label_id
    )
    assert pre_source == "bulk_confirmation"

    # 3. Apply migration 0016
    mig_0016 = next(f for f in all_files if f.name == "0016_rename_police_bulk_and_add_via.sql")
    await apply_one_file(conn, mig_0016)

    # 4. Verify the row was migrated to 'police_bulk'
    post_source = await conn.fetchval(
        "SELECT source FROM labels WHERE label_id = $1", label_id
    )
    assert post_source == "police_bulk"


async def test_bulk_confirmation_rejected_after_migration_0016(isolated_schema: Any) -> None:
    """Verifies that inserting source='bulk_confirmation' is rejected after migration 0016."""
    conn, _ = isolated_schema
    run_id = uuid.uuid4().hex[:16]
    now_utc = datetime(2026, 10, 20, 12, 0, 0, tzinfo=timezone.utc)

    with pytest.raises(asyncpg.exceptions.CheckViolationError):
        await conn.execute(
            """
            INSERT INTO labels (
                run_id, bank_id, label_id, source, account_id, payment_ids,
                verdict, arrival_time
            ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
            """,
            run_id,
            "bank_0",
            f"{run_id}-lbl-bad",
            "bulk_confirmation",
            "acc_001",
            ["pay_001"],
            "fraud",
            now_utc,
        )


async def test_labels_via_check_constraint(isolated_schema: Any) -> None:
    """Verifies that via column accepts NULL and non-blank string, but rejects whitespace-only string."""
    conn, _ = isolated_schema
    run_id = uuid.uuid4().hex[:16]
    now_utc = datetime(2026, 10, 20, 12, 0, 0, tzinfo=timezone.utc)

    insert_sql = """
    INSERT INTO labels (
        run_id, bank_id, label_id, source, account_id, payment_ids,
        verdict, arrival_time, via
    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
    """

    # (c1) via accepts NULL
    await conn.execute(
        insert_sql,
        run_id,
        "bank_0",
        f"{run_id}-lbl-v1",
        "police_bulk",
        "acc_001",
        [],
        "fraud",
        now_utc,
        None,
    )

    # (c2) via accepts non-blank string 'hold_request'
    await conn.execute(
        insert_sql,
        run_id,
        "bank_0",
        f"{run_id}-lbl-v2",
        "police_bulk",
        "acc_002",
        [],
        "fraud",
        now_utc,
        "hold_request",
    )

    # (c3) via rejects blank/whitespace string '  '
    with pytest.raises(asyncpg.exceptions.CheckViolationError):
        await conn.execute(
            insert_sql,
            run_id,
            "bank_0",
            f"{run_id}-lbl-v3",
            "police_bulk",
            "acc_003",
            [],
            "fraud",
            now_utc,
            "  ",
        )


async def test_labels_via_round_trip(isolated_schema: Any) -> None:
    """Verifies round-trip of via through insert_labels and read_labels_known_by."""
    conn, _ = isolated_schema
    run_id = uuid.uuid4().hex[:16]
    now_utc = datetime(2026, 10, 20, 12, 0, 0, tzinfo=timezone.utc)

    row_with_via = LabelRow(
        run_id=run_id,
        bank_id="bank_0",
        label_id=f"{run_id}-lbl-rt1",
        source="police_bulk",
        account_id="acc_rt1",
        payment_ids=(),
        verdict="fraud",
        arrival_time=now_utc,
        via="hold_request",
    )
    row_null_via = LabelRow(
        run_id=run_id,
        bank_id="bank_0",
        label_id=f"{run_id}-lbl-rt2",
        source="police_bulk",
        account_id="acc_rt2",
        payment_ids=(),
        verdict="fraud",
        arrival_time=now_utc,
        via=None,
    )

    inserted = await insert_labels(conn, [row_with_via, row_null_via])
    assert inserted == 2

    read_back = await read_labels_known_by(conn, run_id, "bank_0", now_utc + timedelta(seconds=1))
    assert len(read_back) == 2
    by_id = {r.label_id: r for r in read_back}
    assert by_id[f"{run_id}-lbl-rt1"].via == "hold_request"
    assert by_id[f"{run_id}-lbl-rt2"].via is None
