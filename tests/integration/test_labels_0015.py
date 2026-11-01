# File: tests/integration/test_labels_0015.py
# Purpose: Level 16a Round 1 - integration tests for migration 0015 (labels table, schema, checks, RLS).
import re
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

import asyncpg
import pytest

from features.label_store import (
    LABEL_SOURCES,
    LabelRow,
    insert_labels,
    read_labels_known_by,
)

pytestmark = [pytest.mark.anyio, pytest.mark.needs_stack]


async def test_labels_table_columns_and_pk_exist(isolated_schema: Any) -> None:
    """Verifies that the labels table exists with expected columns, data types, and composite primary key."""
    conn, schema_name = isolated_schema

    col_rows = await conn.fetch(
        """
        SELECT column_name, data_type, is_nullable
        FROM information_schema.columns
        WHERE table_schema = $1 AND table_name = 'labels'
        ORDER BY ordinal_position
        """,
        schema_name,
    )
    col_map = {r["column_name"]: r for r in col_rows}
    expected_cols = {
        "run_id",
        "bank_id",
        "label_id",
        "source",
        "account_id",
        "payment_ids",
        "verdict",
        "arrival_time",
        "region",
        "batch_id",
        "claimed_type",
        "recorded_at",
    }
    assert expected_cols <= set(col_map.keys())

    # Check primary key definition: (bank_id, source, label_id, account_id)
    pk_rows = await conn.fetch(
        """
        SELECT kcu.column_name
        FROM information_schema.table_constraints tc
        JOIN information_schema.key_column_usage kcu
          ON tc.constraint_name = kcu.constraint_name
         AND tc.table_schema = kcu.table_schema
        WHERE tc.table_schema = $1
          AND tc.table_name = 'labels'
          AND tc.constraint_type = 'PRIMARY KEY'
        ORDER BY kcu.ordinal_position
        """,
        schema_name,
    )
    pk_columns = [r["column_name"] for r in pk_rows]
    assert pk_columns == ["bank_id", "source", "label_id", "account_id"]


async def test_labels_invalid_source_rejected(isolated_schema: Any) -> None:
    """Verifies that inserting a label with a source outside allowed sources is rejected."""
    conn, _ = isolated_schema
    run_id = uuid.uuid4().hex[:16]
    label_id = f"{run_id}-lbl-001"
    now_utc = datetime(2026, 10, 20, 12, 0, 0, tzinfo=timezone.utc)

    insert_sql = """
    INSERT INTO labels (
        run_id, bank_id, label_id, source, account_id, payment_ids,
        verdict, arrival_time
    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
    """

    with pytest.raises(asyncpg.exceptions.CheckViolationError):
        await conn.execute(
            insert_sql,
            run_id,
            "bank_0",
            label_id,
            "unsupported_source",
            "acc_001",
            ["pay_001"],
            "fraud",
            now_utc,
        )


async def test_labels_invalid_label_id_prefix_rejected(isolated_schema: Any) -> None:
    """Verifies that label_id must have prefix run_id + '-' to belong to its run namespace."""
    conn, _ = isolated_schema
    run_id = uuid.uuid4().hex[:16]
    other_run = uuid.uuid4().hex[:16]
    bad_label_id = f"{other_run}-lbl-001"
    good_label_id = f"{run_id}-lbl-001"
    now_utc = datetime(2026, 10, 20, 12, 0, 0, tzinfo=timezone.utc)

    insert_sql = """
    INSERT INTO labels (
        run_id, bank_id, label_id, source, account_id, payment_ids,
        verdict, arrival_time
    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
    """

    # Fails when prefix does not match run_id + '-'
    with pytest.raises(asyncpg.exceptions.CheckViolationError):
        await conn.execute(
            insert_sql,
            run_id,
            "bank_0",
            bad_label_id,
            "complaint",
            "acc_001",
            ["pay_001"],
            "fraud",
            now_utc,
        )

    # Succeeds when prefix matches run_id + '-'
    await conn.execute(
        insert_sql,
        run_id,
        "bank_0",
        good_label_id,
        "complaint",
        "acc_001",
        ["pay_001"],
        "fraud",
        now_utc,
    )


async def test_labels_rls_isolation(isolated_schema: Any) -> None:
    """Verifies PostgreSQL Row Level Security enforces tenant isolation on labels for role ashfall_bank_api."""
    conn, _ = isolated_schema
    run_id = uuid.uuid4().hex[:16]
    now_utc = datetime(2026, 10, 20, 12, 0, 0, tzinfo=timezone.utc)

    # Insert two rows as owner/superuser: one for bank_0, one for bank_1
    insert_sql = """
    INSERT INTO labels (
        run_id, bank_id, label_id, source, account_id, payment_ids,
        verdict, arrival_time
    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
    """
    await conn.execute(
        insert_sql,
        run_id,
        "bank_0",
        f"{run_id}-lbl-b0",
        "complaint",
        "acc_b0",
        ["pay_b0"],
        "fraud",
        now_utc,
    )
    await conn.execute(
        insert_sql,
        run_id,
        "bank_1",
        f"{run_id}-lbl-b1",
        "complaint",
        "acc_b1",
        ["pay_b1"],
        "fraud",
        now_utc,
    )

    # 1. Connected as ashfall_bank_api with bank_0: sees only bank_0
    async with conn.transaction():
        await conn.execute("SET LOCAL ROLE ashfall_bank_api")
        await conn.execute("SELECT set_config('app.bank_id', 'bank_0', true)")
        rows_b0 = await conn.fetch("SELECT bank_id, account_id FROM labels ORDER BY bank_id")
        assert len(rows_b0) == 1
        assert rows_b0[0]["bank_id"] == "bank_0"
        assert rows_b0[0]["account_id"] == "acc_b0"

    # 2. Connected as ashfall_bank_api with bank_1: sees only bank_1
    async with conn.transaction():
        await conn.execute("SET LOCAL ROLE ashfall_bank_api")
        await conn.execute("SELECT set_config('app.bank_id', 'bank_1', true)")
        rows_b1 = await conn.fetch("SELECT bank_id, account_id FROM labels ORDER BY bank_id")
        assert len(rows_b1) == 1
        assert rows_b1[0]["bank_id"] == "bank_1"
        assert rows_b1[0]["account_id"] == "acc_b1"

    # 3. Connected as ashfall_bank_api with bank_0: inserting bank_1 row is rejected by RLS WITH CHECK
    async with conn.transaction():
        await conn.execute("SET LOCAL ROLE ashfall_bank_api")
        await conn.execute("SELECT set_config('app.bank_id', 'bank_0', true)")
        with pytest.raises(asyncpg.exceptions.InsufficientPrivilegeError):
            await conn.execute(
                insert_sql,
                run_id,
                "bank_1",
                f"{run_id}-lbl-forbidden",
                "complaint",
                "acc_b1_new",
                ["pay_b1_new"],
                "fraud",
                now_utc,
            )

    # 4. Connected as ashfall_bank_api with unset bank_id: fails closed (0 rows visible)
    async with conn.transaction():
        await conn.execute("SET LOCAL ROLE ashfall_bank_api")
        await conn.execute("SELECT set_config('app.bank_id', '', true)")
        rows_empty = await conn.fetch("SELECT bank_id, account_id FROM labels")
        assert len(rows_empty) == 0


async def test_insert_labels_idempotent_and_empty(isolated_schema: Any) -> None:
    """Verifies that insert_labels is idempotent (inserts 0 on second run) and empty input returns 0."""
    conn, _ = isolated_schema
    run_id = uuid.uuid4().hex[:16]
    now_utc = datetime(2026, 10, 20, 12, 0, 0, tzinfo=timezone.utc)

    # Empty list returns 0 immediately without error
    assert await insert_labels(conn, []) == 0

    rows = [
        LabelRow(
            run_id=run_id,
            bank_id="bank_0",
            label_id=f"{run_id}-lbl-001",
            source="complaint",
            account_id="acc_001",
            payment_ids=("pay_001",),
            verdict="fraud",
            arrival_time=now_utc,
        ),
        LabelRow(
            run_id=run_id,
            bank_id="bank_0",
            label_id=f"{run_id}-lbl-002",
            source="registry",
            account_id="acc_002",
            payment_ids=(),
            verdict="fraud",
            arrival_time=now_utc,
        ),
    ]

    first_count = await insert_labels(conn, rows)
    assert first_count == 2

    # Second insert with same rows inserts 0 rows (idempotent ON CONFLICT DO NOTHING)
    second_count = await insert_labels(conn, rows)
    assert second_count == 0


async def test_three_complaints_on_one_account(isolated_schema: Any) -> None:
    """Verifies that three complaints on the same account insert three distinct rows per Ground Rule G3."""
    conn, _ = isolated_schema
    run_id = uuid.uuid4().hex[:16]
    target_account = "acc_victim_fanin"
    t_base = datetime(2026, 10, 20, 10, 0, 0, tzinfo=timezone.utc)

    rows = [
        LabelRow(
            run_id=run_id,
            bank_id="bank_0",
            label_id=f"{run_id}-cmp-001",
            source="complaint",
            account_id=target_account,
            payment_ids=("pay_001",),
            verdict="fraud",
            arrival_time=t_base,
        ),
        LabelRow(
            run_id=run_id,
            bank_id="bank_0",
            label_id=f"{run_id}-cmp-002",
            source="complaint",
            account_id=target_account,
            payment_ids=("pay_002",),
            verdict="fraud",
            arrival_time=t_base + timedelta(days=7),
        ),
        LabelRow(
            run_id=run_id,
            bank_id="bank_0",
            label_id=f"{run_id}-cmp-003",
            source="complaint",
            account_id=target_account,
            payment_ids=("pay_003",),
            verdict="fraud",
            arrival_time=t_base + timedelta(days=14),
        ),
    ]

    inserted = await insert_labels(conn, rows)
    assert inserted == 3

    labels = await read_labels_known_by(conn, run_id, "bank_0", t_base + timedelta(days=30))
    assert len(labels) == 3
    assert all(lbl.account_id == target_account for lbl in labels)
    assert {lbl.label_id for lbl in labels} == {f"{run_id}-cmp-001", f"{run_id}-cmp-002", f"{run_id}-cmp-003"}


async def test_label_maturity_cutoff_exact_boundary(isolated_schema: Any) -> None:
    """Verifies that a label at cutoff+1s is excluded and one exactly at cutoff is included."""
    conn, _ = isolated_schema
    run_id = uuid.uuid4().hex[:16]
    cutoff = datetime(2026, 10, 20, 12, 0, 0, tzinfo=timezone.utc)

    row_exact = LabelRow(
        run_id=run_id,
        bank_id="bank_0",
        label_id=f"{run_id}-lbl-exact",
        source="complaint",
        account_id="acc_exact",
        payment_ids=("pay_exact",),
        verdict="fraud",
        arrival_time=cutoff,
    )
    row_plus_1s = LabelRow(
        run_id=run_id,
        bank_id="bank_0",
        label_id=f"{run_id}-lbl-future",
        source="complaint",
        account_id="acc_future",
        payment_ids=("pay_future",),
        verdict="fraud",
        arrival_time=cutoff + timedelta(seconds=1),
    )

    await insert_labels(conn, [row_exact, row_plus_1s])

    # Replay as of cutoff: only row_exact is known
    labels_at_cutoff = await read_labels_known_by(conn, run_id, "bank_0", cutoff)
    assert len(labels_at_cutoff) == 1
    assert labels_at_cutoff[0].label_id == f"{run_id}-lbl-exact"

    # Replay at cutoff + 1s: both are known
    labels_later = await read_labels_known_by(conn, run_id, "bank_0", cutoff + timedelta(seconds=1))
    assert len(labels_later) == 2


async def test_read_labels_filters_by_run_id(isolated_schema: Any) -> None:
    """Verifies that read_labels_known_by strictly filters by run_id."""
    conn, _ = isolated_schema
    run_id_a = uuid.uuid4().hex[:16]
    run_id_b = uuid.uuid4().hex[:16]
    now_utc = datetime(2026, 10, 20, 12, 0, 0, tzinfo=timezone.utc)

    row_a = LabelRow(
        run_id=run_id_a,
        bank_id="bank_0",
        label_id=f"{run_id_a}-lbl-a",
        source="complaint",
        account_id="acc_a",
        payment_ids=(),
        verdict="fraud",
        arrival_time=now_utc,
    )
    row_b = LabelRow(
        run_id=run_id_b,
        bank_id="bank_0",
        label_id=f"{run_id_b}-lbl-b",
        source="complaint",
        account_id="acc_b",
        payment_ids=(),
        verdict="fraud",
        arrival_time=now_utc,
    )

    await insert_labels(conn, [row_a, row_b])

    labels_a = await read_labels_known_by(conn, run_id_a, "bank_0", now_utc + timedelta(hours=1))
    assert len(labels_a) == 1
    assert labels_a[0].run_id == run_id_a
    assert labels_a[0].label_id == f"{run_id_a}-lbl-a"


async def test_label_sources_equals_db_check_constraint(isolated_schema: Any) -> None:
    """Verifies that Python LABEL_SOURCES equals the source list parsed from the DB CHECK constraint."""
    conn, schema_name = isolated_schema

    rows = await conn.fetch(
        """
        SELECT pg_get_constraintdef(c.oid) AS condef
        FROM pg_constraint c
        JOIN pg_class t ON c.conrelid = t.oid
        JOIN pg_namespace n ON t.relnamespace = n.oid
        WHERE n.nspname = $1 AND t.relname = 'labels' AND c.contype = 'c'
        """,
        schema_name,
    )
    source_condefs = [r["condef"] for r in rows if "source" in r["condef"]]
    assert len(source_condefs) == 1, f"Expected 1 check constraint on source, found: {source_condefs}"

    # Extract all single-quoted string literals from the constraint definition
    extracted_sources = tuple(re.findall(r"'([^']+)'", source_condefs[0]))
    assert sorted(extracted_sources) == sorted(LABEL_SOURCES)

