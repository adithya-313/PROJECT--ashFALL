# File: tests/integration/test_feature_log_0010.py
# Purpose: Proves migration 0010 bank_feature_log partitioned storage,
#          partition routing, RLS isolation, and missing partition retry.
# Issue: Level 15c watermark and feature log (plan Task 1, D2).
"""
WHAT: Integration tests for migration 0010 bank_feature_log.

WHY: Verifies schema migration application, daily range partition routing,
automatic partition creation retry on 23514 check_violation, and tenant isolation under RLS.
"""

from datetime import datetime, timedelta, timezone

import pytest

from api.v1.bank_scope import bank_scoped_transaction
from features.feature_log import (
    FeatureLogRow,
    insert_feature_log_row,
    read_feature_log_row,
)

pytestmark = pytest.mark.anyio


async def test_feature_log_rls_and_insert_read(isolated_schema) -> None:
    """Verifies that bank_feature_log has RLS enabled and supports insert/read with tenant isolation."""
    connection, schema_name = isolated_schema

    # 1. RLS enabled on bank_feature_log
    relrowsecurity = await connection.fetchval(
        """
        SELECT c.relrowsecurity
        FROM pg_class c
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = $1 AND c.relname = 'bank_feature_log'
        """,
        schema_name,
    )
    assert relrowsecurity is True

    # 2. Insert rows for bank_0 and bank_1
    now = datetime.now(timezone.utc)
    row_b0 = FeatureLogRow(
        bank_id="bank_0",
        payment_id="pay_001",
        decision_id="dec_001",
        decision_time=now,
        arrival_watermark=now,
        status="ok",
        feature_values={"distinct_payers_1h": 2.0},
    )
    row_b1 = FeatureLogRow(
        bank_id="bank_1",
        payment_id="pay_002",
        decision_id="dec_002",
        decision_time=now,
        arrival_watermark=now,
        status="ok",
        feature_values={"distinct_payers_1h": 4.0},
    )

    id_0 = await insert_feature_log_row(connection, row_b0)
    id_1 = await insert_feature_log_row(connection, row_b1)
    assert id_0 > 0
    assert id_1 > 0

    # 3. Read back rows directly
    fetched_b0 = await read_feature_log_row(connection, "bank_0", "dec_001")
    assert fetched_b0 is not None
    assert fetched_b0.bank_id == "bank_0"
    assert fetched_b0.decision_id == "dec_001"
    assert fetched_b0.status == "ok"
    assert fetched_b0.feature_values["distinct_payers_1h"] == 2.0

    # 4. Tenant isolation under bank_scoped_transaction
    async with bank_scoped_transaction(connection, "bank_1") as scoped_conn:
        scoped_rows = await scoped_conn.fetch("SELECT id, bank_id, decision_id FROM bank_feature_log")
        assert len(scoped_rows) == 1
        assert scoped_rows[0]["bank_id"] == "bank_1"
        assert scoped_rows[0]["decision_id"] == "dec_002"


async def test_feature_log_partition_routing(isolated_schema) -> None:
    """Verifies that rows are routed to the specific child partition matching decision_time."""
    connection, _ = isolated_schema
    now = datetime.now(timezone.utc)
    partition_name = f"feature_log_{now.strftime('%Y%m%d')}"

    row = FeatureLogRow(
        bank_id="bank_0",
        payment_id="pay_routed_01",
        decision_id="dec_routed_01",
        decision_time=now,
        arrival_watermark=now,
        status="ok",
        feature_values={"distinct_payers_1h": 1.0},
    )
    await insert_feature_log_row(connection, row)

    # Query child partition directly
    child_count = await connection.fetchval(f"SELECT count(*) FROM {partition_name}")
    assert child_count >= 1


async def test_feature_log_missing_partition_retry(isolated_schema) -> None:
    """Verifies that inserting into a date without an existing partition triggers auto-creation and succeeds."""
    connection, _ = isolated_schema

    # Choose a date 30 days in the future (where migration pre-creation does not cover)
    future_time = datetime.now(timezone.utc) + timedelta(days=30)
    future_partition = f"feature_log_{future_time.strftime('%Y%m%d')}"

    # Verify partition does NOT exist yet
    exists = await connection.fetchval(
        "SELECT to_regclass($1)",
        future_partition,
    )
    assert exists is None

    row = FeatureLogRow(
        bank_id="bank_0",
        payment_id="pay_future_01",
        decision_id="dec_future_01",
        decision_time=future_time,
        arrival_watermark=future_time,
        status="ok",
        feature_values={"distinct_payers_1h": 7.0},
    )

    # insert_feature_log_row catches CheckViolationError, creates partition, and retries
    inserted_id = await insert_feature_log_row(connection, row)
    assert inserted_id > 0

    # Verify partition now exists and contains the row
    exists_now = await connection.fetchval(
        "SELECT to_regclass($1)",
        future_partition,
    )
    assert exists_now is not None

    fetched = await read_feature_log_row(connection, "bank_0", "dec_future_01")
    assert fetched is not None
    assert fetched.payment_id == "pay_future_01"
    assert fetched.feature_values["distinct_payers_1h"] == 7.0
