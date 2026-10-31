# File: tests/integration/test_feature_log_0010.py
# Purpose: Proves migration 0011 feature_log partitioned storage,
#          partition routing, RLS isolation, and missing partition retry.
# Issue: Level 15c watermark and feature log (plan Task 1, D2).
"""
WHAT: Integration tests for migration 0011 feature_log.

WHY: Verifies schema migration application, daily range partition routing,
automatic partition creation retry on 23514 check_violation, and tenant isolation under RLS.
"""

from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from api.v1.bank_scope import bank_scoped_transaction
from features.feature_log import (
    FeatureLogRow,
    canonical_feature_names,
    insert_feature_log_row,
    read_feature_log_row,
)

pytestmark = pytest.mark.anyio


async def test_feature_log_rls_and_insert_read(isolated_schema) -> None:
    """Verifies that feature_log has RLS enabled and supports insert/read with tenant isolation."""
    connection, schema_name = isolated_schema

    # 1. RLS enabled on feature_log
    relrowsecurity = await connection.fetchval(
        """
        SELECT c.relrowsecurity
        FROM pg_class c
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = $1 AND c.relname = 'feature_log'
        """,
        schema_name,
    )
    assert relrowsecurity is True

    # 2. Insert rows for bank_0 and bank_1
    now = datetime.now(timezone.utc)
    names = canonical_feature_names()
    f_b0: dict[str, Any] = {name: None for name in names}
    f_b0["distinct_payers_1h"] = 2.0
    f_b1: dict[str, Any] = {name: None for name in names}
    f_b1["distinct_payers_1h"] = 4.0

    row_b0 = FeatureLogRow(
        bank_id="bank_0",
        payment_id="pay_001",
        run_id="run_01",
        account_id="acc_01",
        decision_time=now,
        decided_at=now,
        arrival_cutoff=now,
        watermark_lag_ms=15,
        status="ok",
        feature_values=f_b0,
    )
    row_b1 = FeatureLogRow(
        bank_id="bank_1",
        payment_id="pay_002",
        run_id="run_02",
        account_id="acc_02",
        decision_time=now,
        decided_at=now,
        arrival_cutoff=now,
        watermark_lag_ms=20,
        status="ok",
        feature_values=f_b1,
    )

    await insert_feature_log_row(connection, row_b0)
    await insert_feature_log_row(connection, row_b1)

    # 3. Read back rows directly
    fetched_b0 = await read_feature_log_row(connection, "bank_0", "pay_001")
    assert fetched_b0 is not None
    assert fetched_b0.bank_id == "bank_0"
    assert fetched_b0.payment_id == "pay_001"
    assert fetched_b0.run_id == "run_01"
    assert fetched_b0.account_id == "acc_01"
    assert fetched_b0.arrival_cutoff == now
    assert fetched_b0.feature_set_version == "m1"
    assert fetched_b0.status == "ok"
    assert fetched_b0.watermark_lag_ms == 15
    assert isinstance(fetched_b0.feature_values, list)

    # 4. Tenant isolation under bank_scoped_transaction
    async with bank_scoped_transaction(connection, "bank_1") as scoped_conn:
        scoped_rows = await scoped_conn.fetch("SELECT bank_id, payment_id FROM feature_log")
        assert len(scoped_rows) == 1
        assert scoped_rows[0]["bank_id"] == "bank_1"
        assert scoped_rows[0]["payment_id"] == "pay_002"


async def test_feature_log_partition_routing(isolated_schema) -> None:
    """Verifies that rows are routed to the specific child partition matching decision_time."""
    connection, _ = isolated_schema
    now = datetime.now(timezone.utc)
    partition_name = f"feature_log_{now.strftime('%Y%m%d')}"

    names = canonical_feature_names()
    f_routed: dict[str, Any] = {name: None for name in names}
    f_routed["distinct_payers_1h"] = 1.0

    row = FeatureLogRow(
        bank_id="bank_0",
        payment_id="pay_routed_01",
        run_id="run_01",
        account_id="acc_01",
        decision_time=now,
        decided_at=now,
        arrival_cutoff=now,
        watermark_lag_ms=5,
        status="ok",
        feature_values=f_routed,
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
        run_id="run_01",
        account_id="acc_01",
        decision_time=future_time,
        decided_at=future_time,
        arrival_cutoff=None,
        watermark_lag_ms=None,
        status="no_watermark",
        feature_values=[],
    )

    # insert_feature_log_row catches CheckViolationError, creates partition, and retries
    await insert_feature_log_row(connection, row)

    # Verify partition now exists and contains the row
    exists_now = await connection.fetchval(
        "SELECT to_regclass($1)",
        future_partition,
    )
    assert exists_now is not None

    fetched = await read_feature_log_row(connection, "bank_0", "pay_future_01")
    assert fetched is not None
    assert fetched.payment_id == "pay_future_01"
    assert fetched.status == "no_watermark"
