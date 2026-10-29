# File: tests/integration/test_bank_api_tables_0007.py
# Purpose: Proves migration 0007 tables, row-level security policies, privilege boundaries,
#          and domain CHECK constraints for commercial bank API records.
from datetime import datetime, timezone

import asyncpg
import pytest

from api.v1.bank_scope import bank_scoped_transaction

pytestmark = pytest.mark.anyio


async def test_0007_tables_have_rls_enabled(isolated_schema) -> None:
    """
    Tables (b-f) have row-level security enabled in pg_class.relrowsecurity,
    while (a) bank_request_nonces has NO SELECT privilege granted to ashfall_bank_api.
    """
    connection, schema_name = isolated_schema
    rls_tables = [
        "bank_idempotency_keys",
        "bank_decision_requests",
        "bank_account_events",
        "bank_complaints",
        "bank_feedback",
    ]

    rows = await connection.fetch(
        """
        SELECT c.relname, c.relrowsecurity
        FROM pg_class c
        JOIN pg_namespace n ON n.oid = c.relnamespace
        WHERE n.nspname = $1 AND c.relname = ANY($2::text[])
        """,
        schema_name,
        rls_tables,
    )
    assert len(rows) == len(rls_tables)
    for row in rows:
        assert row["relrowsecurity"] is True, f"RLS not enabled on {row['relname']}"

    # For (a) bank_request_nonces: the tenant role ashfall_bank_api has no SELECT privilege
    has_priv = await connection.fetchval(
        "SELECT has_table_privilege('ashfall_bank_api', format('%I.bank_request_nonces', $1::text), 'SELECT')",
        schema_name,
    )
    assert has_priv is False, "ashfall_bank_api role must not have SELECT privilege on bank_request_nonces"


async def test_bank_role_cannot_read_other_bank_rows_in_new_tables(isolated_schema) -> None:
    """
    One row inserted per bank in bank_decision_requests.
    Under bank_1 scope, only bank_1's row is visible, while bank_0 and bank_2 rows are hidden.
    """
    connection, _ = isolated_schema

    # Insert one decision request per bank as superuser
    for b_id in ("bank_0", "bank_1", "bank_2"):
        await connection.execute(
            """
            INSERT INTO bank_decision_requests (bank_id, payment_id, request, decision, mode)
            VALUES ($1, $2, '{"amount": 100}', 'unscored', 'shadow')
            """,
            b_id,
            f"pmt_{b_id}_001",
        )

    # Superuser sees all 3 rows
    total_count = await connection.fetchval("SELECT count(*) FROM bank_decision_requests")
    assert total_count == 3

    # Under bank_1 scoped transaction, querying without a bank_id filter returns only bank_1's row
    async with bank_scoped_transaction(connection, "bank_1") as scoped_conn:
        rows = await scoped_conn.fetch("SELECT id, bank_id, payment_id FROM bank_decision_requests")
        assert len(rows) == 1
        assert rows[0]["bank_id"] == "bank_1"
        assert rows[0]["payment_id"] == "pmt_bank_1_001"


async def test_check_constraints_reject_bad_values(isolated_schema) -> None:
    """
    CHECK constraints reject bad values:
    - unknown kind in bank_account_events
    - unknown outcome in bank_feedback
    - empty payment_ids array in bank_complaints
    - decision other than 'unscored' in bank_decision_requests
    """
    connection, _ = isolated_schema
    now = datetime.now(timezone.utc)

    # 1. Unknown kind in bank_account_events
    with pytest.raises(asyncpg.CheckViolationError):
        await connection.execute(
            """
            INSERT INTO bank_account_events (bank_id, event_id, account_id, kind, event_time)
            VALUES ('bank_0', 'evt_bad_001', 'acc_001', 'invalid_kind_here', $1)
            """,
            now,
        )

    # 2. Unknown outcome in bank_feedback
    with pytest.raises(asyncpg.CheckViolationError):
        await connection.execute(
            """
            INSERT INTO bank_feedback (bank_id, feedback_id, payment_id, outcome, event_time)
            VALUES ('bank_0', 'fb_bad_001', 'pmt_001', 'unknown_outcome', $1)
            """,
            now,
        )

    # 3. Empty payment_ids array in bank_complaints (must have cardinality >= 1)
    with pytest.raises(asyncpg.CheckViolationError):
        await connection.execute(
            """
            INSERT INTO bank_complaints (bank_id, complaint_id, complainant_account,
                                        reported_account, reported_bank_id, payment_ids,
                                        claimed_type, received_at)
            VALUES ('bank_0', 'cmp_bad_001', 'acc_c', 'acc_r', 'bank_1', '{}', 'impersonation', $1)
            """,
            now,
        )

    # 4. Decision other than 'unscored' in bank_decision_requests
    with pytest.raises(asyncpg.CheckViolationError):
        await connection.execute(
            """
            INSERT INTO bank_decision_requests (bank_id, payment_id, request, decision, mode)
            VALUES ('bank_0', 'pmt_bad_dec', '{"amount": 50}', 'scored', 'shadow')
            """
        )
