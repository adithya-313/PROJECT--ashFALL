# File: tests/integration/test_upi_schema_0006.py
# Purpose: Proves migration 0006 constraints, UPI columns, backward compatibility, and uniqueness.
import shutil
from datetime import datetime, timezone
from pathlib import Path

import asyncpg
import pytest

from scripts.run_migrations import MIGRATIONS_DIR, apply_pending_migrations

pytestmark = pytest.mark.anyio


async def test_0006_applies_on_top_of_0004_and_old_rows_still_read(empty_schema, tmp_path: Path):
    """Applying 0006 on top of existing 0001-0005 migrations preserves existing rows with new columns NULL."""
    connection, schema_name = empty_schema

    # 1. Set up temporary migrations directory with migrations 0001-0005 only
    copied_dir = tmp_path / "migrations_partial"
    copied_dir.mkdir()
    for f in sorted(MIGRATIONS_DIR.glob("*.sql")):
        if f.name < "0006":
            shutil.copy(f, copied_dir / f.name)

    # Apply 0001 through 0005
    applied_count = await apply_pending_migrations(connection, copied_dir)
    assert applied_count == 5

    # 2. Insert one ecommerce row and one paysim row as superuser
    await connection.execute(
        """
        INSERT INTO transactions (source, account_id, amount)
        VALUES ('ecommerce', 'acc_ecom_old', 120.50)
        """
    )
    await connection.execute(
        """
        INSERT INTO transactions (source, account_id, amount, txn_type, old_balance_orig, new_balance_orig)
        VALUES ('paysim', 'acc_paysim_old', 250.00, 'PAYMENT', 1000.00, 750.00)
        """
    )

    # 3. Apply migration 0006 from real migrations directory
    count_0006 = await apply_pending_migrations(connection, MIGRATIONS_DIR)
    assert count_0006 == 1

    # 4. Verify both existing rows read back with new UPI columns NULL
    rows = await connection.fetch(
        """
        SELECT source, account_id, amount, payment_id, bank_id, side, counterparty_vpa,
               counterparty_bank_id, amount_paise, event_time, status, failure_domain,
               payee_type, channel
        FROM transactions
        ORDER BY id
        """
    )
    assert len(rows) == 2

    ecom_row, paysim_row = rows[0], rows[1]
    assert ecom_row["source"] == "ecommerce"
    assert ecom_row["account_id"] == "acc_ecom_old"
    assert ecom_row["amount"] == 120.50

    paysim_row = rows[1]
    assert paysim_row["source"] == "paysim"
    assert paysim_row["account_id"] == "acc_paysim_old"
    assert paysim_row["amount"] == 250.00

    new_column_names = (
        "payment_id", "bank_id", "side", "counterparty_vpa", "counterparty_bank_id",
        "amount_paise", "event_time", "status", "failure_domain", "payee_type", "channel",
    )
    for col in new_column_names:
        assert ecom_row[col] is None, f"Expected ecom_row[{col}] to be NULL"
        assert paysim_row[col] is None, f"Expected paysim_row[{col}] to be NULL"


async def test_world_row_requires_its_fields(isolated_schema):
    """A 'world' row missing required fields (e.g. event_time) is rejected with CheckViolationError."""
    connection, _ = isolated_schema
    now = datetime.now(timezone.utc)

    # Missing event_time
    with pytest.raises(asyncpg.CheckViolationError):
        await connection.execute(
            """
            INSERT INTO transactions (source, account_id, amount, payment_id, bank_id, side,
                                     amount_paise, event_time, status)
            VALUES ('world', 'acc_w_1', 100.00, 'pmt_req_1', 'bank_0', 'debit', 10000, NULL, 'success')
            """
        )

    # Missing side
    with pytest.raises(asyncpg.CheckViolationError):
        await connection.execute(
            """
            INSERT INTO transactions (source, account_id, amount, payment_id, bank_id, side,
                                     amount_paise, event_time, status)
            VALUES ('world', 'acc_w_1', 100.00, 'pmt_req_1', 'bank_0', NULL, 10000, $1, 'success')
            """,
            now,
        )


async def test_failed_status_requires_failure_domain(isolated_schema):
    """A failed status requires failure_domain, and success status rejects failure_domain."""
    connection, _ = isolated_schema
    now = datetime.now(timezone.utc)

    # Status failed with NO failure_domain -> rejected
    with pytest.raises(asyncpg.CheckViolationError):
        await connection.execute(
            """
            INSERT INTO transactions (source, account_id, amount, payment_id, bank_id, side,
                                     amount_paise, event_time, status, failure_domain)
            VALUES ('world', 'acc_w_2', 50.00, 'pmt_req_2', 'bank_0', 'debit', 5000, $1, 'failed', NULL)
            """,
            now,
        )

    # Status success WITH failure_domain -> rejected
    with pytest.raises(asyncpg.CheckViolationError):
        await connection.execute(
            """
            INSERT INTO transactions (source, account_id, amount, payment_id, bank_id, side,
                                     amount_paise, event_time, status, failure_domain)
            VALUES ('world', 'acc_w_3', 50.00, 'pmt_req_3', 'bank_0', 'debit', 5000, $1, 'success', 'sender_bank')
            """,
            now,
        )

    # Status failed with valid failure_domain -> accepted
    await connection.execute(
        """
        INSERT INTO transactions (source, account_id, amount, payment_id, bank_id, side,
                                 amount_paise, event_time, status, failure_domain)
        VALUES ('world', 'acc_w_4', 50.00, 'pmt_req_4', 'bank_0', 'debit', 5000, $1, 'failed', 'sender_bank')
        """,
        now,
    )


async def test_amount_paise_must_match_amount(isolated_schema):
    """amount_paise must equal amount * 100 and must be strictly positive."""
    connection, _ = isolated_schema
    now = datetime.now(timezone.utc)

    # Matching amount and amount_paise -> accepted
    await connection.execute(
        """
        INSERT INTO transactions (source, account_id, amount, payment_id, bank_id, side,
                                 amount_paise, event_time, status)
        VALUES ('world', 'acc_w_5', 10.50, 'pmt_amt_1', 'bank_0', 'debit', 1050, $1, 'success')
        """,
        now,
    )

    # Mismatched amount and amount_paise -> rejected
    with pytest.raises(asyncpg.CheckViolationError):
        await connection.execute(
            """
            INSERT INTO transactions (source, account_id, amount, payment_id, bank_id, side,
                                     amount_paise, event_time, status)
            VALUES ('world', 'acc_w_6', 10.50, 'pmt_amt_2', 'bank_0', 'debit', 1051, $1, 'success')
            """,
            now,
        )

    # amount_paise = 0 -> rejected
    with pytest.raises(asyncpg.CheckViolationError):
        await connection.execute(
            """
            INSERT INTO transactions (source, account_id, amount, payment_id, bank_id, side,
                                     amount_paise, event_time, status)
            VALUES ('world', 'acc_w_7', 0.00, 'pmt_amt_3', 'bank_0', 'debit', 0, $1, 'success')
            """,
            now,
        )


async def test_same_bank_payment_side_is_unique(isolated_schema):
    """Same (bank_id, payment_id, side) is unique; different side or bank is accepted."""
    connection, _ = isolated_schema
    now = datetime.now(timezone.utc)

    # Insert primary leg
    await connection.execute(
        """
        INSERT INTO transactions (source, account_id, amount, payment_id, bank_id, side,
                                 amount_paise, event_time, status)
        VALUES ('world', 'acc_leg_1', 100.00, 'pmt_uniq_1', 'bank_0', 'debit', 10000, $1, 'success')
        """,
        now,
    )

    # Duplicate leg on same bank, payment_id, side -> UniqueViolationError
    with pytest.raises(asyncpg.UniqueViolationError):
        await connection.execute(
            """
            INSERT INTO transactions (source, account_id, amount, payment_id, bank_id, side,
                                     amount_paise, event_time, status)
            VALUES ('world', 'acc_leg_1', 100.00, 'pmt_uniq_1', 'bank_0', 'debit', 10000, $1, 'success')
            """,
            now,
        )

    # Same payment_id, same bank, DIFFERENT side (credit) -> accepted
    await connection.execute(
        """
        INSERT INTO transactions (source, account_id, amount, payment_id, bank_id, side,
                                 amount_paise, event_time, status)
        VALUES ('world', 'acc_leg_1', 100.00, 'pmt_uniq_1', 'bank_0', 'credit', 10000, $1, 'success')
        """,
        now,
    )

    # Same payment_id, DIFFERENT bank (bank_1), same side (debit) -> accepted
    await connection.execute(
        """
        INSERT INTO transactions (source, account_id, amount, payment_id, bank_id, side,
                                 amount_paise, event_time, status)
        VALUES ('world', 'acc_leg_2', 100.00, 'pmt_uniq_1', 'bank_1', 'debit', 10000, $1, 'success')
        """,
        now,
    )


async def test_unknown_failure_domain_rejected(isolated_schema):
    """An unknown failure domain value is rejected by chk_failure_domain."""
    connection, _ = isolated_schema
    now = datetime.now(timezone.utc)

    with pytest.raises(asyncpg.CheckViolationError):
        await connection.execute(
            """
            INSERT INTO transactions (source, account_id, amount, payment_id, bank_id, side,
                                     amount_paise, event_time, status, failure_domain)
            VALUES ('world', 'acc_w_8', 50.00, 'pmt_req_8', 'bank_0', 'debit', 5000, $1, 'failed', 'alien_invasion')
            """,
            now,
        )
