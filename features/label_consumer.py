# File: features/label_consumer.py
# Purpose: Level 16a Round 1 - transform bank complaints into labels and consume pending complaints.
from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from features.account_keys import extract_run_id
from features.label_store import LabelRow, insert_labels


def label_from_complaint_row(row: Mapping[str, Any]) -> LabelRow | None:
    """
    WHAT: Converts a bank_complaints row mapping into an immutable LabelRow for this bank.

    WHY: Ground Rule 30 distinguishes event time (world simulation time) from transport time
    (database clock). Under Ground Rule 30, arrival_time MUST use row['received_at'], which is
    the simulation world time (IST) the bank learned about the complaint. We NEVER use row['created_at']
    and NEVER max() the two timestamps, because created_at is the database server's real wall clock
    (transport/ingestion audit time), not simulation world time. Using created_at would introduce
    temporal lookahead bias or clock drift during historical replay.

    Mirroring Level 15 same-bank rule (F5), returns None when row['reported_bank_id'] != row['bank_id'].

    Args:
        row: Mapping containing bank_complaints columns.

    Returns:
        LabelRow if complaint concerns an account belonging to this bank, or None if cross-bank.

    Raises:
        ValueError: If complaint_id is invalid or missing run prefix.
    """
    bank_id = row["bank_id"]
    reported_bank_id = row["reported_bank_id"]
    if reported_bank_id != bank_id:
        return None

    complaint_id = row["complaint_id"]
    try:
        run_id = extract_run_id(complaint_id)
    except ValueError as err:
        raise ValueError(f"Invalid complaint_id {complaint_id!r}: {err}") from err

    return LabelRow(
        run_id=run_id,
        bank_id=bank_id,
        label_id=complaint_id,
        source="complaint",
        account_id=row["reported_account"],
        payment_ids=tuple(row["payment_ids"]),
        verdict="fraud",
        arrival_time=row["received_at"],
        region=None,
        batch_id=None,
        claimed_type=row.get("claimed_type"),
        recorded_at=None,
    )


async def consume_complaint_labels_since(
    conn: Any,
    bank_id: str,
    run_id: str,
    after_id: int,
    batch_size: int,
) -> int:
    """
    WHAT: Reads a batch of bank_complaints since after_id and inserts them as fraud labels.

    WHY: Ingests ground-truth customer complaints into the permanent `labels` table for the bank,
    filtering strictly by run_id namespace and advancing the complaint stream cursor.

    Args:
        conn: asyncpg database connection.
        bank_id: Tenant commercial bank identifier.
        run_id: 16-hex simulation run namespace.
        after_id: Last read bank_complaints.id watermark.
        batch_size: Maximum number of bank_complaints rows to process (must be > 0).

    Returns:
        The highest bank_complaints.id read from the database, or after_id if no rows were found.

    Raises:
        ValueError: If batch_size <= 0.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be > 0, got {batch_size}")

    query = """
    SELECT
        id,
        bank_id,
        complaint_id,
        complainant_account,
        reported_account,
        reported_bank_id,
        payment_ids,
        claimed_type,
        received_at,
        created_at
    FROM bank_complaints
    WHERE bank_id = $1 AND id > $2
    ORDER BY id
    LIMIT $3
    """
    rows = await conn.fetch(query, bank_id, after_id, batch_size)
    if not rows:
        return after_id

    labels_to_insert: list[LabelRow] = []
    for row in rows:
        # Skip rows of other runs by exact run-id comparison
        try:
            row_run = extract_run_id(row["complaint_id"])
        except ValueError as err:
            raise ValueError(f"Invalid complaint_id {row['complaint_id']!r}: {err}") from err

        if row_run != run_id:
            continue

        label_row = label_from_complaint_row(row)
        if label_row is not None:
            labels_to_insert.append(label_row)

    if labels_to_insert:
        await insert_labels(conn, labels_to_insert)

    highest_id = max(r["id"] for r in rows)
    return int(highest_id)
