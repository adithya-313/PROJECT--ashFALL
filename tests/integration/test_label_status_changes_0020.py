# File: tests/integration/test_label_status_changes_0020.py
# Purpose: Level 16d-1 - migration 0020 and the label reader's revoke / dispute handling.
"""
WHAT: Checks that a revoke at or before the cutoff drops a label from read_labels_known_by, a revoke after the cutoff
keeps it (the "future trap"), a dispute marks it contested but keeps it, and the history rows are kept while the labels
row is never edited.

WHY: A cutoff reader may only use what the bank knew at the cutoff.

TERM: revoked: the label stops counting from the change's arrival time.
TERM: contested: the label still counts and is reported as contested.
"""

import uuid
from datetime import datetime, timezone
from typing import Any

import pytest

from features.label_store import LabelRow, insert_labels, read_labels_known_by, record_label_status_change

pytestmark = [pytest.mark.anyio, pytest.mark.needs_stack]

T0 = datetime(2026, 10, 1, 12, tzinfo=timezone.utc)
CUTOFF = datetime(2026, 10, 10, tzinfo=timezone.utc)


async def seed_label(conn: Any) -> tuple[str, LabelRow]:
    """Inserts one registry label for bank_0 and returns its run id and row."""
    run_id = uuid.uuid4().hex[:16]
    row = LabelRow(
        run_id=run_id, bank_id="bank_0", label_id=f"{run_id}-S000000:phone", source="registry", account_id="acc_1",
        payment_ids=(), verdict="fraud", arrival_time=T0,
    )
    await insert_labels(conn, [row])
    return run_id, row


async def change(conn: Any, row: LabelRow, kind: str, when: datetime) -> None:
    """Records one status change for the label."""
    await record_label_status_change(
        conn, row.run_id, row.bank_id, row.source, row.label_id, row.account_id, kind, when
    )


async def test_revoke_before_the_cutoff_drops_the_label(isolated_schema: Any) -> None:
    """Checks a revoke that arrived before the cutoff removes the label from the reader."""
    conn, _ = isolated_schema
    run_id, row = await seed_label(conn)
    assert len(await read_labels_known_by(conn, run_id, "bank_0", CUTOFF)) == 1
    await change(conn, row, "revoked", datetime(2026, 10, 5, tzinfo=timezone.utc))
    assert await read_labels_known_by(conn, run_id, "bank_0", CUTOFF) == []


async def test_revoke_after_the_cutoff_keeps_the_label(isolated_schema: Any) -> None:
    """Checks the future trap: a revoke after the cutoff does not exist yet for that cutoff."""
    conn, _ = isolated_schema
    run_id, row = await seed_label(conn)
    await change(conn, row, "revoked", datetime(2026, 10, 20, tzinfo=timezone.utc))
    kept = await read_labels_known_by(conn, run_id, "bank_0", CUTOFF)
    assert [r.label_id for r in kept] == [row.label_id]
    later = await read_labels_known_by(conn, run_id, "bank_0", datetime(2026, 10, 25, tzinfo=timezone.utc))
    assert later == []


async def test_dispute_marks_the_label_contested_and_still_counts_it(isolated_schema: Any) -> None:
    """Checks a contested label is returned, flagged contested only once the dispute has arrived."""
    conn, _ = isolated_schema
    run_id, row = await seed_label(conn)
    await change(conn, row, "contested", datetime(2026, 10, 7, tzinfo=timezone.utc))
    (read,) = await read_labels_known_by(conn, run_id, "bank_0", CUTOFF)
    assert read.contested is True
    (earlier,) = await read_labels_known_by(conn, run_id, "bank_0", datetime(2026, 10, 6, tzinfo=timezone.utc))
    assert earlier.contested is False


async def test_history_is_kept_and_the_label_row_is_never_edited(isolated_schema: Any) -> None:
    """Checks both changes stay as rows, in order, and the labels row is unchanged."""
    conn, _ = isolated_schema
    run_id, row = await seed_label(conn)
    before = dict(await conn.fetchrow("SELECT * FROM labels WHERE label_id = $1", row.label_id))
    await change(conn, row, "contested", datetime(2026, 10, 7, tzinfo=timezone.utc))
    await change(conn, row, "revoked", datetime(2026, 10, 8, tzinfo=timezone.utc))
    history = await conn.fetch("SELECT change FROM label_status_changes WHERE label_id = $1 ORDER BY id", row.label_id)
    assert [h["change"] for h in history] == ["contested", "revoked"]
    assert dict(await conn.fetchrow("SELECT * FROM labels WHERE label_id = $1", row.label_id)) == before
