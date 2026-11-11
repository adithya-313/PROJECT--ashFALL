# File: tests/integration/test_label_consumer_end_to_end.py
# Purpose: Level 16a Round 1 - end-to-end integration test for complaint consumer and label maturity.
import uuid
from datetime import datetime, timedelta, timezone
from typing import Any

import pytest

from features.label_consumer import consume_complaint_labels_since
from features.label_store import read_labels_known_by

pytestmark = [pytest.mark.anyio, pytest.mark.needs_stack]


async def test_complaint_consumer_end_to_end(isolated_schema: Any) -> None:
    """
    End-to-end integration test verifying that:
    (a) one account receiving complaints from 3 victims over 3 weeks yields 3 labels, each with its own arrival_time;
    (b) a cross-bank complaint makes no label (same-bank rule);
    (c) running the consumer twice adds 0 rows (idempotent);
    (d) another run's complaint is skipped;
    (e) read_labels_known_by at three cutoffs returns exactly the labels arrived by each cutoff.
    """
    conn, _ = isolated_schema
    run_id = uuid.uuid4().hex[:16]
    other_run = uuid.uuid4().hex[:16]
    bank_id = "bank_0"
    mule_account = "acc_mule_lifecycle_001"

    t_base = datetime(2026, 10, 1, 10, 0, 0, tzinfo=timezone.utc)
    t_w1 = t_base
    t_w2 = t_base + timedelta(days=7)
    t_w3 = t_base + timedelta(days=14)

    insert_complaint_sql = """
    INSERT INTO bank_complaints (
        bank_id, complaint_id, complainant_account, reported_account,
        reported_bank_id, payment_ids, claimed_type, received_at, created_at
    ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
    """

    # (a) 3 complaints for the same mule account over 3 weeks from 3 distinct victims
    await conn.execute(
        insert_complaint_sql,
        bank_id,
        f"{run_id}-cmp-w1",
        "acc_victim_01",
        mule_account,
        bank_id,
        ["pay_w1_01"],
        "impersonation",
        t_w1,
        t_w1 + timedelta(minutes=15),
    )
    await conn.execute(
        insert_complaint_sql,
        bank_id,
        f"{run_id}-cmp-w2",
        "acc_victim_02",
        mule_account,
        bank_id,
        ["pay_w2_02"],
        "job_scam",
        t_w2,
        t_w2 + timedelta(minutes=20),
    )
    await conn.execute(
        insert_complaint_sql,
        bank_id,
        f"{run_id}-cmp-w3",
        "acc_victim_03",
        mule_account,
        bank_id,
        ["pay_w3_03"],
        "investment_fraud",
        t_w3,
        t_w3 + timedelta(minutes=5),
    )

    # (b) Cross-bank complaint: reported_bank_id is bank_1, not bank_0
    await conn.execute(
        insert_complaint_sql,
        bank_id,
        f"{run_id}-cmp-cross",
        "acc_victim_04",
        "acc_bank1_account",
        "bank_1",
        ["pay_cross_04"],
        "phishing",
        t_w1,
        t_w1 + timedelta(minutes=10),
    )

    # (d) Another run's complaint in the same bank
    await conn.execute(
        insert_complaint_sql,
        bank_id,
        f"{other_run}-cmp-other",
        "acc_victim_05",
        "acc_other_mule",
        bank_id,
        ["pay_other_05"],
        "extortion",
        t_w1,
        t_w1 + timedelta(minutes=10),
    )

    # Run consumer for bank_0 and run_id
    highest_id = await consume_complaint_labels_since(
        conn=conn,
        bank_id=bank_id,
        run_id=run_id,
        after_id=0,
        batch_size=100,
    )
    assert highest_id > 0

    # (a) Verify 3 labels created for mule_account with respective arrival_times
    mule_labels = await conn.fetch(
        "SELECT * FROM labels WHERE bank_id = $1 AND run_id = $2 AND account_id = $3 ORDER BY arrival_time",
        bank_id,
        run_id,
        mule_account,
    )
    assert len(mule_labels) == 3
    assert [r["label_id"] for r in mule_labels] == [
        f"{run_id}-cmp-w1",
        f"{run_id}-cmp-w2",
        f"{run_id}-cmp-w3",
    ]
    assert [r["arrival_time"] for r in mule_labels] == [t_w1, t_w2, t_w3]

    # (b) Cross-bank complaint produced no label
    cross_labels = await conn.fetch(
        "SELECT * FROM labels WHERE account_id = 'acc_bank1_account'",
    )
    assert len(cross_labels) == 0

    # (d) Another run's complaint was skipped
    other_labels = await conn.fetch(
        "SELECT * FROM labels WHERE run_id = $1",
        other_run,
    )
    assert len(other_labels) == 0

    # (c) Running the consumer twice adds 0 rows (idempotent)
    # When called with watermark after_id=highest_id: returns highest_id, reads nothing
    cursor_advance = await consume_complaint_labels_since(
        conn=conn,
        bank_id=bank_id,
        run_id=run_id,
        after_id=highest_id,
        batch_size=100,
    )
    assert cursor_advance == highest_id

    # When re-run from 0: processes rows again but inserts 0 rows (idempotent)
    cursor_replay = await consume_complaint_labels_since(
        conn=conn,
        bank_id=bank_id,
        run_id=run_id,
        after_id=0,
        batch_size=100,
    )
    assert cursor_replay == highest_id

    total_labels = await conn.fetchval(
        "SELECT count(*) FROM labels WHERE bank_id = $1 AND run_id = $2",
        bank_id,
        run_id,
    )
    assert total_labels == 3

    # (e) read_labels_known_by at three cutoffs returns exactly the labels arrived by each
    labels_w1 = await read_labels_known_by(conn, run_id, bank_id, t_w1)
    assert len(labels_w1) == 1
    assert labels_w1[0].label_id == f"{run_id}-cmp-w1"

    labels_w2 = await read_labels_known_by(conn, run_id, bank_id, t_w2)
    assert len(labels_w2) == 2
    assert [r.label_id for r in labels_w2] == [f"{run_id}-cmp-w1", f"{run_id}-cmp-w2"]

    labels_w3 = await read_labels_known_by(conn, run_id, bank_id, t_w3)
    assert len(labels_w3) == 3
    assert [r.label_id for r in labels_w3] == [
        f"{run_id}-cmp-w1",
        f"{run_id}-cmp-w2",
        f"{run_id}-cmp-w3",
    ]
