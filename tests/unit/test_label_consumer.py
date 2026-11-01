# File: tests/unit/test_label_consumer.py
# Purpose: Level 16a Round 1 - unit tests for label_consumer (pure logic and validations).
import uuid
from datetime import datetime, timezone

import pytest

from features.label_consumer import (
    consume_complaint_labels_since,
    label_from_complaint_row,
)


def _sample_complaint_row(
    run_id: str,
    bank_id: str = "bank_0",
    reported_bank_id: str = "bank_0",
) -> dict:
    return {
        "id": 42,
        "bank_id": bank_id,
        "complaint_id": f"{run_id}-cmp-001",
        "complainant_account": "acc_victim_01",
        "reported_account": "acc_mule_01",
        "reported_bank_id": reported_bank_id,
        "payment_ids": ["pay_001", "pay_002", "pay_003"],
        "claimed_type": "phishing",
        "received_at": datetime(2026, 10, 20, 12, 0, 0, tzinfo=timezone.utc),
        "created_at": datetime(2026, 10, 20, 12, 5, 0, tzinfo=timezone.utc),
    }


def test_cross_bank_complaint_returns_none() -> None:
    """Verifies that a complaint against an account of another bank returns None (same-bank rule)."""
    run_id = uuid.uuid4().hex[:16]
    row = _sample_complaint_row(run_id, bank_id="bank_0", reported_bank_id="bank_1")
    assert label_from_complaint_row(row) is None


def test_same_bank_complaint_creates_label_row() -> None:
    """Verifies that a valid same-bank complaint creates an accurate LabelRow."""
    run_id = uuid.uuid4().hex[:16]
    row = _sample_complaint_row(run_id, bank_id="bank_0", reported_bank_id="bank_0")
    label = label_from_complaint_row(row)

    assert label is not None
    assert label.run_id == run_id
    assert label.bank_id == "bank_0"
    assert label.label_id == row["complaint_id"]
    assert label.source == "complaint"
    assert label.account_id == "acc_mule_01"
    assert label.payment_ids == ("pay_001", "pay_002", "pay_003")
    assert label.verdict == "fraud"
    assert label.arrival_time == row["received_at"]
    assert label.region is None
    assert label.batch_id is None
    assert label.claimed_type == "phishing"
    assert label.recorded_at is None


def test_arrival_time_is_received_at_when_created_at_is_later() -> None:
    """Verifies that arrival_time strictly equals received_at (world time) when created_at is later."""
    run_id = uuid.uuid4().hex[:16]
    row = _sample_complaint_row(run_id)
    t_world = datetime(2026, 10, 20, 10, 0, 0, tzinfo=timezone.utc)
    t_wall_later = datetime(2026, 10, 20, 15, 0, 0, tzinfo=timezone.utc)
    row["received_at"] = t_world
    row["created_at"] = t_wall_later

    label = label_from_complaint_row(row)
    assert label is not None
    assert label.arrival_time == t_world
    assert label.arrival_time != t_wall_later


def test_arrival_time_is_received_at_when_created_at_is_earlier() -> None:
    """Verifies that arrival_time strictly equals received_at (world time) even if created_at is earlier."""
    run_id = uuid.uuid4().hex[:16]
    row = _sample_complaint_row(run_id)
    t_world = datetime(2026, 10, 20, 15, 0, 0, tzinfo=timezone.utc)
    t_wall_earlier = datetime(2026, 10, 20, 10, 0, 0, tzinfo=timezone.utc)
    row["received_at"] = t_world
    row["created_at"] = t_wall_earlier

    label = label_from_complaint_row(row)
    assert label is not None
    assert label.arrival_time == t_world
    assert label.arrival_time != t_wall_earlier


def test_bad_complaint_id_raises_value_error() -> None:
    """Verifies that a malformed complaint_id raises ValueError naming the complaint id."""
    run_id = uuid.uuid4().hex[:16]
    row = _sample_complaint_row(run_id)
    row["complaint_id"] = "badcomplaintidwithoutdelimiter"

    with pytest.raises(ValueError, match="badcomplaintidwithoutdelimiter"):
        label_from_complaint_row(row)


def test_payment_ids_order_preserved() -> None:
    """Verifies that payment_ids are preserved in their exact original sequence."""
    run_id = uuid.uuid4().hex[:16]
    row = _sample_complaint_row(run_id)
    row["payment_ids"] = ["pay_gamma", "pay_alpha", "pay_beta"]

    label = label_from_complaint_row(row)
    assert label is not None
    assert label.payment_ids == ("pay_gamma", "pay_alpha", "pay_beta")


@pytest.mark.anyio
async def test_consume_complaint_labels_since_batch_size_validation() -> None:
    """Verifies that consume_complaint_labels_since rejects non-positive batch_size."""
    with pytest.raises(ValueError, match="batch_size must be > 0"):
        await consume_complaint_labels_since(
            conn=None,
            bank_id="bank_0",
            run_id="0123456789abcdef",
            after_id=0,
            batch_size=0,
        )

    with pytest.raises(ValueError, match="batch_size must be > 0"):
        await consume_complaint_labels_since(
            conn=None,
            bank_id="bank_0",
            run_id="0123456789abcdef",
            after_id=0,
            batch_size=-10,
        )
