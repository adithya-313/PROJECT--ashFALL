# File: tests/unit/test_label_store.py
# Purpose: Level 16a Round 1 - unit tests for label_store (LabelRow validation branches and cutoff validation).
import uuid
from datetime import datetime, timezone
from typing import Any

import pytest

from features.label_store import (
    LABEL_SOURCES,
    LABEL_VERDICTS,
    LabelRow,
    read_labels_known_by,
)


def _valid_row_kwargs() -> dict:
    run_id = uuid.uuid4().hex[:16]
    return {
        "run_id": run_id,
        "bank_id": "bank_0",
        "label_id": f"{run_id}-lbl-001",
        "source": "complaint",
        "account_id": "acc_001",
        "payment_ids": ("pay_001",),
        "verdict": "fraud",
        "arrival_time": datetime(2026, 10, 20, 12, 0, 0, tzinfo=timezone.utc),
    }


def test_valid_label_row_creation() -> None:
    """Verifies that a valid LabelRow can be constructed with all fields."""
    kwargs = _valid_row_kwargs()
    kwargs["region"] = 2
    kwargs["batch_id"] = "batch_001"
    kwargs["claimed_type"] = "impersonation"
    kwargs["recorded_at"] = datetime(2026, 10, 20, 12, 0, 1, tzinfo=timezone.utc)

    row = LabelRow(**kwargs)
    assert row.run_id == kwargs["run_id"]
    assert row.bank_id == "bank_0"
    assert row.label_id == kwargs["label_id"]
    assert row.source == "complaint"
    assert row.account_id == "acc_001"
    assert row.payment_ids == ("pay_001",)
    assert row.verdict == "fraud"
    assert row.region == 2
    assert row.batch_id == "batch_001"
    assert row.claimed_type == "impersonation"


def test_valid_label_row_defaults() -> None:
    """Verifies that optional fields default to None."""
    kwargs = _valid_row_kwargs()
    row = LabelRow(**kwargs)
    assert row.region is None
    assert row.batch_id is None
    assert row.claimed_type is None
    assert row.recorded_at is None


def test_payment_ids_converted_to_tuple() -> None:
    """Verifies that passing payment_ids as a list is converted to an immutable tuple."""
    kwargs = _valid_row_kwargs()
    kwargs["payment_ids"] = ["pay_001", "pay_002"]
    row = LabelRow(**kwargs)
    assert isinstance(row.payment_ids, tuple)
    assert row.payment_ids == ("pay_001", "pay_002")


def test_empty_bank_id_raises() -> None:
    """Verifies that an empty or whitespace bank_id raises ValueError."""
    kwargs = _valid_row_kwargs()
    kwargs["bank_id"] = ""
    with pytest.raises(ValueError, match="bank_id cannot be empty"):
        LabelRow(**kwargs)

    kwargs["bank_id"] = "   "
    with pytest.raises(ValueError, match="bank_id cannot be empty"):
        LabelRow(**kwargs)


def test_empty_account_id_raises() -> None:
    """Verifies that an empty or whitespace account_id raises ValueError."""
    kwargs = _valid_row_kwargs()
    kwargs["account_id"] = ""
    with pytest.raises(ValueError, match="account_id cannot be empty"):
        LabelRow(**kwargs)

    kwargs["account_id"] = "   "
    with pytest.raises(ValueError, match="account_id cannot be empty"):
        LabelRow(**kwargs)


def test_invalid_source_raises() -> None:
    """Verifies that a source outside LABEL_SOURCES raises ValueError."""
    kwargs = _valid_row_kwargs()
    kwargs["source"] = "telecom_feed"
    with pytest.raises(ValueError, match="Invalid source"):
        LabelRow(**kwargs)


def test_all_label_sources_accepted() -> None:
    """Verifies that all defined LABEL_SOURCES are valid."""
    for src in LABEL_SOURCES:
        kwargs = _valid_row_kwargs()
        kwargs["source"] = src
        row = LabelRow(**kwargs)
        assert row.source == src


def test_invalid_verdict_raises() -> None:
    """Verifies that a verdict outside LABEL_VERDICTS raises ValueError."""
    kwargs = _valid_row_kwargs()
    kwargs["verdict"] = "suspicious"
    with pytest.raises(ValueError, match="Invalid verdict"):
        LabelRow(**kwargs)


def test_all_label_verdicts_accepted() -> None:
    """Verifies that all defined LABEL_VERDICTS are valid."""
    for v in LABEL_VERDICTS:
        kwargs = _valid_row_kwargs()
        kwargs["verdict"] = v
        row = LabelRow(**kwargs)
        assert row.verdict == v


def test_naive_arrival_time_raises() -> None:
    """Verifies that a naive (no timezone) arrival_time raises ValueError."""
    kwargs = _valid_row_kwargs()
    kwargs["arrival_time"] = datetime(2026, 10, 20, 12, 0, 0)
    with pytest.raises(ValueError, match="arrival_time must be a timezone-aware datetime"):
        LabelRow(**kwargs)


def test_non_datetime_arrival_time_raises() -> None:
    """Verifies that a non-datetime arrival_time raises ValueError."""
    kwargs = _valid_row_kwargs()
    bad_arrival: Any = "2026-10-20T12:00:00Z"
    kwargs["arrival_time"] = bad_arrival
    with pytest.raises(ValueError, match="arrival_time must be a timezone-aware datetime"):
        LabelRow(**kwargs)


def test_label_id_run_id_mismatch_raises() -> None:
    """Verifies that a label_id whose extracted run_id differs from row.run_id raises ValueError."""
    kwargs = _valid_row_kwargs()
    other_run = uuid.uuid4().hex[:16]
    kwargs["label_id"] = f"{other_run}-lbl-001"
    with pytest.raises(ValueError, match="does not match run_id"):
        LabelRow(**kwargs)


def test_label_id_malformed_raises() -> None:
    """Verifies that a label_id lacking hyphen raises ValueError from extract_run_id."""
    kwargs = _valid_row_kwargs()
    kwargs["label_id"] = "nohyphenatall"
    with pytest.raises(ValueError, match="missing '-' delimiter"):
        LabelRow(**kwargs)


def test_negative_region_raises() -> None:
    """Verifies that a negative region raises ValueError."""
    kwargs = _valid_row_kwargs()
    kwargs["region"] = -1
    with pytest.raises(ValueError, match="region must be None or >= 0"):
        LabelRow(**kwargs)


def test_valid_region_zero_and_positive() -> None:
    """Verifies that region=0 and positive integers are accepted."""
    kwargs = _valid_row_kwargs()
    kwargs["region"] = 0
    row0 = LabelRow(**kwargs)
    assert row0.region == 0

    kwargs["region"] = 42
    row42 = LabelRow(**kwargs)
    assert row42.region == 42


@pytest.mark.anyio
async def test_read_labels_known_by_naive_cutoff_raises() -> None:
    """Verifies that read_labels_known_by rejects naive cutoff datetimes."""
    naive_cutoff = datetime(2026, 10, 20, 12, 0, 0)
    with pytest.raises(ValueError, match="cutoff must be a timezone-aware datetime"):
        await read_labels_known_by(
            conn=None,
            run_id="0123456789abcdef",
            bank_id="bank_0",
            cutoff=naive_cutoff,
        )


@pytest.mark.anyio
async def test_read_labels_known_by_non_datetime_cutoff_raises() -> None:
    """Verifies that read_labels_known_by rejects non-datetime cutoff values."""
    bad_cutoff: Any = "2026-10-20T12:00:00Z"
    with pytest.raises(ValueError, match="cutoff must be a timezone-aware datetime"):
        await read_labels_known_by(
            conn=None,
            run_id="0123456789abcdef",
            bank_id="bank_0",
            cutoff=bad_cutoff,
        )
