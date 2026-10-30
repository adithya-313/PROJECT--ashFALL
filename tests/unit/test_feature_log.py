# File: tests/unit/test_feature_log.py
# Purpose: Unit tests for Level 15c canonical feature ordering, vector serialization, and FeatureLogRow.
# Issue: Level 15c watermark and feature log (plan Task 1, D2).
"""
WHAT: Unit tests covering canonical M1 feature names, dense array serialization,
and FeatureLogRow validation.

WHY: Verifies that feature ordering is immutable, missing values map cleanly to None,
and invalid statuses or naive datetimes fail fast.
"""

from datetime import datetime, timezone

import pytest

from features.feature_log import (
    ALLOWED_STATUSES,
    FeatureLogRow,
    canonical_feature_names,
    to_ordered_values,
)

NOW_UTC = datetime(2026, 10, 28, 12, 0, 0, tzinfo=timezone.utc)


def test_canonical_feature_names_order_and_uniqueness() -> None:
    """Verifies that canonical_feature_names returns unique strings in stable schema order."""
    names = canonical_feature_names()
    assert isinstance(names, tuple)
    assert len(names) > 0
    assert len(names) == len(set(names))
    assert "distinct_payers_1h" in names
    assert "distinct_payers_24h" in names


def test_to_ordered_values_full_dict() -> None:
    """Verifies that to_ordered_values maps numeric and boolean features to floats."""
    names = canonical_feature_names()
    features = {name: float(idx) for idx, name in enumerate(names)}
    features["routine_life_present"] = True

    ordered = to_ordered_values(features)
    assert len(ordered) == len(names)

    # Boolean is cast to 1.0
    idx_bool = names.index("routine_life_present")
    assert ordered[idx_bool] == 1.0


def test_to_ordered_values_partial_dict() -> None:
    """Verifies that missing features map to None in ordered values list."""
    names = canonical_feature_names()
    features = {"distinct_payers_1h": 5}
    ordered = to_ordered_values(features)

    assert len(ordered) == len(names)
    assert ordered[0] == 5.0
    assert all(val is None for val in ordered[1:])


def test_to_ordered_values_invalid_type_raises() -> None:
    """Verifies that non-numeric and non-boolean values raise ValueError."""
    with pytest.raises(ValueError, match="invalid non-numeric value"):
        to_ordered_values({"distinct_payers_1h": "five"})


def test_feature_log_row_status_validation() -> None:
    """Verifies allowed status enforcement on FeatureLogRow."""
    for st in ALLOWED_STATUSES:
        row = FeatureLogRow(
            bank_id="bank_0",
            payment_id="pay_01",
            decision_id="dec_01",
            decision_time=NOW_UTC,
            arrival_watermark=NOW_UTC,
            status=st,
            feature_values={},
        )
        assert row.status == st

    with pytest.raises(ValueError, match="Invalid status"):
        FeatureLogRow(
            bank_id="bank_0",
            payment_id="pay_01",
            decision_id="dec_01",
            decision_time=NOW_UTC,
            arrival_watermark=NOW_UTC,
            status="invalid_status",
            feature_values={},
        )


def test_feature_log_row_timezone_validation() -> None:
    """Verifies that timezone-naive datetimes are rejected by FeatureLogRow."""
    naive_dt = datetime(2026, 10, 28, 12, 0, 0)
    with pytest.raises(ValueError, match="decision_time must be timezone-aware"):
        FeatureLogRow(
            bank_id="bank_0",
            payment_id="pay_01",
            decision_id="dec_01",
            decision_time=naive_dt,
            arrival_watermark=NOW_UTC,
            status="ok",
            feature_values={},
        )


def test_feature_log_row_auto_populates_ordered_on_ok() -> None:
    """Verifies that status='ok' automatically serializes feature_values_ordered."""
    row = FeatureLogRow(
        bank_id="bank_0",
        payment_id="pay_01",
        decision_id="dec_01",
        decision_time=NOW_UTC,
        arrival_watermark=NOW_UTC,
        status="ok",
        feature_values={"distinct_payers_1h": 3.0},
    )
    assert row.feature_values_ordered is not None
    assert len(row.feature_values_ordered) == len(canonical_feature_names())
    assert row.feature_values_ordered[0] == 3.0
