# File: tests/unit/test_feature_log.py
# Purpose: Unit tests for Level 15c canonical feature ordering, JSON array serialization, and FeatureLogRow.
# Issue: Level 15c watermark and feature log (plan Task 1, D2).
"""
WHAT: Unit tests covering canonical M1 feature names, JSON array serialization,
and FeatureLogRow validation.

WHY: Verifies that feature ordering is immutable, types are preserved (no string-to-float conversions),
and invalid statuses or naive datetimes fail fast.
"""

from datetime import datetime, timezone

import pytest

from features.feature_log import (
    ALLOWED_STATUSES,
    FeatureLogRow,
    canonical_feature_names,
    to_feature_values_array,
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


def test_to_feature_values_array_preserves_types() -> None:
    """F7: Verifies that to_feature_values_array preserves types (ints, strings, bools, floats, nulls)."""
    names = canonical_feature_names()
    features = {
        "distinct_payers_1h": 5,
        "payee_type": "merchant",
        "routine_life_present": True,
        "pass_through_ratio_7d": 0.85,
    }

    arr = to_feature_values_array(features)
    assert len(arr) == len(names)

    idx_payers = names.index("distinct_payers_1h")
    assert arr[idx_payers] == 5
    assert isinstance(arr[idx_payers], int)

    idx_payee = names.index("payee_type")
    assert arr[idx_payee] == "merchant"
    assert isinstance(arr[idx_payee], str)

    idx_routine = names.index("routine_life_present")
    assert arr[idx_routine] is True
    assert isinstance(arr[idx_routine], bool)

    idx_ratio = names.index("pass_through_ratio_7d")
    assert arr[idx_ratio] == 0.85
    assert isinstance(arr[idx_ratio], float)


def test_to_feature_values_array_missing_fields_are_none() -> None:
    """Verifies that missing features map to None in feature values array."""
    names = canonical_feature_names()
    features = {"distinct_payers_1h": 5}
    arr = to_feature_values_array(features)

    assert len(arr) == len(names)
    assert arr[0] == 5
    assert all(val is None for val in arr[1:])


def test_feature_log_row_status_validation() -> None:
    """Verifies allowed status enforcement on FeatureLogRow."""
    for st in ALLOWED_STATUSES:
        row = FeatureLogRow(
            bank_id="bank_0",
            payment_id="pay_01",
            decision_time=NOW_UTC,
            decided_at=NOW_UTC,
            watermark_lag_ms=10,
            status=st,
            feature_values=[],
        )
        assert row.status == st

    with pytest.raises(ValueError, match="Invalid status"):
        FeatureLogRow(
            bank_id="bank_0",
            payment_id="pay_01",
            decision_time=NOW_UTC,
            decided_at=NOW_UTC,
            watermark_lag_ms=10,
            status="invalid_status",
            feature_values=[],
        )


def test_feature_log_row_timezone_validation() -> None:
    """Verifies that timezone-naive datetimes are rejected by FeatureLogRow."""
    naive_dt = datetime(2026, 10, 28, 12, 0, 0)
    with pytest.raises(ValueError, match="decision_time must be timezone-aware"):
        FeatureLogRow(
            bank_id="bank_0",
            payment_id="pay_01",
            decision_time=naive_dt,
            decided_at=NOW_UTC,
            watermark_lag_ms=10,
            status="ok",
            feature_values=[],
        )

    with pytest.raises(ValueError, match="decided_at must be timezone-aware"):
        FeatureLogRow(
            bank_id="bank_0",
            payment_id="pay_01",
            decision_time=NOW_UTC,
            decided_at=naive_dt,
            watermark_lag_ms=10,
            status="ok",
            feature_values=[],
        )


def test_feature_log_row_converts_dict_to_canonical_array() -> None:
    """Verifies that passing a feature dict automatically converts it to a canonical JSON array."""
    row = FeatureLogRow(
        bank_id="bank_0",
        payment_id="pay_01",
        decision_time=NOW_UTC,
        decided_at=NOW_UTC,
        watermark_lag_ms=25,
        status="ok",
        feature_values={"distinct_payers_1h": 3, "payee_type": "person"},
        missing_request_fields=["receiver_vpa"],
        registry_answers=[{"identifier_hash": "h1", "status": "match"}],
    )
    assert isinstance(row.feature_values, list)
    assert len(row.feature_values) == len(canonical_feature_names())
    assert row.feature_values[0] == 3
    assert row.missing_request_fields == ["receiver_vpa"]
    assert row.registry_answers is not None
    assert len(row.registry_answers) == 1
