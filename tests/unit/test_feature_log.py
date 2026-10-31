# File: tests/unit/test_feature_log.py
# Purpose: Unit tests for Level 15c canonical feature ordering, JSON array serialization, and FeatureLogRow.
# Issue: Level 15c watermark and feature log (plan Task 1, D2).
"""
WHAT: Unit tests covering canonical M1 feature names, JSON array serialization,
and FeatureLogRow validation.

WHY: Verifies that feature ordering is immutable, types are preserved (no string-to-float conversions),
and invalid statuses or naive datetimes fail fast.
"""

import hashlib
from datetime import datetime, timezone
from typing import Any

import pytest

from features.feature_log import (
    ALLOWED_STATUSES,
    FeatureLogRow,
    canonical_feature_names,
    canonical_feature_names_sha256,
    to_ordered_values,
)
from features.feature_set import FEATURE_SET_VERSION

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
    """F7: Verifies that to_ordered_values preserves types (ints, strings, bools, floats, nulls)."""
    names = canonical_feature_names()
    features: dict[str, Any] = {name: None for name in names}
    features.update(
        {
            "distinct_payers_1h": 5,
            "payee_type": "merchant",
            "routine_life_present": True,
            "pass_through_ratio_7d": 0.85,
        }
    )

    arr = to_ordered_values(features)
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


def test_to_ordered_values_requires_exact_canonical_keys() -> None:
    """F7: Verifies that to_ordered_values raises ValueError if key set differs from canonical set."""
    names = canonical_feature_names()
    features = {"distinct_payers_1h": 5}

    with pytest.raises(ValueError, match="Feature dict keys do not match canonical schema"):
        to_ordered_values(features)

    # Extra key also raises
    features_all: dict[str, Any] = {name: None for name in names}
    features_all["unexpected_feature"] = 123
    with pytest.raises(ValueError, match="Feature dict keys do not match canonical schema"):
        to_ordered_values(features_all)


def test_feature_log_row_status_validation() -> None:
    """Verifies allowed status enforcement on FeatureLogRow."""
    for st in ALLOWED_STATUSES:
        row = FeatureLogRow(
            bank_id="bank_0",
            payment_id="pay_01",
            run_id="run_01",
            account_id="acc_01",
            decision_time=NOW_UTC,
            decided_at=NOW_UTC,
            arrival_cutoff=None if st == "no_watermark" else NOW_UTC,
            watermark_lag_ms=10,
            status=st,
            feature_values=[],
            error_type="ValueError" if st == "error" else None,
            error_message="failed" if st == "error" else None,
        )
        assert row.status == st

    with pytest.raises(ValueError, match="Invalid status"):
        FeatureLogRow(
            bank_id="bank_0",
            payment_id="pay_01",
            run_id="run_01",
            account_id="acc_01",
            decision_time=NOW_UTC,
            decided_at=NOW_UTC,
            arrival_cutoff=NOW_UTC,
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
            run_id="run_01",
            account_id="acc_01",
            decision_time=naive_dt,
            decided_at=NOW_UTC,
            arrival_cutoff=NOW_UTC,
            watermark_lag_ms=10,
            status="ok",
            feature_values=[],
        )

    with pytest.raises(ValueError, match="decided_at must be timezone-aware"):
        FeatureLogRow(
            bank_id="bank_0",
            payment_id="pay_01",
            run_id="run_01",
            account_id="acc_01",
            decision_time=NOW_UTC,
            decided_at=naive_dt,
            arrival_cutoff=NOW_UTC,
            watermark_lag_ms=10,
            status="ok",
            feature_values=[],
        )

    with pytest.raises(ValueError, match="arrival_cutoff must be timezone-aware"):
        FeatureLogRow(
            bank_id="bank_0",
            payment_id="pay_01",
            run_id="run_01",
            account_id="acc_01",
            decision_time=NOW_UTC,
            decided_at=NOW_UTC,
            arrival_cutoff=naive_dt,
            watermark_lag_ms=10,
            status="ok",
            feature_values=[],
        )


def test_feature_log_row_converts_dict_to_canonical_array() -> None:
    """Verifies that passing a feature dict automatically converts it to a canonical JSON array."""
    names = canonical_feature_names()
    f_dict: dict[str, Any] = {name: None for name in names}
    f_dict["distinct_payers_1h"] = 3
    f_dict["payee_type"] = "person"
    row = FeatureLogRow(
        bank_id="bank_0",
        payment_id="pay_01",
        run_id="run_01",
        account_id="acc_01",
        decision_time=NOW_UTC,
        decided_at=NOW_UTC,
        arrival_cutoff=NOW_UTC,
        watermark_lag_ms=25,
        status="ok",
        feature_values=f_dict,
        missing_request_fields=["receiver_vpa"],
        registry_answers=[{"identifier_hash": "h1", "status": "match"}],
    )
    assert isinstance(row.feature_values, list)
    assert len(row.feature_values) == len(names)
    assert row.feature_values[0] == 3
    assert row.missing_request_fields == ["receiver_vpa"]
    assert len(row.registry_answers) == 1


def test_feature_set_version_and_sha256() -> None:
    """R5 & G3: Verifies FEATURE_SET_VERSION is 'm1' and feature_names_sha256 identifies the exact list."""
    assert FEATURE_SET_VERSION == "m1"

    names = canonical_feature_names()
    expected_sha256 = hashlib.sha256("\n".join(names).encode("utf-8")).hexdigest()
    assert canonical_feature_names_sha256() == expected_sha256

    row = FeatureLogRow(
        bank_id="bank_0",
        payment_id="pay_01",
        run_id="run_01",
        account_id="acc_01",
        decision_time=NOW_UTC,
        decided_at=NOW_UTC,
        arrival_cutoff=NOW_UTC,
        watermark_lag_ms=0,
        status="ok",
        feature_values=[],
    )
    assert row.feature_set_version == "m1"
    assert row.feature_names_sha256 == expected_sha256

    # Prove hash changes if one canonical name is renamed in a copied list
    modified_names = list(names)
    modified_names[0] = modified_names[0] + "_renamed"
    tampered_sha256 = hashlib.sha256("\n".join(modified_names).encode("utf-8")).hexdigest()
    assert tampered_sha256 != expected_sha256


def test_check_constraints_rejected_on_bad_row() -> None:
    """G2 & G3: Verifies that CHECK constraints reject bad rows."""
    # 1. arrival_cutoff IS NOT NULL OR status = 'no_watermark'
    with pytest.raises(ValueError, match="arrival_cutoff cannot be None unless status is 'no_watermark'"):
        FeatureLogRow(
            bank_id="bank_0",
            payment_id="pay_01",
            run_id="run_01",
            account_id="acc_01",
            decision_time=NOW_UTC,
            decided_at=NOW_UTC,
            arrival_cutoff=None,
            watermark_lag_ms=0,
            status="ok",
            feature_values=[],
        )

    # Allowed when status is no_watermark
    row_no_wm = FeatureLogRow(
        bank_id="bank_0",
        payment_id="pay_01",
        run_id="run_01",
        account_id="acc_01",
        decision_time=NOW_UTC,
        decided_at=NOW_UTC,
        arrival_cutoff=None,
        watermark_lag_ms=None,
        status="no_watermark",
        feature_values=[],
    )
    assert row_no_wm.arrival_cutoff is None

    # 2. Both error_type and error_message must be non-None when status='error'
    with pytest.raises(ValueError, match="Both error_type and error_message must be non-None"):
        FeatureLogRow(
            bank_id="bank_0",
            payment_id="pay_01",
            run_id="run_01",
            account_id="acc_01",
            decision_time=NOW_UTC,
            decided_at=NOW_UTC,
            arrival_cutoff=NOW_UTC,
            watermark_lag_ms=0,
            status="error",
            feature_values=[],
            error_type="ValueError",
            error_message=None,
        )

    with pytest.raises(ValueError, match="Both error_type and error_message must be non-None"):
        FeatureLogRow(
            bank_id="bank_0",
            payment_id="pay_01",
            run_id="run_01",
            account_id="acc_01",
            decision_time=NOW_UTC,
            decided_at=NOW_UTC,
            arrival_cutoff=NOW_UTC,
            watermark_lag_ms=0,
            status="error",
            feature_values=[],
            error_type=None,
            error_message="Something broke",
        )

    # 3. Both error_type and error_message must be None when status != 'error'
    with pytest.raises(ValueError, match="error_type and error_message must be None"):
        FeatureLogRow(
            bank_id="bank_0",
            payment_id="pay_01",
            run_id="run_01",
            account_id="acc_01",
            decision_time=NOW_UTC,
            decided_at=NOW_UTC,
            arrival_cutoff=NOW_UTC,
            watermark_lag_ms=0,
            status="ok",
            feature_values=[],
            error_type="ValueError",
            error_message="unexpected error",
        )

    # 4. missing_request_fields and registry_answers must be list, never None
    with pytest.raises(ValueError, match="missing_request_fields must be a list"):
        FeatureLogRow(
            bank_id="bank_0",
            payment_id="pay_01",
            run_id="run_01",
            account_id="acc_01",
            decision_time=NOW_UTC,
            decided_at=NOW_UTC,
            arrival_cutoff=NOW_UTC,
            watermark_lag_ms=0,
            status="ok",
            feature_values=[],
            missing_request_fields=None,  # type: ignore[arg-type]
        )

    with pytest.raises(ValueError, match="registry_answers must be a list"):
        FeatureLogRow(
            bank_id="bank_0",
            payment_id="pay_01",
            run_id="run_01",
            account_id="acc_01",
            decision_time=NOW_UTC,
            decided_at=NOW_UTC,
            arrival_cutoff=NOW_UTC,
            watermark_lag_ms=0,
            status="ok",
            feature_values=[],
            registry_answers=None,  # type: ignore[arg-type]
        )
