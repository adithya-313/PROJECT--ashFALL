"""
WHAT: Unit tests for scripts/derive_decision_thresholds.py.
Tests calculation of hold/warn quantiles, empirical shares, enforcement of the
sanity gate, and refusal when warn >= hold or row count/threshold mismatch.

WHY: Level 19 decision policy (A-074) requires strictly verified gates and threshold
derivation logic, ensuring erroneous data or inverted thresholds refuse immediately.
"""

from __future__ import annotations

import pytest

from scripts.derive_decision_thresholds import (
    compute_decision_quantiles,
    compute_empirical_shares,
    validate_sanity_gate,
)


def test_synthetic_score_list_known_quantiles() -> None:
    """
    PURPOSE:
        Verifies that compute_decision_quantiles produces expected linear quantiles
        on a known synthetic score distribution.
    """
    # 100 uniformly spaced values from 0.01 to 1.00
    scores = [i / 100.0 for i in range(1, 101)]
    # Upper 0.05 (1 - 0.05 = 0.95 quantile) and Upper 0.20 (1 - 0.20 = 0.80 quantile)
    hold_val, warn_val = compute_decision_quantiles(scores, hold_share=0.05, warn_share=0.20)

    # 0.95 quantile of 1..100 with linear method is 0.9505
    assert hold_val > warn_val
    assert pytest.approx(hold_val, abs=1e-4) == 0.9505
    assert pytest.approx(warn_val, abs=1e-4) == 0.8020

    stats = compute_empirical_shares(scores, hold_val, warn_val)
    assert stats["total_rows"] == 100
    assert stats["held_count"] == 5
    assert stats["hold_share"] == 0.05
    assert stats["warned_count"] == 20
    assert stats["warn_share"] == 0.20


def test_warn_greater_or_equal_hold_refuses() -> None:
    """
    PURPOSE:
        Proves that when warn_share <= hold_share (producing warn threshold >= hold threshold),
        compute_decision_quantiles raises ValueError.
    """
    scores = [0.1, 0.2, 0.3, 0.4, 0.5]
    # warn_share 0.01 is smaller than hold_share 0.05 -> warn_threshold >= hold_threshold
    with pytest.raises(ValueError, match="Warn threshold .* must be strictly less than hold threshold"):
        compute_decision_quantiles(scores, hold_share=0.05, warn_share=0.01)

    # Equal shares
    with pytest.raises(ValueError, match="Warn threshold .* must be strictly less than hold threshold"):
        compute_decision_quantiles(scores, hold_share=0.05, warn_share=0.05)


def test_sanity_gate_row_count_mismatch_refuses() -> None:
    """
    PURPOSE:
        Proves that validate_sanity_gate raises ValueError if row count is not exactly 1,834,135.
    """
    with pytest.raises(ValueError, match="SANITY GATE FAILED: validation row count mismatch"):
        validate_sanity_gate(
            row_count=1834134,
            recomputed_hold=0.1304200652241741,
            expected_rows=1834135,
            expected_hold=0.1304200652241741,
        )


def test_sanity_gate_threshold_mismatch_refuses() -> None:
    """
    PURPOSE:
        Proves that validate_sanity_gate raises ValueError if recomputed hold differs
        from the expected frozen threshold by more than tolerance.
    """
    with pytest.raises(ValueError, match="SANITY GATE FAILED: hold threshold mismatch"):
        validate_sanity_gate(
            row_count=1834135,
            recomputed_hold=0.13042006523,  # diff > 1e-12
            expected_rows=1834135,
            expected_hold=0.1304200652241741,
            tolerance=1e-12,
        )


def test_sanity_gate_exact_match_passes() -> None:
    """
    PURPOSE:
        Proves that validate_sanity_gate succeeds without error when rows and threshold match.
    """
    validate_sanity_gate(
        row_count=1834135,
        recomputed_hold=0.1304200652241741,
        expected_rows=1834135,
        expected_hold=0.1304200652241741,
        tolerance=1e-12,
    )
