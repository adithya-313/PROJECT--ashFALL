"""
WHAT: Unit tests for the Level 12g showcase checks (31 through 37): part g is exactly seven checks in the brief's order
and kinds: normal, edge, edge, edge, negative, negative, consistency. Runs the fast checks 35, 36, and 37 in isolation.

WHY: Guarantees that part g check definitions and their fast verification routines execute correctly in unit test
suites before full showcase execution, verifying training refusal, validator checks, and taxonomic distribution.

TERM: part: One letter naming a slice of a level that is built and reviewed on its own, e.g. "g" for Level 12g.
TERM: check: An automated verification routine evaluating an individual simulation property against strict assertions.
"""

import asyncio

from showcase import level_12
from showcase.framework import validate_checks
from showcase.level_12_checks_g import (
    run_held_out_property_distribution_check,
    run_held_out_validator_check,
    run_training_held_out_refusal_check,
)


def test_part_g_checks_match_the_brief() -> None:
    """
    WHAT: Verifies that Level 12 exports 7 checks with part 'g', named 31 to 37 in order with kinds:
    normal, edge, edge, edge, negative, negative, consistency, that validate_checks passes across all 37 checks.

    WHY: Enforces the showcase framework contract and verifies exact check taxonomy and naming required by the brief.

    Args:
        None.

    Returns:
        None. Fails if check counts, names, categories, or preconditions diverge.

    Example:
        test_part_g_checks_match_the_brief()
    """
    all_checks = level_12.get_checks(quick=True)
    validate_checks(all_checks)
    assert len(all_checks) == 37
    part_g = [check for check in all_checks if check.part == "g"]
    assert len(part_g) == 7
    expected_numbers = ["31", "32", "33", "34", "35", "36", "37"]
    assert [check.name.split(".")[0] for check in part_g] == expected_numbers
    expected_kinds = ["normal", "edge", "edge", "edge", "negative", "negative", "consistency"]
    assert [check.kind for check in part_g] == expected_kinds
    assert asyncio.run(level_12.find_missing_preconditions()) == []


def test_check_35_fast_passes() -> None:
    """
    WHAT: Runs fast negative Check 35 asynchronously and verifies that an interactive training world given held-out
    options raises ValueError with a plain refusal message citing option, family, and split.

    WHY: Proves Check 35 executes successfully and validates held-out option isolation in training splits.

    Args:
        None.

    Returns:
        None. Fails if Check 35 does not report the expected ValueError refusal details.

    Example:
        test_check_35_fast_passes()
    """
    outcome = asyncio.run(run_training_held_out_refusal_check(None))
    assert "drip_piece_max_inr" in outcome.evidence["message"]
    assert "slow_drip" in outcome.evidence["message"]
    assert "training" in outcome.evidence["message"]
    assert outcome.evidence["error_type"] == "ValueError"


def test_check_36_fast_passes() -> None:
    """
    WHAT: Runs fast negative Check 36 asynchronously and verifies that check_held_out_mechanisms rejects an active
    held-out family lacking an implementing block option.

    WHY: Proves Check 36 executes successfully and enforces evaluation configuration mechanism integrity.

    Args:
        None.

    Returns:
        None. Fails if Check 36 does not trigger the expected validator refusal.

    Example:
        test_check_36_fast_passes()
    """
    outcome = asyncio.run(run_held_out_validator_check(None))
    assert "has no implementing block option" in outcome.key_number
    assert outcome.evidence["error_type"] == "ValueError"


def test_check_37_fast_passes() -> None:
    """
    WHAT: Runs consistency Check 37 asynchronously and verifies taxonomic gang distribution across all 680 worlds.
    Confirms 440 training/validation worlds have 0 held-out gangs and 240 test/stress worlds have exactly 1.

    WHY: Proves Check 37 verifies Latin hypercube gang allocation consistency across all planned evaluation splits.

    Args:
        None.

    Returns:
        None. Fails if Check 37 distribution counts diverge from the brief specification.

    Example:
        test_check_37_fast_passes()
    """
    outcome = asyncio.run(run_held_out_property_distribution_check(None))
    expected_key = (
        "N=440 training/validation: 0 held-out gangs; "
        "N=240 test/stress: 240 with one (device_rotation 129, slow_drip 111)"
    )
    assert outcome.key_number == expected_key
    assert outcome.evidence["training_validation"] == 440
    assert outcome.evidence["tv_held_out"] == 0
    assert outcome.evidence["test_stress"] == 240
    assert outcome.evidence["ts_held_out"] == 240
