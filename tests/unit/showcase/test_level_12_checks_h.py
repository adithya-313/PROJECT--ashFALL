"""
WHAT: Unit tests for the Level 12h showcase checks (38 through 41): part h is exactly four checks in the brief's order
and kinds: edge, consistency, normal, edge. Runs the fast checks 38 and 40 in isolation.

WHY: Verifies the showcase framework contract, part-h registration order, category kinds, and fast execution
of the edge-case library and fraud-on world card base rate checks.

TERM: edge case: A specific boundary scenario with defined configuration overrides, gang tactics, and expectations.
TERM: fraud world card: A structured manifest capturing simulation parameters and fraud inflation metrics.
"""

import asyncio

from showcase import level_12
from showcase.framework import validate_checks
from showcase.level_12_checks_h import (
    run_edge_cases_check,
    run_fraud_world_card_check,
)


def test_part_h_checks_match_the_brief() -> None:
    """
    WHAT: Verifies that Level 12 exports 4 checks with part 'h', named 38 to 41 in order with kinds:
    edge, consistency, normal, edge, that validate_checks passes across all 41 checks.

    WHY: Enforces the showcase framework contract and verifies exact check taxonomy and naming required by the brief.

    Args:
        None.

    Returns:
        None. Fails if check counts, names, categories, or preconditions diverge.

    Example:
        test_part_h_checks_match_the_brief()
    """
    all_checks = level_12.get_checks(quick=True)
    validate_checks(all_checks)
    assert len(all_checks) == 41
    part_h = [check for check in all_checks if check.part == "h"]
    assert len(part_h) == 4
    expected_numbers = ["38", "39", "40", "41"]
    assert [check.name.split(".")[0] for check in part_h] == expected_numbers
    expected_kinds = ["edge", "consistency", "normal", "edge"]
    assert [check.kind for check in part_h] == expected_kinds
    assert asyncio.run(level_12.find_missing_preconditions()) == []


def test_check_38_fast_passes() -> None:
    """
    WHAT: Runs fast edge Check 38 asynchronously and verifies that the edge-case library evaluates all 13 cases,
    with 10 runnable passing, 2 later level, and 1 deferred.

    WHY: Proves Check 38 executes successfully and asserts expected boundary behavior for all edge-case scenarios.

    Args:
        None.

    Returns:
        None. Fails if Check 38 outcome fails or reports unexpected case counts.

    Example:
        test_check_38_fast_passes()
    """
    outcome = asyncio.run(run_edge_cases_check(None))
    assert "10 runnable PASS" in outcome.key_number
    assert "2 later level" in outcome.key_number
    assert "1 deferred" in outcome.key_number
    cases = outcome.evidence["cases"]
    assert len(cases) == 13
    runnable = [c for c in cases if c["status"] == "runnable"]
    assert len(runnable) == 10
    for c in runnable:
        assert "PASS" in c["result"]


def test_check_40_fast_passes() -> None:
    """
    WHAT: Runs fast normal Check 40 asynchronously and verifies that build_fraud_world_card records a valid
    base rate with a positive fraud inflation factor.

    WHY: Proves Check 40 executes successfully and confirms that fraud-on world cards report inflation factors.

    Args:
        None.

    Returns:
        None. Fails if Check 40 outcome fails or inflation factor is not strictly greater than 1.0.

    Example:
        test_check_40_fast_passes()
    """
    outcome = asyncio.run(run_fraud_world_card_check(None))
    assert "factor" in outcome.key_number
    assert "(lower bound)" in outcome.key_number
    factor = float(outcome.evidence["factor"])
    assert factor > 1.0
    assert outcome.evidence["complaints"] > 0
    assert outcome.evidence["real_world_ratio"] == 1.05
