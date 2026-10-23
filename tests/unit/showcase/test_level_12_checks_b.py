"""
WHAT: Unit tests verifying Level 12b showcase check runners, framework coverage compliance,
and domain assertion mechanics across Checks 6 through 10.

WHY: Guarantees that each Level 12b check runner executes reliably in isolation, satisfies framework
coverage rules (normal, negative, negative, consistency, edge), and produces valid evidence data structures.

TERM: Level 12: World engine II supporting reactive outside events and interactive fraud simulation.
TERM: check: An automated verification routine evaluating an individual simulation property against strict assertions.
TERM: coverage: The required distribution of checks across normal, edge, and negative categories.
TERM: part: One letter naming a slice of a level that is built and reviewed on its own, e.g. "b" for Level 12b.
"""

import asyncio

from showcase import level_12
from showcase.framework import validate_checks
from showcase.level_12_checks_b import (
    run_gang_consistency_check,
    run_gang_two_layers_check,
    run_over_budget_gang_negative_check,
    run_temporal_truth_edge_check,
    run_truth_isolation_negative_check,
)


def test_level_12_part_b_checks_match_the_brief() -> None:
    """
    WHAT: Verifies that Level 12 exports 5 checks with part 'b' in exact brief order:
    normal, negative, negative, consistency, edge. Asserts that validate_checks passes across
    all 10 checks of the complete level, and that find_missing_preconditions finds no errors.

    WHY: Enforces showcase framework contract requirements and ensures the complete Level 12
    suite satisfies the mandatory coverage rules (at least 1 normal, 2 edge, and 1 negative check).

    Args:
        None.

    Returns:
        None. Fails if check counts, order, categories, or preconditions diverge.

    Example:
        test_level_12_part_b_checks_match_the_brief()
    """
    # 1. Retrieve all checks from level_12 and filter for Part B.
    all_checks = level_12.get_checks(quick=True)
    assert len(all_checks) == 24
    part_b_checks = [check for check in all_checks if check.part == "b"]
    assert len(part_b_checks) == 5

    # 2. Assert check kinds follow the exact brief sequence: normal, negative, negative, consistency, edge.
    expected_kinds = ["normal", "negative", "negative", "consistency", "edge"]
    assert [check.kind for check in part_b_checks] == expected_kinds

    # 3. Assert check names follow the exact brief numbering and titles.
    expected_names = [
        "6. a gang moves fixture money through two layers",
        "7. ashFALL code cannot open the truth",
        "8. an over-budget gang is refused with a plain reason",
        "9. the same seed gives the same gang activity and truth",
        "10. truth is temporal: a rented account is genuine before its rental",
    ]
    assert [check.name for check in part_b_checks] == expected_names

    # 4. Verify framework coverage rules pass across all ten checks and preconditions are met.
    validate_checks(all_checks)
    missing = asyncio.run(level_12.find_missing_preconditions())
    assert not missing, f"Missing preconditions: {missing}"


def test_check_6_gang_moves_money_through_two_layers() -> None:
    """
    WHAT: Executes run_gang_two_layers_check and asserts that the returned CheckOutcome reports
    4 mules at depths 1-2, 4 route payments with stories, and PASS-level evidence structure.

    WHY: Proves Check 6 confirms that gang routing executes correctly through two layers of mules
    while attaching forensic stories and conserving total funds.

    Args:
        None.

    Returns:
        None. Fails if check outcome does not indicate successful validation.

    Example:
        test_check_6_gang_moves_money_through_two_layers()
    """
    # 1. Run Check 6 asynchronously and assert outcome reports mule depth distribution and route payments.
    outcome = asyncio.run(run_gang_two_layers_check(None))
    assert "4 mules at depths 1-2" in outcome.key_number
    assert "route payments, all with stories" in outcome.key_number
    assert outcome.evidence["mule_count"] == 4
    assert outcome.evidence["route_payments"] == 4
    assert len(outcome.evidence["roles_sample"]) > 0
    assert len(outcome.evidence["stories_sample"]) > 0


def test_check_7_truth_isolation_negative() -> None:
    """
    WHAT: Executes run_truth_isolation_negative_check and asserts that runtime caller inspection
    raises PermissionError, static AST scan catches unauthorized import, and real repo is clean.

    WHY: Proves Check 7 enforces multi-layered truth isolation protecting banking models and APIs.

    Args:
        None.

    Returns:
        None. Fails if runtime caller guard or static check fails to report expected isolation status.

    Example:
        test_check_7_truth_isolation_negative()
    """
    # 1. Run Check 7 asynchronously and verify triple-layer isolation outcome.
    outcome = asyncio.run(run_truth_isolation_negative_check(None))
    assert "runtime PermissionError from api.caller" in outcome.key_number
    assert "static scan caught 1/1 in api" in outcome.key_number
    assert "real repo 0 violations" in outcome.key_number
    assert "Operational package 'api' is forbidden" in outcome.evidence["runtime_message"]
    assert outcome.evidence["real_violations_count"] == 0


def test_check_8_over_budget_gang_negative() -> None:
    """
    WHAT: Executes run_over_budget_gang_negative_check and asserts that an over-budget gang
    is refused with a diagnostic sentence naming money, planned sum, and block, with 0 truth files created.

    WHY: Proves Check 8 enforces budget limits and prevents premature world execution.

    Args:
        None.

    Returns:
        None. Fails if over-budget gang is not refused or truth files are generated.

    Example:
        test_check_8_over_budget_gang_negative()
    """
    # 1. Run Check 8 asynchronously and assert budget refusal message and empty truth folder.
    outcome = asyncio.run(run_over_budget_gang_negative_check(None))
    assert "refused:" in outcome.key_number
    assert "money" in outcome.evidence["exception_message"]
    assert "route" in outcome.evidence["exception_message"]
    assert outcome.evidence["truth_folder_empty"] is True


def test_check_9_gang_consistency() -> None:
    """
    WHAT: Executes run_gang_consistency_check and asserts that two runs from the same seed
    produce identical events-v2, 4 truth files, device events, and merged stream (7/7 matches).

    WHY: Proves Check 9 verifies bit-for-bit deterministic simulation repeatability.

    Args:
        None.

    Returns:
        None. Fails if any comparison diverges between repeated executions.

    Example:
        test_check_9_gang_consistency()
    """
    # 1. Run Check 9 asynchronously and assert 7/7 comparisons match identically.
    outcome = asyncio.run(run_gang_consistency_check(None))
    assert "7/7 matches" in outcome.key_number
    assert outcome.evidence["matches"] == 7
    assert outcome.evidence["total"] == 7
    assert outcome.evidence["parity_ratio"] == "7/7"


def test_check_10_temporal_truth_edge() -> None:
    """
    WHAT: Executes run_temporal_truth_edge_check and asserts that a leased citizen mule account
    is genuine before recruitment, becomes a mule at valid_from, and is controlled by its gang device.

    WHY: Proves Check 10 verifies temporal role boundaries and device takeover telemetry flags.

    Args:
        None.

    Returns:
        None. Fails if temporal transition or device telemetry fails validation.

    Example:
        test_check_10_temporal_truth_edge()
    """
    # 1. Run Check 10 asynchronously and assert temporal transition and device telemetry.
    outcome = asyncio.run(run_temporal_truth_edge_check(None))
    assert "genuine at min" in outcome.key_number
    assert "mule at min" in outcome.key_number
    assert "sim_swap=True, remote_access=True" in outcome.key_number
    assert outcome.evidence["sim_swap"] is True
    assert outcome.evidence["remote_access"] is True
