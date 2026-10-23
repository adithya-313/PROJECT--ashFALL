"""
WHAT: Unit tests verifying Level 12c showcase check runners, framework coverage compliance,
and domain assertion mechanics across Checks 11 through 15.

WHY: Guarantees that each Level 12c check runner executes reliably in isolation, satisfies framework
coverage rules (normal, edge, edge, edge, negative), and produces valid evidence data structures.

TERM: Level 12: World engine II supporting reactive outside events and interactive fraud simulation.
TERM: check: An automated verification routine evaluating an individual simulation property against strict assertions.
TERM: coverage: The required distribution of checks across normal, edge, and negative categories.
TERM: part: One letter naming a slice of a level that is built and reviewed on its own, e.g. "c" for Level 12c.
"""

import asyncio

from showcase import level_12
from showcase.framework import validate_checks
from showcase.level_12_checks_c import (
    run_genuine_shop_downstream_check,
    run_mule_lifecycle_check,
    run_rented_mule_normal_life_check,
    run_victim_payment_outage_check,
    run_victim_to_cashout_check,
)


def test_level_12_part_c_checks_match_the_brief() -> None:
    """
    WHAT: Verifies that Level 12 exports 5 checks with part 'c' in exact brief order:
    normal, edge, edge, edge, negative. Asserts that validate_checks passes across
    all 20 checks of the complete level, and that find_missing_preconditions finds no errors.

    WHY: Enforces showcase framework contract requirements and ensures the complete Level 12
    suite satisfies the mandatory coverage rules (at least 1 normal, 2 edge, and 1 negative check).

    Args:
        None.

    Returns:
        None. Fails if check counts, order, categories, or preconditions diverge.

    Example:
        test_level_12_part_c_checks_match_the_brief()
    """
    # 1. Retrieve all checks from level_12 and filter for Part C.
    all_checks = level_12.get_checks(quick=True)
    assert len(all_checks) == 24
    part_c_checks = [check for check in all_checks if check.part == "c"]
    assert len(part_c_checks) == 5

    # 2. Assert check kinds follow the exact brief sequence: normal, edge, edge, edge, negative.
    expected_kinds = ["normal", "edge", "edge", "edge", "negative"]
    assert [check.kind for check in part_c_checks] == expected_kinds

    # 3. Assert check names follow the exact brief numbering and titles.
    expected_names = [
        "11. victim to mules to cash-out for one gang",
        "12. a rented mule keeps its normal life",
        "13. one mule from warm-up to burned",
        "14. a genuine shop receives mule money and stays genuine",
        "15. a victim payment fails in an outage and nothing is forwarded",
    ]
    assert [check.name for check in part_c_checks] == expected_names

    # 4. Verify framework coverage rules pass across all fifteen checks and preconditions are met.
    validate_checks(all_checks)
    missing = asyncio.run(level_12.find_missing_preconditions())
    assert not missing, f"Missing preconditions: {missing}"


def test_check_11_victim_to_cashout() -> None:
    """
    WHAT: Executes run_victim_to_cashout_check and verifies that the CheckOutcome reports
    victim fraud payments, intermediate forwards, diverse cash-out channels, and extracted sum.
    WHY: Proves Check 11 confirms end-to-end victim-to-cash-out money flow for a scam gang.
    Args:
        None.
    Returns:
        None. Fails if the check runner raises or evidence is incomplete.
    Example:
        test_check_11_victim_to_cashout()
    """
    outcome = asyncio.run(run_victim_to_cashout_check(None))
    assert "victim payments ->" in outcome.key_number
    assert "forwards ->" in outcome.key_number
    assert "cash-outs" in outcome.key_number
    assert outcome.evidence["victim_payments"] > 0
    assert outcome.evidence["forwards"] > 0
    assert outcome.evidence["cashouts"] > 0


def test_check_12_rented_mule_normal_life() -> None:
    """
    WHAT: Executes run_rented_mule_normal_life_check and asserts that the CheckOutcome reports
    genuine payments executed by a rented mule alongside paused owner transactions.
    WHY: Proves Check 12 confirms co-existence of authentic routine with paused outgoing payments.
    Args:
        None.
    Returns:
        None. Fails if the check runner raises or expected activity is absent.
    Example:
        test_check_12_rented_mule_normal_life()
    """
    outcome = asyncio.run(run_rented_mule_normal_life_check(None))
    assert "genuine payments made" in outcome.key_number
    assert "paused" in outcome.key_number
    assert outcome.evidence["payments_made_during_rental"] > 0
    assert outcome.evidence["payments_paused_during_rental"] > 0


def test_check_13_mule_lifecycle() -> None:
    """
    WHAT: Executes run_mule_lifecycle_check and asserts that the CheckOutcome reports
    phase_at transitions through warm_up, active, and burned across boundary minutes.
    WHY: Proves Check 13 confirms lifecycle stage intervals conform strictly to temporal specifications.
    Args:
        None.
    Returns:
        None. Fails if the check runner raises or phase states diverge.
    Example:
        test_check_13_mule_lifecycle()
    """
    outcome = asyncio.run(run_mule_lifecycle_check(None))
    assert "warm_up" in outcome.key_number
    assert "active" in outcome.key_number
    assert "burned" in outcome.key_number
    assert outcome.evidence["phases"] == ["warm_up", "active", "burned"]


def test_check_14_genuine_shop_downstream() -> None:
    """
    WHAT: Executes run_genuine_shop_downstream_check and asserts that the CheckOutcome reports
    a genuine merchant receiving mule funds while maintaining role_at = genuine in truth.
    WHY: Proves Check 14 confirms downstream cash-out merchant payments preserve legal innocence.
    Args:
        None.
    Returns:
        None. Fails if the check runner raises or the shop is not genuine.
    Example:
        test_check_14_genuine_shop_downstream()
    """
    outcome = asyncio.run(run_genuine_shop_downstream_check(None))
    assert "role_at = genuine" in outcome.key_number
    assert outcome.evidence["role_at"] == "genuine"
    assert outcome.evidence["amount_paise"] > 0


def test_check_15_victim_payment_outage() -> None:
    """
    WHAT: Executes run_victim_payment_outage_check and asserts that the CheckOutcome reports
    a victim payment declining due to bank downtime and subsequent forward skipping.
    WHY: Proves Check 15 confirms external outage failure handling and downstream skipping.
    Args:
        None.
    Returns:
        None. Fails if the check runner raises or expected failure reasons diverge.
    Example:
        test_check_15_victim_payment_outage()
    """
    outcome = asyncio.run(run_victim_payment_outage_check(None))
    assert "bank_downtime" in outcome.key_number
    assert "no_illicit_money_arrived" in outcome.key_number
    assert outcome.evidence["failed_reason"] == "bank_downtime"
    assert outcome.evidence["skipped_reason"] == "no_illicit_money_arrived"
