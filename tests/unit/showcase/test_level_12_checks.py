"""
WHAT: Unit tests verifying Level 12a showcase check runners, framework coverage compliance,
and precondition validation.

WHY: Guarantees that each Level 12a check runner executes reliably in isolation, satisfies framework
coverage rules (normal, edge, negative distribution), and produces valid evidence data structures.

TERM: Level 12: World engine II supporting reactive outside events and interactive fraud simulation.
TERM: check: An automated verification routine evaluating an individual simulation property against strict assertions.
TERM: coverage: The required distribution of checks across normal, edge, and negative categories.
TERM: part: One letter naming a slice of a level that is built and reviewed on its own, e.g. "a" for Level 12a.
"""

import asyncio

from core.config_validator import load_yaml
from showcase import level_12
from showcase.framework import validate_checks
from showcase.level_12_checks import (
    FRAUD_ON,
    FROZEN_ANCHORS_PATH,
    run_consistency_check,
    run_held_payment_isolation_check,
    run_lien_blocks_debit_check,
    run_split_world_check,
    run_v2_register_negative_check,
)
from showcase.level_12_support import (
    compute_opening_total,
    find_lien_candidate,
    load_level_12_config,
    verify_lien_blocked_debits,
)
from world.fast_mode import FastRunSettings
from world.interactive_runner import prepare_world, run_interactive_world
from world.outside_events import OutsideEvent


def test_level_12_checks_match_the_brief() -> None:
    """
    WHAT: Verifies that Level 12 exports 24 checks across parts 'a', 'b', 'c', 'd', and 'e' in quick and full modes.
    Confirms the checks are tagged five each for parts a to d and four for part e in order, part-a check kinds
    match the brief, validate_checks succeeds, and find_missing_preconditions passes.

    WHY: Enforces showcase framework contract requirements and guarantees that the level definition satisfies
    the mandatory coverage rules (at least 1 normal, 2 edge, and 1 negative check).

    Args:
        None.

    Returns:
        None. Fails if check counts, order, categories, or preconditions diverge.

    Example:
        test_level_12_checks_match_the_brief()  # Verifies 10 checks matching specification
    """
    # 1. Retrieve checks in quick and full modes and verify length and part sequences.
    quick_checks = level_12.get_checks(quick=True)
    full_checks = level_12.get_checks(quick=False)
    assert len(quick_checks) == 24
    assert len(full_checks) == 24
    assert [check.part for check in quick_checks] == ["a"] * 5 + ["b"] * 5 + ["c"] * 5 + ["d"] * 5 + ["e"] * 4

    # 2. Assert check kinds follow the exact brief sequence for Part A.
    part_a_checks = quick_checks[:5]
    expected_kinds = ["consistency", "normal", "edge", "edge", "negative"]
    assert [check.kind for check in part_a_checks] == expected_kinds

    # 3. Verify framework coverage rules pass and preconditions are met.
    validate_checks(quick_checks)
    missing = asyncio.run(level_12.find_missing_preconditions())
    assert not missing, f"Missing preconditions: {missing}"


def test_check_1_engine_consistency() -> None:
    """
    WHAT: Executes run_consistency_check and verifies that the returned CheckOutcome reports a matching
    SHA-256 digest covering all 2,915 events and validates golden fingerprints.

    WHY: Proves Check 1 confirms bit-for-bit repeatability of the baseline world engine against frozen anchors.

    Args:
        None.

    Returns:
        None. Fails if the consistency check outcome does not indicate successful validation.

    Example:
        test_check_1_engine_consistency()  # Asserts 2,915 events match SHA-256
    """
    # Run Check 1 asynchronously and assert outcome reports the golden hash and line count.
    outcome = asyncio.run(run_consistency_check(None))
    assert "SHA-256" in outcome.key_number
    assert "2,915 events" in outcome.key_number
    assert outcome.evidence["lines"] == 2915


def test_check_2_split_world_matches_level_11() -> None:
    """
    WHAT: Executes run_split_world_check and asserts that the split-world benchmark contains at least 3 genuine
    split payments, matches all 5 Level 11 outputs, and shares identical planned payment IDs across switch states.

    WHY: Proves Check 2 confirms requirement F-b: payment splitting over UPI limits preserves full parity.

    Args:
        None.

    Returns:
        None. Fails if split counts fall below threshold or outputs differ.

    Example:
        test_check_2_split_world_matches_level_11()  # Asserts split payments and 5/5 match
    """
    # Run Check 2 asynchronously and assert split counts and output parity.
    outcome = asyncio.run(run_split_world_check(None))
    assert "5/5 match" in outcome.key_number
    assert outcome.evidence["splits"] >= 3
    assert outcome.evidence["parity"] == "5/5"


def test_check_3_lien_blocks_debits_and_money_balances() -> None:
    """
    WHAT: Executes run_lien_blocks_debit_check and asserts that placing a lien blocks candidate and subsequent
    debits on the target account as funds_on_hold while double-entry money conservation rules hold.

    WHY: Proves Check 3 confirms that outside encumbrances prevent outgoing transfers without leaking money.

    Args:
        None.

    Returns:
        None. Fails if blocked count is zero or money rules fail.

    Example:
        test_check_3_lien_blocks_debits_and_money_balances()  # Asserts lien blocks debits
    """
    # Run Check 3 asynchronously and verify candidate and subsequent debits are blocked as funds_on_hold.
    outcome = asyncio.run(run_lien_blocks_debit_check(None))
    assert "debits from" in outcome.key_number
    assert "blocked after a Rs" in outcome.key_number
    assert outcome.evidence["blocked_debits"] >= 1


def test_check_3_fails_when_no_lien_is_placed() -> None:
    """
    WHAT: Verifies that verify_lien_blocked_debits raises an AssertionError when evaluated on a baseline simulation
    run executed without any outside lien encumbrance placed.

    WHY: Proves negative check sensitivity: the verification helper must fail if the expected lien failure
    is absent, ensuring that false positives cannot pass the check suite.

    Args:
        None.

    Returns:
        None. Fails if AssertionError is not raised.

    Example:
        test_check_3_fails_when_no_lien_is_placed()  # Asserts negative detection of absent lien
    """
    # 1. Prepare baseline simulation run without any outside lien encumbrance.
    config = load_level_12_config()
    code_version = config["showcase_run"]["code_version"]
    anchors = load_yaml(FROZEN_ANCHORS_PATH)["frozen_anchors"]["fraud_off_events"]
    settings = FastRunSettings(
        seed=anchors["seed"], n_people=anchors["n_people"], n_shops=anchors["n_shops"],
        n_days=anchors["n_days"], range_name=anchors["range_name"],
    )
    baseline_output = run_interactive_world(settings, FRAUD_ON, code_version=code_version)
    position, debit, balance_before = find_lien_candidate(baseline_output, config["check_3_lien"]["search_start"])
    lien = OutsideEvent(debit.minute, "lien", debit.from_account, balance_before, encumbrance_id="enc-000000")
    opening_total = compute_opening_total(prepare_world(settings, None).world.ledgers)

    # 2. Invoke verification helper with baseline output instead of lien output and assert failure.
    try:
        verify_lien_blocked_debits(baseline_output, baseline_output, position, debit, lien, opening_total)
    except AssertionError:
        pass
    else:
        raise AssertionError("Expected AssertionError was not raised when no lien was placed")


def test_check_4_held_payment_isolation() -> None:
    """
    WHAT: Executes run_held_payment_isolation_check and asserts that a compliance-held payment produces
    0 events, leaves 0 ledger entries, and preserves surrounding events intact.

    WHY: Proves Check 4 confirms transaction isolation during compliance holds without advancing random draws.

    Args:
        None.

    Returns:
        None. Fails if the held payment executes or alters surrounding events.

    Example:
        test_check_4_held_payment_isolation()  # Asserts 0 events for held payment
    """
    # Run Check 4 asynchronously and verify 0 events for the held payment and intact remaining events.
    outcome = asyncio.run(run_held_payment_isolation_check(None))
    assert "0 events and 0 ledger entries" in outcome.key_number
    assert outcome.evidence["held_entries"] == 0
    assert outcome.evidence["baseline_entries"] == 2
    assert outcome.evidence["remaining_events"] == 2914


def test_check_5_v2_register_negative() -> None:
    """
    WHAT: Executes run_v2_register_negative_check and asserts that an invalid issues_v2.yaml missing evidence
    is safely rejected while real configuration files and the v1 28/2/1 invariant remain intact.

    WHY: Proves Check 5 enforces schema validation on corrupt registers without touching repository files.

    Args:
        None.

    Returns:
        None. Fails if the invalid register is accepted or the real file is modified.

    Example:
        test_check_5_v2_register_negative()  # Asserts negative rejection and v1 28/2/1 invariant
    """
    # Run Check 5 asynchronously and verify negative rejection and invariant preservation.
    outcome = asyncio.run(run_v2_register_negative_check(None))
    assert "broken register rejected" in outcome.key_number
    assert outcome.evidence["real_hash_unchanged"] is True
    assert outcome.evidence["v1_status"] == "28/2/1"
    assert "K1: built needs at least one evidence item" in outcome.evidence["captured_message"]

