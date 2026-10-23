"""
WHAT: Unit tests verifying Level 12d showcase check runners, framework coverage compliance,
and domain assertion mechanics across Checks 16 through 20.

WHY: Guarantees that each Level 12d check runner executes reliably in isolation, satisfies framework
coverage rules (edge, edge, normal, edge, negative), and produces valid evidence data structures.

TERM: Level 12: World engine II supporting reactive outside events and interactive fraud simulation.
TERM: check: An automated verification routine evaluating an individual simulation property against strict assertions.
TERM: coverage: The required distribution of checks across normal, edge, and negative categories.
TERM: part: One letter naming a slice of a level that is built and reviewed on its own, e.g. "d" for Level 12d.
"""

import asyncio

from showcase import level_12
from showcase.framework import validate_checks
from showcase.level_12_checks_d import (
    run_account_market_check,
    run_counts_by_family_check,
    run_held_out_in_training_refusal_check,
    run_late_lien_observation_check,
    run_training_world_families_check,
)


def test_level_12_part_d_checks_match_the_brief() -> None:
    """
    WHAT: Verifies that Level 12 exports 5 checks with part 'd' in exact brief order:
    edge, edge, normal, edge, negative. Asserts that validate_checks passes across all 20 checks
    of the complete level, and that find_missing_preconditions finds no errors.

    WHY: Enforces showcase framework contract requirements and ensures the complete Level 12
    suite satisfies the mandatory coverage rules.

    Args:
        None.

    Returns:
        None. Fails if check counts, order, categories, or preconditions diverge.

    Example:
        test_level_12_part_d_checks_match_the_brief()
    """
    all_checks = level_12.get_checks(quick=True)
    assert len(all_checks) == 20
    part_d_checks = [check for check in all_checks if check.part == "d"]
    assert len(part_d_checks) == 5
    assert [check.kind for check in part_d_checks] == ["edge", "edge", "normal", "edge", "negative"]
    assert [check.name for check in part_d_checks] == [
        "16. two gangs share accounts bought from the market",
        "17. a training world contains no held-out family",
        "18. a gang world with counts by family",
        "19. a lien is observed late; money flows until observed, then stops",
        "20. a training world file that names a held-out family is refused",
    ]

    validate_checks(all_checks)
    missing = asyncio.run(level_12.find_missing_preconditions())
    assert not missing, f"Missing preconditions: {missing}"


def test_check_16_account_market() -> None:
    """
    WHAT: Executes run_account_market_check and asserts the outcome names the sold account, the seller,
    the buyer and the resale minute, with the truth owner changing exactly at that minute.
    WHY: Proves Check 16 reports a real resale between two gangs.
    Args:
        None.
    Returns:
        None. Fails if the runner raises or the evidence is incomplete.
    Example:
        test_check_16_account_market()
    """
    outcome = asyncio.run(run_account_market_check(None))
    assert "accounts resold" in outcome.key_number
    assert outcome.evidence["resold_accounts"] >= 1
    assert outcome.evidence["owner_before"] == outcome.evidence["seller"]
    assert outcome.evidence["owner_at_resale"] == outcome.evidence["buyer"]
    assert outcome.evidence["seller"] != outcome.evidence["buyer"]


def test_check_17_training_world_has_no_held_out_family() -> None:
    """
    WHAT: Executes run_training_world_families_check and asserts the families used and the held-out list
    are disjoint, with at least two different families in the world.
    WHY: Proves Check 17 shows a training world free of held-out families.
    Args:
        None.
    Returns:
        None. Fails if the runner raises or the lists overlap.
    Example:
        test_check_17_training_world_has_no_held_out_family()
    """
    outcome = asyncio.run(run_training_world_families_check(None))
    used = set(outcome.evidence["gang_families"])
    held_out = set(outcome.evidence["held_out_families"])
    assert len(used) >= 2
    assert held_out
    assert not used & held_out
    assert "none is held out" in outcome.key_number


def test_check_18_counts_by_family() -> None:
    """
    WHAT: Executes run_counts_by_family_check and asserts every family has fraud payments and gang accounts.
    WHY: Proves Check 18 summarises a gang world by tactic family with real activity in each.
    Args:
        None.
    Returns:
        None. Fails if the runner raises or a family count is zero.
    Example:
        test_check_18_counts_by_family()
    """
    outcome = asyncio.run(run_counts_by_family_check(None))
    assert len(outcome.evidence) >= 2
    for counts in outcome.evidence.values():
        assert counts["payments"] > 0
        assert counts["accounts"] > 0
    assert "payments" in outcome.key_number


def test_check_19_late_lien_observation() -> None:
    """
    WHAT: Executes run_late_lien_observation_check and asserts fraud payments flow through the mule in
    [L, L+D) and none at or after L+D.
    WHY: Proves Check 19 shows a gang acting on a lien only after the feed delay.
    Args:
        None.
    Returns:
        None. Fails if the runner raises or the counts are wrong.
    Example:
        test_check_19_late_lien_observation()
    """
    outcome = asyncio.run(run_late_lien_observation_check(None))
    evidence = outcome.evidence
    assert evidence["seen_minute"] == evidence["lien_minute"] + evidence["delay"]
    assert evidence["delay"] > 0
    assert evidence["payments_before_seen"] > 0
    assert evidence["payments_after_seen"] == 0
    assert f"seen at {evidence['seen_minute']}" in outcome.key_number


def test_check_20_held_out_family_refused() -> None:
    """
    WHAT: Executes run_held_out_in_training_refusal_check and asserts the plain refusal text, naming the gang and
    its held-out family, and that no payment or state was processed before it.
    WHY: Proves Check 20 shows the refusal happens before planning.
    Args:
        None.
    Returns:
        None. Fails if the runner raises or the refusal is missing.
    Example:
        test_check_20_held_out_family_refused()
    """
    outcome = asyncio.run(run_held_out_in_training_refusal_check(None))
    assert outcome.key_number.startswith("refused before planning, 0 events")
    assert outcome.evidence["calls"] == 0
    assert outcome.evidence["family"] in outcome.evidence["refusal"]
    assert "allowed only in test and stress worlds" in outcome.evidence["refusal"]
