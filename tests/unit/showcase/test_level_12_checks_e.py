"""
WHAT: Unit tests verifying Level 12e showcase check runners, framework coverage compliance, and the domain
assertions of Checks 21 through 24 (complaints against the truth, complaint delay by scam type, analyst accuracy
against queue load, and one late bulk-label batch).

WHY: Guarantees each Level 12e check runner executes reliably in isolation, that part e is exactly four checks in the
brief's order, that check 21's complaints appear without forcing any share, and that every chart uses only the
timeline form the report page can draw.

TERM: Level 12: World engine II supporting reactive outside events and interactive fraud simulation.
TERM: check: An automated verification routine evaluating an individual simulation property against strict assertions.
TERM: part: One letter naming a slice of a level that is built and reviewed on its own, e.g. "e" for Level 12e.
TERM: timeline chart: the one chart form showcase/reporting.py draws, with points and optional shaded bands.
"""

import asyncio
from typing import Any

from showcase import level_12
from showcase.framework import validate_checks
from showcase.level_12_checks_e import (
    run_analyst_queue_check,
    run_bulk_batch_check,
    run_complaint_delay_by_type_check,
    run_complaints_against_truth_check,
)
from showcase.level_12_support_e import get_complaints_run
from world.config_reader import read_world_ranges

VICTIM_SHARE_NAME = "victim_complaint_share"


def assert_timeline_chart(chart: dict[str, Any] | None) -> None:
    """Asserts a chart is the timeline form with a title, axis labels and at least two points."""
    assert chart is not None
    assert chart["type"] == "timeline"
    assert chart["title"] and chart["x_label"] and chart["y_label"]
    assert len(chart["points"]) >= 2
    for point in chart["points"]:
        assert len(point) == 2
    for band in chart["shaded"]:
        assert len(band) == 3


def test_level_12_part_e_checks_match_the_brief() -> None:
    """
    WHAT: Verifies that Level 12 exports 4 checks with part 'e', named 21 to 24 in order with kinds normal, edge,
    edge, edge, that validate_checks passes across all 24 checks, and that no precondition is missing.
    WHY: Enforces the showcase framework contract and the order the brief gives for part e.
    Args:
        None.
    Returns:
        None. Fails if the count, names, kinds or preconditions diverge.
    Example:
        test_level_12_part_e_checks_match_the_brief()
    """
    all_checks = level_12.get_checks(quick=True)
    assert len(all_checks) == 24
    part_e_checks = [check for check in all_checks if check.part == "e"]
    assert len(part_e_checks) == 4
    assert [check.kind for check in part_e_checks] == ["normal", "edge", "edge", "edge"]
    assert [check.name for check in part_e_checks] == [
        "21. complaints arriving against the truth",
        "22. complaint delay by scam type",
        "23. analyst accuracy falls as the queue grows",
        "24. a bulk-label batch arrives late for one region",
    ]
    assert "12a-12e" in level_12.LEVEL_TITLE

    validate_checks(all_checks)
    missing = asyncio.run(level_12.find_missing_preconditions())
    assert not missing, f"Missing preconditions: {missing}"


def test_check_21_complaints_against_truth_without_forcing_the_share() -> None:
    """
    WHAT: Executes the check 21 runner and asserts at least one complaint appears with the world's own drawn share,
    which is the unforced range of the configuration, and that the small sample is reported as NOT RUN.
    WHY: Proves Check 21 does not depend on a forced complaint share and never skips its share comparison silently.
    Args:
        None.
    Returns:
        None. Fails if the runner raises, no complaint appears or the ranges were altered.
    Example:
        test_check_21_complaints_against_truth_without_forcing_the_share()
    """
    outcome = asyncio.run(run_complaints_against_truth_check(None))
    output, _labels, drawn = get_complaints_run()
    run_range = output.effective_config["world_configs"]["world_ranges"]["parameters"][VICTIM_SHARE_NAME]
    disk_range = read_world_ranges()["parameters"][VICTIM_SHARE_NAME]
    assert run_range == disk_range
    low, high = disk_range["training"]
    assert low <= drawn.victim_complaint_share <= high
    assert outcome.evidence["complained"] >= 1
    assert "victims complained" in outcome.key_number
    assert outcome.evidence["share_check"].startswith("NOT RUN: ")
    assert "minimum sample" in outcome.evidence["share_check"]
    assert_timeline_chart(outcome.chart)


def test_check_22_complaint_delay_by_scam_type() -> None:
    """
    WHAT: Executes the check 22 runner and asserts the evidence says the inputs are synthetic, investment delays start
    after unauthorised-debit delays end, and the chart has one band per scam type labelled with its name.
    WHY: Proves Check 22 shows the delay ordering from real planner output on clearly labelled synthetic losses.
    Args:
        None.
    Returns:
        None. Fails if the runner raises or the ordering, label or chart is wrong.
    Example:
        test_check_22_complaint_delay_by_scam_type()
    """
    outcome = asyncio.run(run_complaint_delay_by_type_check(None))
    assert outcome.evidence["inputs"].startswith("synthetic")
    assert outcome.evidence["investment_min_minutes"] > outcome.evidence["debit_max_minutes"]
    assert set(outcome.evidence["median_minutes"]) == {"investment", "job", "unauthorised_debit"}
    assert_timeline_chart(outcome.chart)
    assert sorted(label for _start, _end, label in outcome.chart["shaded"]) == [
        "investment", "job", "unauthorised_debit",
    ]


def test_check_23_analyst_accuracy_falls_as_the_queue_grows() -> None:
    """
    WHAT: Executes the check 23 runner and asserts expected accuracy never rises with load, is lower at load 6 than at
    load 1, and the chart has one point per load.
    WHY: Proves Check 23 shows the analyst getting worse as the queue outgrows the team.
    Args:
        None.
    Returns:
        None. Fails if the runner raises or the curve rises.
    Example:
        test_check_23_analyst_accuracy_falls_as_the_queue_grows()
    """
    outcome = asyncio.run(run_analyst_queue_check(None))
    expected = outcome.evidence["expected"]
    loads = outcome.evidence["loads"]
    assert all(later <= earlier for earlier, later in zip(expected, expected[1:]))
    assert expected[loads.index(6.0)] < expected[loads.index(1.0)]
    assert len(outcome.chart["points"]) == len(loads)
    assert "at load 6" in outcome.key_number
    assert_timeline_chart(outcome.chart)


def test_check_24_bulk_batch_arrives_late_for_one_region() -> None:
    """
    WHAT: Executes the check 24 runner and asserts the chosen batch confirmed accounts, that the evidence records the
    batch counts including the batches skipped for having no fraud before their cut-off, that the fixture is labelled
    as not real, and that the chart has one band labelled "lag".
    WHY: Proves Check 24 reports a crackdown that arrives after the fraud it covers, and never silently skips batches.
    Args:
        None.
    Returns:
        None. Fails if the runner raises or the evidence or chart is incomplete.
    Example:
        test_check_24_bulk_batch_arrives_late_for_one_region()
    """
    outcome = asyncio.run(run_bulk_batch_check(None))
    evidence = outcome.evidence
    assert len(evidence["account_ids"]) >= 1
    assert evidence["batches_with_accounts"] >= 1
    assert evidence["batches_total"] >= evidence["batches_with_accounts"]
    assert evidence["batches_skipped_no_fraud_before_cutoff"] >= 0
    assert "not a real range" in evidence["fixture"]
    assert evidence["arrival_minute"] > max(evidence["last_fraud_minutes"])
    low, high = evidence["lag_bounds_minutes"]
    assert low <= evidence["arrival_minute"] - evidence["cutoff_minute"] <= high
    assert "days after the last fraud payment it covers" in outcome.key_number
    assert_timeline_chart(outcome.chart)
    assert [label for _start, _end, label in outcome.chart["shaded"]] == ["lag"]
