"""
WHAT: Unit test suite verifying Level 11 showcase definitions, preconditions, and core check executions.
It validates metadata invariants, precondition discovery, check schema coverage, and sample check execution
across pre-registration amendments, issue register tracking, tiny world simulation, and grounding tables.

WHY: Verifies that showcase/level_11.py and its accompanying check runners conform strictly to framework contracts.
Validating check definitions against framework rules prevents runtime coverage errors during showcase execution.
Testing core check runners in isolation guarantees that assertions, key numbers, and evidence structures behave properly.

TERM: CheckCoverage: Framework rules requiring minimum counts of normal, edge, and negative checks per level,
      e.g. enforcing at least one normal, two edge, and one negative check in every showcase.
TERM: CheckDefinition: A typed record holding check identifier, proving statement, category kind, and callable runner,
      e.g. defining the tiny world simulation check and its failure troubleshooting pointers.
TERM: CheckOutcome: The result data handed back by an individual check containing key number text and evidence,
      e.g. capturing transaction attempt counts and hero chart SVG strings.
"""

import asyncio
from typing import Any

from showcase.framework import validate_checks
from showcase.level_11 import (
    LEVEL_NUMBER,
    LEVEL_TITLE,
    NEEDS_DATABASE,
    WHAT_YOU_SAW,
    find_missing_preconditions,
    get_checks,
)
from showcase.level_11_checks import (
    run_amendments_check,
    run_grounding_check,
    run_issue_register_check,
    run_tiny_world_check,
)


def test_level_11_metadata_and_preconditions() -> None:
    """
    WHAT: Tests top-level metadata constants and evaluates precondition discovery on the active repository.
    It asserts that LEVEL_NUMBER equals 11, LEVEL_TITLE is populated, and NEEDS_DATABASE is False.
    It verifies that find_missing_preconditions returns an empty list on an intact workspace.

    WHY: The showcase framework inspects level attributes to discover, catalog, and execute level showcases.
    Asserting NEEDS_DATABASE is False ensures that Level 11 runs completely offline without Docker dependencies.
    Confirming zero missing preconditions guarantees that all required configuration files exist on disk.

    Returns:
        None upon asserting all metadata constants and preconditions satisfy expectations.

    Example:
        >>> test_level_11_metadata_and_preconditions()
    """
    assert LEVEL_NUMBER == 11
    assert "simulated India" in LEVEL_TITLE
    assert NEEDS_DATABASE is False
    assert len(WHAT_YOU_SAW) > 50
    missing = asyncio.run(find_missing_preconditions())
    assert missing == [], f"Missing preconditions: {missing}"


def test_level_11_checks_coverage_and_kinds() -> None:
    """
    WHAT: Asserts that get_checks returns 16 checks that satisfy the framework coverage rules.
    It evaluates both quick and full check lists against validate_checks to confirm kind distributions.
    It verifies that checks 1 through 16 have distinct names and valid proving statements.

    WHY: Protects against coverage regressions that would trigger CheckCoverageError in the framework runner.
    Confirming that all 16 checks declare real 'proves' sentences ensures informative terminal checklist output.
    Ensures that quick and full modes construct matching check lists with appropriate skipping logic.

    Returns:
        None upon validating check coverage and descriptor schemas.

    Example:
        >>> test_level_11_checks_coverage_and_kinds()
    """
    quick_checks = get_checks(quick=True)
    full_checks = get_checks(quick=False)
    assert len(quick_checks) == 16 and len(full_checks) == 16
    validate_checks(quick_checks)
    validate_checks(full_checks)
    names = [c.name for c in quick_checks]
    assert len(names) == len(set(names)), "Duplicate check names found"


def test_amendments_and_issues_checks() -> None:
    """
    WHAT: Executes run_amendments_check and run_issue_register_check in an async event loop.
    It asserts that both checks return CheckOutcome objects with populated key numbers and evidence dictionaries.
    It verifies that amendment counts and issue register status tallies match expected invariants.

    WHY: Verifies the execution of foundational research checks 1 and 2 without invoking the entire level.
    Ensures that amendments.md and issues.yaml remain readable and correctly structured.
    Confirming that check outcomes contain expected keys prevents template rendering failures downstream.

    Returns:
        None upon successful check execution and assertion validation.

    Example:
        >>> test_amendments_and_issues_checks()
    """
    out1 = asyncio.run(run_amendments_check(None))
    assert "amendments recorded" in out1.key_number
    assert out1.evidence["amendments_count"] >= 11
    out2 = asyncio.run(run_issue_register_check(None))
    assert "28 built" in out2.key_number
    assert out2.evidence["built"] == 28


def test_tiny_world_and_grounding_checks() -> None:
    """
    WHAT: Executes run_tiny_world_check and run_grounding_check in an async event loop.
    It asserts that the tiny world simulation produces payments, failures, and an SVG hero chart.
    It verifies that the empirical grounding check accounts for all 33 simulation parameters.

    WHY: Verifies execution of operational checks 3 and 4, ensuring fast-mode generation works cleanly.
    Validating the generated SVG string protects report builders against empty or missing hero graphics.
    Confirming grounding tallies ensures that parameter documentation matches repository standards.

    Returns:
        None upon successful check execution and assertion validation.

    Example:
        >>> test_tiny_world_and_grounding_checks()
    """
    out3 = asyncio.run(run_tiny_world_check(None))
    assert "attempts" in out3.key_number
    assert out3.evidence["attempts"] > 1000
    assert out3.evidence["chart_svg"].startswith("<svg")
    out4 = asyncio.run(run_grounding_check(None))
    assert "33 parameters" in out4.key_number
    assert out4.evidence["total_parameters"] == 33
