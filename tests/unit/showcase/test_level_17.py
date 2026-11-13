"""
WHAT: Tests for the Level 17 showcase definition: the seven checks, their kinds and the fixture
preconditions.

WHY: The checks themselves replay a world and run the evaluator, so this fast test guards the list
the runner reads: order, coverage kinds, preconditions and the module attributes.
"""

from pathlib import Path

from showcase import level_17, level_17_checks
from showcase.framework import validate_checks

EXPECTED_ORDER = [
    "17-1 each rule fires on its constructed case with its reason code",
    "17-2 tea stall and gig worker: known false trigger counted, real worker quiet",
    "17-3 OS-update wave: OS updates stay quiet, phone changes flag",
    "17-4 fixture worlds produce a complete result.json",
    "17-5 a world with no labels still produces decisions",
    "17-6 untuned rules and a test-split detector are refused",
    "17-7 two runs on the same inputs give one hash",
]


def test_seven_checks_in_the_architects_order() -> None:
    """Checks the showcase lists exactly the seven checks in the fixed order."""
    assert [check.name for check in level_17.get_checks()] == EXPECTED_ORDER


def test_checks_meet_the_framework_coverage_rules() -> None:
    """Checks the list passes validate_checks and carries every required kind."""
    checks = level_17.get_checks()
    validate_checks(checks)
    kinds = [check.kind for check in checks]
    assert kinds.count("normal") >= 1
    assert kinds.count("edge") >= 2
    assert kinds.count("negative") >= 1
    assert kinds.count("consistency") >= 1


def test_module_has_the_attributes_the_runner_needs() -> None:
    """Checks the level module defines the number, title, database flag, story and precondition function."""
    assert level_17.LEVEL_NUMBER == 17 and level_17.NEEDS_DATABASE is False
    assert level_17.LEVEL_TITLE and level_17.WHAT_YOU_SAW and callable(level_17.find_missing_preconditions)


def test_preconditions_name_a_missing_fixture_manifest(tmp_path: Path) -> None:
    """Checks a missing manifest is named by its full path instead of failing later inside a check."""
    missing_root = tmp_path / "level_17"
    problems = level_17_checks.find_missing_preconditions(missing_root)
    assert problems
    assert any(str(missing_root / "FIXTURE_MANIFEST.json") in problem for problem in problems)


def test_committed_fixtures_meet_the_preconditions() -> None:
    """Checks the committed Level 17 fixtures and scikit-learn satisfy the preconditions."""
    assert level_17_checks.find_missing_preconditions() == []
