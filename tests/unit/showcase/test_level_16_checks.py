"""
Tests for the Level 16 showcase definition: 12 checks in the architect's order that meet the framework's coverage rules.

WHY: The checks themselves need the Docker stack and an hour of replay, so this fast test guards their list: count,
order, kinds and the module attributes `python -m showcase --level 16` needs.

TERM: coverage rules: the framework demands at least one normal, two edge and one negative check per level.
"""

from showcase import level_16
from showcase.framework import validate_checks

EXPECTED_ORDER = [
    "16-1 a gang's true mule activity against label arrivals",
    "16-2 datasets at three cutoffs hold only labels known by then",
    "16-3 a falsely reported account is a positive label",
    "16-4 two replays give identical hashes",
    "16-5 the dataset builder is refused truth",
    "16-6 no future leakage, including revokes",
    "16-7 all four label paths reach the right banks",
    "16-8 N1: the innocent holder of a recycled number is labelled",
    "16-9 the registry is wired into decisions",
    "16-10 a superseded dataset is refused unless asked",
    "16-11 G3: one account collects complaints from several victims over weeks",
    "16-12 hygiene: no label column is a feature",
]


def test_twelve_checks_in_the_architects_order() -> None:
    """Checks the showcase lists exactly the 12 checks in the fixed order."""
    assert [c.name for c in level_16.get_checks()] == EXPECTED_ORDER


def test_checks_meet_the_framework_coverage_rules() -> None:
    """Checks the list passes validate_checks (kinds known, proves sentences present, names unique)."""
    validate_checks(level_16.get_checks())


def test_module_has_the_attributes_the_runner_needs() -> None:
    """Checks the level module defines the number, title, database flag, story and precondition function."""
    assert level_16.LEVEL_NUMBER == 16 and level_16.NEEDS_DATABASE is True
    assert level_16.LEVEL_TITLE and level_16.WHAT_YOU_SAW and callable(level_16.find_missing_preconditions)
