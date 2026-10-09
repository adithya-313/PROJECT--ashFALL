"""
File: tests/unit/test_level_09_module.py
Purpose: Proves the Level 9 showcase module has the contract the framework needs and lists its four checks in the
         required order and kinds, and that the checks need no retraining. No data or database is needed.
"""

from showcase import level_09
from showcase.framework import validate_checks

EXPECTED_KINDS = ["normal", "edge", "edge", "negative"]


def test_module_has_the_contract_attributes() -> None:
    """The module defines everything the framework reads."""
    assert level_09.LEVEL_NUMBER == 9
    assert level_09.LEVEL_TITLE == "Audit of model v1"
    assert level_09.WHAT_YOU_SAW
    assert level_09.NEEDS_DATABASE is False
    assert callable(level_09.find_missing_preconditions) and callable(level_09.get_checks)


def test_checks_have_the_required_kinds_in_both_modes() -> None:
    """Both modes list the same four checks, in order, and pass the framework's coverage rules."""
    for quick in (True, False):
        checks = level_09.get_checks(quick)
        validate_checks(checks)
        assert [check.kind for check in checks] == EXPECTED_KINDS
        assert [check.name[0] for check in checks] == ["1", "2", "3", "4"]
        assert all(check.proves and check.likely_cause and check.where_to_look for check in checks)
