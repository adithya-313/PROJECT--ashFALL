"""
File: tests/unit/test_level_08_module.py
Purpose: Proves the Level 8 showcase module has the contract the framework needs, lists its five checks in the required
         order and kinds in both modes, and that the small text helpers behind checks 1, 2 and 3 work. No Docker needed.
"""

from showcase import level_08
from showcase.framework import validate_checks
from showcase.level08_support import (
    count_differences,
    hide,
    last_lines,
    parse_check_summary,
    rename_field_in_text,
)

EXPECTED_NAMES = [
    "1. make check passes and the baselines match",
    "2. a changed response field is caught",
    "3. all artifacts restore from the remote into an empty folder with matching hashes",
    "4. the stack runs with outgoing network blocked and makes zero outbound attempts",
    "5. Level 7's showcase still passes in quick mode",
]
EXPECTED_KINDS = ["normal", "edge", "edge", "negative", "consistency"]

RESPONSE_SOURCE = """class ScoreResponse(BaseModel):
    transaction_id: str
    model_version: str


class Other(BaseModel):
    model_version: str
"""


def test_module_has_the_contract_attributes() -> None:
    """The module defines everything the framework reads."""
    assert level_08.LEVEL_NUMBER == 8
    assert level_08.LEVEL_TITLE == "Safety net"
    assert level_08.WHAT_YOU_SAW
    assert level_08.NEEDS_DATABASE is False
    assert callable(level_08.find_missing_preconditions) and callable(level_08.get_checks)


def test_checks_have_the_required_names_and_kinds_in_both_modes() -> None:
    """Both modes list the same five checks in order, and pass the framework's coverage rules."""
    for quick in (True, False):
        checks = level_08.get_checks(quick)
        validate_checks(checks)
        assert [check.name for check in checks] == EXPECTED_NAMES
        assert [check.kind for check in checks] == EXPECTED_KINDS
        assert all(check.proves and check.likely_cause and check.where_to_look for check in checks)


def test_only_the_field_inside_score_response_is_renamed() -> None:
    """The rename changes exactly one line, inside ScoreResponse, and leaves the other class alone."""
    changed = rename_field_in_text(RESPONSE_SOURCE)
    assert changed is not None
    assert changed.count("model_version_renamed: str") == 1
    assert changed.count("model_version: str") == 1  # the one in class Other
    assert changed.index("model_version_renamed") < changed.index("class Other")


def test_rename_refuses_when_the_field_is_not_found_exactly_once() -> None:
    """No class, no field line, or two field lines in the class give None, so the check fails loudly."""
    assert rename_field_in_text("class Something:\n    pass\n") is None
    assert rename_field_in_text("class ScoreResponse(BaseModel):\n    other: str\n") is None
    twice = "class ScoreResponse(BaseModel):\n    model_version: str\n    model_version: str\n"
    assert rename_field_in_text(twice) is None


def test_difference_count_and_check_summary_are_read_from_output() -> None:
    """The number of differences and the passed/ran step counts are read from the real output formats."""
    assert count_differences("AssertionError: The API's public shape changed (3 difference(s)):") == 3
    assert count_differences("all fine") == 0
    table = "PASS 0.2s ruff\nFAIL 1.0s mypy\nPASS 2.0s tests\nSKIPPED - showcase\n"
    assert parse_check_summary(table) == (2, 3)


def test_text_helpers_hide_private_values_and_keep_the_tail() -> None:
    """A private folder is replaced by a placeholder, and only the last lines are kept."""
    assert hide("error in C:/private/remote now", "C:/private/remote") == "error in <local remote> now"
    assert hide("nothing to hide", "") == "nothing to hide"
    assert last_lines("a\nb\nc\nd", 2) == "c\nd"
