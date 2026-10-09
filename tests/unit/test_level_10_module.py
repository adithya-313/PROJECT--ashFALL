"""
File: tests/unit/test_level_10_module.py
Purpose: Proves the Level 10 showcase module has the contract the framework needs, and that each function in
         core/level10_check.py works on the real repo files (read-only). The broken copies are built in memory.
"""

from datetime import timedelta

from core import level10_check
from showcase import level_10
from showcase.framework import validate_checks

EXPECTED_KINDS = ["normal", "edge", "negative", "edge", "normal"]


def test_module_has_the_contract_attributes() -> None:
    """The module defines everything the framework reads."""
    assert level_10.LEVEL_NUMBER == 10
    assert level_10.LEVEL_TITLE
    assert level_10.WHAT_YOU_SAW
    assert level_10.NEEDS_DATABASE is False
    assert callable(level_10.find_missing_preconditions) and callable(level_10.get_checks)


def test_checks_have_the_required_kinds_in_both_modes() -> None:
    """Both modes list the same five checks, in order, and pass the framework's coverage rules."""
    for quick in (True, False):
        checks = level_10.get_checks(quick)
        validate_checks(checks)
        assert [check.kind for check in checks] == EXPECTED_KINDS
        assert [check.name[0] for check in checks] == ["1", "2", "3", "4", "5"]
        assert all(check.proves and check.likely_cause and check.where_to_look for check in checks)


def test_check_configs_finds_no_errors_and_counts_labels() -> None:
    errors, counts = level10_check.check_configs()
    assert errors == []
    assert set(counts) == {"evaluation.yaml", "world_ranges.yaml", "legal_actions.yaml"}


def test_status_counts() -> None:
    status, hand_checked, tool = level10_check.status_counts()
    assert status == {"confirmed": 11, "changed": 4, "reported": 6, "unverified": 6}
    assert hand_checked == 10
    assert tool == {"none": 12, "single": 4, "both": 9, "disagree": 2}


def test_each_broken_copy_is_rejected_in_memory() -> None:
    rejected = level10_check.broken_copy_is_rejected()
    assert set(rejected) == {"reason removed", "source removed", "training outside test"}
    for name, errors in rejected.items():
        assert len(errors) >= 1, name


def test_preregistration_proof_is_ok_and_honest() -> None:
    proof = level10_check.preregistration_proof()
    assert proof["result"] == "OK"
    assert proof["new_engine_dates"] == {}
    # The evaluation commit is on or after the decisions commit, give or take a minute.
    assert proof["evaluation_date"] >= proof["decisions_date"] - timedelta(minutes=1)
    assert "by design" in proof["sentence"]


def test_spike_verdict_line() -> None:
    assert level10_check.spike_verdict_line() == "Verdict: borrow ideas and use as an outside check"
