"""
Tests for the Level 16 showcase definition: 12 checks in the architect's order that meet the framework's coverage rules.

WHY: The checks themselves need the Docker stack and an hour of replay, so this fast test guards their list: count,
order, kinds and the module attributes `python -m showcase --level 16` needs.

TERM: coverage rules: the framework demands at least one normal, two edge and one negative check per level.
"""

import pytest

from showcase import level_16
from showcase.framework import validate_checks
from showcase.level_16_checks import verify_four_label_paths

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


def test_hold_label_cannot_satisfy_direct_complaint_and_vice_versa() -> None:
    """
    WHAT: Tests that verify_four_label_paths separates direct complaints from hold requests.
    WHY: Commit 1: A hold label (source=complaint, via=hold_request) cannot satisfy direct-complaint
    expectation, and a direct complaint (source=complaint, via=None) cannot satisfy hold-request expectation.
    """
    reg_row = {"bank_id": "bank_0", "account_id": "acc_reg", "source": "registry", "via": None}
    bulk_row = {"bank_id": "bank_0", "account_id": "acc_bulk", "source": "police_bulk", "via": None}
    expected_bulk = {("bank_0", "acc_bulk")}
    participants = ["bank_0"]

    # 1. Hold label supplied, but direct complaint expected:
    hold_row = {"bank_id": "bank_0", "account_id": "acc_victim", "source": "complaint", "via": "hold_request"}
    direct_row = {"bank_id": "bank_0", "account_id": "acc_other", "source": "complaint", "via": None}

    with pytest.raises(AssertionError, match=r"complaint: expected .* but labels hold"):
        verify_four_label_paths(
            rows=[reg_row, bulk_row, hold_row, direct_row],
            expected_complaint={("bank_0", "acc_victim")},
            expected_bulk=expected_bulk,
            expected_hold={("bank_0", "acc_victim")},
            registry_participants=participants,
        )

    # 2. Direct complaint supplied, but hold request expected:
    direct_payee_row = {"bank_id": "bank_0", "account_id": "acc_payee", "source": "complaint", "via": None}
    hold_other_row = {"bank_id": "bank_0", "account_id": "acc_other_hold", "source": "complaint", "via": "hold_request"}

    with pytest.raises(AssertionError, match=r"hold_request: expected .* but labels hold"):
        verify_four_label_paths(
            rows=[reg_row, bulk_row, direct_payee_row, hold_other_row],
            expected_complaint={("bank_0", "acc_payee")},
            expected_bulk=expected_bulk,
            expected_hold={("bank_0", "acc_payee")},
            registry_participants=participants,
        )

