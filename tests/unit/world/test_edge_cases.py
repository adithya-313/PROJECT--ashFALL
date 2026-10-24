"""
WHAT: Unit tests for Level 12h edge-case library loader, validation rules, and schema checks.
It verifies that all 13 scenario files load, status counts match, unknown schema keys are refused,
invalid amendment references are rejected, and layer containment nesting violations raise errors.

WHY: Enforces rigorous structural integrity over the scenario edge-case library so invalid
definitions, corrupted overlays, or unanchored assumptions fail immediately with clear messages.
"""

from pathlib import Path

import pytest
import yaml

from world.edge_cases import (
    load_edge_case,
    load_edge_case_library,
)


def test_all_13_edge_cases_load_and_validate_cleanly() -> None:
    """
    WHAT: Tests that all 13 edge cases in scenarios/edge_cases/ load and validate successfully.
    WHY: Verifies baseline library completeness and confirms each case YAML satisfies closed schema rules.
    """
    cases = load_edge_case_library()
    assert len(cases) == 13
    names = [c.name for c in cases]
    assert len(set(names)) == 13
    assert names[0] == "busy_shop_tiny_payments"
    assert names[-1] == "recycled_phone_number"


def test_counts_by_status_match_the_authored_files() -> None:
    """
    WHAT: Asserts that scenario status counts match the architect's fixed assignment distribution.
    WHY: Verifies the exact partition between runnable (9), later_level (3), and deferred (1) cases.
    """
    cases = load_edge_case_library()
    runnable = [c for c in cases if c.status == "runnable"]
    later_level = [c for c in cases if c.status == "later_level"]
    deferred = [c for c in cases if c.status == "deferred"]

    assert len(runnable) == 9
    assert len(later_level) == 3
    assert len(deferred) == 1

    assert {c.name for c in later_level} == {
        "busy_shop_tiny_payments",
        "trace_outage_mid_trace",
        "recycled_phone_number",
    }
    assert {c.name for c in deferred} == {"mule_imitating_salaried_employee"}


def test_unknown_top_level_key_is_refused(tmp_path: Path) -> None:
    """
    WHAT: Asserts that loading an edge case YAML containing an unapproved key raises ValueError.
    WHY: Closed schema enforcement prevents ad-hoc, untyped parameters from entering the library.
    """
    bad_file = tmp_path / "99-unknown_key.yaml"
    bad_file.write_text(
        yaml.dump({
            "name": "unknown_key",
            "title": "Unknown Key Case",
            "expected_behaviour": "Should be refused by closed schema validator.",
            "reason": "Testing unknown key rejection.",
            "status": "later_level",
            "active_from_level": 15,
            "unsupported_extra_field": "disallowed",
        }),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown key 'unsupported_extra_field'"):
        load_edge_case(bad_file)


def test_deferred_case_citing_unknown_amendment_is_refused(tmp_path: Path) -> None:
    """
    WHAT: Asserts that a deferred case citing a non-existent amendment (e.g. A-999) raises ValueError.
    WHY: Guarantees that every deferred scenario links directly to an approved architectural amendment.
    """
    bad_file = tmp_path / "99-bad_amendment.yaml"
    bad_file.write_text(
        yaml.dump({
            "name": "bad_amendment",
            "title": "Bad Amendment Case",
            "expected_behaviour": "Deferred under A-999; no expectation is checked.",
            "reason": "Testing unknown amendment refusal.",
            "status": "deferred",
            "deferred_by": "A-999",
        }),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="deferred_by amendment 'A-999' not found"):
        load_edge_case(bad_file)


def test_later_level_case_with_no_active_from_level_is_refused(tmp_path: Path) -> None:
    """
    WHAT: Asserts that a later_level edge case omitting active_from_level raises ValueError.
    WHY: Enforces that future milestone requirements explicitly declare their target delivery level.
    """
    bad_file = tmp_path / "99-missing_level.yaml"
    bad_file.write_text(
        yaml.dump({
            "name": "missing_level",
            "title": "Missing Level Case",
            "expected_behaviour": "Should be refused due to missing active level.",
            "reason": "Testing missing active level refusal.",
            "status": "later_level",
        }),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="later_level case requires integer active_from_level"):
        load_edge_case(bad_file)


def test_layer_breaking_nesting_is_refused(tmp_path: Path) -> None:
    """
    WHAT: Asserts that a layer change violating containment nesting (training not in test/stress) raises ValueError.
    WHY: Verifies that edge-case configuration overlays preserve training <= test <= stress containment bounds.
    """
    bad_file = tmp_path / "99-bad_nesting.yaml"
    # Pinned value 0.95 outside stress range [0.05, 0.9]
    bad_file.write_text(
        yaml.dump({
            "name": "bad_nesting",
            "title": "Bad Nesting Case",
            "expected_behaviour": "Should be refused due to broken nesting.",
            "reason": "Testing layer nesting violation refusal.",
            "status": "runnable",
            "split": "training",
            "seed": 101,
            "days": 61,
            "preset": "tiny",
            "layer_changes": {
                "world_ranges": {
                    "victim_complaint_share": {
                        "training": [0.95, 0.95],
                        "test": [0.1, 0.95],
                        "stress": [0.05, 0.9],
                    }
                }
            },
        }),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="test range is not inside the stress range"):
        load_edge_case(bad_file)
