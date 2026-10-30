# File: tests/unit/test_feature_set_m1.py
# Purpose: Unit tests validating the M1 live mule feature set catalog, groups, and schema constraints.
# Issue: Level 15b M1 feature catalog (plan Task 1, R8, R5, G1, G2, E3, R7).
"""
WHAT: Tests loading and validation of data/schemas/feature_set_m1.yaml.
Verifies that all 10 plan feature groups are populated, the four architect R5 device signals
are present, explicit candidate exclusions (emulator_flag, remote_access_flag, channel)
are documented with technical reasons, and malformed schemas (duplicates, unknown groups) are rejected.

WHY: Guarantees the integrity and completeness of the Level 15 M1 live mule feature catalog
before feature computation and ML model training commence.
"""

from pathlib import Path

import pytest
import yaml

from features.feature_set import (
    KNOWN_GROUPS,
    M1FeatureSet,
    load_feature_set_m1,
)


def test_feature_set_m1_loads_successfully() -> None:
    """Verifies that the canonical feature_set_m1.yaml loads and produces a populated M1FeatureSet."""
    feature_set = load_feature_set_m1()
    assert isinstance(feature_set, M1FeatureSet)
    assert len(feature_set.features) > 0
    assert len(feature_set.removed) >= 3
    assert len(feature_set.feature_names) == len(feature_set.features)


def test_every_plan_group_is_present() -> None:
    """Verifies that every approved plan grouping category contains at least one active feature."""
    feature_set = load_feature_set_m1()
    present_groups = {feat.group for feat in feature_set.features}
    assert present_groups == KNOWN_GROUPS, f"Missing groups: {KNOWN_GROUPS - present_groups}"

    # Verify each group specifically has members via features_by_group helper
    for group_name in KNOWN_GROUPS:
        grouped = feature_set.features_by_group(group_name)
        assert len(grouped) > 0, f"Expected features under group {group_name!r}"


def test_four_r5_device_parts_and_history_days_exist() -> None:
    """Verifies that the architect-specified R5 device features and history_days_available are defined."""
    feature_set = load_feature_set_m1()

    # Four R5 device parts
    dev_hardware = feature_set.get_feature("device_hardware_changed")
    assert dev_hardware.group == "control"
    assert "R5" in dev_hardware.issue_ids

    sim_swap = feature_set.get_feature("sim_swapped")
    assert sim_swap.group == "control"
    assert "R5" in sim_swap.issue_ids

    app_reinstall = feature_set.get_feature("app_reinstalled_same_phone")
    assert app_reinstall.group == "control"
    assert "R5" in app_reinstall.issue_ids

    new_dev = feature_set.get_feature("new_device_r5")
    assert new_dev.group == "control"
    assert "R5" in new_dev.issue_ids

    # History days available
    hist_days = feature_set.get_feature("history_days_available")
    assert hist_days.group == "lifecycle"
    assert "summary_horizon_days" in hist_days.window_or_horizon


def test_removed_list_contains_expected_entries_with_reasons() -> None:
    """Verifies that emulator_flag, remote_access_flag, and channel are in removed list with reasons."""
    feature_set = load_feature_set_m1()
    removed_map = {item.name: item.reason for item in feature_set.removed}

    assert "emulator_flag" in removed_map
    assert "not visible to banks in this simulator: no such account-event kind" in removed_map["emulator_flag"]

    assert "remote_access_flag" in removed_map
    assert "not visible to banks in this simulator: no such account-event kind" in removed_map["remote_access_flag"]

    assert "channel" in removed_map
    assert "every simulated payment is upi_app; a constant column teaches nothing" in removed_map["channel"]

    # Verify gap note
    assert any("bill payments are detected as recurring merchant payments" in note for note in feature_set.gap_notes)


def test_duplicate_feature_name_rejected(tmp_path: Path) -> None:
    """Verifies that duplicate feature names in the schema raise ValueError."""
    bad_cfg = {
        "features": [
            {
                "name": "duplicate_name",
                "group": "inflow",
                "definition": "First definition",
                "bank_visible_source_records": "PaymentSideRecord",
                "window_or_horizon": "1h",
                "empty_means": "zero",
            },
            {
                "name": "duplicate_name",
                "group": "outflow",
                "definition": "Second definition",
                "bank_visible_source_records": "PaymentSideRecord",
                "window_or_horizon": "24h",
                "empty_means": "zero",
            },
        ],
        "removed": [],
    }
    test_yaml = tmp_path / "dup_features.yaml"
    with open(test_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(bad_cfg, f)

    with pytest.raises(ValueError, match="Duplicate feature name"):
        load_feature_set_m1(test_yaml)


def test_unknown_group_rejected(tmp_path: Path) -> None:
    """Verifies that a feature entry specifying an unrecognized group raises ValueError."""
    bad_cfg = {
        "features": [
            {
                "name": "invalid_group_feature",
                "group": "unsupported_group_category",
                "definition": "Some definition",
                "bank_visible_source_records": "PaymentSideRecord",
                "window_or_horizon": "1h",
                "empty_means": "zero",
            }
        ],
        "removed": [],
    }
    test_yaml = tmp_path / "bad_group.yaml"
    with open(test_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(bad_cfg, f)

    with pytest.raises(ValueError, match="invalid group"):
        load_feature_set_m1(test_yaml)


def test_feature_in_both_active_and_removed_rejected(tmp_path: Path) -> None:
    """Verifies that a feature appearing in both active and removed lists raises ValueError."""
    bad_cfg = {
        "features": [
            {
                "name": "conflicted_feature",
                "group": "inflow",
                "definition": "Active definition",
                "bank_visible_source_records": "PaymentSideRecord",
                "window_or_horizon": "1h",
                "empty_means": "zero",
            }
        ],
        "removed": [
            {
                "name": "conflicted_feature",
                "reason": "Exclusion reason",
            }
        ],
    }
    test_yaml = tmp_path / "conflicted.yaml"
    with open(test_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(bad_cfg, f)

    with pytest.raises(ValueError, match="appears in both"):
        load_feature_set_m1(test_yaml)
