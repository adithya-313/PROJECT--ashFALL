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

from datetime import datetime, timezone
from pathlib import Path

import fakeredis
import pytest
import yaml

from features.feature_set import (
    KNOWN_GROUPS,
    M1FeatureSet,
    load_feature_set_m1,
)
from features.m1_features import SUPPORTED_GROUPS, compute_account_features


def test_feature_set_m1_loads_successfully() -> None:
    """Verifies that the canonical feature_set_m1.yaml loads and produces a populated M1FeatureSet."""
    feature_set = load_feature_set_m1()
    assert isinstance(feature_set, M1FeatureSet)
    assert len(feature_set.features) > 0
    assert len(feature_set.removed) >= 3
    assert len(feature_set.feature_names) == len(feature_set.features)


def test_every_plan_group_is_present() -> None:
    """Verifies that all active plan grouping categories contain active features (intelligence parked in pending)."""
    feature_set = load_feature_set_m1()
    present_groups = {feat.group for feat in feature_set.features}
    # Intelligence features are parked in pending awaiting architectural rulings
    assert present_groups == (KNOWN_GROUPS - {"intelligence"}), f"Missing groups: {KNOWN_GROUPS - present_groups}"

    for group_name in present_groups:
        grouped = feature_set.features_by_group(group_name)
        assert len(grouped) > 0, f"Expected features under group {group_name!r}"
    assert len(feature_set.features_by_group("intelligence")) == 0


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

    # History days available (in routine_life group per canonical architect list)
    hist_days = feature_set.get_feature("history_days_available")
    assert hist_days.group == "routine_life"
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

    # Verify gap notes
    assert any("bill payments are detected as recurring merchant payments" in note for note in feature_set.gap_notes)
    assert any("crypto cash-out reaches the bank as an ordinary transfer" in note for note in feature_set.gap_notes)


def test_all_canonical_features_present_in_expected_groups() -> None:
    """Verifies that every canonical M1 feature from the plan is present in its approved group without duplicates."""
    feature_set = load_feature_set_m1()

    expected_groups: dict[str, set[str]] = {
        "inflow": {
            "distinct_payers_1h",
            "distinct_payers_24h",
            "distinct_payers_7d",
            "first_time_payer_share_24h",
            "small_credit_count_24h",
            "small_credit_count_7d",
            "below_threshold_credit_count_7d",
        },
        "outflow": {
            "pass_through_ratio_24h",
            "pass_through_ratio_7d",
            "median_credit_to_debit_seconds_7d",
            "top_receiver_share_7d",
            "other_bank_outflow_share_7d",
            "outflow_to_person_paise_7d",
            "outflow_to_merchant_paise_7d",
            "cash_atm_paise_7d",
            "cash_branch_paise_7d",
        },
        "balance_pattern": {
            "current_balance_paise",
            "min_balance_paise_7d",
            "near_zero_cycles_7d",
        },
        "routine_life": {
            "salary_like_payer_count",
            "recurring_merchant_payee_count",
            "regular_transfer_payee_count",
            "routine_life_present",
            "history_days_available",
        },
        "lifecycle": {
            "account_age_days",
            "account_type",
            "dormancy_days_before_recent_activity",
            "volume_vs_own_history_ratio",
        },
        "control": {
            "days_since_device_change",
            "accounts_on_same_device",
            "device_hardware_changed",
            "sim_swapped",
            "app_reinstalled_same_phone",
            "new_device_r5",
            "distinct_fingerprints_lifetime",
            "machine_regular_timing_score",
        },
        "region": {
            "distinct_same_bank_payer_regions_7d",
            "unknown_region_payer_count_7d",
        },
        "network": {
            "hops_to_confirmed_label",
            "shared_counterparties_with_flagged_7d",
            "local_cluster_density",
            "hops_to_complained_account",
        },
        "intelligence": set(),
        "payment_level": {
            "first_payment_to_payee",
            "amount_vs_payer_median_ratio",
            "amount_vs_payer_max_ratio",
            "payee_same_bank_risk",
            "hour_ist",
            "payee_type",
            "customer_failed_attempts_24h",
        },
    }

    # Verify no duplicates across all names
    all_names = [f.name for f in feature_set.features]
    assert len(all_names) == len(set(all_names)), "Duplicate feature names in M1 feature set"

    # Verify each group matches expected canonical members exactly
    for group_name, expected_names in expected_groups.items():
        actual_features = feature_set.features_by_group(group_name)
        actual_names = {f.name for f in actual_features}
        assert actual_names == expected_names, (
            f"Group {group_name!r} mismatch. Missing: {expected_names - actual_names}, "
            f"Unexpected: {actual_names - expected_names}"
        )

    # Verify total count matches
    total_expected = sum(len(names) for names in expected_groups.values())
    assert len(feature_set.features) == total_expected, (
        f"Expected {total_expected} features, got {len(feature_set.features)}"
    )


def test_no_issue_id_outside_allowed_six() -> None:
    """Verifies that no feature in the M1 catalog carries an issue id outside the allowed six."""
    allowed_issue_ids = frozenset({"G1", "G2", "E3", "R5", "R7", "R8"})
    feature_set = load_feature_set_m1()

    for feat in feature_set.features:
        assert len(feat.issue_ids) > 0, f"Feature {feat.name!r} has no issue_ids"
        for issue_id in feat.issue_ids:
            assert issue_id in allowed_issue_ids, (
                f"Feature {feat.name!r} has disallowed issue_id {issue_id!r}; "
                f"must be one of {sorted(allowed_issue_ids)}"
            )


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


def test_pending_section_loads_and_rejects_duplicates(tmp_path: Path) -> None:
    """Verifies that the pending section loads, has expected entries, and rejects duplicates with active/removed."""
    feature_set = load_feature_set_m1()
    assert len(feature_set.pending) == 3
    pending_names = {item.name for item in feature_set.pending}
    assert pending_names == {"payee_registry_match", "registry_match_count", "registry_match_age_days"}

    for item in feature_set.pending:
        assert "Level 13 serves matches through /screen" in item.reason

    # 1. Feature in both active and pending rejected
    bad_cfg_active = {
        "features": [
            {
                "name": "dup_pending_feature",
                "group": "inflow",
                "definition": "Active definition",
                "bank_visible_source_records": "PaymentSideRecord",
                "window_or_horizon": "1h",
                "empty_means": "zero",
            }
        ],
        "removed": [],
        "pending": [
            {
                "name": "dup_pending_feature",
                "reason": "Parked reason",
            }
        ],
    }
    p1 = tmp_path / "dup_active_pending.yaml"
    with open(p1, "w", encoding="utf-8") as f:
        yaml.safe_dump(bad_cfg_active, f)
    with pytest.raises(ValueError, match="appears in both active 'features' and 'pending'"):
        load_feature_set_m1(p1)

    # 2. Feature in both removed and pending rejected
    bad_cfg_removed = {
        "features": [
            {
                "name": "some_feature",
                "group": "inflow",
                "definition": "Active definition",
                "bank_visible_source_records": "PaymentSideRecord",
                "window_or_horizon": "1h",
                "empty_means": "zero",
            }
        ],
        "removed": [
            {
                "name": "dup_removed_feature",
                "reason": "Exclusion reason",
            }
        ],
        "pending": [
            {
                "name": "dup_removed_feature",
                "reason": "Parked reason",
            }
        ],
    }
    p2 = tmp_path / "dup_removed_pending.yaml"
    with open(p2, "w", encoding="utf-8") as f:
        yaml.safe_dump(bad_cfg_removed, f)
    with pytest.raises(ValueError, match="appears in both 'removed' and 'pending'"):
        load_feature_set_m1(p2)


def test_compute_account_features_never_returns_pending_names() -> None:
    """Verifies that compute_account_features never returns names parked in the pending section."""
    client = fakeredis.FakeRedis()
    t = datetime(2026, 10, 28, 12, 0, 0, tzinfo=timezone.utc)
    feature_set = load_feature_set_m1()
    pending_names = {item.name for item in feature_set.pending}

    feats = compute_account_features(
        client=client,
        run_id="66db9aa439b092ee",
        bank_id="bank_0",
        account_id="acc_test",
        decision_time=t,
        arrival_cutoff=t,
        groups=list(SUPPORTED_GROUPS),
    )

    returned_names = set(feats.keys())
    assert not (returned_names & pending_names), f"Returned pending names: {returned_names & pending_names}"

