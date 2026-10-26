"""Unit tests for registry coverage evaluation across hierarchical mule depths.

What it is for:
    Tests the coverage calculation logic in evaluation/registry_coverage.py,
    confirming exact fractions on hand-built cases, handling zero-report and
    zero-mule edge cases without crashing, and verifying reference range comparisons.
"""

import pytest

from evaluation.registry_coverage import compute_registry_coverage
from world.label_records import RegistryReport
from world.truth_records import RoleSpan


def test_hand_built_tiny_case_gives_exact_expected_fractions() -> None:
    """Hand-built case verifies exact fraction calculation for each mule depth tier.

    Recreates a known distribution of layer 1, layer 2, and layer 3 mules with
    a controlled subset appearing in registry reports. Catches any off-by-one or
    misaligned depth partitioning bugs.
    """
    # 4 mules at depth 1, 4 mules at depth 2, 2 mules at depth 3, plus 1 victim
    roles = [
        RoleSpan(account_id="m1_a", role="mule", gang_id="g1", depth=1, valid_from=0, valid_to=100),
        RoleSpan(account_id="m1_b", role="mule", gang_id="g1", depth=1, valid_from=0, valid_to=100),
        RoleSpan(account_id="m1_c", role="mule", gang_id="g1", depth=1, valid_from=0, valid_to=100),
        RoleSpan(account_id="m1_d", role="mule", gang_id="g1", depth=1, valid_from=0, valid_to=100),
        RoleSpan(account_id="m2_a", role="mule", gang_id="g1", depth=2, valid_from=0, valid_to=100),
        RoleSpan(account_id="m2_b", role="mule", gang_id="g1", depth=2, valid_from=0, valid_to=100),
        RoleSpan(account_id="m2_c", role="mule", gang_id="g1", depth=2, valid_from=0, valid_to=100),
        RoleSpan(account_id="m2_d", role="mule", gang_id="g1", depth=2, valid_from=0, valid_to=100),
        RoleSpan(account_id="m3_a", role="mule", gang_id="g1", depth=3, valid_from=0, valid_to=100),
        RoleSpan(account_id="m3_b", role="mule", gang_id="g1", depth=3, valid_from=0, valid_to=100),
        RoleSpan(account_id="v_01", role="victim", gang_id="g1", depth=None, valid_from=0, valid_to=100),
    ]

    # Registry reports:
    # 2 of 4 depth-1 mules reported -> 50%
    # 1 of 4 depth-2 mules reported -> 25%
    # 0 of 2 depth-3 mules reported -> 0%
    # Aggregate deeper: 1 of 6 deeper mules reported -> 16.67%
    reports = [
        RegistryReport(
            report_id="R-01",
            kind="registry_report",
            reported_minute=100,
            reporting_bank_index=0,
            account_id="m1_a",
            device_fingerprint=None,
            confirmed_minute=90,
            complaint_ids=("C-01",),
        ),
        RegistryReport(
            report_id="R-02",
            kind="registry_report",
            reported_minute=110,
            reporting_bank_index=0,
            account_id="m1_b",
            device_fingerprint=None,
            confirmed_minute=95,
            complaint_ids=("C-02",),
        ),
        RegistryReport(
            report_id="R-03",
            kind="registry_report",
            reported_minute=120,
            reporting_bank_index=1,
            account_id="m2_a",
            device_fingerprint=None,
            confirmed_minute=100,
            complaint_ids=("C-03",),
        ),
        # Duplicate report for m1_a to verify set deduplication
        RegistryReport(
            report_id="R-04",
            kind="registry_report",
            reported_minute=130,
            reporting_bank_index=0,
            account_id="m1_a",
            device_fingerprint=None,
            confirmed_minute=90,
            complaint_ids=("C-01", "C-04"),
        ),
    ]

    result = compute_registry_coverage(roles=roles, registry_reports=reports, split="training")

    # Layer 1: 2 reported out of 4 total
    m1 = result.by_depth["1"]
    assert m1.total_mules == 4
    assert m1.reported_mules == 2
    assert m1.coverage == 0.5

    # Layer 2: 1 reported out of 4 total
    m2 = result.by_depth["2"]
    assert m2.total_mules == 4
    assert m2.reported_mules == 1
    assert m2.coverage == 0.25

    # Layer 3+: 0 reported out of 2 total
    m3 = result.by_depth["3+"]
    assert m3.total_mules == 2
    assert m3.reported_mules == 0
    assert m3.coverage == 0.0

    # Aggregate deeper (depth >= 2): 1 reported out of 6 total
    m_deep = result.by_depth["deeper"]
    assert m_deep.total_mules == 6
    assert m_deep.reported_mules == 1
    assert abs(m_deep.coverage - (1.0 / 6.0)) < 1e-6

    # Verify pass rule: layer 1 coverage (0.50) > deeper coverage (0.167)
    assert result.layer1_coverage == 0.5
    assert abs(result.deeper_coverage - (1.0 / 6.0)) < 1e-6
    assert result.passed is True

    # Check table content
    assert "Mule Depth | Reported / Total |  Coverage | Configured Range" in result.table_text
    assert "1          |    2 /    4      |    50.00%" in result.table_text
    assert "2          |    1 /    4      |    25.00%" in result.table_text
    assert "3+         |    0 /    2      |     0.00%" in result.table_text


def test_world_with_no_reports_gives_zero_without_crash() -> None:
    """An empty registry report stream produces 0% coverage and does not raise an error.

    Catches zero-division bugs and verifies safe fallback to 0.0 coverage.
    """
    roles = [
        RoleSpan(account_id="m1", role="mule", gang_id="g1", depth=1, valid_from=0, valid_to=50),
        RoleSpan(account_id="m2", role="mule", gang_id="g1", depth=2, valid_from=0, valid_to=50),
    ]
    reports: list[RegistryReport] = []

    result = compute_registry_coverage(roles=roles, registry_reports=reports, split="training")

    assert result.by_depth["1"].total_mules == 1
    assert result.by_depth["1"].reported_mules == 0
    assert result.by_depth["1"].coverage == 0.0

    assert result.by_depth["2"].total_mules == 1
    assert result.by_depth["2"].reported_mules == 0
    assert result.by_depth["2"].coverage == 0.0

    assert result.by_depth["deeper"].total_mules == 1
    assert result.by_depth["deeper"].reported_mules == 0
    assert result.by_depth["deeper"].coverage == 0.0

    assert result.layer1_coverage == 0.0
    assert result.deeper_coverage == 0.0
    # Equal coverage does not satisfy strictly greater
    assert result.passed is False


def test_world_with_no_mules_gives_zero_without_crash() -> None:
    """A world containing no mules produces 0.0 coverage across all depths.

    Catches division-by-zero errors when total population is empty.
    """
    roles = [
        RoleSpan(account_id="v1", role="victim", gang_id="g1", depth=None, valid_from=0, valid_to=50),
    ]
    reports = [
        RegistryReport(
            report_id="R-99",
            kind="registry_report",
            reported_minute=10,
            reporting_bank_index=0,
            account_id="other_acc",
            device_fingerprint=None,
            confirmed_minute=5,
            complaint_ids=("C-99",),
        ),
    ]

    result = compute_registry_coverage(roles=roles, registry_reports=reports, split="training")

    assert result.by_depth["1"].total_mules == 0
    assert result.by_depth["1"].coverage == 0.0
    assert result.by_depth["deeper"].total_mules == 0
    assert result.by_depth["deeper"].coverage == 0.0
    assert result.passed is False


def test_stress_split_reads_stress_configured_ranges() -> None:
    """Passing split='stress' extracts the stress bounds from configuration."""
    roles = [
        RoleSpan(account_id="m1", role="mule", gang_id="g1", depth=1, valid_from=0, valid_to=50),
        RoleSpan(account_id="m2", role="mule", gang_id="g1", depth=2, valid_from=0, valid_to=50),
    ]
    reports = [
        RegistryReport(
            report_id="R-01",
            kind="registry_report",
            reported_minute=10,
            reporting_bank_index=0,
            account_id="m1",
            device_fingerprint=None,
            confirmed_minute=5,
            complaint_ids=("C-01",),
        ),
    ]

    result = compute_registry_coverage(roles=roles, registry_reports=reports, split="stress")

    # In config/world_ranges.yaml, stress ranges are [0.05, 1.0] for layer1 and [0.0, 0.5] for deeper
    assert result.layer1_range == (0.05, 1.0)
    assert result.deeper_range == (0.0, 0.5)
    assert result.passed is True


def test_invalid_split_raises_value_error() -> None:
    """Unrecognized partition name raises ValueError naming the split."""
    roles = [
        RoleSpan(account_id="m1", role="mule", gang_id="g1", depth=1, valid_from=0, valid_to=50),
    ]
    with pytest.raises(ValueError, match="Split 'unknown_split' not found"):
        compute_registry_coverage(roles=roles, registry_reports=[], split="unknown_split")
