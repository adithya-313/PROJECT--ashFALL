"""
WHAT: Unit tests for Section 23 leakage check 7 on dataset rows.

WHY: A zero hop is only legitimate when the supporting label had already arrived at the snapshot
time; the check must fail on a one-second-early label, a label in another world, or a label of the
wrong kind, and must not report zero feedback values as covered.
"""

from datetime import datetime, timezone
from pathlib import Path

from decisions.mule_account_model import set_mule_model_config_path
from evaluation.leakage_checks import (
    check_label_features_by_arrival,
    combine_label_feature_checks,
    feedback_distinct_values,
)
from tests.unit.mule_model_config_fixture import label_trust_blocks_text, xgboost_device_block_text

SNAPSHOT_AT = datetime(2026, 9, 2, 12, 0, tzinfo=timezone.utc)
ARRIVAL_AT = datetime(2026, 9, 2, 10, 0, tzinfo=timezone.utc)


def _snapshot(
    world_id: str,
    account_id: str,
    moment: datetime,
    hops_complained: int | None = 0,
    hops_confirmed: int | None = None,
    shared_flagged: int | None = None,
) -> dict[str, object]:
    """Builds one daily snapshot row carrying the label-based feature columns."""
    return {
        "world_id": world_id,
        "bank_id": "bank_0",
        "account_id": account_id,
        "snapshot_time": moment,
        "snapshot_kind": "daily",
        "distinct_payers_24h": 1.0,
        "account_age_days": moment.day,
        "hops_to_complained_account": hops_complained,
        "hops_to_confirmed_label": hops_confirmed,
        "shared_counterparties_with_flagged_7d": shared_flagged,
    }


def _label(
    world_id: str,
    account_id: str,
    moment: datetime,
    source: str = "complaint",
    via: str | None = None,
) -> dict[str, object]:
    """Builds one dataset label row as a bank feed would record it."""
    return {
        "world_id": world_id,
        "account_id": account_id,
        "source": source,
        "via": via,
        "verdict": "fraud",
        "arrival_time": moment,
    }


def _config_without_maturity(tmp_path: Path) -> Path:
    """Writes a schema-complete model config whose maturity window drops nothing."""
    path = tmp_path / "mule_model.yaml"
    path.write_text(
        "parameters:\n"
        "  label_cut:\n"
        "    label: policy\n"
        "    value: arrival_time\n"
        "    derived_from: test\n"
        "    reason: test\n"
        "  label_maturity_days:\n"
        "    label: policy\n"
        "    value: 0\n"
        "    derived_from: test\n"
        "    reason: test\n"
        + label_trust_blocks_text()
        + xgboost_device_block_text()
        + "  feature_fill_flag_share:\n"
        "    label: policy\n"
        "    value: 0.999\n"
        "    derived_from: test\n"
        "    reason: test\n"
        "  use_label_network_features:\n"
        "    label: policy\n"
        "    value: true\n"
        "    derived_from: test\n"
        "    reason: test\n"
        "  label_network_features:\n"
        "    label: policy\n"
        "    value:\n"
        "      - hops_to_complained_account\n"
        "      - registry_match_count\n"
        "      - registry_match_age_days\n"
        "    derived_from: test\n"
        "    reason: test\n"
        "  hyperparameter_search:\n"
        "    label: policy\n"
        '    value: "off"\n'
        "    derived_from: test\n"
        "    reason: test\n"
        "  optuna_trials:\n"
        "    label: policy\n"
        "    value: 40\n"
        "    derived_from: test\n"
        "    reason: test\n"
        "  optuna_seed:\n"
        "    label: policy\n"
        "    value: 1717\n"
        "    derived_from: test\n"
        "    reason: test\n"
        "  optuna_min_trials_before_stop:\n"
        "    label: policy\n"
        "    value: 20\n"
        "    derived_from: test\n"
        "    reason: test\n"
        "  optuna_stall_patience_trials:\n"
        "    label: policy\n"
        "    value: 10\n"
        "    derived_from: test\n"
        "    reason: test\n"
        "  optuna_min_improvement:\n"
        "    label: policy\n"
        "    value: 0.001\n"
        "    derived_from: test\n"
        "    reason: test\n"
        "  hyperparameter_space:\n"
        "    label: policy\n"
        "    value:\n"
        "      max_depth: {min: 3, max: 10, log: false}\n"
        "      learning_rate: {min: 0.02, max: 0.3, log: true}\n"
        "      n_estimators: {min: 100, max: 800, log: false}\n"
        "      min_child_weight: {min: 1, max: 20, log: true}\n"
        "      subsample: {min: 0.5, max: 1.0, log: false}\n"
        "      colsample_bytree: {min: 0.5, max: 1.0, log: false}\n"
        "      gamma: {min: 0.0, max: 5.0, log: false}\n"
        "      reg_alpha: {min: 0.001, max: 10.0, log: true}\n"
        "      reg_lambda: {min: 0.1, max: 10.0, log: true}\n"
        "    derived_from: test\n"
        "    reason: test\n",
        encoding="utf-8",
    )
    return path


def test_zero_hops_after_the_complaint_arrived_passes() -> None:
    """A complaint that arrived before the snapshot justifies the zero hop."""
    snapshots = [_snapshot("world-0", "acc_1", SNAPSHOT_AT)]
    labels = [_label("world-0", "acc_1", ARRIVAL_AT)]

    result = check_label_features_by_arrival(snapshots, labels)

    assert result["status"] == "pass"
    assert result["complained"]["snapshots_checked"] == 1
    assert result["complained"]["violations"] == 0
    assert result["feedback_features"]["status"] == "no source data"


def test_zero_hops_one_second_before_arrival_fails_with_the_example(tmp_path: Path) -> None:
    """A complaint arriving one second later must not support the zero hop."""
    snapshots = [_snapshot("world-0", "acc_1", datetime(2026, 9, 2, 9, 59, 59, tzinfo=timezone.utc))]
    labels = [_label("world-0", "acc_1", ARRIVAL_AT)]

    set_mule_model_config_path(_config_without_maturity(tmp_path))
    try:
        result = check_label_features_by_arrival(snapshots, labels)
    finally:
        set_mule_model_config_path(None)

    assert result["status"] == "fail"
    assert result["complained"]["violations"] == 1
    assert result["complained"]["examples"][0]["account_id"] == "acc_1"
    assert result["positive_training_rows_with_zero_complained_hops"] == 1


def test_a_label_in_another_world_does_not_satisfy_the_hop() -> None:
    """The same account_id labelled in another world is a different account."""
    snapshots = [_snapshot("world-0", "acc_1", SNAPSHOT_AT)]
    labels = [_label("world-1", "acc_1", ARRIVAL_AT)]

    result = check_label_features_by_arrival(snapshots, labels)

    assert result["status"] == "fail"
    assert result["complained"]["violations"] == 1


def test_registry_and_hold_request_labels_do_not_satisfy_the_complained_hop() -> None:
    """Only a real complaint fills the complained set; other label kinds are a different set."""
    snapshots = [_snapshot("world-0", "acc_1", SNAPSHOT_AT)]
    labels = [
        _label("world-0", "acc_1", ARRIVAL_AT, source="registry", via="registry"),
        _label(
            "world-0",
            "acc_1",
            ARRIVAL_AT,
            source="complaint",
            via="hold_request",
        ),
        _label("world-0", "acc_1", ARRIVAL_AT, source="police_bulk"),
    ]

    result = check_label_features_by_arrival(snapshots, labels)

    assert result["status"] == "fail"
    assert result["complained"]["violations"] == 1


def test_a_constant_feedback_column_is_no_source_data() -> None:
    """The empty feedback set stores one constant zero, which is no source data rather than a leak."""
    snapshots = [
        _snapshot("world-0", "acc_1", SNAPSHOT_AT, hops_complained=1, shared_flagged=0),
        _snapshot("world-0", "acc_2", SNAPSHOT_AT, hops_complained=1, shared_flagged=0),
    ]

    result = check_label_features_by_arrival(snapshots, [])

    assert result["status"] == "pass"
    assert result["feedback_features"]["status"] == "no source data"
    assert result["feedback_features"]["distinct_value_counts"]["shared_counterparties_with_flagged_7d"] == 1


def test_a_varied_feedback_feature_fails_the_check() -> None:
    """A feedback-derived value that varies has no source in built worlds and fails the check."""
    snapshots = [
        _snapshot("world-0", "acc_1", SNAPSHOT_AT, hops_complained=1, hops_confirmed=2),
        _snapshot("world-0", "acc_2", SNAPSHOT_AT, hops_complained=1, hops_confirmed=3),
    ]

    result = check_label_features_by_arrival(snapshots, [])

    assert result["status"] == "fail"
    assert result["feedback_features"]["status"] == "fail"
    assert result["feedback_features"]["non_null_snapshot_counts"]["hops_to_confirmed_label"] == 2
    assert result["feedback_features"]["distinct_value_counts"]["hops_to_confirmed_label"] == 2
    assert result["complained"]["violations"] == 0


def test_per_world_blocks_combine_to_the_whole_split_block() -> None:
    """Combining world blocks, constants and violations included, equals one whole-split call."""
    world_0 = [
        _snapshot("world-0", "acc_0", SNAPSHOT_AT),
        _snapshot("world-0", "acc_1", SNAPSHOT_AT),
        _snapshot("world-0", "acc_2", SNAPSHOT_AT),
        _snapshot("world-0", "acc_3", SNAPSHOT_AT),
    ]
    world_1 = [
        _snapshot("world-1", "acc_0", SNAPSHOT_AT),
        _snapshot("world-1", "acc_1", SNAPSHOT_AT),
        _snapshot("world-1", "acc_2", SNAPSHOT_AT),
    ]
    # Every row stores the same constant zero, so the split-wide distinct count must stay one even
    # though each world alone also sees one; only a union of the per-world value sets proves that.
    for row in [*world_0, *world_1]:
        row["shared_counterparties_with_flagged_7d"] = 0
    labels = [
        _label("world-0", "acc_0", ARRIVAL_AT),
        _label("world-1", "acc_0", ARRIVAL_AT),
    ]

    blocks = [
        check_label_features_by_arrival(world_0, labels),
        check_label_features_by_arrival(world_1, labels),
    ]
    value_sets = [feedback_distinct_values(world_0), feedback_distinct_values(world_1)]

    combined = combine_label_feature_checks(blocks, value_sets, 2)

    whole = check_label_features_by_arrival([*world_0, *world_1], labels)
    assert combined == {**whole, "world_count": 2}
    assert combined["status"] == "fail"
    assert combined["feedback_features"]["status"] == "no source data"
    assert combined["complained"]["violations"] == 5
    assert combined["complained"]["snapshots_checked"] == 7
    assert len(combined["complained"]["examples"]) == 5
    assert (
        combined["feedback_features"]["distinct_value_counts"]["shared_counterparties_with_flagged_7d"]
        == 1
    )


def test_feedback_values_that_differ_between_worlds_fail_the_split() -> None:
    """Each world constant but different from the other is still a value change across the split."""
    world_0 = [_snapshot("world-0", "acc_1", SNAPSHOT_AT, hops_complained=1, shared_flagged=0)]
    world_1 = [_snapshot("world-1", "acc_1", SNAPSHOT_AT, hops_complained=1, shared_flagged=1)]

    combined = combine_label_feature_checks(
        [
            check_label_features_by_arrival(world_0, []),
            check_label_features_by_arrival(world_1, []),
        ],
        [feedback_distinct_values(world_0), feedback_distinct_values(world_1)],
        2,
    )

    assert combined["status"] == "fail"
    assert combined["feedback_features"]["status"] == "fail"
    assert (
        combined["feedback_features"]["distinct_value_counts"]["shared_counterparties_with_flagged_7d"]
        == 2
    )
