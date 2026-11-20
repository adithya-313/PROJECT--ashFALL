"""
WHAT: Unit tests for the Level 18a4 sanity report: feature fill and concentration flagging, an
oracle scored through the pilot's escalation and metric path, and the CLI report.

WHY: A sparse or constant feature must be visible before a model is blamed; the oracle test proves
the report reaches full precision when the ranking is perfect.
"""

import json
from datetime import datetime, timezone
from pathlib import Path

import orjson
import pytest

from decisions.mule_account_model import account_model_feature_names, load_mule_model_config
from evaluation.dataset_truth_join import join_truth
from evaluation.level18_sanity import (
    build_sanity_report,
    feature_fill_report,
    load_escalation_repeat_policy,
    main,
    oracle_report,
)
from tests.unit.mule_model_config_fixture import label_trust_blocks_text, gpu_policy_blocks_text

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURES = REPO_ROOT / "showcase" / "fixtures" / "level_17"
VALIDATION = FIXTURES / "datasets" / "validation-20261031T1830-e5593f4c3f8e"
TRUTH_ROOT = FIXTURES / "truth"


def _varied_row(index: int) -> dict[str, object]:
    """Builds one snapshot row whose every feature value differs from the other rows."""
    return {name: float(index) for name in account_model_feature_names()}


def _config_body(flag_share: float) -> str:
    """Returns a complete mule model config body carrying the given fill flag share."""
    return (
        "parameters:\n"
        "  label_cut:\n"
        "    label: policy\n"
        "    value: arrival_time\n"
        "    derived_from: test\n"
        "    reason: test\n"
        "  label_maturity_days:\n"
        "    label: policy\n"
        "    value: null\n"
        "    not_found: true\n"
        "    derived_from: test\n"
        "    reason: test\n"
        + label_trust_blocks_text()
        + gpu_policy_blocks_text()
        + "  feature_fill_flag_share:\n"
        "    label: policy\n"
        f"    value: {flag_share}\n"
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
        "    reason: test\n"
    )


def test_feature_fill_flags_a_feature_missing_from_every_row() -> None:
    """A feature that is None on every row is flagged with a zero fill share."""
    rows = [_varied_row(index) for index in range(10)]
    for row in rows:
        row["distinct_payers_24h"] = None

    report = feature_fill_report(rows)

    assert report["feature_fill_flag_share"] == load_mule_model_config().feature_fill_flag_share
    assert report["row_count"] == 10
    assert report["flagged_features"] == [
        {
            "feature": "distinct_payers_24h",
            "rows": 10,
            "fill_share": 0.0,
            "missing_share": 1.0,
            "most_common_value": None,
            "most_common_share": 1.0,
            "distinct_count": 1,
        }
    ]


def test_feature_fill_flags_a_constant_feature_at_the_share() -> None:
    """A feature that is 0 in 999 of 1000 rows is flagged at the 0.999 boundary; 998 is not."""
    rows = [_varied_row(index) for index in range(1000)]
    for row in rows:
        row["distinct_payers_7d"] = 0.0
        row["distinct_payers_24h"] = 0.0
    rows[-1]["distinct_payers_7d"] = 1.0
    for row in rows[-2:]:
        row["distinct_payers_24h"] = 1.0

    report = feature_fill_report(rows)

    flagged = {entry["feature"]: entry for entry in report["flagged_features"]}
    assert set(flagged) == {"distinct_payers_7d"}
    assert flagged["distinct_payers_7d"]["most_common_value"] == 0.0
    assert flagged["distinct_payers_7d"]["most_common_share"] == 0.999
    assert flagged["distinct_payers_7d"]["distinct_count"] == 2


def test_feature_fill_lists_a_mostly_missing_varied_feature_without_flagging_it() -> None:
    """A feature missing from 600 of 1000 rows with varied values is listed, not flagged."""
    rows = [_varied_row(index) for index in range(1000)]
    for row in rows[:600]:
        row["distinct_payers_24h"] = None

    report = feature_fill_report(rows)

    assert "distinct_payers_24h" not in {entry["feature"] for entry in report["flagged_features"]}
    assert {"feature": "distinct_payers_24h", "missing_share": 0.6, "fill_share": 0.4} in report[
        "mostly_missing_features"
    ]


def test_feature_fill_reads_the_flag_share_from_config(tmp_path: Path) -> None:
    """The same rows are unflagged at the committed 0.999 and flagged at a 0.5 config."""
    rows = [_varied_row(index) for index in range(1000)]
    for row in rows[:600]:
        row["distinct_payers_24h"] = 7.0
    config_path = tmp_path / "mule_model.yaml"
    config_path.write_text(_config_body(0.5), encoding="utf-8")

    default_report = feature_fill_report(rows)
    lowered_report = feature_fill_report(rows, config_path=config_path)

    assert default_report["feature_fill_flag_share"] == 0.999
    assert "distinct_payers_24h" not in {entry["feature"] for entry in default_report["flagged_features"]}
    assert lowered_report["feature_fill_flag_share"] == 0.5
    assert "distinct_payers_24h" in {entry["feature"] for entry in lowered_report["flagged_features"]}


def test_oracle_reaches_full_yield_through_the_pilot_path() -> None:
    """With perfect ranking every escalated account-day is a mule day, so measured yield is 1.0."""
    payments, snapshots = join_truth(VALIDATION, TRUTH_ROOT)
    report = oracle_report(snapshots.to_pylist(), payments.to_pylist())

    assert report["escalations"] > 0
    assert report["mule_accounts"] > 0
    assert report["yield_measured"] == 1.0
    assert report["recall_at_capacity"] > 0.0


def _snapshot_row(account_id: str, day: int, role: str, depth: int | None) -> dict[str, object]:
    """Builds one account-day row for a hand-built world with no rule-triggering features."""
    return {
        "world_id": "world-0",
        "bank_id": "bank_0",
        "account_id": account_id,
        "snapshot_time": datetime(2026, 9, day, 12, 0, tzinfo=timezone.utc),
        "snapshot_kind": "daily",
        "truth_role": role,
        "truth_depth": depth,
    }


def _payment_row(payment_id: str, receiver: str, role: str = "mule") -> dict[str, object]:
    """Builds one joined payment row naming a receiver and its truth role."""
    return {
        "payment_id": payment_id,
        "receiver_account_id": receiver,
        "decision_time": datetime(2026, 9, 1, 0, 0, tzinfo=timezone.utc),
        "truth_role_receiver": role,
    }


def test_oracle_misses_count_repeats_and_blocked_capacity_slots() -> None:
    """At capacity one the alphabetically first mule takes every slot, so the second mule is missed."""
    snapshots = [
        row
        for day in (1, 2)
        for row in (
            _snapshot_row("acc_a", day, "mule", 1),
            _snapshot_row("acc_b", day, "mule", 1),
            _snapshot_row("acc_g", day, "genuine", None),
        )
    ]
    payments = [_payment_row("p1", "acc_a"), _payment_row("p2", "acc_b"), _payment_row("p3", "acc_c")]

    report = oracle_report(snapshots, payments)

    misses = report["misses"]
    assert report["recall_at_capacity"] == 0.5
    assert misses["ground_set_mules"] == 2
    assert misses["truth_mule_accounts"] == 3
    assert misses["found_mules"] == 1
    assert misses["missed_with_mule_days"] == ["acc_b"]
    assert misses["missed_without_mule_day"] == ["acc_c"]
    assert misses["blocked_slots_on_missed_mule_days"] == 1
    assert misses["missed_mule_day_slots"] == 2
    assert misses["blocked_slot_share"] == 0.5
    assert misses["total_escalations"] == 2
    assert misses["repeat_escalations"] == 1
    assert misses["repeat_escalation_share"] == 0.5


def test_oracle_misses_count_a_mule_whose_snapshots_never_say_mule() -> None:
    """A mule named only by payment truth is reported as missed without a mule-day snapshot."""
    snapshots = [
        _snapshot_row("acc_a", 1, "mule", 1),
        _snapshot_row("acc_c", 1, "genuine", None),
    ]
    payments = [_payment_row("p1", "acc_a"), _payment_row("p2", "acc_c")]

    report = oracle_report(snapshots, payments)

    misses = report["misses"]
    assert report["recall_at_capacity"] == 1.0
    assert misses["ground_set_mules"] == 1
    assert misses["found_mules"] == 1
    assert misses["missed_with_mule_days"] == []
    assert misses["missed_without_mule_day"] == ["acc_c"]


def test_build_sanity_report_covers_every_validation_world() -> None:
    """The report names the dataset and carries one oracle block per world in its manifest."""
    report = build_sanity_report((VALIDATION,), TRUTH_ROOT)
    manifest = json.loads((VALIDATION / "manifest.json").read_text(encoding="utf-8"))

    assert report["validation_datasets"] == [VALIDATION.name]
    assert [block["world_id"] for block in report["oracle_by_world"]] == [
        str(world["world_id"]) for world in manifest["worlds"]
    ]
    assert report["feature_fill"]["row_count"] > 0


def test_sanity_report_uses_the_configured_repeat_policy(tmp_path: Path) -> None:
    """The committed once_per_world config beats every_day; a temp every_day config reproduces 0.7692."""
    legacy_config = tmp_path / "evaluation.yaml"
    legacy_config.write_text("escalation_repeats:\n  value: every_day\n", encoding="utf-8")

    current = build_sanity_report((VALIDATION,), TRUTH_ROOT)
    legacy = build_sanity_report((VALIDATION,), TRUTH_ROOT, evaluation_config_path=legacy_config)

    assert load_escalation_repeat_policy() == "once_per_world"
    assert current["oracle_by_world"][0]["oracle"]["recall_at_capacity"] == pytest.approx(0.8461538461538461)
    assert current["oracle_by_world"][0]["oracle"]["escalations"] == 11
    assert legacy["oracle_by_world"][0]["oracle"]["recall_at_capacity"] == pytest.approx(0.7692307692307693)
    assert legacy["oracle_by_world"][0]["oracle"]["escalations"] == 140


def test_escalation_repeat_policy_refuses_an_unknown_value(tmp_path: Path) -> None:
    """A policy that is not allowed is an error, never a silent fallback."""
    bad_config = tmp_path / "evaluation.yaml"
    bad_config.write_text("escalation_repeats:\n  value: never_again\n", encoding="utf-8")

    with pytest.raises(ValueError, match="escalation_repeats value must be one of"):
        load_escalation_repeat_policy(bad_config)


def test_cli_writes_the_report(tmp_path: Path) -> None:
    """The CLI writes the same report structure as build_sanity_report."""
    out = tmp_path / "sanity.json"

    exit_code = main(
        [
            "--validation",
            str(VALIDATION),
            "--truth-root",
            str(TRUTH_ROOT),
            "--out",
            str(out),
        ]
    )
    payload = orjson.loads(out.read_bytes())

    assert exit_code == 0
    assert set(payload) == {"feature_fill", "oracle_by_world", "validation_datasets"}
    assert payload["oracle_by_world"][0]["oracle"]["yield_measured"] == 1.0
