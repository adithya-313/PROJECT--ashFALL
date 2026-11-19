"""
WHAT: Integration tests for the Level 18 pilot runner on the committed Level 17 fixtures: the full
run produces all four scorers, the leakage report, the SHAP summary and the per-rule manifest, and
an overlapping split is refused.

WHY: The pilot is only trustworthy if its plumbing works end to end and a split leak cannot slip
through. Tiny fixtures are almost all fraud, so these tests assert structure, never metric values.
"""

import dataclasses
import hashlib
import json
import shutil
from pathlib import Path

import pyarrow.parquet as pq
import pytest
import yaml

import decisions.mule_account_model as mule_account_model
from core.event_clock import IST
from decisions.mule_account_model import _one_row_per_account_day
from decisions.rules_config import load_rules_config
from evaluation.level17_scoring import one_row_per_account_day
from evaluation.run_level_18_pilot import main, run_pilot
from showcase.level_17_checks import TEST_DATASET, TRAINING_DATASET, VALIDATION_DATASET

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURES = REPO_ROOT / "showcase" / "fixtures" / "level_17"
TRAINING = TRAINING_DATASET
VALIDATION = VALIDATION_DATASET
TEST = TEST_DATASET
TRUTH_ROOT = FIXTURES / "truth"
WORLD_CARDS_ROOT = FIXTURES / "world_cards"


def _single_result_dir(out_root: Path) -> Path:
    """Returns the one result folder the runner wrote under out_root."""
    directories = [path for path in out_root.iterdir() if path.is_dir()]
    assert len(directories) == 1
    return directories[0]


def test_pilot_run_on_fixtures_writes_the_expected_files(tmp_path: Path) -> None:
    """A full fixture run finishes and writes result.json, leakage_checks.json and SHAP output."""
    exit_code = run_pilot(
        [TRAINING],
        [VALIDATION],
        [TEST],
        TRUTH_ROOT,
        WORLD_CARDS_ROOT,
        tmp_path,
    )

    assert exit_code == 0
    result_dir = _single_result_dir(tmp_path)
    result = json.loads((result_dir / "result.json").read_text(encoding="utf-8"))
    assert result["simulated"] is True
    assert result["pilot"] is True
    assert set(result["per_world"]["test-0000"]["scorers"]) == {
        "rules",
        "anomaly",
        "rules_and_anomaly",
        "xgboost",
    }
    assert set(result["across_worlds"]["test"]) == {
        "rules",
        "anomaly",
        "rules_and_anomaly",
        "xgboost",
    }

    leakage = json.loads((result_dir / "leakage_checks.json").read_text(encoding="utf-8"))
    assert leakage["pilot"] is True
    assert leakage["checks"]["split_leak"]["status"] == "pass"
    assert leakage["checks"]["truth_leak"]["status"] == "pass"
    assert leakage["checks"]["section_23_checks_4_to_7"]["status"] == "not run"

    shap_payload = json.loads((result_dir / "shap_top_features.json").read_text(encoding="utf-8"))
    assert shap_payload["pilot"] is True
    assert shap_payload["status"] in {"run", "not run"}
    if shap_payload["status"] == "run":
        assert len(shap_payload["top_features"]) == 10
    else:
        assert isinstance(shap_payload["reason"], str)

    per_rule = result["manifest"]["per_rule_standalone"]
    expected_reason_codes = [rule.reason_code for rule in load_rules_config().rules]
    assert [entry["reason_code"] for entry in per_rule] == expected_reason_codes
    assert len(per_rule) == 5


def test_switch_off_records_check_7_as_not_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """With the label network switch off the pilot records check 7 as not run instead of crashing."""
    committed = mule_account_model.load_mule_model_config()
    switched_off = dataclasses.replace(committed, use_label_network_features=False)
    monkeypatch.setattr(
        mule_account_model, "load_mule_model_config", lambda path=None: switched_off
    )

    exit_code = run_pilot(
        [TRAINING],
        [VALIDATION],
        [TEST],
        TRUTH_ROOT,
        WORLD_CARDS_ROOT,
        tmp_path,
    )

    assert exit_code == 0
    leakage = json.loads(
        (_single_result_dir(tmp_path) / "leakage_checks.json").read_text(encoding="utf-8")
    )
    assert leakage["checks"]["section_23_check_7"] == {
        "status": "not run",
        "reason": "label network features are off",
    }


def test_local_account_day_selection_matches_evaluation_selection() -> None:
    """The decisions-local copy selects the same account-day rows in the same order as evaluation."""
    snapshots = pq.read_table(TEST / "snapshots.parquet").to_pylist()

    local_rows = _one_row_per_account_day(snapshots)
    evaluation_rows = one_row_per_account_day(snapshots)

    def row_key(row: dict[str, object]) -> tuple[str, str, object]:
        moment = row["snapshot_time"]
        if not hasattr(moment, "astimezone"):
            raise AssertionError("fixture snapshot_time is not a datetime")
        return (
            str(row["bank_id"]),
            str(row["account_id"]),
            moment.astimezone(IST).date(),
        )

    assert [row_key(row) for row in local_rows] == [row_key(row) for row in evaluation_rows]
    assert all(
        local_row is evaluation_row
        for local_row, evaluation_row in zip(local_rows, evaluation_rows, strict=True)
    )


def test_main_with_switch_off_config_records_the_override_and_clears_it(tmp_path: Path) -> None:
    """--mule-model-config is recorded in the manifest, used by the run, and cleared afterwards."""
    raw = yaml.safe_load((REPO_ROOT / "config" / "mule_model.yaml").read_text(encoding="utf-8"))
    raw["parameters"]["use_label_network_features"]["value"] = False
    switch_off = tmp_path / "mule_model_off.yaml"
    switch_off.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    expected_sha256 = hashlib.sha256(switch_off.read_bytes()).hexdigest()

    exit_code = main(
        [
            "--training",
            str(TRAINING),
            "--validation",
            str(VALIDATION),
            "--test",
            str(TEST),
            "--truth-root",
            str(TRUTH_ROOT),
            "--world-cards-root",
            str(WORLD_CARDS_ROOT),
            "--out-root",
            str(tmp_path),
            "--mule-model-config",
            str(switch_off),
        ]
    )

    assert exit_code == 0
    result_dir = _single_result_dir(tmp_path)
    result = json.loads((result_dir / "result.json").read_text(encoding="utf-8"))
    recorded = result["manifest"]["mule_model_config"]
    assert recorded["overridden"] is True
    assert recorded["use_label_network_features"] is False
    assert recorded["sha256"] == expected_sha256

    leakage = json.loads((result_dir / "leakage_checks.json").read_text(encoding="utf-8"))
    assert leakage["checks"]["section_23_check_7"] == {
        "status": "not run",
        "reason": "label network features are off",
    }

    committed = (REPO_ROOT / "config" / "mule_model.yaml").resolve()
    assert mule_account_model.active_mule_model_config_path().resolve() == committed


def test_overlapping_test_and_validation_splits_are_refused(tmp_path: Path) -> None:
    """A world that appears in both validation and test must stop the run with a failed check."""
    validation_copy = tmp_path / "validation-copy"
    shutil.copytree(TEST, validation_copy)
    manifest_path = validation_copy / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["split"] = "validation"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    out_root = tmp_path / "out"
    exit_code = run_pilot(
        [TRAINING],
        [validation_copy],
        [TEST],
        TRUTH_ROOT,
        WORLD_CARDS_ROOT,
        out_root,
    )

    assert exit_code == 1
    leakage_paths = list(out_root.rglob("leakage_checks.json"))
    assert len(leakage_paths) == 1
    leakage = json.loads(leakage_paths[0].read_text(encoding="utf-8"))
    assert leakage["checks"]["split_leak"]["status"] == "fail"
    assert leakage["checks"]["split_leak"]["overlaps"]


def _write_rules_switch_config(tmp_path: Path, rules_switch: bool) -> Path:
    """Copies the committed mule model config with the rule_6 switch set as given."""
    raw = yaml.safe_load((REPO_ROOT / "config" / "mule_model.yaml").read_text(encoding="utf-8"))
    raw["parameters"]["rules_plus_labels_and_registry"]["value"] = rules_switch
    config_path = tmp_path / f"mule_model_rules_{rules_switch}.yaml"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return config_path


def _run_main_with_mule_config(tmp_path: Path, config_path: Path) -> int:
    """Runs the pilot CLI on the committed fixtures with the given mule model config."""
    return main(
        [
            "--training",
            str(TRAINING),
            "--validation",
            str(VALIDATION),
            "--test",
            str(TEST),
            "--truth-root",
            str(TRUTH_ROOT),
            "--world-cards-root",
            str(WORLD_CARDS_ROOT),
            "--out-root",
            str(tmp_path),
            "--mule-model-config",
            str(config_path),
        ]
    )


def test_switch_on_records_rules_1_to_6_and_rule_6_standalone(tmp_path: Path) -> None:
    """With the rule_6 switch on the manifest records six rules and both rules config hashes."""
    config_path = _write_rules_switch_config(tmp_path, rules_switch=True)

    exit_code = _run_main_with_mule_config(tmp_path, config_path)

    assert exit_code == 0
    result = json.loads((_single_result_dir(tmp_path) / "result.json").read_text(encoding="utf-8"))
    manifest = result["manifest"]
    assert manifest["rules_variant"] == "rules_1_to_6"
    assert manifest["mule_model_config"]["rules_plus_labels_and_registry"] is True
    reason_codes = [entry["reason_code"] for entry in manifest["per_rule_standalone"]]
    assert len(reason_codes) == 6
    assert "RB6_LABEL_OR_REGISTRY_NEIGHBOUR" in reason_codes

    recorded_files = manifest["rules_config_files"]
    assert [entry["path"] for entry in recorded_files] == [
        "config/rules_baseline.yaml",
        "config/rules_labels_extension.yaml",
    ]
    extension_path = REPO_ROOT / "config" / "rules_labels_extension.yaml"
    assert recorded_files[1]["sha256"] == hashlib.sha256(extension_path.read_bytes()).hexdigest()


def test_switch_off_records_rules_1_to_5_without_rule_6(tmp_path: Path) -> None:
    """With the rule_6 switch off the manifest records five rules and only the baseline hash."""
    config_path = _write_rules_switch_config(tmp_path, rules_switch=False)

    exit_code = _run_main_with_mule_config(tmp_path, config_path)

    assert exit_code == 0
    result = json.loads((_single_result_dir(tmp_path) / "result.json").read_text(encoding="utf-8"))
    manifest = result["manifest"]
    assert manifest["rules_variant"] == "rules_1_to_5"
    assert manifest["mule_model_config"]["rules_plus_labels_and_registry"] is False
    reason_codes = [entry["reason_code"] for entry in manifest["per_rule_standalone"]]
    assert len(reason_codes) == 5
    assert "RB6_LABEL_OR_REGISTRY_NEIGHBOUR" not in reason_codes

    recorded_files = manifest["rules_config_files"]
    assert [entry["path"] for entry in recorded_files] == ["config/rules_baseline.yaml"]
    baseline_path = REPO_ROOT / "config" / "rules_baseline.yaml"
    assert recorded_files[0]["sha256"] == hashlib.sha256(baseline_path.read_bytes()).hexdigest()


def _write_pu_config(tmp_path: Path, method: str, n_bags: int | None = None) -> Path:
    """Copies the committed mule model config with the PU method and optional bag count set."""
    raw = yaml.safe_load((REPO_ROOT / "config" / "mule_model.yaml").read_text(encoding="utf-8"))
    raw["parameters"]["pu_method"]["value"] = method
    if n_bags is not None:
        raw["parameters"]["pu_bagging_n_bags"]["value"] = n_bags
    config_path = tmp_path / f"mule_model_{method}.yaml"
    config_path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return config_path


def test_none_method_records_pu_method_none_and_the_previous_keys(tmp_path: Path) -> None:
    """The default method records pu_method none beside every key the supervised manifest had."""
    exit_code = run_pilot(
        [TRAINING],
        [VALIDATION],
        [TEST],
        TRUTH_ROOT,
        WORLD_CARDS_ROOT,
        tmp_path,
    )

    assert exit_code == 0
    result = json.loads((_single_result_dir(tmp_path) / "result.json").read_text(encoding="utf-8"))
    pilot = result["manifest"]["pilot_account_model"]
    assert pilot["pu_method"] == "none"
    assert pilot["pu"]["method"] == "none"
    assert set(pilot["n_estimators_scores"]) == {"100", "200", "400"}
    for key in (
        "chosen_n_estimators",
        "chosen_validation_average_precision",
        "scale_pos_weight",
        "training_row_count",
        "training_positive_count",
        "seed",
        "feature_count",
        "model_path",
        "model_sha256",
        "model_sidecar_path",
        "model_sidecar_sha256",
        "xgboost_threshold",
        "xgboost_validation_rate_pct",
        "xgboost_validation_recall",
        "xgboost_meets_budget",
    ):
        assert key in pilot


def test_bagging_method_records_the_candidate_table_and_member_seeds(tmp_path: Path) -> None:
    """A bagging run scores every candidate, saves the members and records their seeds."""
    config_path = _write_pu_config(tmp_path, "bagging", n_bags=2)

    exit_code = _run_main_with_mule_config(tmp_path, config_path)

    assert exit_code == 0
    result_dir = _single_result_dir(tmp_path)
    result = json.loads((result_dir / "result.json").read_text(encoding="utf-8"))
    pilot = result["manifest"]["pilot_account_model"]
    pu = pilot["pu"]
    assert pilot["pu_method"] == "bagging"
    assert pu["n_bags"] == 2
    assert len(pu["member_seeds"]) == 2
    assert pu["chosen_unlabelled_per_positive"] in (1, 5, 20)
    assert pu["chosen_n_estimators"] in (100, 200, 400)
    assert len(pu["average_precision_scores"]) == 9
    assert len(pu["training_seconds"]) == 9
    ensemble_path = result_dir / "account_model_bags" / "ensemble.json"
    assert ensemble_path.is_file()
    sidecar = json.loads(ensemble_path.read_text(encoding="utf-8"))
    assert sidecar["n_bags"] == 2
    assert len(sidecar["member_seeds"]) == 2


def test_elkan_noto_method_records_c_weights_and_clip_count(tmp_path: Path) -> None:
    """An Elkan-Noto run records c, the weight summary and the clip count for the chosen model."""
    config_path = _write_pu_config(tmp_path, "elkan_noto")

    exit_code = _run_main_with_mule_config(tmp_path, config_path)

    assert exit_code == 0
    result = json.loads((_single_result_dir(tmp_path) / "result.json").read_text(encoding="utf-8"))
    pilot = result["manifest"]["pilot_account_model"]
    pu = pilot["pu"]
    assert pilot["pu_method"] == "elkan_noto"
    assert pu["chosen_n_estimators"] in (100, 200, 400)
    assert pu["c"] is not None
    assert 0.0 < pu["c"] <= 1.0
    assert set(pu["weight_summary"]) == {"min", "median", "max"}
    assert 0.0 <= pu["weight_summary"]["min"] <= pu["weight_summary"]["max"] <= 1.0
    assert isinstance(pu["clip_count"], int)
    assert len(pu["average_precision_scores"]) == 3
    assert len(pu["training_seconds"]) == 3
