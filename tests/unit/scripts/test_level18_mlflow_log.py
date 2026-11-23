"""
File: tests/unit/scripts/test_level18_mlflow_log.py
Purpose: Proves the Level 18 MLflow back-fill is idempotent (a second pass creates no runs), that a failed attempt is
         ended FAILED with no metrics, that a killed attempt is ended KILLED, and that each method's trial children
         are nested under that method's parent run.

The whole run is written into a throwaway SQLite database in tmp_path, with the working directory moved there so
MLflow's default ./mlruns artifacts never touch the repository.
"""

import importlib
import json
from pathlib import Path
from types import ModuleType

import pytest

pytestmark = pytest.mark.ml

FINGERPRINT = {
    "code_version": "abc1234",
    "mule_model_config_sha256": "config-sha",
    "training_dataset_ids": ["training-1"],
    "validation_dataset_ids": ["validation-1"],
    "test_dataset_ids": ["test-1"],
    "stress_dataset_ids": ["stress-1"],
}


@pytest.fixture(scope="module")
def log_script() -> ModuleType:
    pytest.importorskip("mlflow")
    return importlib.import_module("scripts.level18_mlflow_log")


def write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def stage_payload(stage: str, result: dict) -> dict:
    return {"stage": stage, "fingerprint": FINGERPRINT, "seconds": 1.0, "peak_rss_mb": 10.0, "result": result}


def tune_result(method: str, trials: list[dict]) -> dict:
    return {
        "best_score": 0.8,
        "chosen_params": {"max_depth": 3},
        "meets_budget": False,
        "param_importances": {"max_depth": 0.5},
        "pu_method": method,
        "stop_reason": "stalled",
        "stopped_early": True,
        "training_seconds": 1.0,
        "trials": trials,
        "trials_run": len(trials),
        "validation_rate_pct": 40.0,
        "validation_recall": 0.25,
        "validation_truth_average_precision": 0.8,
    }


def trials(start: int = 0, count: int = 2) -> list[dict]:
    return [
        {"number": start + index, "params": {"max_depth": index}, "score": 0.7 + index / 10} for index in range(count)
    ]


def make_stage5(stage5: Path) -> None:
    fit = stage5 / "final" / "fit0001"
    write_json(
        fit / "result.json",
        {
            "manifest": {
                "code_version": "abc1234",
                "datasets": [
                    {"dataset_id": "training-1", "split": "training"},
                    {"dataset_id": "test-1", "split": "test"},
                ],
            },
            "across_worlds": {
                "test": {
                    detector: {"recall_at_capacity": {"median": 0.64, "p10": 0.47, "p90": 0.83, "unit": "fraction"}}
                    for detector in ("anomaly", "rules", "rules_and_anomaly", "xgboost")
                }
            },
        },
    )
    capacity = {
        "0.25": {
            "model": {"median": 0.64, "world_count": 120},
            "normalized": {"median": 0.79, "world_count": 120},
            "oracle": {"median": 0.82, "world_count": 120},
        }
    }
    write_json(
        stage5 / "extra_reporting.json",
        {
            "manifest": {
                "headline_capacity_pct": 0.25,
                "capacity_points_pct": [0.05, 0.25],
                "xgboost_cut": 0.96,
                "amendment": "A-052",
                "consistency_gate": {"status": "pass", "worlds_checked": 120},
                "mule_model_config": {"sha256": "config-sha"},
            },
            "test": {"worlds_processed": 120, "world_cards_checked": 120, "aggregate": {"capacity": capacity}},
            "stress": {"worlds_processed": 120, "world_cards_checked": 120, "aggregate": {"capacity": capacity}},
        },
    )
    write_json(
        stage5 / "stage5_provenance.json",
        {"head": "abcdef", "record": {"mule_model_config_sha256": "config-sha"}},
    )
    write_json(
        stage5 / "leakage_checks_5_6.json",
        {
            "status": "pass",
            "check_5a": {"status": "pass"},
            "check_5b": {"status": "pass"},
            "check_6a": {"status": "pass"},
            "check_6b": {"status": "pass"},
        },
    )
    write_json(fit / "leakage_checks.json", {"checks": {"split_leak": {"status": "pass"}}})
    write_json(fit / "shap_top_features.json", {"status": "pass", "rows_sampled": 100})
    write_json(fit / "optuna_trials.json", {"best_params": {}, "best_score": 0.8, "trial_count": 2})
    write_json(fit / "validation_curves.json", {"curves": []})
    write_text(fit / "checkpoint" / "fit_record.json", "{}")
    write_text(fit / "checkpoint" / "thresholds.json", "{}")
    write_text(fit / "checkpoint" / "model_bags" / "member_000.ubj", "bag")
    write_text(fit / "account_model_bags" / "ensemble.json", "{}")
    write_text(fit / "account_model_bags" / "member_000.ubj", "bag")
    write_text(stage5 / "stages" / "final_mule_model.yaml", "parameters: {}\n")
    write_text(stage5 / "extra_run_a053.log", "a053 run\n")
    write_json(stage5 / "extra_reporting_before_a053_20261123.json", {"older": True})


def make_failed_yaml(yaml_path: Path) -> None:
    yaml_path.write_text(
        "attempts:\n"
        "  - name: run_a_crashed_20261120\n"
        "    folder: run_a_crashed_20261120\n"
        "    reason: crashed\n"
        "    fixed_by: PR #155\n"
        "    derived_from: handoff\n"
        "    status: FAILED\n"
        "    reason_recorded: true\n"
        "    evidence: null\n"
        "  - name: run_a320_stage5_zero_complaints_20261121\n"
        "    folder: run_a320_stage5_zero_complaints_20261121\n"
        "    reason: zero complaint\n"
        "    fixed_by: PR #161\n"
        "    derived_from: handoff\n"
        "    status: FAILED\n"
        "    reason_recorded: true\n"
        "    evidence: null\n"
        "  - name: account_identity_382_merged_accounts\n"
        "    folder: null\n"
        "    reason: merged accounts\n"
        "    fixed_by: PR #159\n"
        "    derived_from: handoff\n"
        "    status: FAILED\n"
        "    reason_recorded: true\n"
        "    evidence: no folder\n"
        "  - name: run_a_aborted_20261120\n"
        "    folder: run_a_aborted_20261120\n"
        "    reason: not recorded\n"
        "    fixed_by: null\n"
        "    derived_from: inventory\n"
        "    status: KILLED\n"
        "    reason_recorded: false\n"
        "    evidence: null\n",
        encoding="utf-8",
    )


def make_tree(tmp_path: Path) -> tuple[Path, Path]:
    runs_root = tmp_path / "runs"

    stages = runs_root / "run_a320" / "stages"
    write_json(
        stages / "maturity_check.json",
        stage_payload(
            "maturity_check",
            {
                "variants": [
                    {
                        "name": "drop_28",
                        "label_maturity_days": 28,
                        "label_trust_weighting": False,
                        "training_row_count": 100,
                        "training_positive_count": 5,
                        "validation_truth_average_precision": 0.5,
                    },
                    {
                        "name": "all_days",
                        "label_maturity_days": 0,
                        "label_trust_weighting": True,
                        "training_row_count": 200,
                        "training_positive_count": 9,
                        "validation_truth_average_precision": 0.6,
                    },
                ],
                "chosen_variant": "all_days",
                "reason": "best AP",
                "tolerance": 0.002,
            },
        ),
    )
    write_json(
        stages / "learning_curve.json",
        stage_payload(
            "learning_curve",
            {
                "points": [
                    {"training_world_count": 40, "validation_truth_average_precision": 0.2, "training_row_count": 10},
                    {"training_world_count": 80, "validation_truth_average_precision": 0.4, "training_row_count": 20},
                ],
                "doubling_improvements": [
                    {"from_training_worlds": 40, "to_training_worlds": 80, "average_precision_improvement": 0.2}
                ],
                "chosen_training_world_count": 80,
                "more_training_worlds_needed": False,
            },
        ),
    )
    for method in ("none", "bagging", "elkan_noto"):
        write_json(
            stages / f"tune_method_{method}.json",
            stage_payload(f"tune_method_{method}", tune_result(method, trials())),
        )
    write_json(
        runs_root / "run_a320_stage4_fixed_c" / "stage4_recomputed.json",
        {
            "chosen_method": "bagging",
            "methods": [
                {
                    "pu_method": "bagging",
                    "validation_rate_pct": 0.49,
                    "validation_recall": 0.87,
                    "meets_budget": True,
                    "retrained_validation_truth_average_precision": 0.83,
                    "saved_validation_rate_pct": 57.9,
                    "saved_validation_recall": 0.47,
                    "saved_meets_budget": False,
                    "saved_validation_truth_average_precision": 0.83,
                }
            ],
            "head": "abcdef",
            "source_code_version": "abc1234",
            "training_world_count": 80,
            "note": "recomputed",
        },
    )
    make_stage5(runs_root / "run_a320_stage5c")

    write_json(
        runs_root / "run_a_crashed_20261120" / "stages" / "maturity_check.json",
        stage_payload("maturity_check", {"variants": [], "chosen_variant": "x", "reason": "r", "tolerance": 0.0}),
    )
    write_text(
        runs_root / "run_a320_stage5_zero_complaints_20261121" / "stages" / "final_mule_model.yaml",
        "parameters: {}\n",
    )
    write_json(
        runs_root / "run_a320_stage5_zero_complaints_20261121" / "final" / "fit9" / "optuna_trials.json",
        {"trial_count": 1},
    )
    write_json(
        runs_root / "run_a_aborted_20261120" / "stages" / "learning_curve.json",
        stage_payload(
            "learning_curve",
            {
                "points": [],
                "doubling_improvements": [],
                "chosen_training_world_count": 0,
                "more_training_worlds_needed": False,
            },
        ),
    )

    yaml_path = tmp_path / "level18_failed_attempts.yaml"
    make_failed_yaml(yaml_path)
    return runs_root, yaml_path


def test_idempotent_second_pass_and_failed_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, log_script: ModuleType
) -> None:
    """A first pass logs every planned run; a second pass creates none; failed/killed runs carry no metrics; trial
    children are nested under their method parent."""
    runs_root, yaml_path = make_tree(tmp_path)
    monkeypatch.setenv("MLFLOW_TRACKING_URI", "sqlite:///" + (tmp_path / "mlflow.db").as_posix())
    monkeypatch.chdir(tmp_path)

    units = log_script.build_units(runs_root, yaml_path)
    planned = sum(log_script.plan_counts(units).values())

    first = log_script.log_units(units, runs_root)
    assert sum(first["created"].values()) == planned
    assert sum(first["skipped"].values()) == 0

    second = log_script.log_units(units, runs_root)
    assert sum(second["created"].values()) == 0
    assert sum(second["skipped"].values()) == planned

    import mlflow

    experiment = mlflow.get_experiment_by_name(log_script.EXPERIMENT)
    assert experiment is not None
    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id], output_format="list", max_results=10_000)
    by_name = {run.data.tags.get("mlflow.runName"): run for run in runs}

    assert len(runs) == planned

    crashed = by_name["run_a_crashed_20261120"]
    assert crashed.info.status == "FAILED"
    assert crashed.data.metrics == {}

    aborted = by_name["run_a_aborted_20261120"]
    assert aborted.info.status == "KILLED"
    assert aborted.data.metrics == {}

    account = by_name["account_identity_382_merged_accounts"]
    assert account.info.status == "FAILED"
    assert account.data.tags["fingerprint_source"] == "none"
    assert account.data.tags["code_version"] == "not recorded"
    assert account.data.tags["evidence"] == "no folder"

    parent = by_name["pu_search_none"]
    children = [run for run in runs if run.data.tags.get("mlflow.parentRunId") == parent.info.run_id]
    assert len(children) == 2
    assert all("score" in child.data.metrics for child in children)

    final = by_name["final_test_and_stress"]
    assert final.data.tags["fingerprint_source"] == "partial"
    assert final.data.tags["simulated"] == "true"
    assert final.data.tags["code_version"] == "abc1234"
    artifact_names = [item.path for item in mlflow.MlflowClient().list_artifacts(final.info.run_id)]
    assert "not_copied_files.json" in artifact_names
