"""
File: tests/unit/scripts/test_level18_mlflow_read.py
Purpose: Proves the Level 18 result readers: a missing stage key raises naming the key, a "not run" importances
         string survives unchanged, trial order is kept, a missing failed-attempt folder raises, a folder with no
         stage file yields "not recorded" with fingerprint_source "none", and disagreeing fingerprints raise.

Every fixture here is a tiny hand-built JSON file in tmp_path; no real run data is read.
"""

import json
from pathlib import Path

import pytest

from scripts.level18_mlflow_read import (
    NOT_RECORDED,
    final_fingerprint_tags,
    final_results,
    fingerprint_tags,
    learning_curve_points,
    maturity_variants,
    pu_method_search,
    read_stage_file,
    single_subdir,
    stage4_recomputed,
    stage_files_with_fingerprint,
)

FINGERPRINT = {
    "code_version": "abc1234",
    "mule_model_config_sha256": "config-sha",
    "training_dataset_ids": ["training-1"],
    "validation_dataset_ids": ["validation-1"],
    "test_dataset_ids": ["test-1"],
    "stress_dataset_ids": ["stress-1"],
}


def write_json(path: Path, payload: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def stage_payload(stage: str, result: dict, fingerprint: dict | None = None) -> dict:
    return {
        "stage": stage,
        "fingerprint": FINGERPRINT if fingerprint is None else fingerprint,
        "seconds": 1.0,
        "peak_rss_mb": 10.0,
        "result": result,
    }


def pu_result(importances, trials) -> dict:
    return {
        "best_score": 0.8,
        "chosen_params": {"max_depth": 3},
        "meets_budget": False,
        "param_importances": importances,
        "pu_method": "none",
        "stop_reason": "stalled",
        "stopped_early": True,
        "training_seconds": 1.0,
        "trials": trials,
        "trials_run": len(trials),
        "validation_rate_pct": 40.0,
        "validation_recall": 0.25,
        "validation_truth_average_precision": 0.8,
    }


def test_read_stage_file_missing_key_raises(tmp_path: Path) -> None:
    """Dropping the result key makes read_stage_file raise a ValueError that names the missing key."""
    payload = stage_payload("maturity_check", {"variants": []})
    del payload["result"]
    path = write_json(tmp_path / "stage.json", payload)

    with pytest.raises(ValueError, match="missing key 'result'"):
        read_stage_file(path)


def test_not_run_importances_string_survives_unchanged(tmp_path: Path) -> None:
    """A "not run: ..." importances string is returned exactly, never coerced to a dict or emptied."""
    message = "not run: no stress worlds supplied"
    path = write_json(tmp_path / "tune.json", stage_payload("tune_method_none", pu_result(message, [])))

    search = pu_method_search(read_stage_file(path))

    assert search.param_importances == message
    assert isinstance(search.param_importances, str)


def test_trial_order_is_kept(tmp_path: Path) -> None:
    """Trials come back in file order, not re-sorted by number."""
    trials = [
        {"number": 2, "params": {"max_depth": 5}, "score": 0.5},
        {"number": 0, "params": {"max_depth": 3}, "score": 0.7},
        {"number": 1, "params": {"max_depth": 4}, "score": 0.6},
    ]
    path = write_json(tmp_path / "tune.json", stage_payload("tune_method_none", pu_result({"max_depth": 0.5}, trials)))

    search = pu_method_search(read_stage_file(path))

    assert tuple(trial.number for trial in search.trials) == (2, 0, 1)
    assert search.trials_run == 3


def test_missing_failed_attempt_folder_raises(tmp_path: Path) -> None:
    """A yaml entry whose folder is not on disk raises a ValueError naming the folder."""
    from scripts.level18_mlflow_read import failed_attempts

    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    yaml_path = tmp_path / "failed.yaml"
    yaml_path.write_text(
        "attempts:\n"
        "  - name: gone\n"
        "    folder: does_not_exist\n"
        "    reason: r\n"
        "    fixed_by: null\n"
        "    derived_from: d\n"
        "    status: FAILED\n"
        "    reason_recorded: true\n"
        "    evidence: null\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="does not exist"):
        failed_attempts(yaml_path, runs_root)


def test_no_stage_file_gives_not_recorded_and_none(tmp_path: Path) -> None:
    """A folder with no stage file yields "not recorded" for all three values and fingerprint_source "none"."""
    folder = tmp_path / "attempt"
    (folder / "stages").mkdir(parents=True)

    stage_files = stage_files_with_fingerprint(folder)
    tags = fingerprint_tags(stage_files)

    assert stage_files == ()
    assert tags.code_version == NOT_RECORDED
    assert tags.config_sha256 == NOT_RECORDED
    assert tags.dataset_ids == NOT_RECORDED
    assert tags.fingerprint_source == "none"
    for value in (tags.code_version, tags.config_sha256, tags.dataset_ids):
        assert value == "not recorded"
        assert value is not None


def test_disagreeing_fingerprints_raise(tmp_path: Path) -> None:
    """Two stage files that disagree on code_version raise, naming the field."""
    first = write_json(tmp_path / "a.json", stage_payload("maturity_check", {"variants": []}))
    other_fingerprint = {**FINGERPRINT, "code_version": "different"}
    second = write_json(
        tmp_path / "b.json", stage_payload("learning_curve", {"points": []}, fingerprint=other_fingerprint)
    )

    with pytest.raises(ValueError, match="code_version"):
        fingerprint_tags((first, second))


def test_single_subdir_requires_exactly_one(tmp_path: Path) -> None:
    """Exactly one subfolder is required; two or zero raise a ValueError."""
    final = tmp_path / "final"
    (final / "one").mkdir(parents=True)
    (final / "two").mkdir()

    with pytest.raises(ValueError, match="expected exactly one subfolder"):
        single_subdir(final)

    empty = tmp_path / "empty"
    empty.mkdir()
    with pytest.raises(ValueError, match="expected exactly one subfolder"):
        single_subdir(empty)


def test_maturity_learning_and_stage4_readers(tmp_path: Path) -> None:
    """The maturity, learning-curve and stage-4 readers return the recorded values."""
    maturity = write_json(
        tmp_path / "maturity_check.json",
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
    learning = write_json(
        tmp_path / "learning_curve.json",
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
    stage4 = write_json(
        tmp_path / "stage4.json",
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

    report = maturity_variants(read_stage_file(maturity))
    curve = learning_curve_points(read_stage_file(learning))
    recomputed = stage4_recomputed(stage4)

    assert [variant.name for variant in report.variants] == ["drop_28", "all_days"]
    assert report.chosen_variant == "all_days"
    assert report.tolerance == 0.002
    assert [point.training_world_count for point in curve.points] == [40, 80]
    assert curve.chosen_training_world_count == 80
    assert curve.more_training_worlds_needed is False
    assert recomputed.chosen_method == "bagging"
    assert recomputed.methods[0].pu_method == "bagging"
    assert recomputed.methods[0].meets_budget is True


def build_stage5_tree(stage5: Path, fit_id: str = "fit0001") -> Path:
    """Write the minimal stage-5 file set the reader needs (result, extra reporting, provenance, leakage, shap)."""
    config_sha = "config-sha"
    fit = stage5 / "final" / fit_id
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
                    "xgboost": {
                        "recall_at_capacity": {"median": 0.64, "p10": 0.47, "p90": 0.83, "unit": "fraction"},
                    }
                }
            },
        },
    )
    write_json(
        stage5 / "extra_reporting.json",
        {
            "manifest": {
                "headline_capacity_pct": 0.25,
                "capacity_points_pct": [0.05, 0.25],
                "xgboost_cut": 0.96,
                "amendment": "A-052",
                "consistency_gate": {"status": "pass", "worlds_checked": 120},
                "mule_model_config": {"sha256": config_sha},
            },
            "test": {
                "worlds_processed": 120,
                "world_cards_checked": 120,
                "aggregate": {
                    "capacity": {
                        "0.25": {
                            "model": {"median": 0.64, "world_count": 120},
                            "normalized": {"median": 0.79, "world_count": 120},
                            "oracle": {"median": 0.82, "world_count": 120},
                        }
                    }
                },
            },
            "stress": {
                "worlds_processed": 120,
                "world_cards_checked": 120,
                "aggregate": {
                    "capacity": {
                        "0.25": {
                            "model": {"median": 0.63, "world_count": 120},
                            "normalized": {"median": 0.85, "world_count": 120},
                            "oracle": {"median": 0.78, "world_count": 120},
                        }
                    }
                },
            },
        },
    )
    write_json(
        stage5 / "stage5_provenance.json",
        {"head": "abcdef", "record": {"mule_model_config_sha256": config_sha}},
    )
    statuses = {
        "check_5a": {"status": "pass"},
        "check_5b": {"status": "pass"},
        "check_6a": {"status": "pass"},
        "check_6b": {"status": "pass"},
    }
    write_json(stage5 / "leakage_checks_5_6.json", {"status": "pass", **statuses})
    write_json(fit / "leakage_checks.json", {"checks": {"split_leak": {"status": "pass"}}})
    write_json(fit / "shap_top_features.json", {"status": "pass", "rows_sampled": 100})
    return stage5


def test_final_results_reads_headline_and_marks_partial(tmp_path: Path) -> None:
    """The final results are read from final/<fit id>/ and their base tags are "partial", not a stage file."""
    stage5 = build_stage5_tree(tmp_path / "stage5")

    results = final_results(stage5)
    tags = final_fingerprint_tags(results)

    assert results.fit_dir.name == "fit0001"
    assert results.code_version == "abc1234"
    assert results.config_sha256 == "config-sha"
    assert results.headline_capacity_pct == 0.25
    assert results.test_result_medians["xgboost"]["recall_at_capacity"] == 0.64
    assert results.test_capacity["0.25"]["oracle"]["median"] == 0.82
    assert results.leakage_5_6_status == "pass"
    assert results.shap_rows_sampled == 100
    assert tags.fingerprint_source == "partial"
    assert tags.dataset_ids == "training-1,test-1"


def test_final_results_requires_single_fit_dir(tmp_path: Path) -> None:
    """Two folders under final/ make the reader refuse instead of guessing."""
    build_stage5_tree(tmp_path / "stage5", fit_id="one")
    (tmp_path / "stage5" / "final" / "two").mkdir()

    with pytest.raises(ValueError, match="expected exactly one subfolder"):
        final_results(tmp_path / "stage5")
