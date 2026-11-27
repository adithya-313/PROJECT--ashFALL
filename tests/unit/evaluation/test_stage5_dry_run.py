"""
WHAT: Tests that the stage-5 dry run stops before any test or stress data is read and says so.

WHY: The dry run is the gate before a five-hour run, and it must be unable to touch the final exam.
These tests make the test and stress loaders raise, so a dry run that reaches them fails loudly.
"""

import json
from pathlib import Path
from typing import Any

import pytest

import evaluation.run_level_18_pilot as pilot_module
from decisions.mule_account_model import active_mule_model_config_path
from evaluation.run_level_18_pilot import run_pilot
from evaluation.stage5_dry_run import main
from showcase.level_17_checks import TEST_DATASET, TRAINING_DATASET, VALIDATION_DATASET

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURES = REPO_ROOT / "showcase" / "fixtures" / "level_17"
TRUTH_ROOT = FIXTURES / "truth"
WORLD_CARDS_ROOT = FIXTURES / "world_cards"
COMMITTED_CONFIG = REPO_ROOT / "config" / "mule_model.yaml"


def _forbid_test_and_stress_loading(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Makes any test or stress dataset load, or test scoring, raise; returns the splits loaded."""
    original_loader = pilot_module.load_split_datasets
    loaded_splits: list[str] = []

    def guarded_loader(split: str, dataset_dirs: Any) -> Any:
        if split in ("test", "stress"):
            raise AssertionError(f"dry run tried to load the {split} split")
        loaded_splits.append(split)
        return original_loader(split, dataset_dirs)

    def forbidden_scoring(*_: Any, **__: Any) -> Any:
        raise AssertionError("dry run tried to score test worlds")

    monkeypatch.setattr(pilot_module, "load_split_datasets", guarded_loader)
    monkeypatch.setattr(pilot_module, "_evaluate_test_worlds", forbidden_scoring)
    return loaded_splits


def _single_dry_run_file(out_root: Path) -> Path:
    """Returns the one dry_run.json the run wrote under out_root."""
    files = list(out_root.rglob("dry_run.json"))
    assert len(files) == 1
    return files[0]


def test_dry_run_stops_before_test_and_records_it(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    loaded_splits = _forbid_test_and_stress_loading(monkeypatch)

    exit_code = run_pilot(
        [TRAINING_DATASET],
        [VALIDATION_DATASET],
        [],
        TRUTH_ROOT,
        WORLD_CARDS_ROOT,
        tmp_path,
        stop_before_test=True,
    )

    assert exit_code == 0
    assert loaded_splits == ["training", "validation"]
    dry_run = json.loads(_single_dry_run_file(tmp_path).read_text(encoding="utf-8"))
    assert dry_run["test_loaded"] is False
    assert dry_run["stress_loaded"] is False
    assert dry_run["note"] == "DRY RUN - not a result"
    assert dry_run["training_row_count"] > 0
    assert 0 < dry_run["detector_kept_row_count"]
    for stage in ("load_training", "detector", "load_validation", "train", "tune"):
        assert stage in dry_run["stages_reached"]
        assert dry_run["stage_seconds"][stage] >= 0.0
        assert dry_run["peak_rss_mb"][stage] > 0.0
    assert "test" not in dry_run["stages_reached"]
    assert "stress" not in dry_run["stages_reached"]
    assert not list(tmp_path.rglob("result.json"))


def test_dry_run_refuses_a_test_or_stress_folder(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _forbid_test_and_stress_loading(monkeypatch)

    with pytest.raises(ValueError, match="stop_before_test"):
        run_pilot(
            [TRAINING_DATASET],
            [VALIDATION_DATASET],
            [TEST_DATASET],
            TRUTH_ROOT,
            WORLD_CARDS_ROOT,
            tmp_path,
            stop_before_test=True,
        )
    with pytest.raises(ValueError, match="stop_before_test"):
        run_pilot(
            [TRAINING_DATASET],
            [VALIDATION_DATASET],
            [],
            TRUTH_ROOT,
            WORLD_CARDS_ROOT,
            tmp_path,
            [TEST_DATASET],
            stop_before_test=True,
        )


def test_cli_runs_the_dry_run_and_restores_the_active_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _forbid_test_and_stress_loading(monkeypatch)
    config_before = active_mule_model_config_path()

    exit_code = main(
        [
            "--training",
            str(TRAINING_DATASET),
            "--validation",
            str(VALIDATION_DATASET),
            "--truth-root",
            str(TRUTH_ROOT),
            "--world-cards-root",
            str(WORLD_CARDS_ROOT),
            "--mule-model-config",
            str(COMMITTED_CONFIG),
            "--out",
            str(tmp_path),
        ]
    )

    assert exit_code == 0
    assert json.loads(_single_dry_run_file(tmp_path).read_text(encoding="utf-8"))["test_loaded"] is False
    assert active_mule_model_config_path() == config_before
