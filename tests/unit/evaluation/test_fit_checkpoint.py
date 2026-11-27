"""
WHAT: Tests the checkpoint run_pilot saves after its search and final fit.

WHY: Stage 5 spends most of its time in the search and the fit, so a crash during test or stress scoring
must not repeat them; and a checkpoint made by different code, config or data must never be reused
silently. A second run must skip the trainer entirely, a changed fingerprint must raise, and a saved
model must predict exactly what the fitted one did.
"""

import json
from pathlib import Path
from typing import Any

import numpy
import pytest
import xgboost

import evaluation.run_level_18_pilot as pilot
from decisions.mule_account_model import train_account_model
from decisions.pu_learning import train_pu_bagged_model
from evaluation.run_level_18_pilot import (
    _fit_checkpoint_fingerprint,
    _FitCheckpoint,
    _load_fit_checkpoint,
    _save_fit_checkpoint,
    run_pilot,
)
from showcase.level_17_checks import TEST_DATASET, TRAINING_DATASET, VALIDATION_DATASET

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURES = REPO_ROOT / "showcase" / "fixtures" / "level_17"
SEED = 11


def _run(out_root: Path) -> int:
    return run_pilot(
        [TRAINING_DATASET], [VALIDATION_DATASET], [TEST_DATASET], FIXTURES / "truth", FIXTURES / "world_cards", out_root
    )


def _checkpoint_record(out_root: Path) -> Path:
    return next(out_root.rglob("fit_record.json"))


def _forbid_training(monkeypatch: pytest.MonkeyPatch) -> None:
    def refuse(*_: Any, **__: Any) -> Any:
        raise AssertionError("the trainer was called although a checkpoint matched")

    monkeypatch.setattr(pilot, "train_account_model", refuse)
    monkeypatch.setattr(pilot, "train_pu_bagged_model", refuse)
    monkeypatch.setattr(pilot, "train_elkan_noto_model", refuse)


def test_a_second_run_with_a_matching_checkpoint_skips_the_trainer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    assert _run(tmp_path) == 0
    first = json.loads(next(tmp_path.rglob("result.json")).read_text(encoding="utf-8"))
    capsys.readouterr()

    _forbid_training(monkeypatch)
    assert _run(tmp_path) == 0

    assert "SKIP search+fit: checkpoint matches" in capsys.readouterr().out
    second = json.loads(next(tmp_path.rglob("result.json")).read_text(encoding="utf-8"))
    assert second["across_worlds"] == first["across_worlds"]
    assert second["per_world"] == first["per_world"]


def test_a_changed_fingerprint_raises_instead_of_reusing_the_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert _run(tmp_path) == 0
    record_path = _checkpoint_record(tmp_path)
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["fingerprint"]["code_version"] = "0000000"
    record_path.write_text(json.dumps(record), encoding="utf-8")

    _forbid_training(monkeypatch)
    with pytest.raises(ValueError, match="different inputs"):
        _run(tmp_path)


def test_a_checkpoint_folder_without_its_record_raises(tmp_path: Path) -> None:
    folder = tmp_path / "checkpoint"
    folder.mkdir()

    with pytest.raises(ValueError, match="no fit_record.json"):
        _load_fit_checkpoint(folder, {"code_version": "x"}, "none")


def test_no_checkpoint_folder_means_nothing_to_load(tmp_path: Path) -> None:
    assert _load_fit_checkpoint(tmp_path / "absent", {"code_version": "x"}, "none") is None


def _small_data() -> tuple[numpy.ndarray, list[int]]:
    matrix = numpy.random.default_rng(2024).random((120, 4)).astype(numpy.float32)
    return matrix, [1 if index % 5 == 0 else 0 for index in range(120)]


def _checkpoint_of(model: Any) -> _FitCheckpoint:
    return _FitCheckpoint(
        model=model,
        best_scale_pos_weight=4.0,
        best_n_estimators=5,
        best_validation_ap=0.5,
        best_unlabelled_per_positive=None,
        best_c=None,
        best_weight_summary=None,
        best_clip_count=None,
        n_estimators_scores={"5": 0.5},
        pu_average_precision_scores={},
        training_seconds={"5": 0.1},
        search_result=None,
    )


def test_a_saved_xgboost_model_predicts_exactly_what_the_fitted_one_did(tmp_path: Path) -> None:
    matrix, targets = _small_data()
    model, _ = train_account_model(matrix, targets, SEED, 5)
    fingerprint = _fit_checkpoint_fingerprint([TRAINING_DATASET], [VALIDATION_DATASET])

    _save_fit_checkpoint(tmp_path, fingerprint, "none", _checkpoint_of(model), SEED)
    restored = _load_fit_checkpoint(tmp_path, fingerprint, "none")

    assert restored is not None
    probe = numpy.random.default_rng(99).random((30, 4)).astype(numpy.float32)
    assert isinstance(restored.model, xgboost.XGBClassifier)
    numpy.testing.assert_array_equal(restored.model.predict_proba(probe), model.predict_proba(probe))
    assert restored.best_n_estimators == 5
    assert restored.training_seconds == {"5": 0.1}


def test_a_saved_bagged_ensemble_predicts_exactly_what_the_fitted_one_did(tmp_path: Path) -> None:
    matrix, targets = _small_data()
    model = train_pu_bagged_model(matrix, targets, SEED, 3, 2, 2)
    fingerprint = _fit_checkpoint_fingerprint([TRAINING_DATASET], [VALIDATION_DATASET])

    _save_fit_checkpoint(tmp_path, fingerprint, "bagging", _checkpoint_of(model), SEED)
    restored = _load_fit_checkpoint(tmp_path, fingerprint, "bagging")

    assert restored is not None
    probe = numpy.random.default_rng(99).random((30, 4)).astype(numpy.float32)
    numpy.testing.assert_array_equal(restored.model.predict_probabilities(probe), model.predict_probabilities(probe))


def test_a_different_pu_method_raises(tmp_path: Path) -> None:
    matrix, targets = _small_data()
    model, _ = train_account_model(matrix, targets, SEED, 5)
    fingerprint = _fit_checkpoint_fingerprint([TRAINING_DATASET], [VALIDATION_DATASET])
    _save_fit_checkpoint(tmp_path, fingerprint, "none", _checkpoint_of(model), SEED)

    with pytest.raises(ValueError, match="different inputs"):
        _load_fit_checkpoint(tmp_path, fingerprint, "bagging")
