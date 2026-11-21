"""
WHAT: Tests the guard and the reproducibility gate of the recompute-from-saved-stage-3 module.

WHY: The module reuses a saved stage 3 and then scores the test set once. That is only safe if it refuses
any source folder or code change that could have altered stage 3 and refuses a retrained model that does
not reproduce the saved truth AP, so each refusal has its own test. The git calls are replaced by the
guard's two seams, so no repository is needed.
"""

from datetime import date
from pathlib import Path
from typing import Any

import orjson
import pytest

from evaluation.hyperparameter_search import validation_truth_average_precision
from evaluation.level17_scoring import ScoredDay
from evaluation.rules_tuning import load_labels_tuning_config
from evaluation.run_level_18_from_saved_stage3 import (
    REPRODUCIBILITY_TOLERANCE,
    SOURCE_REQUIRED_FILES,
    check_source_guard,
    recompute_method_record,
)

SOURCE_CODE_VERSION = "45bab3f"
ALLOWED_PATHS = [
    "evaluation/rules_tuning.py",
    "evaluation/run_level_18_pilot.py",
    "evaluation/run_level_18_from_saved_stage3.py",
    "tests/unit/evaluation/test_account_identity.py",
]


def _write_source(directory: Path, code_version: str = SOURCE_CODE_VERSION) -> Path:
    """Writes a complete fake source stages folder whose fingerprints carry the given code version."""
    directory.mkdir(parents=True, exist_ok=True)
    for name in SOURCE_REQUIRED_FILES:
        if name.endswith(".json"):
            payload = {"fingerprint": {"code_version": code_version}, "result": {}}
            (directory / name).write_bytes(orjson.dumps(payload))
        else:
            (directory / name).write_text("parameters: {}\n", encoding="utf-8")
    return directory


def _guard(source: Path, changed: list[str] | None = None, clean: bool = True) -> str:
    """Runs the guard with the git calls replaced by fixed answers."""
    paths = ALLOWED_PATHS if changed is None else changed
    return check_source_guard(source, changed_paths_of=lambda _version: paths, tree_is_clean=lambda: clean)


def test_guard_accepts_a_complete_clean_source_with_only_allowed_changes(tmp_path: Path) -> None:
    source = _write_source(tmp_path / "stages")

    assert _guard(source) == SOURCE_CODE_VERSION


def test_guard_refuses_a_changed_path_outside_the_allow_list(tmp_path: Path) -> None:
    source = _write_source(tmp_path / "stages")

    with pytest.raises(ValueError, match="outside the allowed files") as raised:
        _guard(source, changed=[*ALLOWED_PATHS, "evaluation/run_level_18_full.py", "config/mule_model.yaml"])

    assert "evaluation/run_level_18_full.py" in str(raised.value)
    assert "config/mule_model.yaml" in str(raised.value)


@pytest.mark.parametrize("missing_name", SOURCE_REQUIRED_FILES)
def test_guard_refuses_a_source_folder_missing_a_required_file(tmp_path: Path, missing_name: str) -> None:
    source = _write_source(tmp_path / "stages")
    (source / missing_name).unlink()

    with pytest.raises(ValueError, match=missing_name):
        _guard(source)


def test_guard_refuses_when_stage_5_already_ran(tmp_path: Path) -> None:
    source = _write_source(tmp_path / "stages")
    (source / "final_validation_and_test.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="already hold final_validation_and_test.json"):
        _guard(source)


def test_guard_refuses_a_dirty_fingerprint(tmp_path: Path) -> None:
    source = _write_source(tmp_path / "stages", code_version=f"{SOURCE_CODE_VERSION}-dirty")

    with pytest.raises(ValueError, match="dirty"):
        _guard(source)


def test_guard_refuses_fingerprints_from_different_code_versions(tmp_path: Path) -> None:
    source = _write_source(tmp_path / "stages")
    other = {"fingerprint": {"code_version": "1234567"}, "result": {}}
    (source / "learning_curve.json").write_bytes(orjson.dumps(other))

    with pytest.raises(ValueError, match="different code versions"):
        _guard(source)


def test_guard_refuses_a_dirty_working_tree(tmp_path: Path) -> None:
    source = _write_source(tmp_path / "stages")

    with pytest.raises(ValueError, match="not clean"):
        _guard(source, clean=False)


def _truth_scored() -> list[ScoredDay]:
    """One world: genuine accounts g1 and g2 and one mule m1, all on one day."""

    def day(account_id: str, role: str) -> ScoredDay:
        return ScoredDay(
            world_id="world_a",
            bank_id="bank_a",
            account_id=account_id,
            day=date(2026, 9, 1),
            rules_score=0,
            rules_fired=(),
            anomaly_score=None,
            role=role,
            depth=1 if role == "mule" else None,
        )

    return [day("g1", "genuine"), day("g2", "genuine"), day("m1", "mule")]


PROBABILITIES = [0.2, 0.1, 0.9]


def _saved_record(saved_average_precision: float) -> dict[str, Any]:
    """A stage 3 style record whose operating-point fields are deliberately wrong."""
    return {
        "pu_method": "none",
        "chosen_params": {"n_estimators": 10},
        "validation_truth_average_precision": saved_average_precision,
        "validation_rate_pct": 99.0,
        "validation_recall": 0.0,
        "meets_budget": False,
    }


def test_gate_stops_on_a_one_in_a_million_average_precision_mismatch() -> None:
    truth_scored = _truth_scored()
    true_average_precision = float(validation_truth_average_precision(PROBABILITIES, truth_scored))
    saved = _saved_record(true_average_precision + 1e-6)

    with pytest.raises(ValueError, match="did not reproduce") as raised:
        recompute_method_record("none", saved, PROBABILITIES, truth_scored, load_labels_tuning_config(), 0.5)

    assert repr(true_average_precision + 1e-6) in str(raised.value)
    assert repr(true_average_precision) in str(raised.value)


def test_gate_passes_an_exact_match_and_recomputes_only_the_operating_point() -> None:
    truth_scored = _truth_scored()
    true_average_precision = float(validation_truth_average_precision(PROBABILITIES, truth_scored))
    saved = _saved_record(true_average_precision)

    record = recompute_method_record("none", saved, PROBABILITIES, truth_scored, load_labels_tuning_config(), 0.5)

    # Two genuine accounts (peaks 0.2 and 0.1) and one mule (peak 0.9): no cut meets a 0.5% budget, so the
    # lowest rate wins: cut 0.2 restricts 1 of 2 genuine accounts and catches the mule.
    assert record["validation_rate_pct"] == 50.0
    assert record["validation_recall"] == 1.0
    assert record["meets_budget"] is False
    assert record["chosen_params"] == {"n_estimators": 10}
    assert record["validation_truth_average_precision"] == true_average_precision
    assert REPRODUCIBILITY_TOLERANCE == 1e-12
