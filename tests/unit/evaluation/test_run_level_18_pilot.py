"""
WHAT: Integration tests for the Level 18 pilot runner on the committed Level 17 fixtures: the full
run produces all four scorers, the leakage report, the SHAP summary and the per-rule manifest, and
an overlapping split is refused.

WHY: The pilot is only trustworthy if its plumbing works end to end and a split leak cannot slip
through. Tiny fixtures are almost all fraud, so these tests assert structure, never metric values.
"""

import json
import shutil
from pathlib import Path

import pyarrow.parquet as pq

from core.event_clock import IST
from decisions.mule_account_model import _one_row_per_account_day
from decisions.rules_config import load_rules_config
from evaluation.level17_scoring import one_row_per_account_day
from evaluation.run_level_18_pilot import run_pilot

REPO_ROOT = Path(__file__).resolve().parents[3]
FIXTURES = REPO_ROOT / "showcase" / "fixtures" / "level_17"
TRAINING = FIXTURES / "datasets" / "training-20261031T1830-7a356ef53e9e"
VALIDATION = FIXTURES / "datasets" / "validation-20261031T1830-9d983967b617"
TEST = FIXTURES / "datasets" / "test-20261031T1830-9a2856ce53b2"
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
