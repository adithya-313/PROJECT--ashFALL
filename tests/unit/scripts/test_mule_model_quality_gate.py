"""
File: tests/unit/scripts/test_mule_model_quality_gate.py
Purpose: Proves the Level 18 "Mule model quality gate" passes faithful cards, fails a degraded card naming its
         threshold id, and treats a missing path, an empty derived_from, a skipped threshold, a duplicate id and
         an entry carrying two rules as failures with exit 1, never as skips.

Every run uses a temporary repo-shaped tree and the real gate function.
"""

import importlib
import json
from pathlib import Path
from types import ModuleType

CARD_DIR = "reports/level_18/model_cards"
CONFIG_PATH = "config/mule_model_quality_gate.yaml"

CONFIG = """version: 1
threshold_count: 4
thresholds:
  - id: account_recall
    card: mule_account_xgb
    path: metrics.recall_at_capacity.median
    at_least: 0.6
    derived_from: comparison.json median 0.642857
  - id: account_ece
    card: mule_account_xgb
    path: calibration.calibrated.ece
    at_most: 0.01
    derived_from: calibration.json calibrated ECE 0.001843
  - id: payment_status
    card: mule_payment_xgb
    path: leakage.run_a_check_6b.status
    equals: "fail"
    derived_from: A-057 check 6b failed
  - id: payment_onnx
    card: mule_payment_xgb
    path: onnx.parity.test.max_abs_diff
    at_most: 1.0e-5
    derived_from: onnx_parity.json 5.364418e-07
"""


def gate() -> ModuleType:
    """Return the real model quality gate module."""
    return importlib.import_module("scripts.mule_model_quality_gate")


def _cards() -> tuple[dict, dict]:
    """Return the two card documents the temporary tree holds."""
    account = {
        "simulated": True,
        "manifest": {},
        "data": {
            "metrics": {"recall_at_capacity": {"median": 0.6428571428571429}},
            "calibration": {"calibrated": {"ece": 0.0018426116879839516}},
        },
    }
    payment = {
        "simulated": True,
        "manifest": {},
        "data": {
            "leakage": {"run_a_check_6b": {"status": "fail"}},
            "onnx": {"parity": {"test": {"max_abs_diff": 5.364418029785156e-07}}},
        },
    }
    return account, payment


def _write_tree(root: Path, config: str = CONFIG) -> None:
    """Write the threshold config and both cards under the temporary root."""
    config_path = root / CONFIG_PATH
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(config, encoding="utf-8")
    account, payment = _cards()
    card_dir = root / CARD_DIR
    card_dir.mkdir(parents=True, exist_ok=True)
    (card_dir / "mule_account_xgb.json").write_text(json.dumps(account), encoding="utf-8")
    (card_dir / "mule_payment_xgb.json").write_text(json.dumps(payment), encoding="utf-8")


def test_all_thresholds_pass(tmp_path, capsys):
    """Four faithful thresholds print four PASS lines, the summary and exit 0."""
    _write_tree(tmp_path)

    assert gate().run(root=tmp_path) == 0

    lines = capsys.readouterr().out.splitlines()
    assert lines[:4] == [
        "PASS account_recall 0.642857 at least 0.6",
        "PASS account_ece 0.00184261 at most 0.01",
        "PASS payment_status fail equals fail",
        "PASS payment_onnx 5.36442e-07 at most 1e-05",
    ]
    assert lines[4] == "4 passed, 0 failed"


def test_degraded_card_fails_with_its_id(tmp_path, capsys):
    """A recall below the floor fails that threshold by name and exits 1."""
    _write_tree(tmp_path)
    path = tmp_path / CARD_DIR / "mule_account_xgb.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    document["data"]["metrics"]["recall_at_capacity"]["median"] = 0.59
    path.write_text(json.dumps(document), encoding="utf-8")

    assert gate().run(root=tmp_path) == 1

    out = capsys.readouterr().out
    assert "FAIL account_recall 0.59 at least 0.6" in out
    assert "3 passed, 1 failed" in out


def test_missing_path_fails(tmp_path, capsys):
    """A card without the threshold path fails by id and exits 1."""
    _write_tree(tmp_path)
    path = tmp_path / CARD_DIR / "mule_account_xgb.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    del document["data"]["calibration"]
    path.write_text(json.dumps(document), encoding="utf-8")

    assert gate().run(root=tmp_path) == 1

    assert "FAIL account_ece: threshold 'account_ece': the card has no value at 'calibration.calibrated.ece'" in (
        capsys.readouterr().out
    )


def test_empty_derived_from_fails(tmp_path, capsys):
    """A threshold without a reason is a config failure and exits 1."""
    _write_tree(tmp_path, CONFIG.replace("derived_from: A-057 check 6b failed", 'derived_from: "  "'))

    assert gate().run(root=tmp_path) == 1

    assert "FAIL config: thresholds[2] has no usable 'derived_from'" in capsys.readouterr().out


def test_skipped_threshold_fails(tmp_path, capsys):
    """Declaring four thresholds while carrying three is a failure, not a quieter gate."""
    _write_tree(tmp_path, CONFIG.replace("threshold_count: 4", "threshold_count: 5"))

    assert gate().run(root=tmp_path) == 1

    assert "carries 4 thresholds but declares 5; a removed threshold is a failure, not a skip" in (
        capsys.readouterr().out
    )


def test_duplicate_ids_fail(tmp_path, capsys):
    """Two thresholds sharing an id fail the config check."""
    _write_tree(tmp_path, CONFIG.replace("id: account_ece", "id: account_recall"))

    assert gate().run(root=tmp_path) == 1

    assert "FAIL config: duplicate threshold ids account_recall" in capsys.readouterr().out


def test_two_rules_fail(tmp_path, capsys):
    """A threshold carrying two rules fails the config check."""
    _write_tree(tmp_path, CONFIG.replace("    at_least: 0.6\n", "    at_least: 0.6\n    at_most: 0.9\n"))

    assert gate().run(root=tmp_path) == 1

    assert "FAIL config: threshold 'account_recall' must carry exactly one of at_least, at_most, equals" in (
        capsys.readouterr().out
    )
