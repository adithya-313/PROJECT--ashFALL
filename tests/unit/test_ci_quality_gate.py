# File: tests/unit/test_ci_quality_gate.py
# Purpose: Checks the 5.11 CI quality gate logic: it passes a good model, fails a low PR-AUC, fails a bad calibration
#          bin, reads the threshold from the environment, and notices stale hashes. No data, model or server needed.
import importlib.util
import json
from pathlib import Path

import pytest

SCRIPTS_DIR = Path(__file__).resolve().parent.parent.parent / "scripts"
FIXTURE_DIR = Path(__file__).resolve().parent.parent / "fixtures"


def load_gate():
    """Load scripts/ci_quality_gate.py by file path (scripts/ is not a package)."""
    spec = importlib.util.spec_from_file_location("ci_quality_gate", SCRIPTS_DIR / "ci_quality_gate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


gate = load_gate()


def good_bins() -> dict:
    return {"split": "calibration", "bins": [
        {"bin": "0.0-0.1", "count": 1000, "deviation": 0.0001},
        {"bin": "0.1-0.2", "count": 0, "deviation": None},  # an empty bin cannot be checked and is skipped
        {"bin": "0.9-1.0", "count": 50, "deviation": 0.05},
    ]}


def write_json(path: Path, content: dict) -> Path:
    path.write_text(json.dumps(content))
    return path


def test_good_pr_auc_passes_and_low_pr_auc_fails():
    assert gate.check_pr_auc({"pr_auc": 0.9947}, 0.92)[0] is True
    assert gate.check_pr_auc({"pr_auc": 0.92}, 0.92)[0] is True  # exactly at the floor still passes
    ok, message = gate.check_pr_auc({"pr_auc": 0.5}, 0.92)
    assert ok is False and "0.5000 < threshold 0.9200" in message


def test_missing_or_nan_pr_auc_fails():
    assert gate.check_pr_auc({}, 0.92)[0] is False
    assert gate.check_pr_auc({"pr_auc": float("nan")}, 0.92)[0] is False


def test_calibration_passes_when_all_bins_within_ten_points():
    ok, lines = gate.check_calibration(good_bins())
    assert ok is True and "2 non-empty bins" in lines[0]


def test_one_bad_calibration_bin_fails_and_is_named():
    bins = good_bins()
    bins["bins"].append({"bin": "0.5-0.6", "count": 80, "deviation": 0.1500})
    ok, lines = gate.check_calibration(bins)
    assert ok is False and any("0.5-0.6" in line for line in lines)


def test_calibration_with_no_bins_fails():
    assert gate.check_calibration({"bins": []})[0] is False


def test_threshold_comes_from_environment_then_default(monkeypatch, tmp_path):
    monkeypatch.setattr(gate, "ENV_FILE_PATH", tmp_path / "missing.env")  # no .env file in this test
    monkeypatch.delenv("MODEL_AUC_THRESHOLD", raising=False)
    assert gate.read_threshold() == 0.92
    monkeypatch.setenv("MODEL_AUC_THRESHOLD", "0.97")
    assert gate.read_threshold() == 0.97


def test_threshold_can_come_from_the_env_file(monkeypatch, tmp_path):
    env_file = tmp_path / ".env"
    env_file.write_text("OTHER=1\nMODEL_AUC_THRESHOLD=0.88\n")
    monkeypatch.setattr(gate, "ENV_FILE_PATH", env_file)
    monkeypatch.delenv("MODEL_AUC_THRESHOLD", raising=False)
    assert gate.read_threshold() == 0.88


def test_stale_hash_is_detected():
    real = {key: gate.read_dvc_md5(gate.REPO_ROOT / path) for key, path in gate.HASH_FILES.items()}
    assert gate.check_hashes(real)[0] is True
    stale = {**real, "dvc_model_onnx_md5": "0" * 32}
    assert gate.check_hashes(stale)[0] is False
    assert gate.check_hashes({})[0] is False  # a file with no hashes cannot vouch for anything


def test_main_exit_codes_with_alternative_metrics_file(tmp_path, monkeypatch):
    monkeypatch.delenv("MODEL_AUC_THRESHOLD", raising=False)
    monkeypatch.setattr(gate, "ENV_FILE_PATH", tmp_path / "missing.env")
    bins = write_json(tmp_path / "bins.json", good_bins())
    good = write_json(tmp_path / "good.json", {"model_name": "x", "pr_auc": 0.99})
    low = write_json(tmp_path / "low.json", {"model_name": "x", "pr_auc": 0.40})
    assert gate.main(["--metrics", str(good), "--bins", str(bins)]) == 0
    assert gate.main(["--metrics", str(low), "--bins", str(bins)]) == 1
    # The committed degraded fixture used by the CI proof step must fail too.
    assert gate.main(["--metrics", str(FIXTURE_DIR / "metrics_degraded.json"), "--bins", str(bins)]) == 1


def test_main_fails_on_bad_bin_even_when_pr_auc_is_good(tmp_path, monkeypatch):
    monkeypatch.setattr(gate, "ENV_FILE_PATH", tmp_path / "missing.env")
    bad = good_bins()
    bad["bins"].append({"bin": "0.5-0.6", "count": 80, "deviation": 0.3})
    good = write_json(tmp_path / "good.json", {"pr_auc": 0.99})
    assert gate.main(["--metrics", str(good), "--bins", str(write_json(tmp_path / "bad.json", bad))]) == 1


def test_missing_metrics_file_stops_the_gate(tmp_path):
    with pytest.raises(SystemExit) as stopped:
        gate.main(["--metrics", str(tmp_path / "nope.json")])
    assert "cannot read" in str(stopped.value)
