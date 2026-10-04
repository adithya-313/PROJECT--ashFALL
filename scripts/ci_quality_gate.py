# File: scripts/ci_quality_gate.py
# Purpose: Sub-phase 5.11 - the CI model-quality gate. Exits 0 when the model's committed numbers are good enough,
#          exits 1 (with a clear message) when they are not.
#
# What it checks:
#   1. PR-AUC in reports/model/metrics.json is at least MODEL_AUC_THRESHOLD (default 0.92).
#   2. Every non-empty bin in reports/calibration/calibration_bins.json is within 0.10 of the diagonal
#      (the calibration-split 10 percentage point rule).
#   3. The DVC hashes written inside metrics.json still match the .dvc files in this commit (catches a STALE file).
#
# Usage:
#   python scripts/ci_quality_gate.py                          # check the committed files
#   python scripts/ci_quality_gate.py --metrics other.json     # check a different metrics file (used by tests/CI proof)
#
# Where the threshold comes from, first match wins: the MODEL_AUC_THRESHOLD environment variable, then the
# MODEL_AUC_THRESHOLD line in .env, then 0.92.
#
# HONEST LIMITATION - read this before trusting a green gate:
#   CI has no data, no GPU and no MLflow store, so it CANNOT re-score the model. This gate only enforces that the
#   COMMITTED number meets the threshold. It cannot prove the number really came from the committed model. The
#   hash check (3) catches a stale file (model changed, metrics not refreshed) but not a falsified one.
#
# Standard library only, so the CI job needs no pip install.
import argparse
import json
import math
import os
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_METRICS_PATH = REPO_ROOT / "reports" / "model" / "metrics.json"
DEFAULT_BINS_PATH = REPO_ROOT / "reports" / "calibration" / "calibration_bins.json"
ENV_FILE_PATH = REPO_ROOT / ".env"

DEFAULT_THRESHOLD = 0.92  # PR-AUC floor. The current model scores about 0.9947, so this leaves margin.
BIN_TOLERANCE = 0.10  # every calibration bin must be within 10 percentage points of the diagonal

# metrics.json key -> the .dvc file whose md5 it must equal
HASH_FILES = {
    "dvc_data_md5": "data/enriched/paysim_enriched.parquet.dvc",
    "dvc_model_ubj_md5": "models/xgboost/model.ubj.dvc",
    "dvc_model_onnx_md5": "models/xgboost/model.onnx.dvc",
}


def read_threshold() -> float:
    """Return the PR-AUC floor: environment variable, else the .env file, else the default."""
    text = os.environ.get("MODEL_AUC_THRESHOLD")
    if text is None and ENV_FILE_PATH.exists():
        match = re.search(r"^MODEL_AUC_THRESHOLD=(\S+)", ENV_FILE_PATH.read_text(), re.MULTILINE)
        text = match.group(1) if match else None
    if text is None:
        return DEFAULT_THRESHOLD
    try:
        return float(text)
    except ValueError:
        sys.exit(f"GATE ERROR: MODEL_AUC_THRESHOLD must be a number, got '{text}'")


def load_json(path: Path) -> dict:
    """Read a JSON file; a missing or unreadable file fails the gate with a plain message."""
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        sys.exit(f"GATE FAIL: cannot read {path}: {error}")


def check_pr_auc(metrics: dict, threshold: float) -> tuple[bool, str]:
    """Pass when PR-AUC is a real number at or above the threshold."""
    value = metrics.get("pr_auc")
    if not isinstance(value, (int, float)) or math.isnan(value):
        return False, f"FAIL PR-AUC: metrics file has no usable 'pr_auc' (got {value!r})"
    ok = value >= threshold
    return ok, f"{'PASS' if ok else 'FAIL'} PR-AUC {value:.4f} {'>=' if ok else '<'} threshold {threshold:.4f}"


def check_calibration(bins_file: dict, tolerance: float = BIN_TOLERANCE) -> tuple[bool, list[str]]:
    """Pass when every non-empty bin is within `tolerance` of the diagonal (deviation = |predicted - actual|)."""
    rows = [row for row in bins_file.get("bins", []) if row.get("deviation") is not None]
    if not rows:
        return False, ["FAIL calibration: no non-empty bins to check"]
    lines = []
    for row in rows:
        bin_ok = row["deviation"] <= tolerance
        if not bin_ok:
            lines.append(f"  bin {row['bin']}: deviation {row['deviation']:.4f} > {tolerance} ({row['count']} rows)")
    worst = max(rows, key=lambda row: row["deviation"])
    ok = not lines
    summary = (f"{'PASS' if ok else 'FAIL'} calibration ({bins_file.get('split', '?')} split): {len(rows)} non-empty "
               f"bins, worst deviation {worst['deviation']:.4f} (bin {worst['bin']}), limit {tolerance}")
    return ok, [summary] + lines


def read_dvc_md5(dvc_path: Path) -> str | None:
    """Pull the first md5 out of a .dvc file (plain text search, so no YAML library is needed)."""
    if not dvc_path.exists():
        return None
    match = re.search(r"md5:\s*([0-9a-f]+)", dvc_path.read_text())
    return match.group(1) if match else None


def check_hashes(metrics: dict) -> tuple[bool, list[str]]:
    """Pass when each hash recorded in metrics.json equals the md5 in the matching .dvc file of this commit."""
    lines, ok = [], True
    for key, dvc_file in HASH_FILES.items():
        recorded, actual = metrics.get(key), read_dvc_md5(REPO_ROOT / dvc_file)
        match = recorded is not None and recorded == actual
        ok = ok and match
        lines.append(f"  {'ok  ' if match else 'STALE'} {key}: metrics.json {recorded} vs {dvc_file} {actual}")
    return ok, [f"{'PASS' if ok else 'FAIL'} hashes in metrics.json match the .dvc files"] + lines


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description="CI model-quality gate (reads committed metrics files).")
    parser.add_argument("--metrics", type=Path, default=None, help="metrics JSON to check (default: committed file)")
    parser.add_argument("--bins", type=Path, default=DEFAULT_BINS_PATH, help="calibration bins JSON to check")
    args = parser.parse_args(argv)

    threshold = read_threshold()
    metrics_path = args.metrics or DEFAULT_METRICS_PATH
    metrics = load_json(metrics_path)
    print(f"Quality gate: metrics {metrics_path}, model '{metrics.get('model_name')}', threshold {threshold}")

    pr_auc_ok, pr_auc_line = check_pr_auc(metrics, threshold)
    results = [(pr_auc_ok, [pr_auc_line]), check_calibration(load_json(args.bins))]
    if args.metrics is None:
        results.append(check_hashes(metrics))
    else:
        print("  (hash check skipped: an alternative --metrics file is not tied to the committed artifacts)")

    for _, lines in results:
        print("\n".join(lines))
    passed = all(ok for ok, _ in results)
    print("GATE PASSED" if passed else "GATE FAILED: the model does not meet the quality bar")
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
