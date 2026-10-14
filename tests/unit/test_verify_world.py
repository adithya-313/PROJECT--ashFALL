"""
WHAT: Unit tests verifying scripts/verify_world.py behavior under normal and mutated configurations.
WHY: Ensures golden check runner detects deviations properly and exits with appropriate return codes.
TERM: Mutation Testing: deliberately corrupting expected golden baseline values to verify failure detection,
      e.g. replacing valid SHA-256 fingerprint with zeroes to confirm script exits with code 1.
TERM: Golden Baseline: frozen reference data catalog storing canonical simulation hashes and event counts,
      e.g. scripts/golden_values.yaml holding ground truth outputs.
"""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "verify_world.py"
REAL_GOLDEN_PATH = REPO_ROOT / "scripts" / "golden_values.yaml"


def test_verify_world_passes_with_real_yaml():
    """
    WHAT: Asserts scripts/verify_world.py exits with status 0 and prints all golden check passes.
    WHY: Confirms that unchanged repository codebase satisfies all golden baseline values end-to-end.
    Example:
        # Runs verify_world.py against real golden_values.yaml
        # assert proc.returncode == 0
        # assert "ALL GOLDEN CHECKS PASSED" in proc.stdout
    """
    cmd = [sys.executable, "-B", str(SCRIPT_PATH), "--golden-file", str(REAL_GOLDEN_PATH)]
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    assert proc.returncode == 0, f"Expected 0, got {proc.returncode}:\n{proc.stdout}\n{proc.stderr}"
    assert "ALL GOLDEN CHECKS PASSED" in proc.stdout
    for check_name in ("PASS FP", "PASS G1", "PASS G2", "PASS G3", "PASS ISSUES"):
        assert check_name in proc.stdout, f"Missing '{check_name}' in output:\n{proc.stdout}"


def test_verify_world_fails_when_hash_altered(tmp_path: Path):
    """
    WHAT: Mutates expected golden fingerprint and asserts scripts/verify_world.py reports FAIL and exits 1.
    WHY: Verifies the integrity of the golden test harness so tampered values cannot pass undetected.
    Args: tmp_path: Pytest temporary directory fixture used to isolate mutated YAML configuration.
    Example:
        # Mutates FP expected hash to zeroes in temporary file
        # assert proc.returncode == 1
    """
    content = REAL_GOLDEN_PATH.read_text(encoding="utf-8")
    # Mutate FP expected hash in temporary file copy so real config remains untouched.
    mutated = content.replace("0f09c489d4055fa0343316c6619fd5e254e074779c0e30d9a4bc315308f5986f", "0" * 64)
    temp_yaml = tmp_path / "mutated_golden.yaml"
    temp_yaml.write_text(mutated, encoding="utf-8")

    cmd = [sys.executable, "-B", str(SCRIPT_PATH), "--golden-file", str(temp_yaml)]
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    assert proc.returncode == 1, f"Expected 1, got {proc.returncode}:\n{proc.stdout}"
    assert "FAIL FP" in proc.stdout, f"Expected 'FAIL FP' in output:\n{proc.stdout}"
    assert "CHECK(S) FAILED" in proc.stdout, f"Expected failure tally line in output:\n{proc.stdout}"
