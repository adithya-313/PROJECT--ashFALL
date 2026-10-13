"""
File: tests/unit/test_verify_world.py
Purpose: Tests that scripts/verify_world.py passes on valid golden values and fails when values diverge.
"""

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
SCRIPT_PATH = REPO_ROOT / "scripts" / "verify_world.py"
REAL_GOLDEN_PATH = REPO_ROOT / "scripts" / "golden_values.yaml"


def test_verify_world_passes_with_real_yaml():
    """Verify that running verify_world.py against real golden values succeeds with exit code 0."""
    cmd = [sys.executable, "-B", str(SCRIPT_PATH), "--golden-file", str(REAL_GOLDEN_PATH)]
    proc = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    assert proc.returncode == 0, f"Expected 0, got {proc.returncode}:\n{proc.stdout}\n{proc.stderr}"
    assert "ALL GOLDEN CHECKS PASSED" in proc.stdout
    for check_name in ("PASS FP", "PASS G1", "PASS G2", "PASS G3", "PASS ISSUES"):
        assert check_name in proc.stdout, f"Missing '{check_name}' in output:\n{proc.stdout}"


def test_verify_world_fails_when_hash_altered(tmp_path: Path):
    """Verify that altering an expected golden value causes verify_world.py to report failure and exit 1."""
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
