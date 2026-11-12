"""
WHAT: Unit tests for writing optional personas.jsonl ground truth file in world/truth_writer.py.
Verifies that:
a) when disabled (default and write_personas=False), personas.jsonl is not written and manifest bytes are identical;
b) when enabled (write_personas=True), personas.jsonl contains one row per person and shop sorted by account_id,
   with valid persona names and matching manifest SHA-256 and row count metadata.

WHY: Level 17 genuine harm evaluation requires account-level persona attribution without altering historical
truth store byte reproducibility for existing simulations.
"""

import hashlib
import json
from pathlib import Path

from core.config_validator import load_yaml
from world.fast_mode import FastRunSettings
from world.interactive_records import RunnerSettings
from world.interactive_runner import run_interactive_world
from world.run_identity import REPO_ROOT
from world.truth_writer import write_truth_files

# Tiny world configuration from frozen anchors
FROZEN = load_yaml(REPO_ROOT / "config" / "frozen_anchors.yaml")["frozen_anchors"]["fraud_off_events"]
TINY = FastRunSettings(FROZEN["seed"], FROZEN["n_people"], FROZEN["n_shops"], FROZEN["n_days"], FROZEN["range_name"])
TEST_CODE_VERSION = "test-version-truth-personas"


def test_truth_personas_switch_off_produces_identical_manifest_and_no_personas_file(tmp_path: Path) -> None:
    """
    Test (a):
    Switch off: file list and manifest.json bytes are identical between default write and explicit write_personas=False.
    Asserts personas.jsonl is absent in both.
    """
    output = run_interactive_world(TINY, RunnerSettings(fraud_enabled=False), code_version=TEST_CODE_VERSION)

    root_default = tmp_path / "truth_default"
    root_explicit = tmp_path / "truth_explicit"

    dir_default = write_truth_files(output, root_default)
    dir_explicit = write_truth_files(output, root_explicit, write_personas=False)

    # 1. Assert personas.jsonl is absent
    assert not (dir_default / "personas.jsonl").exists(), "personas.jsonl must not exist when switch is off"
    assert not (dir_explicit / "personas.jsonl").exists(), "personas.jsonl must not exist with write_personas=False"

    # 2. Assert file lists match
    files_default = sorted(p.name for p in dir_default.iterdir())
    files_explicit = sorted(p.name for p in dir_explicit.iterdir())
    assert files_default == files_explicit
    assert "personas.jsonl" not in files_default

    # 3. Assert manifest.json bytes are bit-for-bit identical
    manifest_default_bytes = (dir_default / "manifest.json").read_bytes()
    manifest_explicit_bytes = (dir_explicit / "manifest.json").read_bytes()
    assert manifest_default_bytes == manifest_explicit_bytes

    sha_default = hashlib.sha256(manifest_default_bytes).hexdigest()
    sha_explicit = hashlib.sha256(manifest_explicit_bytes).hexdigest()

    print(f"\n[EVIDENCE] manifest.json sha256 with switch off (default): {sha_default}")
    print(f"[EVIDENCE] manifest.json sha256 with switch off (write_personas=False): {sha_explicit}")


def test_truth_personas_switch_on_writes_rows_for_people_and_shops(tmp_path: Path) -> None:
    """
    Test (b):
    Switch on: row count == people + shops; every person row's persona is a real persona name;
    manifest lists personas.jsonl with its sha256 and row count.
    """
    output = run_interactive_world(TINY, RunnerSettings(fraud_enabled=False), code_version=TEST_CODE_VERSION)

    root_on = tmp_path / "truth_on"
    dir_on = write_truth_files(output, root_on, write_personas=True)

    personas_path = dir_on / "personas.jsonl"
    assert personas_path.exists(), "personas.jsonl must exist when write_personas=True"

    raw_lines = personas_path.read_text(encoding="utf-8").splitlines()
    rows = [json.loads(line) for line in raw_lines]

    n_people = len(output.world.people)
    n_shops = len(output.world.shops)
    total_expected_rows = n_people + n_shops

    print(f"\n[EVIDENCE] personas.jsonl row count in test (b): {len(rows)}, people: {n_people}, shops: {n_shops}")

    assert len(rows) == total_expected_rows

    # Verify rows are sorted strictly by account_id
    account_ids = [r["account_id"] for r in rows]
    assert account_ids == sorted(account_ids)
    assert len(set(account_ids)) == len(account_ids), "All account identifiers must be unique"

    people_by_acc = {p.account_id: p for p in output.world.people}
    shops_by_acc = {s.account_id: s for s in output.world.shops}

    person_rows = [r for r in rows if r["holder_type"] == "person"]
    shop_rows = [r for r in rows if r["holder_type"] == "shop"]

    assert len(person_rows) == n_people
    assert len(shop_rows) == n_shops

    # Assert every person row's persona matches their real persona name
    for r in person_rows:
        acc_id = r["account_id"]
        assert acc_id in people_by_acc
        expected_persona = people_by_acc[acc_id].persona_name
        assert r["persona"] == expected_persona
        assert len(r["persona"]) > 0

    # Assert every shop row's persona matches shop:<kind>
    for r in shop_rows:
        acc_id = r["account_id"]
        assert acc_id in shops_by_acc
        expected_persona = f"shop:{shops_by_acc[acc_id].kind}"
        assert r["persona"] == expected_persona

    # Assert manifest lists personas.jsonl with matching sha256 and row count
    manifest = json.loads((dir_on / "manifest.json").read_text(encoding="utf-8"))
    assert "personas.jsonl" in manifest["files"]
    personas_meta = manifest["files"]["personas.jsonl"]
    assert personas_meta["row_count"] == total_expected_rows
    assert personas_meta["sha256"] == hashlib.sha256(personas_path.read_bytes()).hexdigest()
