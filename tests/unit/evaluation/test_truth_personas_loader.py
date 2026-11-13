"""
WHAT: Unit tests for loading optional personas.jsonl ground truth file in evaluation/truth.py.
Verifies that:
c) when switch is on, load_truth sets has_personas True, persona_of returns the correct person persona,
   and querying an unknown account returns "unknown";
d) when switch is off, has_personas is False, and persona_of raises ValueError;
e) when switch is on and personas.jsonl is altered by 1 byte, load_truth raises ValueError
   on cryptographic integrity check.

WHY: Ensures tamper-proof evaluation loader behavior and secure persona resolution for Level 17 genuine harm metrics.
"""

from pathlib import Path

import pytest

from core.config_validator import load_yaml
from evaluation.truth import load_truth
from world.fast_mode import FastRunSettings
from world.interactive_records import RunnerSettings
from world.interactive_runner import run_interactive_world
from world.run_identity import REPO_ROOT
from world.truth_writer import write_truth_files

# Tiny world configuration from frozen anchors
FROZEN = load_yaml(REPO_ROOT / "config" / "frozen_anchors.yaml")["frozen_anchors"]["fraud_off_events"]
TINY = FastRunSettings(FROZEN["seed"], FROZEN["n_people"], FROZEN["n_shops"], FROZEN["n_days"], FROZEN["range_name"])
TEST_CODE_VERSION = "test-version-truth-personas-loader"


def test_truth_loader_with_personas_enabled(tmp_path: Path) -> None:
    """
    Test (c):
    Switch on: load_truth -> has_personas True; persona_of(a person) correct; unknown account -> 'unknown'.
    """
    output = run_interactive_world(TINY, RunnerSettings(fraud_enabled=False), code_version=TEST_CODE_VERSION)
    run_id = output.result.run_id
    truth_root = tmp_path / "truth_on"

    write_truth_files(output, truth_root, write_personas=True)

    store = load_truth(truth_root, run_id)
    assert store.has_personas is True

    # Assert known person returns their correct persona string
    first_person = output.world.people[0]
    persona_result = store.persona_of(first_person.account_id)
    assert persona_result == first_person.persona_name
    assert persona_result != "unknown"

    # Assert known shop returns shop:<kind>
    first_shop = output.world.shops[0]
    shop_persona_result = store.persona_of(first_shop.account_id)
    assert shop_persona_result == f"shop:{first_shop.kind}"

    # Assert unknown account returns "unknown"
    unknown_result = store.persona_of("acc_nonexistent_999999")
    assert unknown_result == "unknown"


def test_truth_loader_with_personas_disabled_raises_on_query(tmp_path: Path) -> None:
    """
    Test (d):
    Switch off: has_personas False; persona_of raises ValueError.
    """
    output = run_interactive_world(TINY, RunnerSettings(fraud_enabled=False), code_version=TEST_CODE_VERSION)
    run_id = output.result.run_id
    truth_root = tmp_path / "truth_off"

    write_truth_files(output, truth_root, write_personas=False)

    store = load_truth(truth_root, run_id)
    assert store.has_personas is False

    first_person = output.world.people[0]
    with pytest.raises(ValueError, match="Personas truth was not loaded"):
        store.persona_of(first_person.account_id)


def test_truth_loader_tamper_detection_on_personas_file(tmp_path: Path) -> None:
    """
    Test (e):
    Switch on: change one byte of personas.jsonl -> load_truth raises ValueError (tamper check).
    """
    output = run_interactive_world(TINY, RunnerSettings(fraud_enabled=False), code_version=TEST_CODE_VERSION)
    run_id = output.result.run_id
    truth_root = tmp_path / "truth_tamper"

    run_dir = write_truth_files(output, truth_root, write_personas=True)
    personas_file = run_dir / "personas.jsonl"
    assert personas_file.exists()

    # Tamper with 1 byte in personas.jsonl
    raw_bytes = bytearray(personas_file.read_bytes())
    raw_bytes[5] ^= 0xFF
    personas_file.write_bytes(raw_bytes)

    with pytest.raises(ValueError, match="Cryptographic hash mismatch for 'personas.jsonl'"):
        load_truth(truth_root, run_id)
