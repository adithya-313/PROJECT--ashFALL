"""
WHAT: One guarded test file for every value that Level 11 pinned and that later levels must never move.
Each test re-measures one value from the real code and compares it with config/frozen_anchors.yaml.

WHY: Level 12 adds fraud on top of the Level 11 world. With fraud switched off, the world must stay exactly
as Level 11 shipped it (ground rule 27). Gathering the pinned values in one place means one failing test
names the value that moved, instead of a slow drift spread over many files.

TERM: frozen anchor: a number or hash that Level 11 pinned and that stays fixed forever,
      e.g. the 6 stress scenario files, or the hash of the fraud-off events.
TERM: golden fingerprint: a hash saved from an audited run (FP, G1, G2, G3 in scripts/golden_values.yaml),
      e.g. G2 is the hash of the 2,915 payment events of the 60-person world with seed 101.
TERM: SHA-256 hash: a 64-character fingerprint of some text; changing one character of the text changes it,
      e.g. the fraud-off events hash starts with 24fc0e25.
TERM: canonical form: one fixed way of writing data (sorted keys, no spaces) so the same data always gives
      the same text, e.g. so a file with Windows line endings and one with Linux line endings hash the same.
TERM: issue register: the list of 31 research issues in config/issues.yaml with a status for each,
      e.g. R7 is "planned" and R9 is "deferred".
TERM: grounded parameter: a setting in the three realism config files that carries a label saying where its
      number came from, e.g. "assumption" (our own guess) or "reported" (seen in a published source).
"""

import hashlib
import json
from pathlib import Path
from typing import Any

import yaml

from core.config_validator import load_yaml
from core.issue_register import check_register, load_register
from scripts.verify_world import check_fp, check_g1, check_g2, check_g3
from world.fast_mode import FastRunSettings, event_lines, run_fast_world
from world.grounding import count_by_label, grounding_rows
from world.world_configs import read_world_configs

# The repo root, found from this file's own location so the tests work from any working directory.
REPO_ROOT: Path = Path(__file__).resolve().parents[3]
FROZEN_ANCHORS_PATH: Path = REPO_ROOT / "config" / "frozen_anchors.yaml"


def load_frozen_anchors() -> dict[str, Any]:
    """
    WHAT: Reads config/frozen_anchors.yaml and returns its one block, frozen_anchors, as a dictionary.
    Every test in this file takes its expected numbers from here, never from literals in the test code.

    WHY: Keeping the pinned numbers in one labelled config file means there is exactly one place that says
    what Level 11 promised. A test that hard-coded its own copy could quietly disagree with the config.

    Args:
        None.

    Returns:
        Dictionary with keys such as "fraud_off_events", "level_11_register" and "stress_scenario_files".

    Example:
        >>> load_frozen_anchors()["stress_scenario_files"]
        6
        >>> load_frozen_anchors()["fraud_off_events"]["sha256"][:8]
        '24fc0e25'
    """
    # The file holds a single top-level block; return only that block.
    return load_yaml(FROZEN_ANCHORS_PATH)["frozen_anchors"]


def canonical_yaml_sha256(yaml_text: str) -> str:
    """
    WHAT: Parses YAML text, writes the data back out in canonical form, and returns the SHA-256 hash of that.
    Canonical form here means JSON with sorted keys and no spaces after commas or colons.

    WHY: Hashing the raw file bytes would give a different answer on Windows (CRLF line endings) and on the
    Linux CI runner (LF line endings) even though the values are the same. Parsing first removes that
    difference, so the hash only changes when a real value changes.

    Args:
        yaml_text: The full text of a YAML file, for example the text of scripts/golden_values.yaml.

    Returns:
        A 64-character lowercase hexadecimal SHA-256 hash.

    Example:
        >>> canonical_yaml_sha256("b: 2\\na: 1\\n") == canonical_yaml_sha256("a: 1\\r\\nb: 2\\r\\n")
        True
        >>> len(canonical_yaml_sha256("a: 1"))
        64
    """
    # Turn the YAML text into plain Python data, so line endings and key order no longer matter.
    data = yaml.safe_load(yaml_text)
    # Write that data in one fixed text form: keys sorted, no optional spaces.
    canonical_text = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical_text.encode("utf-8")).hexdigest()


def test_fraud_off_events_hash_is_unchanged() -> None:
    """
    WHAT: Runs the 60-person, 12-shop, 61-day world with seed 101 in fast mode, exactly as Level 11 check 11
    does, and hashes its 2,915 event lines. The hash must equal the frozen sha256.

    WHY: This hash covers every payment, failure and device stamp of the fraud-off world. If it moves, the
    world that Level 11 promised is no longer the world being produced.

    Args:
        None.

    Returns:
        None. The test fails with an explanation if the hash differs.

    Example:
        >>> test_fraud_off_events_hash_is_unchanged()  # computes 24fc0e25009d79462d50... and passes
    """
    # Read the world settings and the expected hash from the frozen config.
    frozen = load_frozen_anchors()["fraud_off_events"]
    settings = FastRunSettings(
        seed=frozen["seed"], n_people=frozen["n_people"], n_shops=frozen["n_shops"],
        n_days=frozen["n_days"], range_name=frozen["range_name"],
    )
    # Run the world and hash all event lines joined together, the same recipe Level 11 check 11 uses.
    lines = event_lines(run_fast_world(settings))
    actual_hash = hashlib.sha256("".join(lines).encode("utf-8")).hexdigest()
    assert actual_hash == frozen["sha256"], (
        "The fraud-off world changed: something touched Level 11's code path or random streams 0-3. "
        "Revert it; never refresh the hash."
    )


def test_level_11_showcase_config_still_pins_the_same_hash() -> None:
    """
    WHAT: Reads config/showcase_level_11.yaml and checks that its expected_invariants.events_sha256 is still
    the same hash as the frozen one.

    WHY: Someone could make Level 11's showcase pass again by editing its pinned hash instead of fixing the
    code. This test catches that, because the two files must keep agreeing.

    Args:
        None.

    Returns:
        None. The test fails with an explanation if the two hashes differ.

    Example:
        >>> test_level_11_showcase_config_still_pins_the_same_hash()  # both start with 24fc0e25, so it passes
    """
    # Read the hash the Level 11 showcase compares against.
    showcase_config = load_yaml(REPO_ROOT / "config" / "showcase_level_11.yaml")
    showcase_hash = showcase_config["expected_invariants"]["events_sha256"]
    # It must match the frozen hash exactly.
    assert showcase_hash == load_frozen_anchors()["fraud_off_events"]["sha256"], (
        "config/showcase_level_11.yaml no longer pins the frozen fraud-off hash: it was refreshed. "
        "Revert that edit."
    )


def test_golden_fingerprints_reproduce() -> None:
    """
    WHAT: Re-runs the four golden checks FP, G1, G2 and G3 using scripts/verify_world.py's own functions,
    with the golden config loaded the same way Level 11 check 16 loads it.

    WHY: FP covers the people, shops and banks the world is built with; G1 the monthly ledger; G2 the calendar
    payment events; G3 the device stamps and phones. Reusing the existing check functions means this test can
    never disagree with the verify_world script about what "reproduced" means.

    Args:
        None.

    Returns:
        None. The test fails and names every fingerprint that did not reproduce.

    Example:
        >>> test_golden_fingerprints_reproduce()  # all four of FP, G1, G2, G3 return True, so it passes
    """
    # Load the golden values exactly as Level 11 check 16 does.
    golden_config = load_yaml(REPO_ROOT / "scripts" / "golden_values.yaml")
    # Run each check; each returns (passed, expected text, actual text).
    results = {
        "FP": check_fp(golden_config), "G1": check_g1(golden_config),
        "G2": check_g2(golden_config), "G3": check_g3(golden_config),
    }
    # Collect the names of the fingerprints that did not reproduce.
    failed = [name for name, (passed, _, _) in results.items() if not passed]
    assert not failed, (
        f"Golden fingerprints {failed} no longer reproduce: world building, ledgers, payments or devices changed. "
        "Revert the change; never refresh the golden values."
    )


def test_golden_values_file_is_unchanged() -> None:
    """
    WHAT: Hashes scripts/golden_values.yaml in canonical form and compares it with the frozen
    golden_values_canonical_sha256, measured from the file at commit 7757091.

    WHY: The fingerprint test above passes if code and golden file are changed together. This test pins the
    golden file itself, so refreshing a golden value is caught even when the code was changed to match it.

    Args:
        None.

    Returns:
        None. The test fails with an explanation if the golden file's values changed.

    Example:
        >>> test_golden_values_file_is_unchanged()  # computes 324cecb443c646b0... and passes
    """
    # Hash the golden file's values, ignoring line endings and key order.
    golden_text = (REPO_ROOT / "scripts" / "golden_values.yaml").read_text(encoding="utf-8")
    actual_hash = canonical_yaml_sha256(golden_text)
    assert actual_hash == load_frozen_anchors()["golden_values_canonical_sha256"], (
        "scripts/golden_values.yaml was edited: a golden value was refreshed. Revert it to the 7757091 values."
    )


def test_level_11_issue_register_is_unchanged() -> None:
    """
    WHAT: Loads config/issues.yaml with core.issue_register's own functions, checks it has no errors, and
    checks the number of IDs and how many are built, planned and deferred.

    WHY: Level 12 keeps its new issues in a separate register. The Level 11 register must stay exactly as
    shipped (31 IDs: 28 built, 2 planned, 1 deferred), because Level 11 check 2 asserts those numbers.

    Args:
        None.

    Returns:
        None. The test fails with an explanation if errors appear or a count moved.

    Example:
        >>> test_level_11_issue_register_is_unchanged()  # 31 IDs, 28 + 2 + 1, no errors, so it passes
    """
    # Read the expected counts and the real register.
    frozen = load_frozen_anchors()["level_11_register"]
    register = load_register(REPO_ROOT / "config" / "issues.yaml")
    # The register's own checker must find nothing wrong.
    errors = check_register(register, REPO_ROOT)
    assert errors == [], f"The Level 11 issue register no longer validates: {errors}. Revert the edit to it."
    # Count each status the same way core.issue_register.main does.
    counts = {
        status: sum(1 for entry in register.values() if isinstance(entry, dict) and entry.get("status") == status)
        for status in ("built", "planned", "deferred")
    }
    expected = {status: frozen[status] for status in ("built", "planned", "deferred")}
    assert len(register) == frozen["total_ids"] and counts == expected, (
        f"The Level 11 issue register moved ({len(register)} IDs, {counts}): Level 12 issues belong in "
        "config/issues_v2.yaml, not in config/issues.yaml."
    )


def test_grounded_parameter_counts_are_unchanged() -> None:
    """
    WHAT: Builds the grounding table with world.grounding's own functions and counts its rows by label.
    It expects 33 rows: 30 labelled assumption and 3 labelled reported.

    WHY: The grounding table only reads the background, failure and device realism files. If the count moves,
    one of those Level 11 files was edited, which changes the fraud-off world's settings.

    Args:
        None.

    Returns:
        None. The test fails with an explanation if any count moved.

    Example:
        >>> test_grounded_parameter_counts_are_unchanged()  # 33 rows, 30 + 3, so it passes
    """
    # Read the expected counts and build the real grounding table.
    frozen = load_frozen_anchors()["grounded_parameters"]
    rows = grounding_rows(read_world_configs())
    label_counts = count_by_label(rows)
    # Compare the total and both label counts in one go.
    actual = (len(rows), label_counts.get("assumption", 0), label_counts.get("reported", 0))
    expected = (frozen["total"], frozen["assumption"], frozen["reported"])
    assert actual == expected, (
        f"Grounded parameters moved (total, assumption, reported = {actual}): a Level 11 realism config file "
        "was edited. Put new Level 12 numbers in new config files instead."
    )


def test_stress_scenario_file_count_is_unchanged() -> None:
    """
    WHAT: Counts the YAML files directly inside scenarios/stress/ (not in sub-folders) and expects 6.
    The path is built from the repo root, so the result does not depend on the working directory.

    WHY: Level 11's stress library has exactly 6 labelled layers, and Level 11 check 15 asserts that. New fraud
    stress files belong in scenarios/stress_fraud/, a sibling folder this count deliberately ignores.

    Args:
        None.

    Returns:
        None. The test fails with an explanation if the count moved.

    Example:
        >>> test_stress_scenario_file_count_is_unchanged()  # finds 6 files, so it passes
    """
    # List only the top level of scenarios/stress/, the same non-recursive pattern Level 11 uses.
    stress_files = sorted((REPO_ROOT / "scenarios" / "stress").glob("*.yaml"))
    assert len(stress_files) == load_frozen_anchors()["stress_scenario_files"], (
        f"scenarios/stress/ now holds {len(stress_files)} files: a Level 11 stress file was added or removed. "
        "Fraud stress files go in scenarios/stress_fraud/."
    )
