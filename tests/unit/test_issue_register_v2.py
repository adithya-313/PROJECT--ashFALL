"""
WHAT: Tests for core/issue_register_v2.py, the checker of config/issues_v2.yaml. One test shows that the real
register passes. Then each rule is shown to bite: a broken in-memory copy of the register (or a broken copy
of a config file in a temporary folder) is rejected, and each rejection has its own plain message.

WHY: A checker that never fails is worthless. These tests prove every rule can fail. They never edit the
real files, and the last test proves that the Level 11 register and its counts are untouched.

TERM: broken copy: a copy of the register with one deliberate mistake, e.g. one ID deleted.
TERM: revision_2 ID: an ID first used in Revision 2 of the report, e.g. "K1" or "12-7".
TERM: level_11 ID: an ID that already exists in config/issues.yaml, e.g. "G1".
TERM: evidence: a config block, test function, amendment or document text that proves a status,
      e.g. test_part_a_runs_only_part_a_checks for 12-20.
TERM: tag: the issue_ids_v2 list inside a config block, naming the register IDs it belongs to.
"""

import copy
from pathlib import Path
from typing import Any

from core.config_validator import load_yaml
from core.issue_register import EXPECTED_IDS
from core.issue_register import check_register as check_v1_register
from core.issue_register import load_register as load_v1_register
from core.issue_register_v2 import (
    EXPECTED_V2_IDS,
    REGISTER_PATH,
    check_config_tags,
    check_register,
    count_statuses,
)

REPO_ROOT: Path = Path(__file__).resolve().parents[2]


def real_register() -> dict[str, Any]:
    """
    WHAT: Loads config/issues_v2.yaml and returns a fresh deep copy, so a test can break the copy without
    any chance of changing the real data or another test's copy.

    WHY: Every broken-copy test starts from the real register and makes exactly one mistake. A shared
    object that one test edited would make the next test fail for the wrong reason.

    Args:
        None.

    Returns:
        A dictionary with the keys "meta" and "issues".

    Example:
        >>> copy_one = real_register()
        >>> copy_one["issues"]["K1"]["status"]
        'built'
        >>> del copy_one["issues"]["K1"]
        >>> real_register()["issues"]["K1"]["status"]
        'built'
    """
    return copy.deepcopy(load_yaml(REGISTER_PATH))


def errors_of(register: dict[str, Any]) -> str:
    """
    WHAT: Runs the full v2 check on a register against the real repository and returns all error messages
    joined into one text, so a test can look for a phrase in it.

    WHY: Each test asserts on the wording of one message. Joining them keeps those assertions short and
    makes a failure print every message at once.

    Args:
        register: A register dictionary, usually a broken copy.

    Returns:
        All error strings joined by newlines. An empty text means no errors.

    Example:
        >>> errors_of(real_register())
        ''
        >>> broken = real_register(); del broken["issues"]["K1"]
        >>> "Missing revision_2 IDs (1): ['K1']" in errors_of(broken)
        True
    """
    return "\n".join(check_register(register, REPO_ROOT))


def test_real_register_passes() -> None:
    """
    WHAT: Checks that the real config/issues_v2.yaml has no errors, covers exactly the 77 revision_2 IDs
    from Section 27, and holds the 16 level_11 IDs that later levels carry on (93 entries in total).

    WHY: This is the main promise of the register. It also fixes the totals, so an ID cannot be added
    or dropped without this test noticing.

    Returns:
        None. The test fails and prints every error if the register is not valid.

    Example:
        77 revision_2 IDs + 16 level_11 IDs = 93 entries; the counts are built 18, partial 11,
        planned 63 and deferred 1, which also add up to 93. Level 12b moved 12-1 to built and
        12-17 and K5 to partial; Level 12c C-1 moved 12-19 to partial; Level 12c C-2 moved G6 to partial;
        Level 12c C-3a moved G1 to partial and 12-19 to built;
        Level 12c C-3b moved G1 to built and G3 to partial;
        Level 12c C-4 moved G2 to partial, and A2 and 12-17 to built.
        Level 12d D-1 moved G3 to built and G4 to partial.
    """
    register = real_register()
    assert check_register(register, REPO_ROOT) == []
    issues = register["issues"]
    assert len(EXPECTED_V2_IDS) == 77
    assert len(issues) == 93
    assert count_statuses(issues) == {"built": 19, "partial": 11, "planned": 62, "deferred": 1}


def test_missing_id_is_rejected() -> None:
    """
    WHAT: Deletes revision_2 ID "13-17" from a copy of the register and checks the checker names it as missing.

    WHY: A forgotten ID would silently drop one finding from the plan. Naming the missing ID makes the fix
    obvious.

    Returns:
        None.

    Example:
        With "13-17" removed, 76 of 77 revision_2 IDs remain, and the message says
        "Missing revision_2 IDs (1): ['13-17']".
    """
    register = real_register()
    del register["issues"]["13-17"]
    assert "Missing revision_2 IDs (1): ['13-17']" in errors_of(register)


def test_extra_unknown_id_is_rejected() -> None:
    """
    WHAT: Adds a revision_2 entry called "Z9", which is not in Section 27, and checks the checker calls it unknown.

    WHY: An invented ID would claim a finding the report never made. The ID set must equal Section 27 exactly,
    in both directions.

    Returns:
        None.

    Example:
        With "Z9" added there are 78 revision_2 IDs against 77 expected, and the message says
        "Unknown revision_2 IDs (1): ['Z9']".
    """
    register = real_register()
    register["issues"]["Z9"] = copy.deepcopy(register["issues"]["K2"])
    assert "Unknown revision_2 IDs (1): ['Z9']" in errors_of(register)


def test_revision_2_id_colliding_with_level_11_is_rejected() -> None:
    """
    WHAT: Changes the origin of entry "G1" to revision_2, so a revision_2 entry now uses a Level 11 ID,
    and checks the checker reports the collision.

    WHY: If "G1" could mean two things, a status in one register could be read as the other's. Level 11's
    31 IDs are reserved for Level 11.

    Returns:
        None.

    Example:
        G1 is one of the 31 IDs in EXPECTED_IDS, so the message is
        "G1: revision_2 ID collides with a Level 11 ID".
    """
    register = real_register()
    register["issues"]["G1"]["origin"] = "revision_2"
    assert "G1: revision_2 ID collides with a Level 11 ID" in errors_of(register)


def test_level_11_id_not_in_v1_is_rejected() -> None:
    """
    WHAT: Adds a level_11 entry called "Q7", which config/issues.yaml does not have, and checks that the
    checker rejects it.

    WHY: A level_11 entry is a promise to continue a Level 11 issue. If the issue does not exist in Level 11,
    there is nothing to continue.

    Returns:
        None.

    Example:
        "Q7" is not among the 31 Level 11 IDs, so the message is
        "Q7: level_11 ID is not a Level 11 ID".
    """
    register = real_register()
    register["issues"]["Q7"] = copy.deepcopy(register["issues"]["G1"])
    assert "Q7: level_11 ID is not a Level 11 ID" in errors_of(register)


def test_v1_status_mismatch_is_rejected() -> None:
    """
    WHAT: Sets the copied v1_status of "G1" to planned while the live Level 11 register says built, and checks
    the checker reports the difference.

    WHY: The copy is only trustworthy while it equals the live value. This is the check that makes the copy
    fail the day Level 11's register changes.

    Returns:
        None.

    Example:
        config/issues.yaml says G1 is built, so the message is
        "G1: v1_status is planned but config/issues.yaml says built".
    """
    register = real_register()
    register["issues"]["G1"]["v1_status"] = "planned"
    assert "G1: v1_status is planned but config/issues.yaml says built" in errors_of(register)


def test_bad_status_is_rejected() -> None:
    """
    WHAT: Gives entry "K2" the status word "finished", which is not in meta.statuses, and checks the checker
    rejects it.

    WHY: The status vocabulary is closed (built, partial, planned, deferred). A made-up word could not be
    counted or planned from.

    Returns:
        None.

    Example:
        The message is "K2: status 'finished' is not one of ['built', 'partial', 'planned', 'deferred']".
    """
    register = real_register()
    register["issues"]["K2"]["status"] = "finished"
    assert "K2: status 'finished' is not one of" in errors_of(register)


def test_built_without_evidence_is_rejected() -> None:
    """
    WHAT: Removes the evidence list from the built entry "K1" and checks the checker demands evidence.

    WHY: "Built" without anything to look up is just a claim. The register exists to stop claims
    without proof.

    Returns:
        None.

    Example:
        K1 has 2 evidence items; with them removed the message is
        "K1: built needs at least one evidence item".
    """
    register = real_register()
    del register["issues"]["K1"]["evidence"]
    assert "K1: built needs at least one evidence item" in errors_of(register)


def test_evidence_path_that_does_not_exist_is_rejected() -> None:
    """
    WHAT: Points the first evidence item of "K1" at a test file that does not exist and checks the checker
    reports the missing file.

    WHY: A renamed or deleted file would otherwise leave a status resting on nothing.

    Returns:
        None.

    Example:
        The message is "K1: file tests/unit/world/test_no_such_file.py not found".
    """
    register = real_register()
    # The first evidence item may be a config block; the file check applies to every kind.
    register["issues"]["K1"]["evidence"][0]["file"] = "tests/unit/world/test_no_such_file.py"
    assert "K1: file tests/unit/world/test_no_such_file.py not found" in errors_of(register)


def test_test_function_that_does_not_exist_is_rejected() -> None:
    """
    WHAT: Renames the test function named by the first evidence item of "K1" to one that does not exist in
    that file, and checks the checker reports it.

    WHY: A file can exist while the one test that backs the claim has been renamed or removed. Naming the
    function closes that gap.

    Returns:
        None.

    Example:
        The message starts "K1: test function test_not_a_real_test not found in".
    """
    register = real_register()
    # Pick the first evidence item that is a test, because only a test names a function.
    test_item = next(item for item in register["issues"]["K1"]["evidence"] if item["kind"] == "test")
    test_item["name"] = "test_not_a_real_test"
    assert "K1: test function test_not_a_real_test not found in" in errors_of(register)


def test_planned_without_later_work_is_rejected() -> None:
    """
    WHAT: Removes the later_work list from the planned entry "K2" and checks the checker demands it.

    WHY: A planned item with no later_work has no owner and no level, so nothing would ever build it.

    Returns:
        None.

    Example:
        K2 normally lists 1 note, "Level 4: ..."; with the list removed the message is
        "K2: planned needs a non-empty later_work".
    """
    register = real_register()
    del register["issues"]["K2"]["later_work"]
    assert "K2: planned needs a non-empty later_work" in errors_of(register)


def test_amendment_reference_to_a_missing_amendment_is_rejected() -> None:
    """
    WHAT: Points the amendment evidence of "12-6" first at A-099, which does not exist, then at A-012, which
    exists but lists only 12-7, and checks that each gets its own message.

    WHY: Amendment evidence only proves something if the amendment really exists and really names the ID
    in its "| Issue IDs v2 |" row.

    Returns:
        None.

    Example:
        A-099 gives "12-6: amendment A-099 not found". A-012 gives
        "12-6: amendment A-012 does not list this ID".
    """
    register = real_register()
    amendment_item = register["issues"]["12-6"]["evidence"][0]
    assert amendment_item["kind"] == "amendment"
    amendment_item["id"] = "A-099"
    assert "12-6: amendment A-099 not found" in errors_of(register)
    amendment_item["id"] = "A-012"
    assert "12-6: amendment A-012 does not list this ID" in errors_of(register)


def write_config(tmp_path: Path, text: str) -> Path:
    """
    WHAT: Creates a temporary root folder with a config folder that holds one YAML file with the given text,
    and returns the root.

    WHY: The config-tag checks scan every file in root/config. Building a tiny fake root keeps the real config
    folder untouched while still testing the real scanner.

    Args:
        tmp_path: A temporary folder given by pytest.
        text: The YAML text to write into config/sample.yaml.

    Returns:
        The temporary root folder.

    Example:
        write_config(tmp_path, "block:\\n  issue_ids_v2: [K1]\\n") returns tmp_path, and
        (tmp_path / "config" / "sample.yaml") then holds those 2 lines.
    """
    (tmp_path / "config").mkdir()
    (tmp_path / "config" / "sample.yaml").write_text(text, encoding="utf-8")
    return tmp_path


def test_tag_naming_an_unknown_id_is_rejected(tmp_path: Path) -> None:
    """
    WHAT: Writes a config block tagged issue_ids_v2: [99-99], an ID the register does not have, and checks
    the scanner reports the unknown ID.

    WHY: A tag is a claim that the block belongs to a register entry. A tag naming nothing is a typo
    that would hide the block from every search.

    Args:
        tmp_path: A temporary folder given by pytest.

    Returns:
        None.

    Example:
        The message is "config/sample.yaml: issue_ids_v2 names unknown ID '99-99'".
    """
    root = write_config(tmp_path, 'block:\n  issue_ids_v2: ["99-99"]\n  derived_from: "x"\n')
    errors = check_config_tags(root, set(real_register()["issues"]))
    assert errors == ["config/sample.yaml: issue_ids_v2 names unknown ID '99-99'"]


def test_tagged_block_without_derived_from_is_rejected(tmp_path: Path) -> None:
    """
    WHAT: Writes a config block tagged issue_ids_v2: [K1] with no derived_from, and checks the scanner
    reports the missing note.

    WHY: Ground rule 2: every number must say where it came from. A tagged block without derived_from breaks
    that rule.

    Args:
        tmp_path: A temporary folder given by pytest.

    Returns:
        None.

    Example:
        The message is "config/sample.yaml: block with issue_ids_v2 ['K1'] is missing derived_from".
    """
    root = write_config(tmp_path, "block:\n  issue_ids_v2: [K1]\n")
    errors = check_config_tags(root, set(real_register()["issues"]))
    assert errors == ["config/sample.yaml: block with issue_ids_v2 ['K1'] is missing derived_from"]


def test_revision_2_id_under_the_old_key_is_rejected(tmp_path: Path) -> None:
    """
    WHAT: Writes a config block with issue_ids: [K1], the old Level 11 key, and checks the scanner says plainly
    that revision_2 IDs belong under issue_ids_v2.

    WHY: The Level 11 checker rejects unknown IDs under issue_ids, so a revision_2 ID there would break Level
    11's own check. The message tells the author what to change.

    Args:
        tmp_path: A temporary folder given by pytest.

    Returns:
        None.

    Example:
        The message is "config/sample.yaml: revision_2 ID 'K1' is under issue_ids; use issue_ids_v2".
    """
    root = write_config(tmp_path, "block:\n  issue_ids: [K1]\n")
    errors = check_config_tags(root, set(real_register()["issues"]))
    assert errors == ["config/sample.yaml: revision_2 ID 'K1' is under issue_ids; use issue_ids_v2"]


def test_level_11_register_is_untouched() -> None:
    """
    WHAT: After all the broken copies above, runs the Level 11 checker with its own functions and checks it
    still reports 31 IDs: built 28, planned 2 and deferred 1, with no errors.

    WHY: The whole point of a separate module is that Level 11's register, checker and counts stay exactly
    as Level 11 shipped them. This test proves the new module changed none of them.

    Returns:
        None.

    Example:
        28 built + 2 planned + 1 deferred = 31, the same as len(EXPECTED_IDS).
    """
    v1_register = load_v1_register(REPO_ROOT / "config" / "issues.yaml")
    assert check_v1_register(v1_register, REPO_ROOT) == []
    assert len(v1_register) == len(EXPECTED_IDS) == 31
    assert count_statuses(v1_register) == {"built": 28, "planned": 2, "deferred": 1}
