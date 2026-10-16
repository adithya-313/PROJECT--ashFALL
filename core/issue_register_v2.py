"""
WHAT: Checks the version 2 issue register, config/issues_v2.yaml, against reality. It holds the findings of
Revision 2 of the project report (Section 27) and the Level 11 issues that later levels carry on. Every
claim of "built" or "partial" is looked up: the named config block, test function, amendment or document
text must really exist.

WHY: The Level 11 register (config/issues.yaml, checked by core/issue_register.py) has a fixed set of 31 IDs
that tests pin, so new findings cannot be added to it. This module is a separate checker, so the Level 11
register, its checker and its counts stay exactly as Level 11 shipped them. It only READS EXPECTED_IDS and
load_register from the old module and never changes them.

TERM: issue register: a list of known problems, each with a status, so none is silently forgotten,
      e.g. "12-6" is "built" and "12-8" is "partial".
TERM: revision_2 ID: an ID first used in Revision 2 of the report, e.g. "K1", "F4" or "12-7".
TERM: level_11 ID: an ID that already exists in config/issues.yaml, e.g. "G1".
      Its v1_status in the new register must equal the live status in the old one.
TERM: evidence: something a script can open to show a status is true,
      e.g. a test function name inside a test file.
TERM: tag: a list called issue_ids_v2 inside a config block, naming the register IDs the block belongs to,
      e.g. issue_ids_v2: [12-7] on world_fraud_inflation_factor.
TERM: derived_from: a short note saying where a tagged block's numbers came from,
      e.g. "amendment A-012".

Use from the command line:
    python -m core.issue_register_v2
"""

import re
import sys
from pathlib import Path
from typing import Any, Iterator

from core.config_validator import load_yaml
from core.issue_register import EXPECTED_IDS, find_amendment_issue_ids, load_register

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
REGISTER_PATH: Path = REPO_ROOT / "config" / "issues_v2.yaml"

ALLOWED_ORIGINS: tuple[str, ...] = ("revision_2", "level_11")
ALLOWED_KINDS: tuple[str, ...] = ("config", "amendment", "doc", "test")
V2_ROW_NAME: str = "| Issue IDs v2 |"

# The revision_2 IDs, exactly as listed in Section 27 of the report. Where Section 27 groups IDs on one row,
# each ID is listed here separately. Level 13, 14 and 15 skip numbers on purpose (for example 14-1 and 15-4).
EXPECTED_V2_IDS: frozenset[str] = frozenset(
    [f"K{number}" for number in range(1, 10)]
    + [f"X{number}" for number in range(1, 9)]
    + [f"F{number}" for number in range(1, 7)]
    + [f"12-{number}" for number in range(1, 21)]
    + [f"13-{number}" for number in (2, 3, 5, 9, 10, 11, 12, 14, 15, 16, 17)]
    + [f"14-{number}" for number in (2, 4, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16)]
    + [f"15-{number}" for number in (1, 2, 3, 5, 6, 7, 8, 9, 10, 11, 12)]
)


def load_v2_register(path: Path) -> dict[str, Any]:
    """
    WHAT: Reads the version 2 register file and returns it as a dictionary. Unlike the Level 11 loader it does
    not hide a broken file: a missing file or bad YAML raises an error, which main() turns into a message.

    WHY: An empty answer for a broken file would look the same as "no errors found". Raising keeps a
    broken register from passing the check by accident.

    Args:
        path: Path to the register file, normally config/issues_v2.yaml.

    Returns:
        The parsed YAML as a dictionary with the top-level keys "meta" and "issues".

    Example:
        >>> sorted(load_v2_register(REGISTER_PATH))
        ['issues', 'meta']
    """
    # load_yaml raises on a missing file or invalid YAML, which is what we want here.
    return load_yaml(path)


def check_top_level(register: Any) -> list[str]:
    """
    WHAT: Checks that the register is a dictionary with a "meta" block and an "issues" block, that meta carries
    a policy label, a reason, a derived_from note and a statuses list, and that issues is a non-empty dictionary.

    WHY: Every other check reads these blocks. Failing early with a plain message is better than a crash
    with a confusing KeyError halfway through.

    Args:
        register: The parsed register, whatever the YAML turned out to be.

    Returns:
        A list of error strings. Empty means the shape is fine.

    Example:
        >>> check_top_level({"meta": {}, "issues": {}})[-1]
        'issues: block is missing or empty'
        >>> len(check_top_level({"meta": {}, "issues": {}}))
        5
        >>> check_top_level([])
        ['register must be a dictionary with meta and issues']
    """
    if not isinstance(register, dict):
        return ["register must be a dictionary with meta and issues"]
    meta = register.get("meta")
    meta = meta if isinstance(meta, dict) else {}
    errors: list[str] = []
    # The meta block follows the project rule that every decision carries a label and a reason.
    if meta.get("label") != "policy":
        errors.append("meta: label must be policy")
    for key in ("reason", "derived_from"):
        if not meta.get(key):
            errors.append(f"meta: {key} is missing")
    if not isinstance(meta.get("statuses"), list) or not meta.get("statuses"):
        errors.append("meta: statuses must be a non-empty list")
    if not isinstance(register.get("issues"), dict) or not register.get("issues"):
        errors.append("issues: block is missing or empty")
    return errors


def check_id_set(issues: dict[str, Any]) -> list[str]:
    """
    WHAT: Compares the revision_2 IDs in the register with EXPECTED_V2_IDS, and names every missing and every
    extra ID. Entries whose origin is level_11 are left out of this comparison.

    WHY: The set of IDs comes from Section 27 of the report. A forgotten ID would silently drop a finding, and
    an invented ID would claim a finding the report never made.

    Args:
        issues: The "issues" dictionary of the register, keyed by ID.

    Returns:
        A list of error strings, one for the missing IDs and one for the extra IDs, when there are any.

    Example:
        >>> check_id_set({"K1": {"origin": "revision_2"}})[0][:38]
        "Missing revision_2 IDs (76): ['12-1', "
        >>> len(EXPECTED_V2_IDS)
        77
    """
    found = {
        issue_id
        for issue_id, entry in issues.items()
        if isinstance(entry, dict) and entry.get("origin") == "revision_2"
    }
    errors: list[str] = []
    missing = sorted(EXPECTED_V2_IDS - found)
    extra = sorted(found - EXPECTED_V2_IDS)
    # Report the two kinds of mistake separately, with the IDs named, so the fix is obvious.
    if missing:
        errors.append(f"Missing revision_2 IDs ({len(missing)}): {missing}")
    if extra:
        errors.append(f"Unknown revision_2 IDs ({len(extra)}): {extra}")
    return errors


def check_origins(issues: dict[str, Any], v1_register: dict[str, Any]) -> list[str]:
    """
    WHAT: Checks how the register links to the Level 11 register. A revision_2 ID must not be a Level 11 ID.
    A level_11 ID must be a real Level 11 ID, and its v1_status must equal that ID's live status in
    config/issues.yaml. An origin that is neither word is also an error.

    WHY: Two registers that reuse an ID would make "G1" mean two things. A copied status that is not compared
    with the live one would go stale the day Level 11's register changes.

    Args:
        issues: The "issues" dictionary of the version 2 register.
        v1_register: The Level 11 register as loaded from config/issues.yaml.

    Returns:
        A list of error strings, each starting with the issue ID.

    Example:
        >>> check_origins({"G1": {"origin": "revision_2"}}, {"G1": {"status": "built"}})
        ['G1: revision_2 ID collides with a Level 11 ID']
        >>> check_origins({"G1": {"origin": "level_11", "v1_status": "planned"}}, {"G1": {"status": "built"}})
        ['G1: v1_status is planned but config/issues.yaml says built']
    """
    errors: list[str] = []
    for issue_id, entry in issues.items():
        origin = entry.get("origin") if isinstance(entry, dict) else None
        if origin == "revision_2" and issue_id in EXPECTED_IDS:
            errors.append(f"{issue_id}: revision_2 ID collides with a Level 11 ID")
        elif origin == "level_11":
            errors.extend(check_level_11_link(issue_id, entry, v1_register))
        elif origin != "revision_2":
            errors.append(f"{issue_id}: origin must be one of {list(ALLOWED_ORIGINS)}, got {origin!r}")
    return errors


def check_level_11_link(issue_id: str, entry: dict[str, Any], v1_register: dict[str, Any]) -> list[str]:
    """
    WHAT: Checks one level_11 entry: its ID must be one of the 31 known Level 11 IDs, and its v1_status must
    equal the status that config/issues.yaml holds for that ID right now.

    WHY: This is the link that stops the two registers drifting apart. If Level 11's register ever changes,
    this check fails until the copy is updated on purpose.

    Args:
        issue_id: The ID being checked, for example "R9".
        entry: The version 2 entry for that ID.
        v1_register: The Level 11 register as loaded from config/issues.yaml.

    Returns:
        A list of error strings, empty when the link is correct.

    Example:
        >>> check_level_11_link("ZZ", {"v1_status": "built"}, {})
        ['ZZ: level_11 ID is not a Level 11 ID']
        >>> check_level_11_link("R9", {"v1_status": "deferred"}, {"R9": {"status": "deferred"}})
        []
    """
    if issue_id not in EXPECTED_IDS or issue_id not in v1_register:
        return [f"{issue_id}: level_11 ID is not a Level 11 ID"]
    live_status = v1_register[issue_id].get("status")
    # The copy must match the live status exactly, so a change in Level 11's register cannot go unnoticed.
    if entry.get("v1_status") != live_status:
        return [f"{issue_id}: v1_status is {entry.get('v1_status')} but config/issues.yaml says {live_status}"]
    return []


def check_levels(issue_id: str, entry: dict[str, Any]) -> list[str]:
    """
    WHAT: Checks that the entry's levels value is a non-empty list of whole numbers, and that parts, when
    present, is a list of text such as "12a". True and False are not accepted as numbers.

    WHY: The levels say which later levels own the work. Text such as "Level 12" or an empty list could not be
    sorted or searched, so the register would stop being useful as a plan.

    Args:
        issue_id: The ID being checked, used to start each message.
        entry: The register entry.

    Returns:
        A list of error strings, empty when levels and parts are fine.

    Example:
        >>> check_levels("K1", {"levels": [12, 13]})
        []
        >>> check_levels("K1", {"levels": []})
        ['K1: levels must be a non-empty list of whole numbers']
    """
    levels = entry.get("levels")
    errors: list[str] = []
    # bool is a kind of int in Python, so it is rejected by name.
    whole_numbers = isinstance(levels, list) and all(isinstance(n, int) and not isinstance(n, bool) for n in levels)
    valid_levels = whole_numbers and bool(levels)
    if not valid_levels:
        errors.append(f"{issue_id}: levels must be a non-empty list of whole numbers")
    parts = entry.get("parts", [])
    if not isinstance(parts, list) or not all(isinstance(part, str) for part in parts):
        errors.append(f"{issue_id}: parts must be a list of text such as 12a")
    return errors


def check_entry_fields(issue_id: str, entry: Any, statuses: list[str]) -> list[str]:
    """
    WHAT: Checks the plain fields of one entry: a title, a status from meta.statuses, valid levels and parts,
    a non-empty later_work for partial, planned and deferred entries, and a Level-N shape for each later_work
    note. A deferred entry also needs a reason. Evidence is checked separately.

    WHY: A planned item with no later_work is a promise nobody owns. Checking the shape of every entry is what
    makes the register trustworthy enough to plan from.

    Args:
        issue_id: The ID being checked.
        entry: The register entry, expected to be a dictionary.
        statuses: The allowed status words, from meta.statuses.

    Returns:
        A list of error strings, each starting with the issue ID.

    Example:
        >>> check_entry_fields("K2", {"title": "t", "status": "planned", "levels": [4]}, ["planned"])
        ['K2: planned needs a non-empty later_work']
        >>> check_entry_fields("K2", {"title": "t", "status": "odd", "levels": [4]}, ["planned"])
        ["K2: status 'odd' is not one of ['planned']"]
    """
    if not isinstance(entry, dict):
        return [f"{issue_id}: entry must be a dictionary"]
    errors = check_levels(issue_id, entry)
    if not isinstance(entry.get("title"), str) or not entry["title"].strip():
        errors.append(f"{issue_id}: title is missing or empty")
    status = entry.get("status")
    if status not in statuses:
        errors.append(f"{issue_id}: status {status!r} is not one of {statuses}")
    later_work = entry.get("later_work", [])
    if status in ("partial", "planned", "deferred") and not later_work:
        errors.append(f"{issue_id}: {status} needs a non-empty later_work")
    if status == "deferred" and not entry.get("reason"):
        errors.append(f"{issue_id}: deferred needs a reason")
    for note in later_work if isinstance(later_work, list) else []:
        if not isinstance(note, str) or not note.startswith("Level "):
            errors.append(f"{issue_id}: later_work note {note!r} must start with 'Level '")
    return errors


def find_v2_amendment_ids(root: Path) -> dict[str, list[str]]:
    """
    WHAT: Reads docs/research/amendments.md and returns, for each "## A-nnn" heading, the IDs listed in its
    "| Issue IDs v2 |" row. An amendment without that row is left out.

    WHY: The Level 11 parser only reads the "| Issue IDs |" row, so version 2 IDs live in their own row and
    this function reads that one. The two rows never get mixed up because their first cell text differs.

    Args:
        root: The repository root (or a copy of it) that holds docs/research/amendments.md.

    Returns:
        A dictionary such as {"A-012": ["12-7"], "A-014": ["12-8"]}.

    Example:
        >>> find_v2_amendment_ids(REPO_ROOT)["A-012"]
        ['12-7']
        >>> find_v2_amendment_ids(REPO_ROOT)["A-016"]
        ['12-9']
    """
    result: dict[str, list[str]] = {}
    amendments_file = root / "docs" / "research" / "amendments.md"
    if not amendments_file.exists():
        return result
    current = None
    for line in amendments_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        # A heading starts a new amendment; the version 2 row, if it exists, comes after it.
        if line.startswith("## A-"):
            current = line[3:].strip()
        elif current and line.startswith(V2_ROW_NAME):
            cells = line.split("|")
            result[current] = [item.strip() for item in cells[2].split(",") if item.strip()]
    return result


def check_amendment_evidence(issue_id: str, origin: str, item: dict[str, Any], root: Path) -> list[str]:
    """
    WHAT: Checks that the named amendment exists and lists this ID. For a revision_2 ID the "| Issue IDs v2 |"
    row is read; for a level_11 ID the old "| Issue IDs |" row is read, through the Level 11 parser.

    WHY: An amendment is the written record of why a change was made. Evidence that points at an amendment
    which does not mention the ID would look like proof but prove nothing.

    Args:
        issue_id: The ID being checked, for example "12-7" or "E1".
        origin: "revision_2" or "level_11", which picks the row to read.
        item: The evidence item, with an "id" such as "A-012".
        root: The repository root that holds docs/research/amendments.md.

    Returns:
        A list of error strings, empty when the amendment lists the ID.

    Example:
        >>> check_amendment_evidence("12-7", "revision_2", {"id": "A-012"}, REPO_ROOT)
        []
        >>> check_amendment_evidence("12-7", "revision_2", {"id": "A-099"}, REPO_ROOT)
        ['12-7: amendment A-099 not found']
    """
    amendment_id = item.get("id", "")
    if origin == "level_11":
        listed = find_amendment_issue_ids(root)
    else:
        listed = find_v2_amendment_ids(root)
    if amendment_id not in listed:
        return [f"{issue_id}: amendment {amendment_id} not found"]
    if issue_id not in listed[amendment_id]:
        return [f"{issue_id}: amendment {amendment_id} does not list this ID"]
    return []


def check_config_evidence(issue_id: str, item: dict[str, Any], root: Path) -> list[str]:
    """
    WHAT: Checks that the named config file exists and, when a block name is given, that the block is a
    top-level key of the file or one of its "parameters".

    WHY: A register that points at a config block which was renamed or removed would keep claiming the work
    exists. Opening the file catches that.

    Args:
        issue_id: The ID being checked.
        item: The evidence item, with "file" and optionally "name".
        root: The repository root.

    Returns:
        A list of error strings, empty when the file and block exist.

    Example:
        >>> item = {"file": "config/tactic_costs.yaml", "name": "tactic_cost_units"}
        >>> check_config_evidence("12-9", item, REPO_ROOT)
        []
        >>> check_config_evidence("12-9", {"file": "config/nope.yaml"}, REPO_ROOT)
        ['12-9: file config/nope.yaml not found']
    """
    path = root / item.get("file", "")
    if not path.is_file():
        return [f"{issue_id}: file {item.get('file')} not found"]
    name = item.get("name")
    if name is None:
        return []
    data = load_yaml(path)
    parameters = data.get("parameters", {}) if isinstance(data, dict) else {}
    # A block may sit at the top of the file or inside its "parameters" block (as in world_ranges.yaml).
    if name not in data and name not in parameters:
        return [f"{issue_id}: block {name} not found in {item.get('file')}"]
    return []


def check_doc_evidence(issue_id: str, item: dict[str, Any], root: Path) -> list[str]:
    """
    WHAT: Checks that the named file exists and, when a "text" is given, that the file contains that text.

    WHY: A document or script is cheap to point at, so the text check is what makes the pointer mean
    something: the exact words that prove the claim must still be there.

    Args:
        issue_id: The ID being checked.
        item: The evidence item, with "file" and optionally "text".
        root: The repository root.

    Returns:
        A list of error strings, empty when the file exists and holds the text.

    Example:
        >>> item = {"file": "docs/research/amendments.md", "text": "## A-016"}
        >>> check_doc_evidence("F6", item, REPO_ROOT)
        []
        >>> check_doc_evidence("F6", {"file": "docs/none.md"}, REPO_ROOT)
        ['F6: file docs/none.md not found']
    """
    path = root / item.get("file", "")
    if not path.is_file():
        return [f"{issue_id}: file {item.get('file')} not found"]
    text = item.get("text")
    if text and text not in path.read_text(encoding="utf-8"):
        return [f"{issue_id}: text {text!r} not found in {item.get('file')}"]
    return []


def check_test_evidence(issue_id: str, item: dict[str, Any], root: Path) -> list[str]:
    """
    WHAT: Checks that the named test file exists and defines a function with the named test name, found by
    looking for a line that starts with "def <name>(".

    WHY: A test file that exists proves little: the claim rests on one named test. If that test is renamed
    or deleted, the status must be re-examined, so the lookup fails loudly.

    Args:
        issue_id: The ID being checked.
        item: The evidence item, with "file" and "name".
        root: The repository root.

    Returns:
        A list of error strings, empty when the function exists in the file.

    Example:
        >>> item = {"file": "tests/unit/test_showcase_part.py", "name": "test_part_a_runs_only_part_a_checks"}
        >>> check_test_evidence("12-20", item, REPO_ROOT)
        []
        >>> check_test_evidence("12-20", {"file": "tests/unit/test_showcase_part.py", "name": "test_zzz"}, REPO_ROOT)
        ['12-20: test function test_zzz not found in tests/unit/test_showcase_part.py']
    """
    path = root / item.get("file", "")
    name = str(item.get("name", ""))
    if not path.is_file():
        return [f"{issue_id}: file {item.get('file')} not found"]
    pattern = rf"^\s*def {re.escape(name)}\("
    if not name or not re.search(pattern, path.read_text(encoding="utf-8"), flags=re.MULTILINE):
        return [f"{issue_id}: test function {name} not found in {item.get('file')}"]
    return []


def check_evidence(issue_id: str, entry: dict[str, Any], root: Path) -> list[str]:
    """
    WHAT: For a built or partial entry, requires at least one evidence item and checks every item by its kind
    (config, amendment, doc or test). Entries with any other status are not checked here.

    WHY: "Built" is a claim. This is the function that refuses to accept the claim without something real
    to look at, which is the whole reason the register exists.

    Args:
        issue_id: The ID being checked.
        entry: The register entry.
        root: The repository root that the evidence paths are relative to.

    Returns:
        A list of error strings, empty when every item checks out.

    Example:
        >>> check_evidence("K1", {"status": "built", "origin": "revision_2"}, REPO_ROOT)
        ['K1: built needs at least one evidence item']
        >>> check_evidence("K1", {"status": "planned"}, REPO_ROOT)
        []
    """
    if entry.get("status") not in ("built", "partial"):
        return []
    items = entry.get("evidence")
    if not items or not isinstance(items, list):
        return [f"{issue_id}: {entry['status']} needs at least one evidence item"]
    errors: list[str] = []
    origin = str(entry.get("origin"))
    for item in items:
        kind = item.get("kind") if isinstance(item, dict) else None
        if kind == "amendment":
            errors.extend(check_amendment_evidence(issue_id, origin, item, root))
        elif kind == "config":
            errors.extend(check_config_evidence(issue_id, item, root))
        elif kind == "doc":
            errors.extend(check_doc_evidence(issue_id, item, root))
        elif kind == "test":
            errors.extend(check_test_evidence(issue_id, item, root))
        else:
            errors.append(f"{issue_id}: unknown evidence kind {kind!r}, allowed {list(ALLOWED_KINDS)}")
    return errors


def walk_dicts(value: Any) -> Iterator[dict[str, Any]]:
    """
    WHAT: Walks through any nested mix of dictionaries and lists and yields every dictionary it meets,
    the outer one first and then the ones inside it.

    WHY: Tagged blocks sit at different depths in different config files: at the top of one file, inside
    "parameters" in another. One walker finds them all without caring about the layout.

    Args:
        value: Any value from parsed YAML.

    Returns:
        An iterator over dictionaries.

    Example:
        >>> list(walk_dicts({"a": {"b": 1}, "c": [{"d": 2}]}))
        [{'a': {'b': 1}, 'c': [{'d': 2}]}, {'b': 1}, {'d': 2}]
    """
    if isinstance(value, dict):
        yield value
        for inner in value.values():
            yield from walk_dicts(inner)
    elif isinstance(value, list):
        for inner in value:
            yield from walk_dicts(inner)


def check_block_tags(file_name: str, block: dict[str, Any], known_ids: set[str]) -> list[str]:
    """
    WHAT: Checks the issue tags of one config block. Every ID under issue_ids_v2 must be in the register,
    and a block with issue_ids_v2 must also carry a non-empty derived_from. A revision_2 ID under the old
    issue_ids key is an error.

    WHY: The Level 11 checker rejects unknown IDs under issue_ids, so a revision_2 ID put there would break
    it. derived_from is ground rule 2: every number must say where it came from.

    Args:
        file_name: The config file name, used to start each message.
        block: One dictionary found inside the file.
        known_ids: Every ID in the version 2 register.

    Returns:
        A list of error strings, empty when the block is fine.

    Example:
        >>> check_block_tags("a.yaml", {"issue_ids_v2": ["9-9"], "derived_from": "x"}, {"K1"})
        ["a.yaml: issue_ids_v2 names unknown ID '9-9'"]
        >>> check_block_tags("a.yaml", {"issue_ids": ["K1"]}, {"K1"})
        ["a.yaml: revision_2 ID 'K1' is under issue_ids; use issue_ids_v2"]
    """
    errors: list[str] = []
    tagged = block.get("issue_ids_v2")
    if tagged is not None:
        for tag in tagged if isinstance(tagged, list) else [tagged]:
            if tag not in known_ids:
                errors.append(f"{file_name}: issue_ids_v2 names unknown ID {tag!r}")
        # Ground rule 2: a tagged block must say where its numbers came from.
        derived = block.get("derived_from")
        if not isinstance(derived, str) or not derived.strip():
            errors.append(f"{file_name}: block with issue_ids_v2 {tagged} is missing derived_from")
    old_tags = block.get("issue_ids")
    for tag in old_tags if isinstance(old_tags, list) else []:
        if tag in EXPECTED_V2_IDS:
            errors.append(f"{file_name}: revision_2 ID {tag!r} is under issue_ids; use issue_ids_v2")
    return errors


def check_config_tags(root: Path, known_ids: set[str]) -> list[str]:
    """
    WHAT: Reads every YAML file in root/config and checks the issue tags of every block inside it with
    check_block_tags. A file that cannot be read is reported as an error rather than skipped.

    WHY: Tags are only useful if they are all checked, including those in files added later. A glob finds new
    files without anyone remembering to register them. This only reads keys and counts nothing that a
    test pins.

    Args:
        root: The repository root (or a copy of it) that holds a config folder.
        known_ids: Every ID in the version 2 register.

    Returns:
        A list of error strings from all files, in file-name order.

    Example:
        >>> check_config_tags(REPO_ROOT, EXPECTED_V2_IDS | EXPECTED_IDS)
        []
    """
    errors: list[str] = []
    for path in sorted((root / "config").glob("*.yaml")):
        name = f"config/{path.name}"
        try:
            data = load_yaml(path)
        except Exception as problem:
            errors.append(f"{name}: cannot be read ({problem})")
            continue
        for block in walk_dicts(data):
            errors.extend(check_block_tags(name, block, known_ids))
    return errors


def check_register(register: dict[str, Any], root: Path) -> list[str]:
    """
    WHAT: Runs every check on a loaded register and returns all errors together: the top-level shape, the ID
    set, the links to the Level 11 register, each entry's fields and evidence, and the tags in config files.

    WHY: One function that runs everything gives the command line and the tests the same answer. Running all
    checks, instead of stopping at the first error, shows the whole list of problems in one go.

    Args:
        register: The parsed version 2 register.
        root: The repository root. config/issues.yaml (the Level 11 register), the amendments and every
            evidence path are read from here.

    Returns:
        A list of error strings. Empty means the register is valid.

    Example:
        >>> check_register(load_v2_register(REGISTER_PATH), REPO_ROOT)
        []
        >>> check_register([], REPO_ROOT)
        ['register must be a dictionary with meta and issues']
    """
    errors = check_top_level(register)
    if errors:
        return errors
    issues = register["issues"]
    v1_register = load_register(root / "config" / "issues.yaml")
    errors.extend(check_id_set(issues))
    errors.extend(check_origins(issues, v1_register))
    for issue_id, entry in issues.items():
        errors.extend(check_entry_fields(issue_id, entry, register["meta"]["statuses"]))
        if isinstance(entry, dict):
            errors.extend(check_evidence(issue_id, entry, root))
    errors.extend(check_config_tags(root, set(issues)))
    return errors


def count_statuses(issues: dict[str, Any]) -> dict[str, int]:
    """
    WHAT: Counts how many entries have each status word (built, partial, planned, deferred).

    WHY: The summary line printed by main() is what a person reads in CI, so it must be computed from
    the register and never typed by hand.

    Args:
        issues: The "issues" dictionary of the register.

    Returns:
        A dictionary from status word to count, for example {"built": 2, "planned": 1}.

    Example:
        >>> count_statuses({"A": {"status": "built"}, "B": {"status": "built"}, "C": {"status": "planned"}})
        {'built': 2, 'planned': 1}
    """
    counts: dict[str, int] = {}
    for entry in issues.values():
        status = entry.get("status") if isinstance(entry, dict) else None
        counts[str(status)] = counts.get(str(status), 0) + 1
    return counts


def main(argv: list[str] | None = None) -> int:
    """
    WHAT: Command line entry point. Loads the register (or the file named as the first argument), checks it,
    prints each error and returns 1, or prints one OK line with the counts and returns 0.

    WHY: CI runs this as `python -m core.issue_register_v2`, and the exit code is what makes the
    job fail. The OK line shows at a glance that the counts did not drift.

    Args:
        argv: Command line arguments; when None, sys.argv[1:] is used.

    Returns:
        0 when the register is valid, 1 otherwise.

    Example:
        Running it on the real register prints
        "OK: 93 issues checked (built 13, partial 7, planned 72, deferred 1)" and returns 0.
    """
    args = argv if argv is not None else sys.argv[1:]
    path = Path(args[0]) if args else REGISTER_PATH
    try:
        register = load_v2_register(path)
    except Exception as problem:
        print(f"Cannot read {path}: {problem}")
        return 1
    errors = check_register(register, REPO_ROOT)
    if errors:
        for error in errors:
            print(error)
        return 1
    counts = count_statuses(register["issues"])
    total = sum(counts.values())
    print(
        f"OK: {total} issues checked (built {counts.get('built', 0)}, partial {counts.get('partial', 0)}, "
        f"planned {counts.get('planned', 0)}, deferred {counts.get('deferred', 0)})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
