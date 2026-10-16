"""
WHAT: Checks the Level 12a tactic family file (config/tactic_families.yaml) and the tactic cost file
(config/tactic_costs.yaml) against the files that own the facts they repeat: config/evaluation.yaml owns
the held-out list, and docs/research/mule_typologies.md owns the typology rows and findings.

WHY: Amendment A-014 freezes the family definitions before any gang world exists. If a held-out name drifts
away from evaluation.yaml, or a family cites a typology row that does not exist, a later result could not be
trusted. These tests make that drift fail loudly instead of silently.

TERM: tactic family: one named way a mule gang moves stolen money,
      e.g. "fast_pass_through" is money that lands in an account and leaves again quickly.
TERM: held-out family: a family used only in test and stress worlds,
      e.g. "slow_drip" is one of the four names listed in config/evaluation.yaml.
TERM: typology row: a numbered row of the table in docs/research/mule_typologies.md,
      e.g. row 1 is "Fast pass-through into a new or dormant account" and is confirmed.
TERM: finding id: a short name for one verified source in the same document,
      e.g. "bengaluru_rented_mule_suppliers".
TERM: nearest training family: the training family a held-out family looks most like,
      e.g. "shared_device_ring" for "device_rotation".
"""

import re
from pathlib import Path
from typing import Any

from core.config_validator import load_yaml

# The repo root, found from this file's own location so the tests work from any working directory.
REPO_ROOT: Path = Path(__file__).resolve().parents[2]
FAMILIES_PATH: Path = REPO_ROOT / "config" / "tactic_families.yaml"
COSTS_PATH: Path = REPO_ROOT / "config" / "tactic_costs.yaml"
EVALUATION_PATH: Path = REPO_ROOT / "config" / "evaluation.yaml"
TYPOLOGY_PATH: Path = REPO_ROOT / "docs" / "research" / "mule_typologies.md"


def load_families() -> dict[str, Any]:
    """
    WHAT: Reads config/tactic_families.yaml and returns its one block, tactic_families, as a dictionary.
    Every test below takes the family names and citations from here.

    WHY: One loader means every test reads the same file the same way, so a test cannot pass by looking at
    a different copy than the one the rest of the project reads.

    Args:
        None.

    Returns:
        Dictionary with the keys "training", "traits", "held_out" and the tags of the block.

    Example:
        >>> sorted(load_families()["held_out"])
        ['device_rotation', 'fake_routine_life', 'pseudo_merchant_camouflage', 'slow_drip']
        >>> len(load_families()["training"])
        7
    """
    # The file holds a single top-level block, so return only that block.
    return load_yaml(FAMILIES_PATH)["tactic_families"]


def read_typology_statuses() -> tuple[dict[int, str], dict[str, str]]:
    """
    WHAT: Parses docs/research/mule_typologies.md and returns two lookups: the status of each numbered
    typology row (rows 1 to 7), and the status of each finding id in the verification table.
    A table line is recognised by its first cell: a whole number, or a snake_case finding id.

    WHY: The tests must not hard-code the rows. If someone edits the document, for example by changing
    a status, the family file has to be updated to match, and parsing the real file is how we notice.

    Args:
        None.

    Returns:
        A pair (row_status, finding_status). row_status maps a row number to a word such as "confirmed".
        finding_status maps a finding id to a word such as "reported".

    Example:
        >>> row_status, finding_status = read_typology_statuses()
        >>> (row_status[1], row_status[4], finding_status["bengaluru_rented_mule_suppliers"])
        ('confirmed', 'unverified', 'reported')
    """
    row_status: dict[int, str] = {}
    finding_status: dict[str, str] = {}
    # Walk every line of the document and look only at table lines.
    for line in TYPOLOGY_PATH.read_text(encoding="utf-8").splitlines():
        cells = [cell.strip() for cell in line.split("|")]
        # A table line starts and ends with "|", so it has at least two empty edge cells.
        if len(cells) < 4 or cells[0] != "":
            continue
        # The typology table has seven columns and its status sits in column 5 (after the edge cell).
        if cells[1].isdigit() and len(cells) >= 8:
            row_status[int(cells[1])] = cells[5]
        # The finding table keeps the status in column 3 and starts with a snake_case id.
        elif re.fullmatch(r"[a-z0-9_]+", cells[1]) and len(cells) >= 5:
            finding_status[cells[1]] = cells[3]
    return row_status, finding_status


def test_held_out_names_match_evaluation_yaml() -> None:
    """
    WHAT: Checks that the held-out family names in config/tactic_families.yaml are exactly the names in
    config/evaluation.yaml held_out_tactics, and that basic_merchant_mule is a training family.

    WHY: evaluation.yaml owns the held-out list and was committed first. If the family file could hold a
    different set, a tactic could be "held out" in one file and trained on in the other.

    Returns:
        None. The test fails with an assertion message if the sets differ.

    Example:
        With evaluation.yaml listing slow_drip, device_rotation, pseudo_merchant_camouflage and
        fake_routine_life, both sets have 4 names and are equal, and "basic_merchant_mule" is
        one of the 7 training family names.
    """
    families = load_families()
    evaluation = load_yaml(EVALUATION_PATH)["held_out_tactics"]
    # The two files must agree about which tactics are held out.
    assert set(families["held_out"]) == set(evaluation["tactics"])
    # The training tactic named in evaluation.yaml must be a real training family.
    for name in evaluation["kept_in_training"]:
        assert name in families["training"], name


def test_no_name_is_both_training_and_held_out() -> None:
    """
    WHAT: Checks that no family name appears in both the training group and the held-out group, and that
    the trait names do not reuse a family name either.

    WHY: A name in both groups would let a held-out tactic leak into training worlds, which defeats the
    whole purpose of holding it out.

    Returns:
        None. The test fails and prints the overlapping names if there are any.

    Example:
        With 7 training names and 4 held-out names, the overlap is the empty set, so
        the test passes. Giving "slow_drip" to both groups would leave the overlap {"slow_drip"}.
    """
    families = load_families()
    training = set(families["training"])
    held_out = set(families["held_out"])
    traits = set(families["traits"])
    # Nothing may be in both groups, and a trait is not a family, so it must not share a family name.
    assert training & held_out == set()
    assert traits & (training | held_out) == set()


def test_training_families_cite_existing_typology_sources() -> None:
    """
    WHAT: Checks that every training family and trait cites at least one typology row or finding id, that
    each cited row or finding exists in docs/research/mule_typologies.md, and that the status copied into
    the family file equals the status in that document.

    WHY: The families repeat facts about how well each pattern is sourced. A copied status that drifts
    from the document would let a weakly sourced pattern look confirmed, so it is compared with the real file.

    Returns:
        None. The test fails and names the family whose citation is wrong.

    Example:
        fast_pass_through cites row 1 with status "confirmed", and the document row 1 says "confirmed",
        so it passes. basic_merchant_mule is the one training family with no row: it is project design and
        carries a note instead, so it is skipped here.
    """
    families = load_families()
    row_status, finding_status = read_typology_statuses()
    cited_blocks = {**families["training"], **families["traits"]}
    for name, block in cited_blocks.items():
        rows = block.get("typology_rows", [])
        findings = block.get("finding_ids", [])
        # Only basic_merchant_mule may cite nothing, because it is project design with a note saying so.
        if not rows and not findings:
            assert name == "basic_merchant_mule" and block.get("note"), name
            continue
        # Every cited row and finding must exist, with the same status as in the document.
        for cited in rows:
            assert row_status.get(cited["row"]) == cited["status"], (name, cited)
        for cited in findings:
            assert finding_status.get(cited["id"]) == cited["status"], (name, cited)


def test_held_out_families_name_a_real_nearest_training_family() -> None:
    """
    WHAT: Checks that every held-out family names a nearest training family that exists, and that its
    what_differs and forbidden_in_training texts are not empty.

    WHY: A held-out tactic only means something when we know which training tactic it is a variation of,
    what the variation is, and what must be kept out of training. Empty text would hide a missing decision.

    Returns:
        None. The test fails and names the held-out family with the gap.

    Example:
        device_rotation names shared_device_ring, which is one of the 7 training families, and both of its
        text fields have words in them, so it passes. An empty forbidden_in_training would fail.
    """
    families = load_families()
    for name, block in families["held_out"].items():
        # The nearest family must be one of the training families, not a held-out one.
        assert block["nearest_training_family"] in families["training"], name
        # Both explanations must say something.
        for key in ("what_differs", "forbidden_in_training"):
            assert isinstance(block[key], str) and block[key].strip(), (name, key)


def test_cost_parameters_are_assumptions_with_ranges_inside_zero_to_one() -> None:
    """
    WHAT: Checks that both tactic cost parameters (account_supplier_fee_share and operator_share) are
    labelled assumption, and that each range has a minimum below its maximum, both inside 0 to 1.

    WHY: The source figures are reported and not confirmed, so calling them anything but an assumption
    would overstate what we know. A share must be a fraction, so a value outside 0 to 1 is a typing mistake.

    Returns:
        None. The test fails and names the parameter that breaks a rule.

    Example:
        account_supplier_fee_share has range [0.03, 0.06]: 0.03 < 0.06 and both lie in [0, 1], so it passes.
        operator_share has range [0.20, 0.30], which also passes. A range [0.06, 0.03] would fail.
    """
    parameters = load_yaml(COSTS_PATH)["tactic_cost_units"]["parameters"]
    assert set(parameters) == {"account_supplier_fee_share", "operator_share"}
    for name, block in parameters.items():
        low, high = block["range"]
        # Reported but unconfirmed figures are assumptions, never sourced.
        assert block["label"] == "assumption", name
        # A share of money is a fraction between 0 and 1, and a range needs its minimum below its maximum.
        assert 0 <= low < high <= 1, (name, block["range"])
