"""
WHAT: Tactic family catalogue loader, cross-specification validation, and gang family enforcement (Level 12d, D-4).
Provides centralized loading for config/tactic_families.yaml, verifies alignment between held-out family lists
and pre-registered evaluation policies in config/evaluation.yaml, and enforces simulation partition constraints.

WHY: Amendment A-014 pre-registers tactic family dividing lines, and amendment A-015 isolates model evaluation
partitions. Validating gang tactic families before world execution ensures that held-out adversarial evasions
cannot execute in training or validation worlds, preventing data snooping and premature pattern leakage.

TERM: tactic family: one named way a mule syndicate moves illicit funds, e.g. "fast_pass_through".
TERM: held-out family: an evasion tactic reserved exclusively for test and stress evaluation worlds.
TERM: training family: a tactic typology permitted in training and validation simulation worlds.
"""

import sys
from pathlib import Path
from typing import Any, Sequence

from core.config_validator import load_yaml
from world.gang_config import GangConfig
from world.run_identity import REPO_ROOT
from world.tactic_config import load_tactic_blocks

FAMILIES_PATH: Path = REPO_ROOT / "config" / "tactic_families.yaml"
EVALUATION_PATH: Path = REPO_ROOT / "config" / "evaluation.yaml"


def load_tactic_families(path: Path | None = None) -> dict[str, Any]:
    """
    WHAT: Reads config/tactic_families.yaml and returns its parsed tactic_families block.

    WHY: Provides a centralized entry point ensuring all consumers inspect identical canonical definitions.

    Args:
        path: Optional file path overriding the default config/tactic_families.yaml location.

    Returns:
        Dictionary containing the parsed tactic_families configuration block.

    Example:
        families = load_tactic_families()
    """
    target = path if path is not None else FAMILIES_PATH
    loaded = load_yaml(target)
    if "tactic_families" not in loaded:
        raise ValueError(f"'tactic_families' key is missing from {target}")
    return loaded["tactic_families"]


def load_evaluation_config(path: Path | None = None) -> dict[str, Any]:
    """
    WHAT: Reads config/evaluation.yaml and returns its pre-registered evaluation configuration dictionary.

    WHY: Supplies pre-registered evaluation metrics and held-out tactic lists for cross-validation checks.

    Args:
        path: Optional file path overriding the default config/evaluation.yaml location.

    Returns:
        Dictionary representing the parsed evaluation.yaml document.

    Example:
        evaluation = load_evaluation_config()
    """
    target = path if path is not None else EVALUATION_PATH
    return load_yaml(target)


def check_family_lists_agree(
    families: dict[str, Any],
    evaluation: dict[str, Any],
) -> None:
    """
    WHAT: Cross-validates tactic family catalogues against the pre-registered evaluation policy.
    Asserts that catalogue held-out names equal active tactics union deferred names, that active
    and deferred lists are disjoint, that catalogue statuses match, and that training and held-out
    lists do not overlap.

    WHY: Ground rule 17 and amendments A-014 / A-025 require held-out evasions and deferrals to be
    explicitly pre-registered, catalogued, and disjoint from training typologies.

    Args:
        families: Dictionary loaded from config/tactic_families.yaml.
        evaluation: Dictionary loaded from config/evaluation.yaml.

    Returns:
        None.

    Raises:
        ValueError: If held-out tactic lists disagree, overlap, or status does not match.

    Example:
        check_family_lists_agree(load_tactic_families(), load_evaluation_config())
    """
    if "held_out" not in families:
        raise ValueError("tactic_families block is missing 'held_out' key")
    if "training" not in families:
        raise ValueError("tactic_families block is missing 'training' key")
    if "held_out_tactics" not in evaluation:
        raise ValueError("evaluation config is missing 'held_out_tactics' key")
    if "tactics" not in evaluation["held_out_tactics"]:
        raise ValueError("evaluation config is missing 'held_out_tactics.tactics' key")

    held_out_names = set(families["held_out"].keys())
    training_names = set(families["training"].keys())
    active_tactics = set(evaluation["held_out_tactics"]["tactics"])
    deferred_items = evaluation["held_out_tactics"].get("deferred", [])
    deferred_names = {item["name"] if isinstance(item, dict) else item for item in deferred_items}

    # 1. tactics and deferred are disjoint
    disjoint_overlap = active_tactics & deferred_names
    if disjoint_overlap:
        raise ValueError(
            f"Active tactics and deferred lists are not disjoint: {sorted(disjoint_overlap)}."
        )

    # 2. the catalogue's held-out names == tactics ∪ deferred names
    expected_held_out = active_tactics | deferred_names
    if held_out_names != expected_held_out:
        raise ValueError(
            f"Held-out tactics lists disagree: tactic_families has {sorted(held_out_names)}, "
            f"evaluation.yaml has {sorted(expected_held_out)}."
        )

    # 3. each name's status in the catalogue matches the list it is in
    for name in active_tactics:
        cat_item = families["held_out"].get(name, {})
        status = cat_item.get("status") if isinstance(cat_item, dict) else None
        if status != "active":
            raise ValueError(
                f"Held-out family '{name}' is in active tactics list but has status '{status}' in catalogue."
            )

    for name in deferred_names:
        cat_item = families["held_out"].get(name, {})
        status = cat_item.get("status") if isinstance(cat_item, dict) else None
        if status != "deferred":
            raise ValueError(
                f"Held-out family '{name}' is in deferred list but has status '{status}' in catalogue."
            )

    # 4. keep existing training/held-out overlap check
    overlap = held_out_names & training_names
    if overlap:
        raise ValueError(
            f"Family name cannot be both training and held-out: {sorted(overlap)}."
        )


def check_held_out_mechanisms(
    families: dict[str, Any],
    evaluation: dict[str, Any],
    blocks: dict[str, Any],
) -> list[str]:
    """
    WHAT: Enforces that active held-out families have implementing block options, deferred families
    have none and cite valid amendments, and returns formatted status lines for each family.

    WHY: Level 12g (amendment A-025) requires mechanism-level checks ensuring that active held-out
    tactics possess concrete implementations in tactic_blocks.yaml while deferred tactics remain unbuilt.

    Args:
        families: Dictionary parsed from config/tactic_families.yaml.
        evaluation: Dictionary parsed from config/evaluation.yaml.
        blocks: Dictionary parsed from config/tactic_blocks.yaml.

    Returns:
        List of formatted status strings for each held-out family.

    Raises:
        ValueError: If active family has no block option, if marker names inactive family,
            if deferred family has block options, or if deferred metadata is invalid.

    Example:
        lines = check_held_out_mechanisms(fams, eval_cfg, blocks)
    """
    if "held_out" not in families:
        raise ValueError("tactic_families block is missing 'held_out' key")
    if "held_out_tactics" not in evaluation:
        raise ValueError("evaluation config is missing 'held_out_tactics' key")
    if "tactics" not in evaluation["held_out_tactics"]:
        raise ValueError("evaluation config is missing 'held_out_tactics.tactics' key")

    held_out_cat = families["held_out"]
    active_names = set(evaluation["held_out_tactics"]["tactics"])
    deferred_items = evaluation["held_out_tactics"].get("deferred", [])
    deferred_map: dict[str, dict[str, Any]] = {}
    for item in deferred_items:
        if isinstance(item, dict) and "name" in item:
            deferred_map[str(item["name"])] = item

    # Gather block options with held_out_family marker
    blocks_dict = blocks
    if "tactic_blocks" in blocks_dict and isinstance(blocks_dict["tactic_blocks"], dict):
        blocks_dict = blocks_dict["tactic_blocks"]
    if "blocks" in blocks_dict and isinstance(blocks_dict["blocks"], dict):
        blocks_dict = blocks_dict["blocks"]
    family_options: dict[str, list[str]] = {name: [] for name in held_out_cat}

    for block_name, block_def in sorted(blocks_dict.items()):
        if not isinstance(block_def, dict):
            continue
        params = block_def.get("parameters", {})
        if not isinstance(params, dict):
            continue
        for param_name, param_def in sorted(params.items()):
            if isinstance(param_def, dict) and "held_out_family" in param_def:
                target_family = str(param_def["held_out_family"])
                if target_family not in active_names:
                    raise ValueError(
                        f"held_out_family marker on '{block_name}.{param_name}' names family "
                        f"'{target_family}' which is not active"
                    )
                family_options.setdefault(target_family, []).append(f"{block_name}.{param_name}")

    # 1. An active name with no option in tactic_blocks.yaml carrying held_out_family == that name -> ValueError
    for name in sorted(active_names):
        opts = family_options.get(name, [])
        if not opts:
            raise ValueError(
                f"held-out family '{name}' is active but has no implementing block option; "
                "a name without a mechanism fails"
            )

    # 2. A deferred name with any implementing option -> ValueError
    for name in sorted(deferred_map.keys()):
        opts = family_options.get(name, [])
        if opts:
            raise ValueError(
                f"held-out family '{name}' is deferred but has implementing block option(s): {opts}; "
                "it should be active"
            )

        # 3. Deferred item validation: reason, returns_when, amendment
        item = deferred_map[name]
        for field in ("reason", "returns_when", "amendment"):
            val = item.get(field)
            if not val or not str(val).strip():
                raise ValueError(
                    f"deferred held-out family '{name}' is missing required field '{field}'"
                )

        amendment_id = str(item["amendment"]).strip()
        amendments_file = REPO_ROOT / "docs" / "research" / "amendments.md"
        if not amendments_file.exists():
            raise ValueError(f"amendments documentation not found at {amendments_file}")
        content = amendments_file.read_text(encoding="utf-8")
        headings = {line.strip() for line in content.splitlines() if line.startswith("## ")}
        if f"## {amendment_id}" not in headings:
            raise ValueError(
                f"deferred held-out family '{name}' cites amendment '{amendment_id}' "
                f"which does not appear as a '## {amendment_id}' heading in {amendments_file}"
            )

    # Build status lines
    status_lines: list[str] = []
    for name in sorted(held_out_cat.keys()):
        if name in active_names:
            opts_str = ", ".join(sorted(family_options[name]))
            status_lines.append(f"{name}: ACTIVE — {opts_str}")
        elif name in deferred_map:
            item = deferred_map[name]
            amend = item["amendment"]
            reason = item["reason"]
            status_lines.append(f"{name}: DEFERRED ({amend}) — {reason}")

    return status_lines


def validate_gang_families(
    gang_configs: Sequence[GangConfig],
    range_name: str,
    families: dict[str, Any] | None = None,
) -> None:
    """
    WHAT: Validates tactic families assigned to gang configurations against declarative catalogues.
    Asserts that every specified family exists in training or held-out categories, and that held-out
    tactics are strictly rejected in training range worlds.

    WHY: Ground rule 17 and amendment A-014 prohibit held-out tactics from executing in training worlds
    to prevent data leakage into models. Validation worlds use training ranges (amendment A-015), so
    this check covers them as well.

    Args:
        gang_configs: Sequence of GangConfig instances to validate.
        range_name: Simulation world range name ("training", "test", or "stress").
        families: Optional loaded tactic families dictionary; loaded from config if omitted.

    Returns:
        None.

    Raises:
        ValueError: If a gang specifies an unknown family or uses a held-out family in training ranges.

    Example:
        validate_gang_families([gang], "test")
    """
    if families is None:
        families = load_tactic_families()
    if "training" not in families:
        raise ValueError("tactic_families block is missing 'training' key")
    if "held_out" not in families:
        raise ValueError("tactic_families block is missing 'held_out' key")
    training_names = set(families["training"].keys())
    held_out_names = set(families["held_out"].keys())
    known_names = training_names | held_out_names

    for gang in gang_configs:
        if gang.family is None:
            continue
        if gang.family not in known_names:
            raise ValueError(f"Gang '{gang.gang_id}' uses unknown family '{gang.family}'.")
        # Validation worlds use training ranges (A-015), so held-out families are blocked there too.
        if range_name in ("training", "validation") and gang.family in held_out_names:
            raise ValueError(
                f"Gang '{gang.gang_id}' uses held-out family '{gang.family}', "
                "which is allowed only in test and stress worlds."
            )


if __name__ == "__main__":
    try:
        loaded_families = load_tactic_families()
        loaded_evaluation = load_evaluation_config()
        loaded_blocks = load_tactic_blocks()
        check_family_lists_agree(loaded_families, loaded_evaluation)
        lines = check_held_out_mechanisms(loaded_families, loaded_evaluation, loaded_blocks)
        for output_line in lines:
            print(output_line)
        sys.exit(0)
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
