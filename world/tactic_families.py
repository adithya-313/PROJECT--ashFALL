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

from pathlib import Path
from typing import Any, Sequence

from core.config_validator import load_yaml
from world.gang_config import GangConfig
from world.run_identity import REPO_ROOT

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
    Asserts that held-out names in config/tactic_families.yaml match held_out_tactics.tactics in
    config/evaluation.yaml, and enforces that no family name appears in both training and held-out lists.

    WHY: Ground rule 17 and amendment A-014 require held-out evasions to be pre-registered and strictly
    disjoint from training typologies so models cannot be evaluated on training tactics disguised as held-out.

    Args:
        families: Dictionary loaded from config/tactic_families.yaml.
        evaluation: Dictionary loaded from config/evaluation.yaml.

    Returns:
        None.

    Raises:
        ValueError: If held-out tactic lists disagree or if any family name overlaps between training and held-out.

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
    eval_held_out = set(evaluation["held_out_tactics"]["tactics"])

    if held_out_names != eval_held_out:
        raise ValueError(
            f"Held-out tactics lists disagree: tactic_families has {sorted(held_out_names)}, "
            f"evaluation.yaml has {sorted(eval_held_out)}."
        )

    overlap = held_out_names & training_names
    if overlap:
        raise ValueError(
            f"Family name cannot be both training and held-out: {sorted(overlap)}."
        )


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
