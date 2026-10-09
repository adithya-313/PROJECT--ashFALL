"""Checks that config/world_ranges.yaml and config/legal_actions.yaml keep their agreed shape.

These two files hold the research catalogue: where each number came from, and which actions are legal.
"""
from pathlib import Path

from core.config_validator import find_labelled_blocks, load_yaml, validate_file

CONFIG_FOLDER = Path(__file__).resolve().parents[2] / "config"
WORLD_RANGES_PATH = CONFIG_FOLDER / "world_ranges.yaml"
LEGAL_ACTIONS_PATH = CONFIG_FOLDER / "legal_actions.yaml"

LABELLED_BLOCK_NAMES = ["precondition", "legal_basis", "max_duration_days", "requires_human_approval"]
DRAFT_ACTION = "temporary_debit_hold_rbi_draft"


# ---- World ranges ----


def count_parameter_labels(parameters):
    """Count the labels of the blocks under "parameters"."""
    counts = {}
    for _, block in find_labelled_blocks(parameters):
        label = block["label"]
        counts[label] = counts.get(label, 0) + 1
    return counts


def test_world_ranges_file_is_valid():
    assert validate_file(WORLD_RANGES_PATH) == []


def test_world_ranges_has_ten_parameters():
    assert len(load_yaml(WORLD_RANGES_PATH)["parameters"]) == 10


def test_world_ranges_label_counts():
    parameters = load_yaml(WORLD_RANGES_PATH)["parameters"]
    assert count_parameter_labels(parameters) == {"sourced": 5, "assumption": 5}


def test_world_ranges_parameters_have_unit_and_all_three_ranges():
    parameters = load_yaml(WORLD_RANGES_PATH)["parameters"]
    for name, block in parameters.items():
        for key in ["unit", "training", "test", "stress"]:
            assert key in block, f"{name} has no {key}"


def test_world_ranges_lists_what_was_not_found():
    not_found = load_yaml(WORLD_RANGES_PATH)["searched_but_not_found"]
    assert isinstance(not_found, list) and len(not_found) > 0


# ---- Legal actions ----


def test_legal_actions_file_is_valid():
    assert validate_file(LEGAL_ACTIONS_PATH) == []


def test_legal_actions_has_seven_actions():
    assert len(load_yaml(LEGAL_ACTIONS_PATH)["actions"]) == 7


def test_every_action_has_its_four_blocks_scope_and_in_force():
    actions = load_yaml(LEGAL_ACTIONS_PATH)["actions"]
    for name, action in actions.items():
        assert "scope" in action, name
        assert "in_force" in action, name
        for block_name in LABELLED_BLOCK_NAMES:
            assert "label" in action[block_name], f"{name} has no labelled {block_name}"


def test_rbi_draft_is_not_in_force_and_has_a_60_day_limit():
    draft = load_yaml(LEGAL_ACTIONS_PATH)["actions"][DRAFT_ACTION]
    assert draft["in_force"] is False
    assert draft["max_duration_days"]["value"] == 60


def test_lien_has_a_sourced_legal_basis():
    lien = load_yaml(LEGAL_ACTIONS_PATH)["actions"]["lien_on_disputed_amount"]
    assert lien["legal_basis"]["label"] == "sourced"


def test_only_the_draft_has_a_maximum_duration():
    actions = load_yaml(LEGAL_ACTIONS_PATH)["actions"]
    for name, action in actions.items():
        if name != DRAFT_ACTION:
            assert action["max_duration_days"]["value"] is None, name


def test_legal_actions_has_notes():
    notes = load_yaml(LEGAL_ACTIONS_PATH)["notes"]
    assert isinstance(notes, list) and len(notes) > 0
