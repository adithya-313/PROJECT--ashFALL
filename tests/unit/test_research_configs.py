"""Checks that config/world_ranges.yaml and config/legal_actions.yaml keep their agreed shape.

These two files hold the research catalogue: where each number came from, and which actions are legal.
"""
from pathlib import Path

from core.config_validator import find_labelled_blocks, load_yaml, validate_file

CONFIG_FOLDER = Path(__file__).resolve().parents[2] / "config"
WORLD_RANGES_PATH = CONFIG_FOLDER / "world_ranges.yaml"
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
    assert count_parameter_labels(parameters) == {"sourced": 3, "reported": 2, "assumption": 5}


def test_world_ranges_parameters_have_unit_and_all_three_ranges():
    parameters = load_yaml(WORLD_RANGES_PATH)["parameters"]
    for name, block in parameters.items():
        for key in ["unit", "training", "test", "stress"]:
            assert key in block, f"{name} has no {key}"


def test_world_ranges_lists_what_was_not_found():
    not_found = load_yaml(WORLD_RANGES_PATH)["searched_but_not_found"]
    assert isinstance(not_found, list) and len(not_found) > 0
