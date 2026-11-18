"""
WHAT: Unit tests for the positive-unlabelled switches in config/mule_model.yaml.

WHY: The PU method is a modelling choice read from config, so a missing or malformed value must be
refused naming the key, and an older config that predates the switches must still load with the
switches unset.
"""

from pathlib import Path

import pytest
import yaml

from decisions.mule_account_model import load_mule_model_config

REPO_ROOT = Path(__file__).resolve().parents[3]
MULE_MODEL_CONFIG = REPO_ROOT / "config" / "mule_model.yaml"


def _committed_body() -> dict[str, object]:
    """Returns the parsed body of the committed mule model config."""
    raw = yaml.safe_load(MULE_MODEL_CONFIG.read_text(encoding="utf-8"))
    assert isinstance(raw, dict)
    return raw


def _write_modified(directory: Path, key: str, value: object) -> Path:
    """Writes the committed config with one PU parameter value replaced."""
    raw = _committed_body()
    parameters = raw["parameters"]
    assert isinstance(parameters, dict)
    block = parameters[key]
    assert isinstance(block, dict)
    block["value"] = value
    path = directory / f"mule_model_{key}.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


def test_committed_config_parses_the_pu_switches() -> None:
    """The committed switches default the method off and carry the committed grid and guard."""
    config = load_mule_model_config(MULE_MODEL_CONFIG)

    assert config.pu_method == "none"
    assert config.pu_bagging_n_bags == 50
    assert config.pu_unlabelled_per_positive_candidates == (1, 5, 20)
    assert config.pu_probability_clip == pytest.approx(0.999999)


def test_config_without_pu_switches_loads_with_them_unset(tmp_path: Path) -> None:
    """A config that predates the PU switches still loads, leaving every PU value None."""
    raw = _committed_body()
    parameters = raw["parameters"]
    assert isinstance(parameters, dict)
    for key in (
        "pu_method",
        "pu_bagging_n_bags",
        "pu_unlabelled_per_positive_candidates",
        "pu_probability_clip",
    ):
        parameters.pop(key)
    path = tmp_path / "mule_model_without_pu.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    config = load_mule_model_config(path)

    assert config.pu_method is None
    assert config.pu_bagging_n_bags is None
    assert config.pu_unlabelled_per_positive_candidates is None
    assert config.pu_probability_clip is None


def test_unknown_pu_method_is_refused_naming_the_key(tmp_path: Path) -> None:
    """A pu_method outside the allowed values is refused and the key is named."""
    config_path = _write_modified(tmp_path, "pu_method", "random_forest")

    with pytest.raises(ValueError, match="pu_method .* must be one of"):
        load_mule_model_config(config_path)


def test_non_positive_bag_count_is_refused_naming_the_key(tmp_path: Path) -> None:
    """A bag count of zero or below is refused and the key is named."""
    config_path = _write_modified(tmp_path, "pu_bagging_n_bags", 0)

    with pytest.raises(ValueError, match="pu_bagging_n_bags .* must be a positive integer"):
        load_mule_model_config(config_path)


def test_boolean_bag_count_is_refused_naming_the_key(tmp_path: Path) -> None:
    """A boolean bag count is not an integer and is refused."""
    config_path = _write_modified(tmp_path, "pu_bagging_n_bags", True)

    with pytest.raises(ValueError, match="pu_bagging_n_bags .* must be a positive integer"):
        load_mule_model_config(config_path)


def test_empty_candidate_list_is_refused_naming_the_key(tmp_path: Path) -> None:
    """An empty candidate list makes the search impossible, so it is refused and named."""
    config_path = _write_modified(tmp_path, "pu_unlabelled_per_positive_candidates", [])

    with pytest.raises(ValueError, match="pu_unlabelled_per_positive_candidates .* non-empty list"):
        load_mule_model_config(config_path)


def test_non_positive_candidate_is_refused_naming_the_key(tmp_path: Path) -> None:
    """A candidate ratio of zero is refused and the key is named."""
    config_path = _write_modified(tmp_path, "pu_unlabelled_per_positive_candidates", [1, 0])

    with pytest.raises(ValueError, match="pu_unlabelled_per_positive_candidates .* positive integers"):
        load_mule_model_config(config_path)


def test_probability_clip_outside_the_open_unit_interval_is_refused(tmp_path: Path) -> None:
    """A clip of 1 would divide by zero in the Elkan-Noto weight, so it is refused and named."""
    config_path = _write_modified(tmp_path, "pu_probability_clip", 1.0)

    with pytest.raises(ValueError, match="pu_probability_clip .* must be a number in"):
        load_mule_model_config(config_path)
