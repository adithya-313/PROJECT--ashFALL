"""
WHAT: Tests the single process-wide override point for the account model config: every loader
follows it, an explicit path still wins, a missing path is refused and clearing restores the
committed config.

WHY: The pilot must be able to run against a switch-off config on a clean tree, and one override
point is only safe if every caller of load_mule_model_config sees the same file.
"""

from collections.abc import Iterator
from pathlib import Path

import pyarrow.parquet as pq
import pytest
import yaml

from decisions.mule_account_model import (
    account_model_feature_names,
    active_mule_model_config_path,
    build_training_rows,
    load_mule_model_config,
    set_mule_model_config_path,
)
from showcase.level_17_checks import TEST_DATASET

REPO_ROOT = Path(__file__).resolve().parents[2]
COMMITTED_CONFIG = REPO_ROOT / "config" / "mule_model.yaml"


def _switch_off_copy(directory: Path) -> Path:
    """Writes a copy of the committed config with only the label network switch turned off."""
    raw = yaml.safe_load(COMMITTED_CONFIG.read_text(encoding="utf-8"))
    raw["parameters"]["use_label_network_features"]["value"] = False
    path = directory / "mule_model_off.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


@pytest.fixture(autouse=True)
def clear_override() -> Iterator[None]:
    """Starts and ends every test on the committed config so state cannot leak between tests."""
    set_mule_model_config_path(None)
    yield
    set_mule_model_config_path(None)


def test_no_override_loads_the_committed_config() -> None:
    """Without an override the loader reads exactly the file the explicit committed path reads."""
    assert load_mule_model_config() == load_mule_model_config(COMMITTED_CONFIG)


def test_override_to_switch_off_copy_changes_features_and_matrix_columns(tmp_path: Path) -> None:
    """A switch-off override reaches both the feature-name list and the built training matrix."""
    switch_off = _switch_off_copy(tmp_path)
    set_mule_model_config_path(switch_off)

    assert len(account_model_feature_names()) == 39

    snapshots = pq.read_table(TEST_DATASET / "snapshots.parquet").to_pylist()
    labels = pq.read_table(TEST_DATASET / "labels.parquet").to_pylist()
    matrix, targets, keys = build_training_rows(snapshots, labels)

    assert matrix
    assert len(matrix[0]) == 39
    assert len(matrix) == len(targets) == len(keys)

    set_mule_model_config_path(None)
    assert len(account_model_feature_names()) == 42


def test_missing_override_path_is_refused_and_keeps_the_previous_override(tmp_path: Path) -> None:
    """A missing path raises naming it and leaves the previously configured override in place."""
    switch_off = _switch_off_copy(tmp_path)
    set_mule_model_config_path(switch_off)
    missing = tmp_path / "missing_mule_model.yaml"

    with pytest.raises(FileNotFoundError, match="missing_mule_model.yaml"):
        set_mule_model_config_path(missing)

    assert active_mule_model_config_path() == switch_off


def test_explicit_path_argument_wins_over_the_override(tmp_path: Path) -> None:
    """An explicit load_mule_model_config path ignores the process-wide override."""
    switch_off = _switch_off_copy(tmp_path)
    set_mule_model_config_path(switch_off)

    assert load_mule_model_config().use_label_network_features is False
    assert load_mule_model_config(COMMITTED_CONFIG).use_label_network_features is True
