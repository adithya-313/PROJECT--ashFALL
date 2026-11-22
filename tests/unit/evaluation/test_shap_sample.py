"""
WHAT: Tests the fixed-seed row reservoir behind the SHAP sample and the config keys that size it.

WHY: SHAP explains a sample of the test account-days, so the sample must be exactly the configured size,
reproducible from its seed, and every row when there are fewer rows than the limit.
"""

from pathlib import Path

import pytest
import yaml

from decisions.mule_account_model import load_mule_model_config
from evaluation.run_level_18_pilot import _RowReservoir

REPO_ROOT = Path(__file__).resolve().parents[3]
COMMITTED_CONFIG = REPO_ROOT / "config" / "mule_model.yaml"


def _fill(reservoir: _RowReservoir, count: int) -> None:
    for index in range(count):
        reservoir.add({"row": index})


def test_the_same_seed_gives_the_same_sample() -> None:
    first = _RowReservoir(50, 1717)
    second = _RowReservoir(50, 1717)

    _fill(first, 1000)
    _fill(second, 1000)

    assert first.rows == second.rows


def test_a_different_seed_gives_a_different_sample() -> None:
    first = _RowReservoir(50, 1717)
    second = _RowReservoir(50, 1718)

    _fill(first, 1000)
    _fill(second, 1000)

    assert first.rows != second.rows


def test_the_sample_has_exactly_the_requested_size_and_counts_every_row() -> None:
    reservoir = _RowReservoir(50, 1717)

    _fill(reservoir, 1000)

    assert len(reservoir.rows) == 50
    assert reservoir.rows_seen == 1000
    assert len({row["row"] for row in reservoir.rows}) == 50


def test_fewer_rows_than_the_limit_uses_all_of_them_in_order() -> None:
    reservoir = _RowReservoir(50, 1717)

    _fill(reservoir, 30)

    assert reservoir.rows == [{"row": index} for index in range(30)]
    assert reservoir.rows_seen == 30


def test_a_non_positive_size_is_refused() -> None:
    with pytest.raises(ValueError, match="positive size"):
        _RowReservoir(0, 1717)


def test_committed_config_carries_the_ruled_shap_sample_policy() -> None:
    config = load_mule_model_config(COMMITTED_CONFIG)

    assert config.shap_sample_rows == 20000
    assert config.shap_sample_seed == 1717


def test_an_older_config_without_the_keys_reads_the_committed_values(tmp_path: Path) -> None:
    raw = yaml.safe_load(COMMITTED_CONFIG.read_text(encoding="utf-8"))
    del raw["parameters"]["shap_sample_rows"]
    del raw["parameters"]["shap_sample_seed"]
    older = tmp_path / "older_config.yaml"
    older.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    config = load_mule_model_config(older)

    assert (config.shap_sample_rows, config.shap_sample_seed) == (20000, 1717)


@pytest.mark.parametrize("bad_value", [0, -5, 2.5, "many", True])
def test_a_bad_sample_size_is_refused(tmp_path: Path, bad_value: object) -> None:
    raw = yaml.safe_load(COMMITTED_CONFIG.read_text(encoding="utf-8"))
    raw["parameters"]["shap_sample_rows"]["value"] = bad_value
    path = tmp_path / "bad_config.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")

    with pytest.raises(ValueError, match="shap_sample_rows"):
        load_mule_model_config(path)
