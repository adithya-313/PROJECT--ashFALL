"""
WHAT: Tests the fixed-seed row reservoir behind the SHAP sample and the config keys that size it.

WHY: SHAP explains a sample of the test account-days, so the sample must be exactly the configured size,
reproducible from its seed, and every row when there are fewer rows than the limit.
"""

from pathlib import Path

import pytest
import yaml

from evaluation.run_level_18_pilot import _load_shap_sample_policy, _RowReservoir

REPO_ROOT = Path(__file__).resolve().parents[3]
SHAP_CONFIG = REPO_ROOT / "config" / "shap_sample.yaml"


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


def test_committed_shap_config_carries_the_ruled_policy() -> None:
    assert _load_shap_sample_policy(SHAP_CONFIG) == (20000, 1717)
    assert _load_shap_sample_policy() == (20000, 1717)


def _write_config(tmp_path: Path, mutate: object = None) -> Path:
    raw = yaml.safe_load(SHAP_CONFIG.read_text(encoding="utf-8"))
    if callable(mutate):
        mutate(raw)
    path = tmp_path / "shap_sample.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


def test_a_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="not found"):
        _load_shap_sample_policy(tmp_path / "absent.yaml")


@pytest.mark.parametrize("missing_key", ["shap_sample_rows", "shap_sample_seed"])
def test_a_missing_key_raises(tmp_path: Path, missing_key: str) -> None:
    path = _write_config(tmp_path, lambda raw: raw.pop(missing_key))

    with pytest.raises(ValueError, match=missing_key):
        _load_shap_sample_policy(path)


@pytest.mark.parametrize("bad_rows", [0, -5, 2.5, "many", True, None])
def test_a_bad_row_count_raises(tmp_path: Path, bad_rows: object) -> None:
    path = _write_config(tmp_path, lambda raw: raw["shap_sample_rows"].update(value=bad_rows))

    with pytest.raises(ValueError, match="shap_sample_rows"):
        _load_shap_sample_policy(path)


@pytest.mark.parametrize("bad_seed", [1.5, "seed", True, None])
def test_a_bad_seed_raises(tmp_path: Path, bad_seed: object) -> None:
    path = _write_config(tmp_path, lambda raw: raw["shap_sample_seed"].update(value=bad_seed))

    with pytest.raises(ValueError, match="shap_sample_seed"):
        _load_shap_sample_policy(path)


@pytest.mark.parametrize("missing_field", ["label", "derived_from", "reason"])
def test_a_block_without_its_metadata_raises(tmp_path: Path, missing_field: str) -> None:
    path = _write_config(tmp_path, lambda raw: raw["shap_sample_rows"].pop(missing_field))

    with pytest.raises(ValueError, match=missing_field):
        _load_shap_sample_policy(path)


def test_a_fixture_pilot_run_records_the_sample_in_shap_top_features(tmp_path: Path) -> None:
    import json

    from evaluation.run_level_18_pilot import run_pilot
    from showcase.level_17_checks import TEST_DATASET, TRAINING_DATASET, VALIDATION_DATASET

    fixtures = REPO_ROOT / "showcase" / "fixtures" / "level_17"

    exit_code = run_pilot(
        [TRAINING_DATASET],
        [VALIDATION_DATASET],
        [TEST_DATASET],
        fixtures / "truth",
        fixtures / "world_cards",
        tmp_path,
    )

    assert exit_code == 0
    shap_file = next(tmp_path.rglob("shap_top_features.json"))
    payload = json.loads(shap_file.read_text(encoding="utf-8"))
    assert payload["seed"] == 1717
    assert payload["rows_total"] > 0
    # The fixture has far fewer rows than the 20000 limit, so every row is sampled.
    assert payload["rows_sampled"] == payload["rows_total"]
