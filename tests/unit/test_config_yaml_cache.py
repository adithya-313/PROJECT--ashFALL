"""
WHAT: Unit tests for content-addressed YAML caching in core/config_validator.py.

WHY: High-volume simulation and fraud planning repeatedly load static YAML configurations.
Caching parsed representations keyed by byte content SHA-256 avoids repeated parse overhead
while ensuring immediate invalidation upon content changes and strict isolation via deep copies.

ISSUE IDs: 12-perf-yaml
"""

from pathlib import Path
from typing import Any

import pytest
import yaml

import core.config_validator as cv
from core.config_validator import clear_yaml_cache, load_yaml


@pytest.fixture(autouse=True)
def reset_cache_between_tests() -> None:
    """Ensures each unit test executes with a clean, isolated YAML cache."""
    clear_yaml_cache()


def test_repeated_loads_return_equal_dicts_with_distinct_object_identities(tmp_path: Path) -> None:
    """Verifies that loading the same file twice produces equal data structures that are distinct object instances."""
    sample_path = tmp_path / "sample.yaml"
    sample_path.write_text("server:\n  host: 127.0.0.1\n  port: 8080\n", encoding="utf-8")

    first_load = load_yaml(sample_path)
    second_load = load_yaml(sample_path)

    assert first_load == second_load
    assert first_load is not second_load
    assert first_load["server"] is not second_load["server"]


def test_mutating_earlier_load_result_does_not_affect_subsequent_loads(tmp_path: Path) -> None:
    """Verifies that modifying a returned dictionary does not mutate the cached representation or future results."""
    sample_path = tmp_path / "mutable.yaml"
    sample_path.write_text("limits:\n  rate: 100\n  burst: 200\n", encoding="utf-8")

    first = load_yaml(sample_path)
    first["limits"]["rate"] = 999
    first["limits"]["extra_key"] = "leaked_state"

    second = load_yaml(sample_path)
    assert second["limits"]["rate"] == 100
    assert "extra_key" not in second["limits"]


def test_overwriting_same_path_with_new_content_returns_updated_data(tmp_path: Path) -> None:
    """Verifies that writing new content to the same filesystem path returns updated data even with identical length."""
    target_path = tmp_path / "config.yaml"
    # Both strings are exactly 20 characters long: len("feature_toggle: true") == len("feature_toggle: fals")
    content_a = "feature_toggle: true\n"
    content_b = "feature_toggle: fals\n"
    assert len(content_a) == len(content_b)

    target_path.write_text(content_a, encoding="utf-8")
    loaded_a = load_yaml(target_path)
    assert loaded_a == {"feature_toggle": True}

    target_path.write_text(content_b, encoding="utf-8")
    loaded_b = load_yaml(target_path)
    assert loaded_b == {"feature_toggle": "fals"}


def test_different_paths_with_identical_content_return_independent_copies(tmp_path: Path) -> None:
    """Verifies that two distinct file paths with identical content share cached parse but return separate copies."""
    path_1 = tmp_path / "config_1.yaml"
    path_2 = tmp_path / "config_2.yaml"
    shared_content = "settings:\n  enabled: true\n"
    path_1.write_text(shared_content, encoding="utf-8")
    path_2.write_text(shared_content, encoding="utf-8")

    result_1 = load_yaml(path_1)
    result_2 = load_yaml(path_2)

    assert result_1 == result_2
    assert result_1 is not result_2


def test_missing_file_and_invalid_yaml_raise_expected_exceptions(tmp_path: Path) -> None:
    """Verifies that missing files raise FileNotFoundError and invalid YAML syntax raises yaml.YAMLError."""
    missing_path = tmp_path / "non_existent.yaml"
    with pytest.raises(FileNotFoundError):
        load_yaml(missing_path)

    invalid_path = tmp_path / "malformed.yaml"
    invalid_path.write_text("invalid: yaml: [unbalanced", encoding="utf-8")
    with pytest.raises(yaml.YAMLError):
        load_yaml(invalid_path)


def test_yaml_safe_load_executes_once_per_unique_content(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Verifies via monkeypatching that yaml.safe_load executes once for duplicate loads and once more on change."""
    test_file = tmp_path / "cache_probe.yaml"
    test_file.write_text("key: value_1\n", encoding="utf-8")

    parse_call_count = 0
    original_safe_load = yaml.safe_load

    def counting_safe_load(stream: Any) -> Any:
        """Mock wrapper intercepting and incrementing safe_load invocation counters."""
        nonlocal parse_call_count
        parse_call_count += 1
        return original_safe_load(stream)

    monkeypatch.setattr(cv.yaml, "safe_load", counting_safe_load)

    # 3 loads of identical content should invoke yaml.safe_load exactly once.
    load_yaml(test_file)
    load_yaml(test_file)
    load_yaml(test_file)
    assert parse_call_count == 1

    # Overwriting with different content should trigger exactly a second parse call.
    test_file.write_text("key: value_2\n", encoding="utf-8")
    load_yaml(test_file)
    assert parse_call_count == 2
