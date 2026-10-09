"""Tests for core/config_validator.py.

All fixtures are written inline here. These tests never read a real config file.
"""
import yaml

from core.config_validator import check_block, count_labels, find_labelled_blocks, validate_file

LONG_WORDS_25 = " ".join(["word"] * 25)


def make_sourced_block():
    return {
        "label": "sourced",
        "reason": "Taken from a regulator page.",
        "source": {"url": "https://example.org/page", "date": "2026-01-01", "quote": "a short quote"},
    }


def make_reported_block():
    return {
        "label": "reported",
        "reason": "Seen in a secondary source.",
        "source": {"url": "https://example.org/page", "date": "2026-01-01"},
    }


def make_range_block():
    return {
        "label": "assumption",
        "reason": "A guess.",
        "training": [2, 5],
        "test": [1, 7],
        "stress": [1, 10],
    }


def write_yaml(tmp_path, data):
    """Save a dictionary as a YAML file and return its path."""
    file_path = tmp_path / "fixture.yaml"
    file_path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return file_path


def errors_text(errors):
    return " | ".join(errors)


# ---- Blocks that must pass ----

def test_sourced_block_passes():
    assert check_block("a", make_sourced_block()) == []


def test_reported_block_passes():
    assert check_block("a", make_reported_block()) == []


def test_assumption_block_passes():
    assert check_block("a", {"label": "assumption", "reason": "A guess."}) == []


def test_policy_block_passes():
    assert check_block("a", {"label": "policy", "reason": "Our choice."}) == []


def test_range_block_passes():
    assert check_block("a", make_range_block()) == []


def test_empty_value_with_not_found_passes():
    block = {"label": "assumption", "reason": "Nothing found.", "value": None, "not_found": True}
    assert check_block("a", block) == []


# ---- Blocks that must be rejected ----

def test_missing_reason_is_rejected():
    block = {"label": "policy"}
    assert "reason is missing or empty" in errors_text(check_block("a", block))


def test_empty_reason_is_rejected():
    block = {"label": "policy", "reason": "   "}
    assert "reason is missing or empty" in errors_text(check_block("a", block))


def test_unknown_label_is_rejected():
    block = {"label": "guess", "reason": "x"}
    assert "label 'guess' is not allowed" in errors_text(check_block("a", block))


def test_sourced_without_url_is_rejected():
    block = make_sourced_block()
    del block["source"]["url"]
    assert "sourced block has no source.url" in errors_text(check_block("a", block))


def test_sourced_with_long_quote_is_rejected():
    block = make_sourced_block()
    block["source"]["quote"] = LONG_WORDS_25
    assert "must be under 25" in errors_text(check_block("a", block))


def test_reported_without_date_is_rejected():
    block = make_reported_block()
    del block["source"]["date"]
    assert "no source.date" in errors_text(check_block("a", block))


def test_empty_value_without_not_found_is_rejected():
    block = {"label": "assumption", "reason": "x", "value": None}
    assert "not_found is not true" in errors_text(check_block("a", block))


def test_range_with_min_not_below_max_is_rejected():
    block = make_range_block()
    block["training"] = [5, 5]
    assert "not below max" in errors_text(check_block("a", block))


def test_training_outside_test_is_rejected():
    block = make_range_block()
    block["training"] = [0, 5]
    assert "training range is not inside the test range" in errors_text(check_block("a", block))


def test_test_outside_stress_is_rejected():
    block = make_range_block()
    block["test"] = [1, 20]
    assert "test range is not inside the stress range" in errors_text(check_block("a", block))


def test_file_with_no_labelled_blocks_is_rejected(tmp_path):
    file_path = write_yaml(tmp_path, {"just": {"some": "data"}})
    assert validate_file(file_path) == ["the file has no labelled blocks"]


# ---- Whole-file helpers ----

def test_validate_file_accepts_a_good_file(tmp_path):
    file_path = write_yaml(tmp_path, {"one": make_sourced_block(), "two": make_range_block()})
    assert validate_file(file_path) == []


def test_find_labelled_blocks_looks_inside_lists(tmp_path):
    data = {"items": [{"label": "policy", "reason": "x"}]}
    blocks = find_labelled_blocks(data)
    assert [path for path, _ in blocks] == ["items[0]"]


def test_count_labels(tmp_path):
    data = {
        "a": make_sourced_block(),
        "b": make_reported_block(),
        "c": make_reported_block(),
        "d": {"label": "assumption", "reason": "x"},
    }
    counts = count_labels(write_yaml(tmp_path, data))
    assert counts == {"sourced": 1, "reported": 2, "assumption": 1, "policy": 0}
