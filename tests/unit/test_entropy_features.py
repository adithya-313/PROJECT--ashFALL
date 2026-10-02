# File: tests/unit/test_entropy_features.py
# Purpose: Unit tests for merchant entropy, device ID parsing and device ratio in features/entropy.py.
import pytest

from features.entropy import compute_device_ratio, compute_merchant_entropy, parse_device_id


def test_entropy_empty_list_is_zero():
    assert compute_merchant_entropy([]) == 0.0


def test_entropy_single_merchant_is_zero():
    assert compute_merchant_entropy(["A", "A", "A"]) == 0.0


def test_entropy_two_equal_merchants_is_one_bit():
    assert compute_merchant_entropy(["A", "B"]) == pytest.approx(1.0)


def test_entropy_four_equal_merchants_is_two_bits():
    assert compute_merchant_entropy(["A", "B", "C", "D"]) == pytest.approx(2.0)


def test_parse_device_id_valid():
    assert parse_device_id("dev_id=abc123|os=Windows|browser=Chrome") == "abc123"


def test_parse_device_id_not_first_segment():
    assert parse_device_id("os=Windows|dev_id=xyz|browser=Chrome") == "xyz"


def test_parse_device_id_missing_raises():
    with pytest.raises(ValueError):
        parse_device_id("no_dev_id_here")


def test_device_ratio_two_unique_of_three():
    fingerprints = [
        "dev_id=A|os=X|browser=Y",
        "dev_id=A|os=X|browser=Y",
        "dev_id=B|os=X|browser=Y",
    ]
    assert compute_device_ratio(fingerprints) == pytest.approx(2 / 3)


def test_device_ratio_empty_is_zero():
    assert compute_device_ratio([]) == 0.0


def test_device_ratio_skips_malformed_entries():
    # The malformed entry is skipped for the unique count but still counts in the denominator: 1 unique / 2 total.
    assert compute_device_ratio(["dev_id=A|os=X", "garbage"]) == pytest.approx(0.5)
