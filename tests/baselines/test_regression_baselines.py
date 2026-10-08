# File: tests/baselines/test_regression_baselines.py
# Purpose: Regression baselines. Each test builds a fresh snapshot of something that must not change by accident (the
#          model v1 numbers, the calibration bins, the API's public shape, the Docker Compose health setup) and
#          compares it with the snapshot saved in this folder. No Docker, database or Kafka is needed.
#
# If a test fails because of a change you made on purpose, run `python scripts/update_baselines.py` and
# explain the diff in the pull request. See tests/baselines/README.md.
from pathlib import Path
from typing import Any

import orjson

from scripts.update_baselines import (
    CALIBRATION_BASELINE_FILE,
    COMPOSE_BASELINE_FILE,
    MODEL_BASELINE_FILE,
    OPENAPI_BASELINE_FILE,
    build_calibration_baseline,
    build_compose_baseline,
    build_model_baseline,
    build_openapi_snapshot,
)

BASELINE_DIR = Path(__file__).resolve().parent
FLOAT_TOLERANCE = 1e-9


def load_baseline(file_name: str) -> Any:
    """Read one saved baseline file from this folder."""
    return orjson.loads((BASELINE_DIR / file_name).read_bytes())


def explain(what_changed: str, file_name: str) -> str:
    """The message shown when a baseline test fails, in plain English."""
    return (
        f"{what_changed}\n"
        f"The saved baseline is tests/baselines/{file_name}.\n"
        "If this change is NOT intended, undo it. If it IS intended, run `python scripts/update_baselines.py`, "
        "look at the git diff of tests/baselines/, and explain the change in your pull request."
    )


def find_differences(expected: Any, actual: Any, where: str = "") -> list[str]:
    """
    Compare two JSON-like values and list every difference in words.

    Floats may differ by FLOAT_TOLERANCE (to ignore tiny rounding); everything else, including hashes, must be equal.
    """
    if isinstance(expected, dict) and isinstance(actual, dict):
        differences = []
        for key in sorted(set(expected) | set(actual)):
            if key not in actual:
                differences.append(f"{where}/{key}: was in the baseline but is now missing")
            elif key not in expected:
                differences.append(f"{where}/{key}: is new, not in the baseline")
            else:
                differences.extend(find_differences(expected[key], actual[key], f"{where}/{key}"))
        return differences
    if isinstance(expected, list) and isinstance(actual, list):
        if len(expected) != len(actual):
            return [f"{where}: the list had {len(expected)} items in the baseline, now {len(actual)}"]
        differences = []
        for index, (old_item, new_item) in enumerate(zip(expected, actual)):
            differences.extend(find_differences(old_item, new_item, f"{where}[{index}]"))
        return differences
    if isinstance(expected, float) and isinstance(actual, float):
        if abs(expected - actual) > FLOAT_TOLERANCE:
            return [f"{where}: baseline {expected!r}, now {actual!r}"]
        return []
    if expected != actual:
        return [f"{where}: baseline {expected!r}, now {actual!r}"]
    return []


def check_against_baseline(file_name: str, current: Any, what: str) -> None:
    """Fail with a helpful message if the current snapshot differs from the saved one."""
    differences = find_differences(load_baseline(file_name), current)
    shown = "\n".join(f"  - {line}" for line in differences[:10])
    assert not differences, explain(f"{what} changed ({len(differences)} difference(s)):\n{shown}", file_name)


def test_model_v1_metrics_match_the_baseline():
    """The committed model v1 metrics (PR-AUC, ROC-AUC, row counts, DVC hashes) equal the saved baseline."""
    check_against_baseline(MODEL_BASELINE_FILE, build_model_baseline(), "The model v1 metrics")


def test_calibration_bins_match_the_baseline():
    """The committed calibration bins equal the saved baseline."""
    check_against_baseline(CALIBRATION_BASELINE_FILE, build_calibration_baseline(), "The calibration bins")


def test_api_shape_matches_the_snapshot():
    """The API's routes and the request/response shapes of /health and /score equal the saved snapshot."""
    check_against_baseline(OPENAPI_BASELINE_FILE, build_openapi_snapshot(), "The API's public shape")


def test_compose_health_setup_matches_the_baseline():
    """Every Docker Compose service and whether it has a healthcheck equals the saved baseline."""
    check_against_baseline(COMPOSE_BASELINE_FILE, build_compose_baseline(), "The Docker Compose health setup")


def test_the_difference_finder_reports_changes_in_words():
    """The helper itself: a changed float, a changed hash, a missing key and a new key are each reported."""
    expected = {"score": 0.5, "hash": "abc", "gone": 1}
    actual = {"score": 0.6, "hash": "abd", "new": 2}
    report = " ".join(find_differences(expected, actual))
    assert "/score" in report and "/hash" in report and "/gone" in report and "/new" in report
    assert find_differences({"score": 0.5}, {"score": 0.5 + 1e-12}) == []
