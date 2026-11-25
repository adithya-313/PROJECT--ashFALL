"""
WHAT: Tests for the Level 18 showcase: the seven read-only console checks, the envelope loader and
the fixed order, kinds and parts A-065 registered.

WHY: The checks display frozen console numbers and never score anything, so this fast test guards
the numbers the report shows and the refusals that keep a malformed console document out of it.
"""

import asyncio
import json
from pathlib import Path
from typing import Any

import pytest

from showcase import level_18, level_18_checks
from showcase.framework import CheckOutcome, validate_checks

EXPECTED_ORDER = [
    "18-1 comparison: recall at capacity with p10 and p90 spreads",
    "18-2 held-out families: device rotation and slow drip at capacity and at the cut",
    "18-3 hidden depth: recall by hidden mule depth 1 to 5",
    "18-4 leakage: all seven published checks, including the recorded 6b failure",
    "18-5 calibration: reliability bins before and after isotonic, with a chart",
    "18-6 ONNX vs native: parity on validation and test within the 1e-5 tolerance",
    "18-7 yield: measured against E1 re-weighting with the inflation factor",
]
EXPECTED_KINDS = ["normal", "normal", "edge", "negative", "normal", "consistency", "edge"]
EXPECTED_PARTS = ["a", "a", "a", "b", "b", "b", "a"]


def test_seven_checks_in_the_registered_order() -> None:
    """Checks the showcase lists exactly the seven A-065 checks in their fixed order."""
    assert [check.name for check in level_18.get_checks()] == EXPECTED_ORDER


def test_checks_carry_the_registered_kinds_and_parts() -> None:
    """Checks the list passes validate_checks and carries the registered kind and part tags."""
    checks = level_18.get_checks()
    validate_checks(checks)
    assert [check.kind for check in checks] == EXPECTED_KINDS
    assert [check.part for check in checks] == EXPECTED_PARTS


def test_module_has_the_attributes_the_runner_needs() -> None:
    """Checks the level module defines the number, title, database flag, story and precondition function."""
    assert level_18.LEVEL_NUMBER == 18 and level_18.NEEDS_DATABASE is False
    assert level_18.LEVEL_TITLE and level_18.WHAT_YOU_SAW and callable(level_18.find_missing_preconditions)


def test_preconditions_name_every_missing_console_file(tmp_path: Path) -> None:
    """Checks every one of the seven required console files is named by its full path when absent."""
    problems = level_18_checks.find_missing_preconditions(tmp_path)
    assert len(problems) == len(level_18_checks.CHECK_FILES) == 7
    for name in level_18_checks.CHECK_FILES.values():
        assert f"Missing required console file: {tmp_path / name}" in problems


def test_committed_console_meets_the_preconditions() -> None:
    """Checks the committed Level 18 console folder holds all seven required files."""
    assert level_18_checks.find_missing_preconditions() == []


def test_load_console_refuses_missing_unsimulated_and_dataless_documents(tmp_path: Path) -> None:
    """Checks the envelope loader refuses a missing file, a non-true simulated flag and a document without data."""
    with pytest.raises(ValueError, match="console file is missing"):
        level_18_checks.load_console("18-1", tmp_path)

    source = level_18_checks.CONSOLE_ROOT / level_18_checks.CHECK_FILES["18-1"]
    document: dict[str, Any] = json.loads(source.read_text(encoding="utf-8"))
    document["simulated"] = False
    (tmp_path / "comparison.json").write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match="'simulated' is missing or not true"):
        level_18_checks.load_console("18-1", tmp_path)

    document["simulated"] = True
    del document["data"]
    (tmp_path / "comparison.json").write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(ValueError, match=":data: expected an object"):
        level_18_checks.load_console("18-1", tmp_path)


def test_check_18_1_reports_the_frozen_comparison() -> None:
    """Checks 18-1 reports the xgboost recall spread against both rules baselines over 120 worlds."""
    outcome = asyncio.run(level_18_checks.check_18_1_comparison(None))
    assert isinstance(outcome, CheckOutcome)
    assert outcome.key_number == (
        "xgboost 0.642857 (p10 0.477045, p90 0.833333) vs rules 0.368112 "
        "and rules+anomaly 0.375000, 120 worlds"
    )
    assert outcome.evidence["test_world_count"] == 120
    assert outcome.evidence["spreads"]["xgboost"]["recall_at_capacity"]["median"] == pytest.approx(
        0.6428571428571429
    )
    assert outcome.evidence["spreads"]["rules"]["recall_at_capacity"]["median"] == pytest.approx(
        0.36811201445347785
    )


def test_check_18_2_reports_the_held_out_families() -> None:
    """Checks 18-2 reports both families with every world counted as used or absent."""
    outcome = asyncio.run(level_18_checks.check_18_2_held_out_families(None))
    assert outcome.key_number == (
        "device_rotation 0.666667 at capacity / 0.930288 at cut (60 used, 60 absent), "
        "slow_drip 0.800000 / 0.909091 (59 used, 61 absent)"
    )
    assert outcome.evidence["worlds_processed"] == 120
    assert outcome.evidence["families"]["device_rotation"]["accounts_total"] == 448
    assert outcome.evidence["families"]["slow_drip"]["worlds_absent"] == 61


def test_check_18_3_reports_the_hidden_depth_recall() -> None:
    """Checks 18-3 reports depths 1 to 5 with the worlds used per depth."""
    outcome = asyncio.run(level_18_checks.check_18_3_hidden_depth(None))
    assert outcome.key_number == (
        "medians depth 1-5: 1.000000, 0.944444, 1.000000, 0.857143, 1.000000; 30 to 120 worlds"
    )
    assert outcome.evidence["depths"]["5"]["worlds_used"] == 30
    assert outcome.evidence["depths"]["1"]["accounts_total"] == 704
    assert outcome.evidence["world_cards_checked"] == 120


def test_check_18_4_reports_every_leakage_check() -> None:
    """Checks 18-4 requires all seven published statuses plus check 4 and shows 6b as the A-051 finding."""
    outcome = asyncio.run(level_18_checks.check_18_4_leakage(None))
    assert outcome.key_number == (
        "8 recorded: 7 pass, check_6b FINDING — documented limitation (A-051) 10.415126 > 2.000000; "
        "check 4 (trivial world) pass 0.028270 < 0.900000"
    )
    assert [entry["status"] for entry in outcome.evidence["checks"]] == [
        "pass",
        "pass",
        "pass",
        "pass",
        "pass",
        "fail",
        "pass",
    ]
    assert outcome.evidence["check_4"]["status"] == "pass"
    assert outcome.evidence["check_4"]["best_direction_ap"] == pytest.approx(0.028269987287482604)
    assert outcome.evidence["check_4_source"] == (
        "runs/full/run_a320_stage5c/final/45c496799170/leakage_checks.json:checks.trivial_world"
    )
    assert outcome.evidence["check_6b"]["average_precision_over_base_rate"] == pytest.approx(10.415125553116175)


def test_check_18_5_reports_the_calibration_and_its_chart() -> None:
    """Checks 18-5 reports both models' ECEs and attaches the four-series reliability scatter chart."""
    outcome = asyncio.run(level_18_checks.check_18_5_calibration(None))
    assert outcome.key_number == (
        "account ECE 0.127137 to 0.001843, payment 0.012284 to 0.012821 (1395478 and 1846587 rows)"
    )
    assert outcome.evidence["summaries"]["account"]["calibrated"]["ece"] == pytest.approx(0.0018426116879839516)
    chart = outcome.chart
    assert chart is not None
    assert chart["type"] == "scatter"
    assert [series["name"] for series in chart["series"]] == [
        "account raw",
        "account calibrated",
        "payment raw",
        "payment calibrated",
    ]
    assert all(len(series["points"]) == 10 for series in chart["series"])


def test_check_18_6_reports_the_onnx_parity() -> None:
    """Checks 18-6 reports both models' validation and test parity within the file's tolerance."""
    outcome = asyncio.run(level_18_checks.check_18_6_onnx_parity(None))
    assert outcome.key_number == (
        "account 1.788418e-07 / 1.678902e-07, payment 4.470348e-07 / 5.364418e-07 "
        "vs 1.000000e-05; 0 rows over"
    )
    assert outcome.evidence["parity"]["account"]["validation"]["max_abs_diff"] == pytest.approx(
        1.7884182645921953e-07
    )
    assert outcome.evidence["merge_proof"]["decision_flips"] == 0
    assert outcome.evidence["tolerance"] == pytest.approx(1e-05)


def test_check_18_7_reports_the_e1_reweighted_yield() -> None:
    """Checks 18-7 reports the xgboost measured-to-reweighted medians and the positive factor count."""
    outcome = asyncio.run(level_18_checks.check_18_7_yield(None))
    assert outcome.key_number == (
        "xgboost 0.923077 measured to 0.272955 reweighted; median factor 39.316691 over 118 of 120 worlds"
    )
    assert outcome.evidence["worlds"] == 120
    assert outcome.evidence["worlds_positive_factor"] == 118
    assert outcome.evidence["worlds_zero_factor"] == 2
