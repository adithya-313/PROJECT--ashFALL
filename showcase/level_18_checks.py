# File: showcase/level_18_checks.py
# Purpose: Level 18 showcase checks: seven read-only views of the committed Level 18 console files.
"""
WHAT: The check functions Level 18's showcase runs. Each one reads one console file under
reports/level_18/console/ and asserts the facts A-065 fixed for it: ordered spreads, complete world
counts, every published leakage outcome, calibrated reliability bins, the ONNX parity within the
console's own tolerance, and the E1 re-weighting against the inflation factor.

WHY: The Level 18 results are frozen and already published in the console files. The showcase must
show exactly those numbers and fail when a file is missing or contradicts them; nothing here scores,
trains or re-measures anything, and no dataset row is read.
"""

from __future__ import annotations

import json
import math
import statistics
from pathlib import Path
from typing import Any

from showcase.framework import CheckOutcome

REPO_ROOT = Path(__file__).resolve().parent.parent
CONSOLE_ROOT = REPO_ROOT / "reports" / "level_18" / "console"

# The console file each check reads, in the A-065 order.
CHECK_FILES: dict[str, str] = {
    "18-1": "comparison.json",
    "18-2": "held_out_families.json",
    "18-3": "hidden_depth.json",
    "18-4": "leakage.json",
    "18-5": "calibration.json",
    "18-6": "onnx_parity.json",
    "18-7": "yield_e1.json",
}

TEST_SCORERS = ("anomaly", "rules", "rules_and_anomaly", "xgboost")
TEST_METRICS = (
    "genuine_restriction_rate",
    "median_time_to_detection_hours",
    "recall_at_capacity",
    "yield_measured",
    "yield_reweighted",
)
DETECTION_METRIC = "median_time_to_detection_hours"
EXPECTED_WORLDS = 120
EXPECTED_FAMILIES = ("device_rotation", "slow_drip")
EXPECTED_DEPTHS = ("1", "2", "3", "4", "5")
EXPECTED_LEAKAGE_CHECKS = (
    "truth_leak",
    "split_leak",
    "check_5a",
    "check_5b",
    "check_6a",
    "check_6b",
    "section_23_check_7",
)
CALIBRATION_SERIES = (
    ("account", "raw"),
    ("account", "calibrated"),
    ("payment", "raw"),
    ("payment", "calibrated"),
)
PARITY_SPLITS = ("validation", "test")
PARITY_MODELS = ("account", "payment")
BINS_PER_SERIES = 10


def _mapping(value: Any, where: str) -> dict[str, Any]:
    """Return value as an object, or raise ValueError naming where it should have been."""
    if not isinstance(value, dict):
        raise ValueError(f"{where}: expected an object, got {type(value).__name__}")
    return value


def _sequence(value: Any, where: str) -> list[Any]:
    """Return value as a list, or raise ValueError naming where it should have been."""
    if not isinstance(value, list):
        raise ValueError(f"{where}: expected a list, got {type(value).__name__}")
    return value


def _number(value: Any, where: str) -> float:
    """Return value as a float, refusing booleans and everything that is not a number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{where}: expected a number, got {type(value).__name__}")
    return float(value)


def _count(value: Any, where: str) -> int:
    """Return value as a non-negative integer count, refusing anything else."""
    number = _number(value, where)
    if number < 0 or number != int(number):
        raise ValueError(f"{where}: expected a non-negative whole number, got {value!r}")
    return int(number)


def _text(value: Any, where: str) -> str:
    """Return value as a non-empty string, or raise ValueError naming where it should have been."""
    if not isinstance(value, str) or not value:
        raise ValueError(f"{where}: expected a non-empty string")
    return value


def load_console(name: str, console_root: Path = CONSOLE_ROOT) -> dict[str, Any]:
    """
    WHAT: Reads one console file and returns its data block after checking the A-063 envelope.
    A missing file, malformed JSON, a simulated flag that is not true, a manifest without a generator
    or sources mapping, or a missing data block raises ValueError naming the file and the key.

    WHY: The showcase must never display a number from a file it could not identify as a labelled
    console document; a plain refusal is better than a default or a silent skip.

    Args:
        name: The check key, such as "18-1", naming the file in CHECK_FILES.
        console_root: Folder holding the console files; the committed one by default.

    Returns:
        The file's data object.
    """
    if name not in CHECK_FILES:
        raise ValueError(f"unknown console document {name!r}; known: {sorted(CHECK_FILES)}")
    path = console_root / CHECK_FILES[name]
    if not path.is_file():
        raise ValueError(f"console file is missing: {path}")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as error:
        raise ValueError(f"{path}: not valid JSON ({error})") from error
    document = _mapping(document, str(path))
    if document.get("simulated") is not True:
        raise ValueError(f"{path}: 'simulated' is missing or not true")
    manifest = _mapping(document.get("manifest"), f"{path}:manifest")
    _text(manifest.get("generator"), f"{path}:manifest.generator")
    if not _mapping(manifest.get("sources"), f"{path}:manifest.sources"):
        raise ValueError(f"{path}: 'manifest.sources' is empty")
    return _mapping(document.get("data"), f"{path}:data")


def _spread(block: Any, where: str) -> dict[str, float]:
    """Return p10, median and p90 of one published spread and refuse a spread that is out of order."""
    block = _mapping(block, where)
    p10 = _number(block.get("p10"), f"{where}.p10")
    median = _number(block.get("median"), f"{where}.median")
    p90 = _number(block.get("p90"), f"{where}.p90")
    if not p10 <= median <= p90:
        raise ValueError(f"{where}: p10 {p10} > median {median} or median {median} > p90 {p90}")
    return {"p10": p10, "median": median, "p90": p90}


def _fraction(value: Any, where: str) -> float:
    """Return a fraction in [0, 1], refusing anything outside it."""
    number = _number(value, where)
    if not 0.0 <= number <= 1.0:
        raise ValueError(f"{where}: expected a fraction in [0, 1], got {number}")
    return number


async def check_18_1_comparison(workspace: Any) -> CheckOutcome:
    """
    WHAT: Reads the account comparison console and checks the 120-world test spreads of the four
    scorers, including that the xgboost recall at capacity median is above both rules baselines.

    WHY: This is the frozen result the Level 18 account model is judged by; a spread that is out of
    order or a model that no longer sits above the rules would be invisible if the showcase only
    printed the numbers.
    """
    data = load_console("18-1")
    worlds = _count(data.get("test_world_count"), "comparison:data.test_world_count")
    if worlds != EXPECTED_WORLDS:
        raise ValueError(f"comparison:data.test_world_count is {worlds}, expected {EXPECTED_WORLDS}")
    test = _mapping(data.get("test"), "comparison:data.test")
    spreads: dict[str, dict[str, dict[str, float]]] = {}
    for scorer in TEST_SCORERS:
        block = _mapping(test.get(scorer), f"comparison:test.{scorer}")
        spreads[scorer] = {}
        for metric in TEST_METRICS:
            spread = _spread(block.get(metric), f"comparison:test.{scorer}.{metric}")
            if metric == DETECTION_METRIC:
                if spread["median"] < 0.0:
                    raise ValueError(f"comparison:test.{scorer}.{metric}: negative median {spread['median']}")
            else:
                for key, value in spread.items():
                    _fraction(value, f"comparison:test.{scorer}.{metric}.{key}")
            spreads[scorer][metric] = spread

    xgboost_recall = spreads["xgboost"]["recall_at_capacity"]
    for baseline in ("rules", "rules_and_anomaly"):
        baseline_median = spreads[baseline]["recall_at_capacity"]["median"]
        if xgboost_recall["median"] <= baseline_median:
            raise ValueError(
                f"xgboost recall at capacity median {xgboost_recall['median']} is not above "
                f"{baseline} {baseline_median}"
            )

    key_number = (
        f"xgboost {xgboost_recall['median']:.6f} (p10 {xgboost_recall['p10']:.6f}, p90 {xgboost_recall['p90']:.6f}) "
        f"vs rules {spreads['rules']['recall_at_capacity']['median']:.6f} "
        f"and rules+anomaly {spreads['rules_and_anomaly']['recall_at_capacity']['median']:.6f}, {worlds} worlds"
    )
    return CheckOutcome(
        key_number=key_number,
        evidence={"test_world_count": worlds, "spreads": spreads, "decision_cut": data.get("decision_cut")},
    )


async def check_18_2_held_out_families(workspace: Any) -> CheckOutcome:
    """
    WHAT: Reads the held-out family console and checks device rotation and slow drip: every world is
    either used or absent, and both recall spreads (at capacity and at the cut) are ordered.

    WHY: Held-out families are the only families the model never trained on; losing worlds from the
    count or hiding a spread would make the family result look better than it is.
    """
    data = load_console("18-2")
    processed = _count(data.get("worlds_processed"), "held_out_families:data.worlds_processed")
    families = _mapping(data.get("families"), "held_out_families:data.families")
    if tuple(families) != EXPECTED_FAMILIES:
        raise ValueError(
            f"held_out_families:data.families has {sorted(families)}, expected {sorted(EXPECTED_FAMILIES)}"
        )

    result: dict[str, Any] = {}
    for family in EXPECTED_FAMILIES:
        block = _mapping(families.get(family), f"held_out_families:{family}")
        used = _count(block.get("worlds_used"), f"held_out_families:{family}.worlds_used")
        absent = _count(block.get("worlds_absent"), f"held_out_families:{family}.worlds_absent")
        if used + absent != processed:
            raise ValueError(f"held_out_families:{family}: {used} used + {absent} absent != {processed} processed")
        accounts = _count(block.get("accounts_total"), f"held_out_families:{family}.accounts_total")
        if accounts == 0:
            raise ValueError(f"held_out_families:{family}.accounts_total is 0")
        capacity = _spread(block.get("recall_at_capacity"), f"held_out_families:{family}.recall_at_capacity")
        at_cut = _spread(block.get("recall_at_cut"), f"held_out_families:{family}.recall_at_cut")
        result[family] = {
            "accounts_total": accounts,
            "worlds_used": used,
            "worlds_absent": absent,
            "recall_at_capacity": capacity,
            "recall_at_cut": at_cut,
        }

    device = result["device_rotation"]
    drip = result["slow_drip"]
    key_number = (
        f"device_rotation {device['recall_at_capacity']['median']:.6f} at capacity / "
        f"{device['recall_at_cut']['median']:.6f} at cut "
        f"({device['worlds_used']} used, {device['worlds_absent']} absent), "
        f"slow_drip {drip['recall_at_capacity']['median']:.6f} / {drip['recall_at_cut']['median']:.6f} "
        f"({drip['worlds_used']} used, {drip['worlds_absent']} absent)"
    )
    return CheckOutcome(
        key_number=key_number,
        evidence={"worlds_processed": processed, "families": result},
    )


async def check_18_3_hidden_depth(workspace: Any) -> CheckOutcome:
    """
    WHAT: Reads the hidden depth console and checks depths 1 to 5: every recall spread is ordered and
    the worlds used equal the worlds counted for that depth.

    WHY: Depth says how far a hidden mule sits from the surface of a world; a depth with fewer worlds
    than it counts would quietly change what its recall means.
    """
    data = load_console("18-3")
    hidden = _mapping(data.get("hidden_by_depth"), "hidden_depth:data.hidden_by_depth")
    if tuple(hidden) != EXPECTED_DEPTHS:
        raise ValueError(f"hidden_depth:data.hidden_by_depth has {sorted(hidden)}, expected {list(EXPECTED_DEPTHS)}")

    result: dict[str, Any] = {}
    medians: list[float] = []
    world_counts: list[int] = []
    for depth in EXPECTED_DEPTHS:
        block = _mapping(hidden.get(depth), f"hidden_depth:depth {depth}")
        recall_block = _mapping(block.get("recall"), f"hidden_depth:depth {depth}.recall")
        recall = _spread(recall_block, f"hidden_depth:depth {depth}.recall")
        used = _count(recall_block.get("worlds_used"), f"hidden_depth:depth {depth}.recall.worlds_used")
        counted = _count(recall_block.get("world_count"), f"hidden_depth:depth {depth}.recall.world_count")
        if used != counted:
            raise ValueError(f"hidden_depth:depth {depth}: worlds_used {used} != world_count {counted}")
        accounts = _count(block.get("accounts_total"), f"hidden_depth:depth {depth}.accounts_total")
        result[depth] = {"accounts_total": accounts, "worlds_used": used, "recall": recall}
        medians.append(recall["median"])
        world_counts.append(used)

    key_number = (
        f"medians depth 1-5: {', '.join(f'{median:.6f}' for median in medians)}; "
        f"{min(world_counts)} to {max(world_counts)} worlds"
    )
    return CheckOutcome(
        key_number=key_number,
        evidence={
            "worlds_processed": _count(data.get("worlds_processed"), "hidden_depth:data.worlds_processed"),
            "world_cards_checked": _count(data.get("world_cards_checked"), "hidden_depth:data.world_cards_checked"),
            "depths": result,
        },
    )


async def check_18_4_leakage(workspace: Any) -> CheckOutcome:
    """
    WHAT: Reads the leakage console and checks all seven published outcomes in their fixed order,
    including that the one recorded failure is check_6b and that its average precision over the base
    rate really exceeds its own limit, and that check 4 is recorded as not run.

    WHY: A leakage report is only honest if a failure stays visible and a missing check is named
    instead of being dropped; the showcase must show the failure, not work around it.
    """
    data = load_console("18-4")
    checks = _sequence(data.get("checks"), "leakage:data.checks")
    names = []
    statuses: dict[str, str] = {}
    for entry_index, entry in enumerate(checks):
        block = _mapping(entry, f"leakage:data.checks[{entry_index}]")
        name = _text(block.get("check"), f"leakage:data.checks[{entry_index}].check")
        status = _text(block.get("status"), f"leakage:data.checks[{entry_index}].status")
        names.append(name)
        statuses[name] = status
    if tuple(names) != EXPECTED_LEAKAGE_CHECKS:
        raise ValueError(f"leakage:data.checks is {names}, expected {list(EXPECTED_LEAKAGE_CHECKS)}")

    failures = [name for name, status in statuses.items() if status == "fail"]
    if failures != ["check_6b"]:
        raise ValueError(f"leakage:data.checks has failures {failures}, expected exactly ['check_6b']")

    six_b = _mapping(checks[5].get("detail"), "leakage:check_6b.detail")
    observed = _number(
        six_b.get("average_precision_over_base_rate"), "leakage:check_6b.average_precision_over_base_rate"
    )
    limit = _number(six_b.get("max_ap_over_base_rate"), "leakage:check_6b.max_ap_over_base_rate")
    if not observed > limit:
        raise ValueError(f"leakage:check_6b: {observed} over the base rate does not exceed its limit {limit}")

    trivial = _mapping(data.get("trivial_world"), "leakage:data.trivial_world")
    if _text(trivial.get("status"), "leakage:trivial_world.status") != "pass":
        raise ValueError(f"leakage:trivial_world.status is {trivial.get('status')!r}, expected 'pass'")
    best = _number(trivial.get("best_direction_ap"), "leakage:trivial_world.best_direction_ap")
    threshold = _number(trivial.get("fail_threshold"), "leakage:trivial_world.fail_threshold")
    if not best < threshold:
        raise ValueError(f"leakage:trivial_world: best feature {best} reaches the fail threshold {threshold}")

    section_23 = _mapping(data.get("section_23_checks_4_to_7"), "leakage:data.section_23_checks_4_to_7")
    if _text(section_23.get("status"), "leakage:section_23_checks_4_to_7.status") != "not run":
        raise ValueError(f"leakage:section_23_checks_4_to_7.status is {section_23.get('status')!r}, expected 'not run'")
    not_implemented = _mapping(data.get("not_implemented"), "leakage:data.not_implemented")
    check_4 = _mapping(not_implemented.get("check_4"), "leakage:data.not_implemented.check_4")
    if _text(check_4.get("status"), "leakage:not_implemented.check_4.status") != "not run":
        raise ValueError(f"leakage:not_implemented.check_4.status is {check_4.get('status')!r}, expected 'not run'")

    passed = sum(1 for status in statuses.values() if status == "pass")
    key_number = (
        f"{len(names)} checks: {passed} pass, {len(failures)} fail "
        f"(check_6b {observed:.6f} > {limit:.6f}); check 4 not run"
    )
    return CheckOutcome(
        key_number=key_number,
        evidence={
            "checks": [{"check": name, "status": statuses[name]} for name in names],
            "check_6b": six_b,
            "trivial_world": trivial,
            "section_23_checks_4_to_7": section_23,
            "not_implemented": not_implemented,
        },
    )


async def check_18_5_calibration(workspace: Any) -> CheckOutcome:
    """
    WHAT: Reads the calibration console and checks the four reliability series (account and payment,
    raw and calibrated), the bin counts against the published rows, the bin order, and that the
    account calibrated ECE is below its raw ECE; attaches the reliability scatter chart.

    WHY: A model card can claim calibration only if the bins behind it are complete and ordered; the
    chart shows the raw and calibrated curves the ECE numbers summarise.
    """
    data = load_console("18-5")
    points = _sequence(data.get("points"), "calibration:data.points")
    summaries = _mapping(data.get("summaries"), "calibration:data.summaries")

    series: list[dict[str, Any]] = []
    rows_by_series: dict[str, int] = {}
    for model, state in CALIBRATION_SERIES:
        bins = [
            _mapping(point, f"calibration:points[{index}]")
            for index, point in enumerate(points)
            if isinstance(point, dict) and point.get("model") == model and point.get("state") == state
        ]
        if len(bins) != BINS_PER_SERIES:
            raise ValueError(f"calibration:{model}/{state} has {len(bins)} bins, expected {BINS_PER_SERIES}")
        bins.sort(key=lambda bin_block: _count(bin_block.get("bin"), f"calibration:{model}/{state}.bin"))
        counts = 0
        chart_points: list[list[float]] = []
        for bin_block in bins:
            low = _fraction(bin_block.get("min_probability"), f"calibration:{model}/{state}.min_probability")
            mean = _fraction(bin_block.get("mean_probability"), f"calibration:{model}/{state}.mean_probability")
            high = _fraction(bin_block.get("max_probability"), f"calibration:{model}/{state}.max_probability")
            observed = _fraction(bin_block.get("observed_rate"), f"calibration:{model}/{state}.observed_rate")
            if not low <= mean <= high:
                raise ValueError(f"calibration:{model}/{state}: min {low} > mean {mean} or mean > max {high}")
            counts += _count(bin_block.get("count"), f"calibration:{model}/{state}.count")
            chart_points.append([mean, observed])
        summary = _mapping(summaries.get(model), f"calibration:summaries.{model}")
        published_rows = _count(
            _mapping(summary.get(state), f"calibration:summaries.{model}.{state}").get("rows"),
            f"calibration:summaries.{model}.{state}.rows",
        )
        if counts != published_rows:
            raise ValueError(f"calibration:{model}/{state}: bin counts sum to {counts}, rows say {published_rows}")
        rows_by_series[f"{model}/{state}"] = counts
        series.append({"name": f"{model} {state}", "points": chart_points})

    ece: dict[str, dict[str, float]] = {}
    for model in ("account", "payment"):
        model_summary = _mapping(summaries.get(model), f"calibration:summaries.{model}")
        ece[model] = {}
        for state in ("raw", "calibrated"):
            state_summary = _mapping(model_summary.get(state), f"calibration:summaries.{model}.{state}")
            ece[model][state] = _fraction(state_summary.get("ece"), f"calibration:summaries.{model}.{state}.ece")
            for field in ("rows", "positives"):
                _count(state_summary.get(field), f"calibration:summaries.{model}.{state}.{field}")
        raw = _mapping(model_summary.get("raw"), f"calibration:summaries.{model}.raw")
        calibrated = _mapping(model_summary.get("calibrated"), f"calibration:summaries.{model}.calibrated")
        if raw.get("rows") != calibrated.get("rows") or raw.get("positives") != calibrated.get("positives"):
            raise ValueError(f"calibration:{model}: raw and calibrated states cover different rows or positives")
    if not ece["account"]["calibrated"] < ece["account"]["raw"]:
        raise ValueError(
            "calibration:account calibrated ECE "
            f"{ece['account']['calibrated']} is not below raw {ece['account']['raw']}"
        )

    key_number = (
        f"account ECE {ece['account']['raw']:.6f} to {ece['account']['calibrated']:.6f}, "
        f"payment {ece['payment']['raw']:.6f} to {ece['payment']['calibrated']:.6f} "
        f"({rows_by_series['account/raw']} and {rows_by_series['payment/raw']} rows)"
    )
    return CheckOutcome(
        key_number=key_number,
        evidence={
            "summaries": summaries,
            "calibrator": data.get("calibrator"),
            "bins_used": BINS_PER_SERIES,
        },
        chart={
            "type": "scatter",
            "title": "Reliability: mean predicted probability against observed rate",
            "x_label": "mean predicted probability",
            "y_label": "observed rate",
            "series": series,
        },
    )


async def check_18_6_onnx_parity(workspace: Any) -> CheckOutcome:
    """
    WHAT: Reads the ONNX parity console and checks both models on validation and test against the
    tolerance the file records, plus the account merge proof (no decision flips, top set identical,
    Spearman at or above the floor).

    WHY: The served model is the ONNX file; parity is the claim that the served probabilities are the
    native ones, so it must be checked against the console's own tolerance rather than a new one.
    """
    data = load_console("18-6")
    tolerances = _mapping(data.get("tolerances"), "onnx_parity:data.tolerances")
    tolerance = _number(tolerances.get("parity"), "onnx_parity:tolerances.parity")
    spearman_floor = _number(tolerances.get("spearman_min"), "onnx_parity:tolerances.spearman_min")
    models = _mapping(data.get("models"), "onnx_parity:data.models")

    parity: dict[str, dict[str, dict[str, float]]] = {}
    for model in PARITY_MODELS:
        model_block = _mapping(models.get(model), f"onnx_parity:models.{model}")
        splits = _mapping(model_block.get("parity"), f"onnx_parity:models.{model}.parity")
        parity[model] = {}
        for split in PARITY_SPLITS:
            block = _mapping(splits.get(split), f"onnx_parity:models.{model}.parity.{split}")
            rows = _count(block.get("rows"), f"onnx_parity:{model}.{split}.rows")
            if rows == 0:
                raise ValueError(f"onnx_parity:{model}.{split}.rows is 0")
            over = _count(block.get("rows_over_tolerance"), f"onnx_parity:{model}.{split}.rows_over_tolerance")
            if over != 0:
                raise ValueError(f"onnx_parity:{model}.{split}: {over} of {rows} rows exceed the tolerance")
            maximum = _number(block.get("max_abs_diff"), f"onnx_parity:{model}.{split}.max_abs_diff")
            mean = _number(block.get("mean_abs_diff"), f"onnx_parity:{model}.{split}.mean_abs_diff")
            if maximum > tolerance:
                raise ValueError(f"onnx_parity:{model}.{split}: max diff {maximum} exceeds the tolerance {tolerance}")
            if mean > maximum:
                raise ValueError(f"onnx_parity:{model}.{split}: mean diff {mean} exceeds the max diff {maximum}")
            parity[model][split] = {"rows": rows, "max_abs_diff": maximum, "mean_abs_diff": mean}

    merge = _mapping(_mapping(models.get("account"), "onnx_parity:models.account").get("merge_proof"),
                     "onnx_parity:models.account.merge_proof")
    flips = _count(merge.get("decision_flips"), "onnx_parity:merge_proof.decision_flips")
    if flips != 0:
        raise ValueError(f"onnx_parity:merge_proof.decision_flips is {flips}, expected 0")
    if merge.get("top_set_identical") is not True:
        raise ValueError(
            f"onnx_parity:merge_proof.top_set_identical is {merge.get('top_set_identical')!r}, expected true"
        )
    spearman = _number(merge.get("spearman"), "onnx_parity:merge_proof.spearman")
    if spearman < spearman_floor:
        raise ValueError(f"onnx_parity:merge_proof.spearman {spearman} is below the floor {spearman_floor}")

    account = parity["account"]
    payment = parity["payment"]
    key_number = (
        f"account {account['validation']['max_abs_diff']:.6e} / {account['test']['max_abs_diff']:.6e}, "
        f"payment {payment['validation']['max_abs_diff']:.6e} / {payment['test']['max_abs_diff']:.6e} "
        f"vs {tolerance:.6e}; 0 rows over"
    )
    return CheckOutcome(
        key_number=key_number,
        evidence={
            "tolerance": tolerance,
            "spearman_floor": spearman_floor,
            "parity": parity,
            "merge_proof": merge,
            "converter": data.get("converter"),
        },
    )


async def check_18_7_yield(workspace: Any) -> CheckOutcome:
    """
    WHAT: Reads the E1 console and checks the 120 worlds: the re-weighted yield is present exactly
    when the inflation factor is positive, the factor counts and median agree with the published
    summary, and the xgboost re-weighted median is below its measured median.

    WHY: The E1 re-weighting is the only correction between simulated complaints and a bank base
    rate; if a world's factor and its re-weighted yield stopped agreeing, the published yield would
    mean something other than it says.
    """
    data = load_console("18-7")
    worlds = _sequence(data.get("worlds"), "yield_e1:data.worlds")
    if len(worlds) != EXPECTED_WORLDS:
        raise ValueError(f"yield_e1:data.worlds has {len(worlds)} worlds, expected {EXPECTED_WORLDS}")

    factors: list[float] = []
    measured_values: list[float] = []
    reweighted_values = 0
    for world_index, entry in enumerate(worlds):
        world = _mapping(entry, f"yield_e1:worlds[{world_index}]")
        world_key = _text(world.get("world_key"), f"yield_e1:worlds[{world_index}].world_key")
        factor = _number(world.get("fraud_inflation_factor"), f"yield_e1:{world_key}.fraud_inflation_factor")
        if factor < 0:
            raise ValueError(f"yield_e1:{world_key}.fraud_inflation_factor is negative: {factor}")
        measured = _fraction(world.get("measured"), f"yield_e1:{world_key}.measured")
        reweighted = world.get("reweighted")
        if (factor > 0) != (reweighted is not None):
            raise ValueError(
                f"yield_e1:{world_key}: factor {factor} positive={factor > 0} but reweighted is {reweighted!r}"
            )
        if reweighted is not None:
            _fraction(reweighted, f"yield_e1:{world_key}.reweighted")
            reweighted_values += 1
        factors.append(factor)
        measured_values.append(measured)

    positive = [factor for factor in factors if factor > 0]
    zeros = [factor for factor in factors if factor == 0]
    inflation = _mapping(data.get("inflation_factor"), "yield_e1:data.inflation_factor")
    defined = _count(inflation.get("worlds_defined"), "yield_e1:inflation_factor.worlds_defined")
    zero_count = _count(inflation.get("worlds_zero_factor"), "yield_e1:inflation_factor.worlds_zero_factor")
    if defined != len(positive) or zero_count != len(zeros):
        raise ValueError(
            f"yield_e1:inflation_factor counts {defined} defined and {zero_count} zero, "
            f"worlds hold {len(positive)} positive and {len(zeros)} zero"
        )
    median_defined = _number(inflation.get("median_defined"), "yield_e1:inflation_factor.median_defined")
    recomputed = statistics.median(positive)
    if not math.isclose(recomputed, median_defined, rel_tol=1e-12, abs_tol=0.0):
        raise ValueError(f"yield_e1:inflation_factor.median_defined {median_defined} != recomputed {recomputed}")

    scorers = _mapping(data.get("scorers"), "yield_e1:data.scorers")
    xgboost = _mapping(scorers.get("xgboost"), "yield_e1:data.scorers.xgboost")
    measured_median = _number(
        _mapping(xgboost.get("yield_measured"), "yield_e1:scorers.xgboost.yield_measured").get("median"),
        "yield_e1:scorers.xgboost.yield_measured.median",
    )
    reweighted_median = _number(
        _mapping(xgboost.get("yield_reweighted"), "yield_e1:scorers.xgboost.yield_reweighted").get("median"),
        "yield_e1:scorers.xgboost.yield_reweighted.median",
    )
    if not reweighted_median < measured_median:
        raise ValueError(
            f"yield_e1: xgboost reweighted median {reweighted_median} is not below measured {measured_median}"
        )

    key_number = (
        f"xgboost {measured_median:.6f} measured to {reweighted_median:.6f} reweighted; "
        f"median factor {median_defined:.6f} over {len(positive)} of {len(worlds)} worlds"
    )
    return CheckOutcome(
        key_number=key_number,
        evidence={
            "worlds": len(worlds),
            "worlds_positive_factor": len(positive),
            "worlds_zero_factor": len(zeros),
            "reweighted_values": reweighted_values,
            "inflation_factor": inflation,
            "xgboost": xgboost,
            "scorers": data.get("scorers"),
            "metric": data.get("metric"),
        },
    )


def find_missing_preconditions(console_root: Path = CONSOLE_ROOT) -> list[str]:
    """
    WHAT: Names every console file the seven checks read and does not find.

    WHY: A missing file must stop the run with the exact path named, so the checks report NOT RUN
    with a reason instead of failing later inside a check.
    """
    return [
        f"Missing required console file: {console_root / name}"
        for name in CHECK_FILES.values()
        if not (console_root / name).is_file()
    ]
