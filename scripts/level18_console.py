# File: scripts/level18_console.py
# Purpose: Level 18 task 3 - build the console JSON files from frozen result files only.
#
# What it does:
#   Reads the already-written Level 18 result files (Run A, payment test, calibration, reason codes,
#   ONNX parity, leakage) and the 120 world cards, and writes nine files under
#   reports/level_18/console/. Every file carries "simulated": true and a manifest with the sha256 of
#   each input it read and the generating commit hash. No model is fitted, no row is scored, no SHAP
#   or Triton work runs and no dataset is read.
#
# Rules: docs/research/amendments.md, A-063 and A-067.
#
# Usage:
#   PYTHONPATH=<repo root> py -3.12 -B scripts/level18_console.py

from __future__ import annotations

import hashlib
import json
import statistics
import sys
from pathlib import Path
from typing import Any

import orjson

from world.run_identity import REPO_ROOT, _run_git

GENERATOR = "scripts/level18_console.py"

RUN_A_RESULT = "runs/full/run_a320_stage5c/final/45c496799170/result.json"
RUN_A_EXTRA = "runs/full/run_a320_stage5c/extra_reporting.json"
RUN_A_LEAKAGE = "runs/full/run_a320_stage5c/final/45c496799170/leakage_checks.json"
RUN_A_LEAKAGE_5_6 = "runs/full/run_a320_stage5c/leakage_checks_5_6.json"
PAYMENT_RESULT = "runs/full/payment_model_v2/test_score/result.json"
PAYMENT_METADATA = "runs/full/payment_model_v2/C1/model_metadata.json"
PAYMENT_PREVIOUS_RESULT = "runs/full/payment_model_v1/test_score/result.json"
PAYMENT_DIAGNOSIS = "runs/full/payment_model_v1/diagnosis/diagnosis.json"
CALIBRATION = "runs/full/level18c/calibration/calibration_data.json"
REASON_ACCOUNT = "runs/full/level18c/reason_codes/account_model.json"
REASON_PAYMENT = "runs/full/level18c/reason_codes/payment_model.json"
ONNX_EXPORT = "runs/full/level18c/onnx/onnx_export.json"
WORLD_CARDS_DIR = "runs/full/world_cards"
OUT_DIR = "reports/level_18/console"

# Run B, the behaviour-only fallback account model (A-071, A-072).
RUN_B_RESULT = "runs/full/run_b320/final/45c496799170/result.json"
RUN_B_LEAKAGE = "runs/full/run_b320/final/45c496799170/leakage_checks.json"
RUN_B_THRESHOLDS = "runs/full/run_b320/final/45c496799170/checkpoint/thresholds.json"
RUN_B_STAGE5 = "runs/full/run_b320/stages/final_validation_and_test.json"
RUN_B_LEAKAGE_5_6 = "runs/full/run_b320/leakage_checks_5_6.json"
RUN_B_SERVING = "runs/full/run_b320/serving_validation.json"
FALLBACK_INPUT_PATHS: tuple[str, ...] = (
    RUN_A_RESULT,
    RUN_B_RESULT,
    RUN_B_LEAKAGE,
    RUN_B_THRESHOLDS,
    RUN_B_STAGE5,
    RUN_B_LEAKAGE_5_6,
    RUN_B_SERVING,
)
FALLBACK_FILE = "fallback_account.json"
FALLBACK_METRICS: tuple[str, ...] = (
    "recall_at_capacity",
    "genuine_restriction_rate",
    "median_time_to_detection_hours",
    "yield_measured",
    "yield_reweighted",
)
TRIVIAL_NUMBERS: tuple[str, ...] = ("best_feature", "best_direction_ap", "fail_threshold")
CHECK_5B_NUMBERS: tuple[str, ...] = (
    "model_feature_count",
    "missing_model_feature_count",
    "simulator_only_violation_count",
)
CHECK_6B_NUMBERS: tuple[str, ...] = (
    "validation_average_precision",
    "validation_base_rate",
    "average_precision_over_base_rate",
    "max_ap_over_base_rate",
)
CHECK_6B_SHOWN_AS = "FINDING — documented limitation (A-051)"

INPUT_PATHS: tuple[str, ...] = (
    RUN_A_RESULT,
    RUN_A_EXTRA,
    RUN_A_LEAKAGE,
    RUN_A_LEAKAGE_5_6,
    PAYMENT_RESULT,
    PAYMENT_METADATA,
    PAYMENT_PREVIOUS_RESULT,
    PAYMENT_DIAGNOSIS,
    CALIBRATION,
    REASON_ACCOUNT,
    REASON_PAYMENT,
    ONNX_EXPORT,
)

# The seven published leakage outcomes, in the fixed order fixed by A-063 ruling 9.
LEAKAGE_CHECKS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("truth_leak", RUN_A_LEAKAGE, ("checks", "truth_leak")),
    ("split_leak", RUN_A_LEAKAGE, ("checks", "split_leak")),
    ("check_5a", RUN_A_LEAKAGE_5_6, ("check_5a",)),
    ("check_5b", RUN_A_LEAKAGE_5_6, ("check_5b",)),
    ("check_6a", RUN_A_LEAKAGE_5_6, ("check_6a",)),
    ("check_6b", RUN_A_LEAKAGE_5_6, ("check_6b",)),
    ("section_23_check_7", RUN_A_LEAKAGE, ("checks", "section_23_check_7")),
)

PARITY_BLOCKS: tuple[str, ...] = ("validation", "test")
YIELD_KEYS: tuple[str, ...] = ("measured", "reweighted")


def _git_head() -> str:
    """Return the current commit hash, refusing an empty answer."""
    head = _run_git("git", REPO_ROOT, ["rev-parse", "HEAD"], 5)
    if not head:
        raise ValueError("git rev-parse HEAD returned no commit hash")
    return head


def _sha256_of_file(path: Path) -> str:
    """Return the sha256 hex digest of a file's bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _document(root: Path, rel_path: str) -> dict[str, Any]:
    """Read one frozen JSON input, refusing a missing file, invalid JSON or a non-object document."""
    path = root / rel_path
    if not path.is_file():
        raise ValueError(f"console input is missing: {rel_path}")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"console input is not valid JSON: {rel_path}: {error}") from error
    if not isinstance(document, dict):
        raise ValueError(f"console input must be a JSON object: {rel_path}")
    return document


def _mapping(value: Any, where: str) -> dict[str, Any]:
    """Return an object value, refusing anything else."""
    if not isinstance(value, dict):
        raise ValueError(f"expected an object at {where}, got {type(value).__name__}")
    return value


def _entry(mapping: dict[str, Any], key: str, where: str) -> Any:
    """Return a required key, refusing its absence."""
    if key not in mapping:
        raise ValueError(f"missing key '{key}' at {where}")
    return mapping[key]


def _number(value: Any, where: str) -> float:
    """Return a finite number, refusing booleans, strings and null."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"expected a number at {where}, got {value!r}")
    return float(value)


def _text(value: Any, where: str) -> str:
    """Return a non-empty string, refusing anything else."""
    if not isinstance(value, str) or not value:
        raise ValueError(f"expected a non-empty string at {where}, got {value!r}")
    return value


def _walk(document: dict[str, Any], keys: tuple[str, ...], rel_path: str) -> dict[str, Any]:
    """Follow a key path inside one input document and return the block found there."""
    block = document
    for index, key in enumerate(keys):
        where = f"{rel_path}:{'/'.join(keys[: index + 1])}"
        block = _mapping(_entry(block, key, where), where)
    return block


def _across_worlds_test(run_a: dict[str, Any]) -> dict[str, Any]:
    """Return the Run A `across_worlds.test` block with every scorer block checked to be an object."""
    across = _mapping(_entry(run_a, "across_worlds", RUN_A_RESULT), f"{RUN_A_RESULT}:across_worlds")
    test = _mapping(_entry(across, "test", f"{RUN_A_RESULT}:across_worlds"), f"{RUN_A_RESULT}:across_worlds.test")
    for scorer in test:
        _mapping(test[scorer], f"{RUN_A_RESULT}:across_worlds.test.{scorer}")
    return test


def _extra_aggregate(extra: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return the extra-reporting `test` block and its `aggregate` block."""
    test = _mapping(_entry(extra, "test", RUN_A_EXTRA), f"{RUN_A_EXTRA}:test")
    aggregate = _mapping(_entry(test, "aggregate", f"{RUN_A_EXTRA}:test"), f"{RUN_A_EXTRA}:test.aggregate")
    return test, aggregate


def _world_card_paths(run_a: dict[str, Any]) -> tuple[list[tuple[str, str]], dict[str, dict[str, Any]]]:
    """Return (world_key, run_id) pairs and the card path of each Run A test world."""
    per_world = _mapping(_entry(run_a, "per_world", RUN_A_RESULT), f"{RUN_A_RESULT}:per_world")
    pairs: list[tuple[str, str]] = []
    paths: dict[str, dict[str, Any]] = {}
    for world_key in sorted(per_world):
        where = f"{RUN_A_RESULT}:per_world.{world_key}"
        entry = _mapping(per_world[world_key], where)
        run_id = _text(_entry(entry, "run_id", where), f"{where}.run_id")
        pairs.append((world_key, run_id))
        paths[run_id] = {"path": f"{WORLD_CARDS_DIR}/{run_id}/base_rate.json"}
    return pairs, paths


def collect_comparison(run_a: dict[str, Any], extra: dict[str, Any]) -> dict[str, Any]:
    """Copy the Run A test and stress scorer blocks, the frozen cut and the world count."""
    test = _across_worlds_test(run_a)
    across = _mapping(_entry(run_a, "across_worlds", RUN_A_RESULT), f"{RUN_A_RESULT}:across_worlds")
    stress = _mapping(_entry(across, "stress", f"{RUN_A_RESULT}:across_worlds"), f"{RUN_A_RESULT}:across_worlds.stress")
    manifest = _mapping(_entry(extra, "manifest", RUN_A_EXTRA), f"{RUN_A_EXTRA}:manifest")
    cut = _number(_entry(manifest, "xgboost_cut", f"{RUN_A_EXTRA}:manifest"), f"{RUN_A_EXTRA}:manifest.xgboost_cut")
    per_world = _mapping(_entry(run_a, "per_world", RUN_A_RESULT), f"{RUN_A_RESULT}:per_world")
    decision_cut = {"value": cut, "source": RUN_A_EXTRA, "key": "manifest.xgboost_cut"}
    return {"test": test, "stress": stress, "decision_cut": decision_cut, "test_world_count": len(per_world)}


def collect_hidden_depth(extra: dict[str, Any]) -> dict[str, Any]:
    """Copy the published hidden-mule recall by depth, with its provenance counts."""
    test, aggregate = _extra_aggregate(extra)
    hidden = _mapping(
        _entry(aggregate, "hidden_by_depth", f"{RUN_A_EXTRA}:test.aggregate"),
        f"{RUN_A_EXTRA}:test.aggregate.hidden_by_depth",
    )
    for depth in hidden:
        _mapping(hidden[depth], f"{RUN_A_EXTRA}:test.aggregate.hidden_by_depth.{depth}")
    worlds = _number(_entry(test, "worlds_processed", f"{RUN_A_EXTRA}:test"), f"{RUN_A_EXTRA}:test.worlds_processed")
    checked = _number(
        _entry(test, "world_cards_checked", f"{RUN_A_EXTRA}:test"), f"{RUN_A_EXTRA}:test.world_cards_checked"
    )
    return {"hidden_by_depth": hidden, "worlds_processed": int(worlds), "world_cards_checked": int(checked)}


def collect_held_out_families(extra: dict[str, Any]) -> dict[str, Any]:
    """Copy the published held-out tactic families, with its provenance count."""
    test, aggregate = _extra_aggregate(extra)
    families = _mapping(
        _entry(aggregate, "families", f"{RUN_A_EXTRA}:test.aggregate"), f"{RUN_A_EXTRA}:test.aggregate.families"
    )
    for family in families:
        _mapping(families[family], f"{RUN_A_EXTRA}:test.aggregate.families.{family}")
    worlds = _number(_entry(test, "worlds_processed", f"{RUN_A_EXTRA}:test"), f"{RUN_A_EXTRA}:test.worlds_processed")
    return {"families": families, "worlds_processed": int(worlds)}


def _calibration_points(model: str, state_block: dict[str, Any], state: str, where: str) -> list[dict[str, Any]]:
    """Flatten one calibration state's bins into chart points, copying every value."""
    bins = _entry(state_block, "bins", where)
    if not isinstance(bins, list) or not bins:
        raise ValueError(f"expected a non-empty bin list at {where}.bins")
    points = []
    for index, raw in enumerate(bins):
        row = _mapping(raw, f"{where}.bins[{index}]")
        points.append(
            {
                "model": model,
                "split": "test",
                "state": state,
                "bin": int(_number(_entry(row, "bin", f"{where}.bins[{index}]"), f"{where}.bins[{index}].bin")),
                "count": int(_number(_entry(row, "count", f"{where}.bins[{index}]"), f"{where}.bins[{index}].count")),
                "mean_probability": _number(
                    _entry(row, "mean_probability", f"{where}.bins[{index}]"), "mean_probability"
                ),
                "observed_rate": _number(_entry(row, "observed_rate", f"{where}.bins[{index}]"), "observed_rate"),
                "min_probability": _number(_entry(row, "min_probability", f"{where}.bins[{index}]"), "min_probability"),
                "max_probability": _number(_entry(row, "max_probability", f"{where}.bins[{index}]"), "max_probability"),
            }
        )
    return points


def _calibration_summary(state_block: dict[str, Any], where: str) -> dict[str, Any]:
    """Copy the row, positive, Brier and ECE summary of one calibration state."""
    return {
        "rows": int(_number(_entry(state_block, "rows", where), f"{where}.rows")),
        "positives": int(_number(_entry(state_block, "positives", where), f"{where}.positives")),
        "brier": _number(_entry(state_block, "brier", where), f"{where}.brier"),
        "ece": _number(_entry(state_block, "ece", where), f"{where}.ece"),
    }


def _collect_calibration_model(
    model: str, block: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    """Collect one model's calibration points, summaries and calibrator provenance."""
    test = _mapping(_entry(block, "test", f"calibration.models.{model}"), f"calibration.models.{model}.test")
    points: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    for state, source_key in (("raw", "before_calibration"), ("calibrated", "after_calibration")):
        where = f"calibration.models.{model}.test.{source_key}"
        state_block = _mapping(_entry(test, source_key, f"calibration.models.{model}.test"), where)
        points.extend(_calibration_points(model, state_block, state, where))
        summaries[state] = _calibration_summary(state_block, where)
    calibrator = _mapping(
        _entry(block, "calibrator", f"calibration.models.{model}"), f"calibration.models.{model}.calibrator"
    )
    entry = {
        "file": Path(_text(_entry(calibrator, "path", "calibrator"), "calibrator.path")).name,
        "sha256": _text(_entry(calibrator, "sha256", "calibrator"), "calibrator.sha256"),
        "application": _text(_entry(calibrator, "application", "calibrator"), "calibrator.application"),
    }
    return points, summaries, entry


def collect_calibration(document: dict[str, Any]) -> dict[str, Any]:
    """Copy both models' test calibration points, summaries and calibrator provenance."""
    models = _mapping(_entry(document, "models", CALIBRATION), f"{CALIBRATION}:models")
    points: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    calibrators: dict[str, Any] = {}
    for model in sorted(models):
        block = _mapping(models[model], f"{CALIBRATION}:models.{model}")
        model_points, model_summaries, calibrator = _collect_calibration_model(model, block)
        points.extend(model_points)
        summaries[model] = model_summaries
        calibrators[model] = calibrator
    return {"points": points, "summaries": summaries, "calibrator": calibrators}


def _reason_code_block(document: dict[str, Any], rel_path: str) -> tuple[str, dict[str, Any]]:
    """Collect one reason-code model's frequencies and provenance."""
    model = _text(_entry(document, "model", rel_path), f"{rel_path}:model")
    frequency = _mapping(_entry(document, "frequency", rel_path), f"{rel_path}:frequency")
    codes = _entry(frequency, "codes", f"{rel_path}:frequency")
    if not isinstance(codes, list) or not codes:
        raise ValueError(f"expected a non-empty code list at {rel_path}:frequency.codes")
    for index, raw in enumerate(codes):
        code = _mapping(raw, f"{rel_path}:frequency.codes[{index}]")
        _text(_entry(code, "code", "codes"), "code")
        _number(_entry(code, "rows", "codes"), "rows")
        _number(_entry(code, "share", "codes"), "share")
    manifest = _mapping(_entry(document, "manifest", rel_path), f"{rel_path}:manifest")
    block = {
        "feature_count": int(_number(_entry(document, "feature_count", rel_path), f"{rel_path}:feature_count")),
        "feature_sha256": _text(_entry(document, "feature_sha256", rel_path), f"{rel_path}:feature_sha256"),
        "sample_rows": int(_number(_entry(manifest, "sample_rows", rel_path), f"{rel_path}:manifest.sample_rows")),
        "seed": int(_number(_entry(manifest, "seed", rel_path), f"{rel_path}:manifest.seed")),
        "frequency": frequency,
    }
    return model, block


def collect_reason_codes(account: dict[str, Any], payment: dict[str, Any]) -> dict[str, Any]:
    """Copy both models' reason-code frequencies keyed by the source's own model field."""
    models: dict[str, Any] = {}
    for rel_path, document in ((REASON_ACCOUNT, account), (REASON_PAYMENT, payment)):
        model, block = _reason_code_block(document, rel_path)
        if model in models:
            raise ValueError(f"duplicate reason-code model '{model}' in {rel_path}")
        models[model] = block
    return {"models": models}


def _payment_split(block: dict[str, Any], where: str) -> dict[str, Any]:
    """Copy one payment split's aggregate, NaN share and NaN reason counts."""
    aggregate = _mapping(_entry(block, "aggregate", where), f"{where}.aggregate")
    for metric in aggregate:
        _mapping(aggregate[metric], f"{where}.aggregate.{metric}")
    entry: dict[str, Any] = {
        "aggregate": aggregate,
        "nan_payee_risk_share": _number(_entry(block, "nan_payee_risk_share", where), f"{where}.nan_payee_risk_share"),
    }
    if "nan_reason_counts" in block:
        entry["nan_reason_counts"] = _mapping(block["nan_reason_counts"], f"{where}.nan_reason_counts")
    return entry


def _payment_leakage(leakage: dict[str, Any]) -> dict[str, Any]:
    """Copy the payment diagnosis leakage statuses, leaving the evidence lists in the source."""
    items = _mapping(_entry(leakage, "items", f"{PAYMENT_DIAGNOSIS}:leakage"), "leakage.items")
    id_leak = _mapping(_entry(items, "id_leak", "leakage.items"), "leakage.items.id_leak")
    return {
        "status": _text(_entry(leakage, "status", f"{PAYMENT_DIAGNOSIS}:leakage"), "leakage.status"),
        "item_status": _mapping(_entry(leakage, "item_status", "leakage"), "leakage.item_status"),
        "run_a_check_6b": _mapping(_entry(id_leak, "run_a_check_6b", "leakage.items.id_leak"), "run_a_check_6b"),
    }


def collect_payment_results(
    result: dict[str, Any],
    metadata: dict[str, Any],
    previous_result: dict[str, Any],
    diagnosis: dict[str, Any],
) -> dict[str, Any]:
    """Copy the v2 payment results beside v1's previous-version block, its provenance and leakage."""
    splits = _mapping(_entry(result, "splits", PAYMENT_RESULT), f"{PAYMENT_RESULT}:splits")
    out_splits = {
        split: _payment_split(
            _mapping(_entry(splits, split, f"{PAYMENT_RESULT}:splits"), f"{PAYMENT_RESULT}:splits.{split}"),
            f"payment_result.splits.{split}",
        )
        for split in ("test", "stress")
    }
    diag_splits = _mapping(_entry(diagnosis, "splits", PAYMENT_DIAGNOSIS), f"{PAYMENT_DIAGNOSIS}:splits")
    diag_test = _mapping(_entry(diag_splits, "test", "diagnosis.splits"), "diagnosis.splits.test")
    feature_truth_ap = {
        split: _mapping(
            _entry(
                _mapping(_entry(block, "feature_truth_ap", f"diagnosis.splits.{split}"), "feature_truth_ap"),
                "payee_risk",
                "feature_truth_ap",
            ),
            "feature_truth_ap.payee_risk",
        )
        for split, block in (
            ("test", diag_test),
            ("stress", _mapping(_entry(diag_splits, "stress", "diagnosis.splits"), "diagnosis.splits.stress")),
        )
    }
    stress_block = _mapping(_entry(splits, "stress", f"{PAYMENT_RESULT}:splits"), f"{PAYMENT_RESULT}:splits.stress")
    alone = _mapping(
        _entry(stress_block, "payee_risk_alone_truth_ap", f"{PAYMENT_RESULT}:splits.stress"),
        f"{PAYMENT_RESULT}:splits.stress.payee_risk_alone_truth_ap",
    )
    for key in ("median", "p10", "p90", "worlds_used"):
        expected = _number(_entry(alone, key, "payee_risk_alone_truth_ap"), f"payee_risk_alone_truth_ap.{key}")
        recorded = _number(_entry(feature_truth_ap["stress"], key, "feature_truth_ap.stress"), key)
        if recorded != expected:
            raise ValueError(
                f"v1 diagnosis payee_risk stress {key} {recorded!r} disagrees with the v2 result's {expected!r}"
            )
    artifacts = _mapping(_entry(metadata, "artifacts", PAYMENT_METADATA), f"{PAYMENT_METADATA}:artifacts")
    best_params = _mapping(
        _entry(
            _mapping(_entry(metadata, "best_params", PAYMENT_METADATA), f"{PAYMENT_METADATA}:best_params"),
            "all",
            f"{PAYMENT_METADATA}:best_params",
        ),
        f"{PAYMENT_METADATA}:best_params.all",
    )
    frozen_model = {
        "sha256": _text(_entry(artifacts, "frozen_model.json", "artifacts"), "artifacts.frozen_model.json"),
        "feature_sha256": _text(_entry(metadata, "feature_sha256", PAYMENT_METADATA), "feature_sha256"),
        "feature_count": int(_number(_entry(metadata, "feature_count", PAYMENT_METADATA), "feature_count")),
        "rounds": int(_number(_entry(best_params, "n_estimators", "best_params.all"), "best_params.all.n_estimators")),
        "seed": int(_number(_entry(metadata, "seed", PAYMENT_METADATA), "seed")),
    }
    previous_manifest = _mapping(
        _entry(previous_result, "manifest", PAYMENT_PREVIOUS_RESULT), f"{PAYMENT_PREVIOUS_RESULT}:manifest"
    )
    previous_model = _mapping(
        _entry(previous_manifest, "model", f"{PAYMENT_PREVIOUS_RESULT}:manifest"),
        f"{PAYMENT_PREVIOUS_RESULT}:manifest.model",
    )
    previous_splits = _mapping(
        _entry(previous_result, "splits", PAYMENT_PREVIOUS_RESULT), f"{PAYMENT_PREVIOUS_RESULT}:splits"
    )
    previous_version = {
        "version": "v1",
        "git_hash": _text(
            _entry(previous_manifest, "git_hash", "manifest"), f"{PAYMENT_PREVIOUS_RESULT}:manifest.git_hash"
        ),
        "threshold": _number(
            _entry(previous_result, "threshold", PAYMENT_PREVIOUS_RESULT), f"{PAYMENT_PREVIOUS_RESULT}:threshold"
        ),
        "hold_share": _number(
            _entry(previous_result, "hold_share", PAYMENT_PREVIOUS_RESULT), f"{PAYMENT_PREVIOUS_RESULT}:hold_share"
        ),
        "splits": {
            split: _payment_split(
                _mapping(
                    _entry(previous_splits, split, f"{PAYMENT_PREVIOUS_RESULT}:splits"),
                    f"{PAYMENT_PREVIOUS_RESULT}:splits.{split}",
                ),
                f"previous_result.splits.{split}",
            )
            for split in ("test", "stress")
        },
        "payee_precision": _mapping(
            _entry(diag_test, "payee_precision", "diagnosis.splits.test"), "diagnosis.splits.test.payee_precision"
        ),
        "feature_truth_ap": feature_truth_ap,
        "frozen_model": {
            "sha256": _text(_entry(previous_model, "sha256", "manifest.model"), "manifest.model.sha256"),
            "feature_count": int(
                _number(_entry(previous_model, "feature_count", "manifest.model"), "manifest.model.feature_count")
            ),
            "rounds": int(_number(_entry(previous_model, "rounds", "manifest.model"), "manifest.model.rounds")),
            "seed": int(_number(_entry(previous_model, "seed", "manifest.model"), "manifest.model.seed")),
        },
    }
    leakage = _mapping(_entry(diagnosis, "leakage", PAYMENT_DIAGNOSIS), f"{PAYMENT_DIAGNOSIS}:leakage")
    test_block = _mapping(_entry(splits, "test", f"{PAYMENT_RESULT}:splits"), f"{PAYMENT_RESULT}:splits.test")
    return {
        "version": "v2",
        "threshold": _number(_entry(result, "threshold", PAYMENT_RESULT), f"{PAYMENT_RESULT}:threshold"),
        "hold_share": _number(_entry(result, "hold_share", PAYMENT_RESULT), f"{PAYMENT_RESULT}:hold_share"),
        "splits": out_splits,
        "payee_precision": _mapping(
            _entry(test_block, "payee_precision", f"{PAYMENT_RESULT}:splits.test"),
            f"{PAYMENT_RESULT}:splits.test.payee_precision",
        ),
        "feature_truth_ap": feature_truth_ap,
        "frozen_model": frozen_model,
        "leakage": _payment_leakage(leakage),
        "previous_version": previous_version,
    }


def collect_leakage(leakage: dict[str, Any], leakage_5_6: dict[str, Any]) -> dict[str, Any]:
    """Copy the seven published leakage outcomes, the trivial-world check and the check-4 record."""
    documents = {RUN_A_LEAKAGE: leakage, RUN_A_LEAKAGE_5_6: leakage_5_6}
    checks = []
    for name, rel_path, keys in LEAKAGE_CHECKS:
        detail = _walk(documents[rel_path], keys, rel_path)
        checks.append(
            {
                "check": name,
                "status": _text(_entry(detail, "status", rel_path), f"{rel_path}.{name}.status"),
                "source": rel_path,
                "detail": detail,
            }
        )
    checks_block = _mapping(_entry(leakage, "checks", RUN_A_LEAKAGE), f"{RUN_A_LEAKAGE}:checks")
    trivial = _mapping(_entry(checks_block, "trivial_world", RUN_A_LEAKAGE), "checks.trivial_world")
    container = _mapping(
        _entry(checks_block, "section_23_checks_4_to_7", RUN_A_LEAKAGE), "checks.section_23_checks_4_to_7"
    )
    not_implemented = {
        "check_4": {
            "status": "not run",
            "reason": _text(_entry(container, "reason", "section_23_checks_4_to_7"), "checks_4_to_7.reason"),
            "source": RUN_A_LEAKAGE,
        }
    }
    return {
        "checks": checks,
        "trivial_world": trivial,
        "section_23_checks_4_to_7": container,
        "not_implemented": not_implemented,
    }


def _parity_block(parity: dict[str, Any], where: str) -> dict[str, Any]:
    """Check one published parity block carries its row, difference and over-tolerance counts."""
    for key in ("rows", "nan_rows", "max_abs_diff", "mean_abs_diff", "rows_over_tolerance"):
        _number(_entry(parity, key, where), f"{where}.{key}")
    return parity


def _onnx_model(model: str, block: dict[str, Any]) -> dict[str, Any]:
    """Collect one model's ONNX file identity, parity blocks and (account) merge proof."""
    where = f"{ONNX_EXPORT}:{model}"
    onnx = _mapping(_entry(block, "onnx", where), f"{where}.onnx")
    parity = _mapping(_entry(block, "parity", where), f"{where}.parity")
    entry: dict[str, Any] = {
        "file": Path(_text(_entry(onnx, "path", f"{where}.onnx"), f"{where}.onnx.path")).name,
        "sha256": _text(_entry(onnx, "sha256", f"{where}.onnx"), f"{where}.onnx.sha256"),
        "parity": {
            name: _parity_block(_mapping(_entry(parity, name, f"{where}.parity"), f"{where}.parity.{name}"), name)
            for name in PARITY_BLOCKS
        },
    }
    if model == "account":
        entry["merge_proof"] = _mapping(_entry(block, "merge_proof", where), f"{where}.merge_proof")
    return entry


def collect_onnx_parity(document: dict[str, Any]) -> dict[str, Any]:
    """Copy both models' ONNX identities and parity blocks, the tolerances and the converter versions."""
    models = {
        model: _onnx_model(model, _mapping(_entry(document, model, ONNX_EXPORT), model))
        for model in ("account", "payment")
    }
    return {
        "models": models,
        "tolerances": _mapping(_entry(document, "tolerances", ONNX_EXPORT), f"{ONNX_EXPORT}:tolerances"),
        "converter": _mapping(_entry(document, "converter", ONNX_EXPORT), f"{ONNX_EXPORT}:converter"),
    }


def _yield_world(world_key: str, entry: dict[str, Any], card: dict[str, Any]) -> dict[str, Any]:
    """Collect one world's card factor and frozen measured/re-weighted yield, proving they agree."""
    where = f"{RUN_A_RESULT}:per_world.{world_key}"
    run_id = _text(_entry(entry, "run_id", where), f"{where}.run_id")
    factor = _number(_entry(card, "fraud_inflation_factor", f"world card {run_id}"), "fraud_inflation_factor")
    scorers = _mapping(_entry(entry, "scorers", where), f"{where}.scorers")
    xgboost = _mapping(_entry(scorers, "xgboost", f"{where}.scorers"), f"{where}.scorers.xgboost")
    yield_block = _mapping(_entry(xgboost, "yield_rate", f"{where}.scorers.xgboost"), "yield_rate")
    measured = _number(_entry(yield_block, "measured", "yield_rate"), "yield_rate.measured")
    reweighted = yield_block.get("reweighted")
    if reweighted is not None:
        reweighted = _number(reweighted, "yield_rate.reweighted")
    if (factor > 0) == (reweighted is None):
        raise ValueError(f"world {run_id}: card factor {factor!r} and frozen reweighted yield {reweighted!r} disagree")
    return {
        "world_key": world_key,
        "run_id": run_id,
        "fraud_inflation_factor": factor,
        "complaints": int(_number(_entry(card, "complaints", f"card {run_id}"), "complaints")),
        "complaints_per_100000": _number(
            _entry(card, "complaints_per_100000", f"card {run_id}"), "complaints_per_100000"
        ),
        "measured": measured,
        "reweighted": reweighted,
    }


def collect_yield_e1(run_a: dict[str, Any], cards: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Copy each world's E1 inflation factor and yields, and compute the median of the defined factors."""
    per_world = _mapping(_entry(run_a, "per_world", RUN_A_RESULT), f"{RUN_A_RESULT}:per_world")
    worlds = []
    for world_key in sorted(per_world):
        entry = _mapping(per_world[world_key], f"{RUN_A_RESULT}:per_world.{world_key}")
        run_id = _text(_entry(entry, "run_id", world_key), f"per_world.{world_key}.run_id")
        if run_id not in cards:
            raise ValueError(f"no world card was loaded for run id {run_id}")
        worlds.append(_yield_world(world_key, entry, cards[run_id]))
    factors = [world["fraud_inflation_factor"] for world in worlds if world["fraud_inflation_factor"] > 0]
    if not factors:
        raise ValueError("no world has a positive E1 inflation factor")
    test = _across_worlds_test(run_a)
    scorers = {
        scorer: {
            "yield_measured": _mapping(_entry(block, "yield_measured", scorer), f"{scorer}.yield_measured"),
            "yield_reweighted": _mapping(_entry(block, "yield_reweighted", scorer), f"{scorer}.yield_reweighted"),
        }
        for scorer, block in test.items()
    }
    inflation = {
        "worlds_defined": len(factors),
        "worlds_zero_factor": sum(1 for world in worlds if world["fraud_inflation_factor"] <= 0),
        "median_defined": statistics.median(factors),
    }
    return {
        "metric": "yield_rate",
        "config": "config/evaluation.yaml#base_rate_reweighting",
        "worlds": worlds,
        "inflation_factor": inflation,
        "scorers": scorers,
    }


def _write(root: Path, output_dir: Path, name: str, data: dict[str, Any], sources: list[str], git_hash: str) -> Path:
    """Write one console file with its envelope, refusing a source that is not a file under the root."""
    if len(set(sources)) != len(sources):
        raise ValueError(f"{name}: duplicate source path")
    document = {
        "simulated": True,
        "manifest": {
            "generator": GENERATOR,
            "git_hash": git_hash,
            "sources": {rel_path: _sha256_of_file(root / rel_path) for rel_path in sorted(sources)},
        },
        "data": data,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / name
    path.write_bytes(orjson.dumps(document, option=orjson.OPT_SORT_KEYS | orjson.OPT_INDENT_2) + b"\n")
    return path


def _load_world_cards(root: Path, run_a: dict[str, Any]) -> tuple[list[str], dict[str, dict[str, Any]]]:
    """Load every Run A test world's card and return its path list and the cards keyed by run id."""
    _, entries = _world_card_paths(run_a)
    paths: list[str] = []
    cards: dict[str, dict[str, Any]] = {}
    for run_id in sorted(entries):
        rel_path = entries[run_id]["path"]
        cards[run_id] = _document(root, rel_path)
        paths.append(rel_path)
    return paths, cards


def build(root: Path = REPO_ROOT, output_dir: Path | None = None) -> list[Path]:
    """Read every frozen input under root and write the nine console files."""
    documents = {rel_path: _document(root, rel_path) for rel_path in INPUT_PATHS}
    run_a = documents[RUN_A_RESULT]
    card_paths, cards = _load_world_cards(root, run_a)
    git_hash = _git_head()
    target = output_dir if output_dir is not None else root / OUT_DIR
    outputs: tuple[tuple[str, dict[str, Any], list[str]], ...] = (
        ("comparison.json", collect_comparison(run_a, documents[RUN_A_EXTRA]), [RUN_A_RESULT, RUN_A_EXTRA]),
        ("hidden_depth.json", collect_hidden_depth(documents[RUN_A_EXTRA]), [RUN_A_EXTRA]),
        ("held_out_families.json", collect_held_out_families(documents[RUN_A_EXTRA]), [RUN_A_EXTRA]),
        ("calibration.json", collect_calibration(documents[CALIBRATION]), [CALIBRATION]),
        (
            "reason_codes.json",
            collect_reason_codes(documents[REASON_ACCOUNT], documents[REASON_PAYMENT]),
            [REASON_ACCOUNT, REASON_PAYMENT],
        ),
        (
            "payment_results.json",
            collect_payment_results(
                documents[PAYMENT_RESULT],
                documents[PAYMENT_METADATA],
                documents[PAYMENT_PREVIOUS_RESULT],
                documents[PAYMENT_DIAGNOSIS],
            ),
            [PAYMENT_RESULT, PAYMENT_METADATA, PAYMENT_PREVIOUS_RESULT, PAYMENT_DIAGNOSIS],
        ),
        (
            "leakage.json",
            collect_leakage(documents[RUN_A_LEAKAGE], documents[RUN_A_LEAKAGE_5_6]),
            [RUN_A_LEAKAGE, RUN_A_LEAKAGE_5_6],
        ),
        ("onnx_parity.json", collect_onnx_parity(documents[ONNX_EXPORT]), [ONNX_EXPORT]),
        ("yield_e1.json", collect_yield_e1(run_a, cards), [RUN_A_RESULT, *card_paths]),
    )
    return [_write(root, target, name, data, sources, git_hash) for name, data, sources in outputs]


def _fallback_check(name: str, rel_path: str, block: dict[str, Any], keys: tuple[str, ...]) -> dict[str, Any]:
    """Copy one Run B leakage check's recorded status and the named numbers."""
    where = f"{rel_path}.{name}"
    return {
        "check": name,
        "status": _text(_entry(block, "status", where), f"{where}.status"),
        "source": rel_path,
        "numbers": {key: _entry(block, key, where) for key in keys},
    }


def _fallback_leakage(leakage: dict[str, Any], leakage_5_6: dict[str, Any]) -> list[dict[str, Any]]:
    """Run B's recorded checks. Check 6b must be recorded as failed; A-072 fixes how it is shown, not its status."""
    checks_block = _mapping(_entry(leakage, "checks", RUN_B_LEAKAGE), f"{RUN_B_LEAKAGE}:checks")
    published = [
        _fallback_check(name, rel_path, _walk(document, keys, rel_path), numbers)
        for name, rel_path, document, keys, numbers in (
            ("truth_leak", RUN_B_LEAKAGE, checks_block, ("truth_leak",), ()),
            ("split_leak", RUN_B_LEAKAGE, checks_block, ("split_leak",), ()),
            ("trivial_world", RUN_B_LEAKAGE, checks_block, ("trivial_world",), TRIVIAL_NUMBERS),
            ("check_5a", RUN_B_LEAKAGE_5_6, leakage_5_6, ("check_5a",), ("hit_count",)),
            ("check_5b", RUN_B_LEAKAGE_5_6, leakage_5_6, ("check_5b",), CHECK_5B_NUMBERS),
            ("check_6a", RUN_B_LEAKAGE_5_6, leakage_5_6, ("check_6a",), ("hit_count",)),
            ("check_6b", RUN_B_LEAKAGE_5_6, leakage_5_6, ("check_6b",), CHECK_6B_NUMBERS),
            ("section_23_check_7", RUN_B_LEAKAGE, checks_block, ("section_23_check_7",), ()),
        )
    ]
    six_b = next(check for check in published if check["check"] == "check_6b")
    if six_b["status"] != "fail":
        raise ValueError(f"A-072 records Run B check 6b as failed, but the source says {six_b['status']!r}")
    six_b["shown_as"] = CHECK_6B_SHOWN_AS
    return published


def _fallback_test(result: dict[str, Any], rel_path: str) -> dict[str, Any]:
    """The five published test spreads of the xgboost scorer."""
    block = _walk(result, ("across_worlds", "test", "xgboost"), rel_path)
    return {metric: _mapping(_entry(block, metric, rel_path), f"{rel_path}:{metric}") for metric in FALLBACK_METRICS}


def collect_fallback_account(documents: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Collect Run B, the behaviour-only fallback model, beside Run A, with the A-071 serving numbers."""
    serving = documents[RUN_B_SERVING]
    served = _text(_entry(serving, "serve", RUN_B_SERVING), f"{RUN_B_SERVING}:serve")
    if served != "run_a":
        raise ValueError(f"Run B is published as the fallback only when Run A is served; got serve={served!r}")
    validation = {}
    for run in ("run_a", "run_b"):
        block = _mapping(_entry(serving, run, RUN_B_SERVING), f"{RUN_B_SERVING}:{run}")
        validation[run] = {
            "median_recall_at_capacity": _number(
                _entry(block, "median_recall_at_capacity", run), f"{run}.median_recall_at_capacity"
            ),
            "defined_worlds": int(_number(_entry(block, "defined_worlds", run), f"{run}.defined_worlds")),
        }
    thresholds = documents[RUN_B_THRESHOLDS]
    stage = _walk(documents[RUN_B_STAGE5], ("result", "success_criterion"), RUN_B_STAGE5)
    run_b = documents[RUN_B_RESULT]
    return {
        "role": "fallback model for banks with no complaint history",
        "serving": {
            "served": served,
            "amendment": "A-071",
            "capacity_pct": _number(_entry(serving, "capacity_pct", RUN_B_SERVING), "capacity_pct"),
            "validation": validation,
        },
        "test": {
            "run_a": _fallback_test(documents[RUN_A_RESULT], RUN_A_RESULT),
            "run_b": _fallback_test(run_b, RUN_B_RESULT),
        },
        "test_world_count": len(_mapping(_entry(run_b, "per_world", RUN_B_RESULT), f"{RUN_B_RESULT}:per_world")),
        "decision_cut": {
            "value": _number(_entry(thresholds, "xgboost_cut", RUN_B_THRESHOLDS), "xgboost_cut"),
            "source": RUN_B_THRESHOLDS,
            "key": "xgboost_cut",
        },
        "success_criterion_status": _text(_entry(stage, "status", RUN_B_STAGE5), "success_criterion.status"),
        "leakage": _fallback_leakage(documents[RUN_B_LEAKAGE], documents[RUN_B_LEAKAGE_5_6]),
    }


def build_fallback(root: Path = REPO_ROOT, output_dir: Path | None = None) -> Path:
    """Read Run B's frozen files and Run A's result, and write the fallback account console file."""
    documents = {rel_path: _document(root, rel_path) for rel_path in FALLBACK_INPUT_PATHS}
    target = output_dir if output_dir is not None else root / OUT_DIR
    data = collect_fallback_account(documents)
    return _write(root, target, FALLBACK_FILE, data, list(FALLBACK_INPUT_PATHS), _git_head())


def main() -> int:
    """Write the console files and print each path relative to the repository root."""
    for path in sorted([*build(), build_fallback()]):
        print(f"wrote {path.relative_to(REPO_ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
