# File: scripts/level18_model_cards.py
# Purpose: Level 18 task 10 - build the two mule model cards from the committed console files only.
#
# What it does:
#   Reads seven of the nine Level 18 console files (A-063) and writes
#   reports/level_18/model_cards/{mule_account_xgb,mule_payment_xgb}.{json,md}. Each card carries the intended
#   use, the SIMULATED data label, metrics with spreads, calibration before and after, the leakage outcomes
#   including check 6b, ONNX parity and five known limitations stated plainly. No model is loaded or fitted,
#   no row is scored and no other file is read.
#
# Rules: docs/research/amendments.md, A-064 and A-067.
#
# Usage:
#   PYTHONPATH=<repo root> py -3.12 -B scripts/level18_model_cards.py

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import orjson

from scripts.level18_console import _document, _entry, _git_head, _mapping, _number, _sha256_of_file, _text
from world.run_identity import REPO_ROOT

GENERATOR = "scripts/level18_model_cards.py"
CONSOLE_DIR = "reports/level_18/console"
COMPARISON = f"{CONSOLE_DIR}/comparison.json"
HIDDEN_DEPTH = f"{CONSOLE_DIR}/hidden_depth.json"
HELD_OUT_FAMILIES = f"{CONSOLE_DIR}/held_out_families.json"
CALIBRATION = f"{CONSOLE_DIR}/calibration.json"
LEAKAGE = f"{CONSOLE_DIR}/leakage.json"
PAYMENT_RESULTS = f"{CONSOLE_DIR}/payment_results.json"
ONNX_PARITY = f"{CONSOLE_DIR}/onnx_parity.json"
INPUT_PATHS: tuple[str, ...] = (
    COMPARISON,
    HIDDEN_DEPTH,
    HELD_OUT_FAMILIES,
    CALIBRATION,
    LEAKAGE,
    PAYMENT_RESULTS,
    ONNX_PARITY,
)
OUT_DIR = "reports/level_18/model_cards"
# Run B beside Run A; only the account card reads it, through build_with_fallback.
FALLBACK = f"{CONSOLE_DIR}/fallback_account.json"

ACCOUNT_MODEL = "mule_account_xgb"
PAYMENT_MODEL = "mule_payment_xgb"
ACCOUNT_SOURCES: tuple[str, ...] = (COMPARISON, HIDDEN_DEPTH, HELD_OUT_FAMILIES, CALIBRATION, LEAKAGE, ONNX_PARITY)
PAYMENT_SOURCES: tuple[str, ...] = (PAYMENT_RESULTS, CALIBRATION, ONNX_PARITY)
PARITY_SPLITS: tuple[str, ...] = ("validation", "test")
PAYMENT_AGGREGATES: tuple[str, ...] = (
    "truth_ap",
    "precision",
    "recall",
    "hold_rate",
    "same_bank_truth_ap",
    "other_bank_truth_ap",
)


def _console(root: Path, rel_path: str) -> dict[str, Any]:
    """Return one console file's data block, refusing a file that is missing or lacks its envelope."""
    return _mapping(_entry(_document(root, rel_path), "data", rel_path), f"{rel_path}:data")


def _spread(block: dict[str, Any], where: str) -> dict[str, Any]:
    """Return one published median/p10/p90 block with its world count and any defined/undefined counts."""
    spread: dict[str, Any] = {
        "median": _number(_entry(block, "median", where), f"{where}.median"),
        "p10": _number(_entry(block, "p10", where), f"{where}.p10"),
        "p90": _number(_entry(block, "p90", where), f"{where}.p90"),
    }
    for key in ("world_count", "worlds_used"):
        if key in block:
            spread["worlds"] = int(_number(block[key], f"{where}.{key}"))
            break
    else:
        raise ValueError(f"missing world count at {where}")
    for key in ("worlds_defined", "worlds_undefined"):
        if key in block:
            spread[key] = int(_number(block[key], f"{where}.{key}"))
    return spread


def _flat(block: dict[str, Any], where: str) -> dict[str, Any]:
    """Copy a flat block of scalars, refusing a nested object or list."""
    result: dict[str, Any] = {}
    for key, value in block.items():
        if isinstance(value, (dict, list)):
            raise ValueError(f"expected a scalar at {where}.{key}, got {type(value).__name__}")
        result[key] = value
    return result


def _depth_metrics(hidden: dict[str, Any]) -> dict[str, Any]:
    """Copy the published hidden-mule-depth blocks, each as a spread plus its account counts."""
    depths = _mapping(_entry(hidden, "hidden_by_depth", HIDDEN_DEPTH), f"{HIDDEN_DEPTH}:hidden_by_depth")
    result: dict[str, Any] = {}
    for key in sorted(depths):
        where = f"{HIDDEN_DEPTH}:hidden_by_depth.{key}"
        block = _mapping(depths[key], where)
        spread = _spread(_mapping(_entry(block, "recall", where), f"{where}.recall"), f"{where}.recall")
        spread["accounts_median"] = _number(_entry(block, "accounts_median", where), f"{where}.accounts_median")
        spread["accounts_total"] = int(_number(_entry(block, "accounts_total", where), f"{where}.accounts_total"))
        result[key] = spread
    return result


def _family_metrics(families_document: dict[str, Any]) -> dict[str, Any]:
    """Copy each held-out typology family's recall at cut and at capacity with its world counts."""
    families = _mapping(_entry(families_document, "families", HELD_OUT_FAMILIES), f"{HELD_OUT_FAMILIES}:families")
    result: dict[str, Any] = {}
    for name in sorted(families):
        where = f"{HELD_OUT_FAMILIES}:families.{name}"
        block = _mapping(families[name], where)
        result[name] = {
            "accounts_total": int(_number(_entry(block, "accounts_total", where), f"{where}.accounts_total")),
            "recall_at_cut": _spread(_mapping(_entry(block, "recall_at_cut", where), where), f"{where}.recall_at_cut"),
            "recall_at_capacity": _spread(
                _mapping(_entry(block, "recall_at_capacity", where), where), f"{where}.recall_at_capacity"
            ),
            "worlds_absent": int(_number(_entry(block, "worlds_absent", where), f"{where}.worlds_absent")),
            "worlds_used": int(_number(_entry(block, "worlds_used", where), f"{where}.worlds_used")),
        }
    return result


def _calibration(calibration_document: dict[str, Any], model: str) -> dict[str, Any]:
    """Copy one model's raw and calibrated Brier/ECE summaries and its calibrator file identity."""
    summaries = _mapping(_entry(calibration_document, "summaries", CALIBRATION), f"{CALIBRATION}:summaries")
    summary = _mapping(_entry(summaries, model, f"{CALIBRATION}:summaries"), f"{CALIBRATION}:summaries.{model}")
    states: dict[str, Any] = {}
    for state in ("raw", "calibrated"):
        block = _mapping(_entry(summary, state, f"summaries.{model}"), f"summaries.{model}.{state}")
        states[state] = {
            "brier": _number(_entry(block, "brier", state), f"summaries.{model}.{state}.brier"),
            "ece": _number(_entry(block, "ece", state), f"summaries.{model}.{state}.ece"),
            "rows": int(_number(_entry(block, "rows", state), f"summaries.{model}.{state}.rows")),
            "positives": int(_number(_entry(block, "positives", state), f"summaries.{model}.{state}.positives")),
        }
    if (states["raw"]["rows"], states["raw"]["positives"]) != (
        states["calibrated"]["rows"],
        states["calibrated"]["positives"],
    ):
        raise ValueError(f"{model}: raw and calibrated summaries describe different rows")
    calibrators = _mapping(_entry(calibration_document, "calibrator", CALIBRATION), f"{CALIBRATION}:calibrator")
    calibrator = _mapping(_entry(calibrators, model, f"{CALIBRATION}:calibrator"), f"{CALIBRATION}:calibrator.{model}")
    calibrator_where = f"{CALIBRATION}:calibrator.{model}"
    return {
        **states,
        "calibrator": {
            "file": _text(_entry(calibrator, "file", calibrator_where), f"{calibrator_where}.file"),
            "sha256": _text(_entry(calibrator, "sha256", calibrator_where), f"{calibrator_where}.sha256"),
            "application": _text(
                _entry(calibrator, "application", calibrator_where), f"{calibrator_where}.application"
            ),
        },
    }


def _onnx(onnx_document: dict[str, Any], model: str) -> dict[str, Any]:
    """Copy one model's ONNX file identity, parity on both splits and the merge proof when present."""
    models = _mapping(_entry(onnx_document, "models", ONNX_PARITY), f"{ONNX_PARITY}:models")
    block = _mapping(_entry(models, model, f"{ONNX_PARITY}:models"), f"{ONNX_PARITY}:models.{model}")
    where = f"{ONNX_PARITY}:models.{model}"
    parity = _mapping(_entry(block, "parity", where), f"{where}.parity")
    parity_out: dict[str, Any] = {}
    for split in PARITY_SPLITS:
        split_block = _mapping(_entry(parity, split, f"{where}.parity"), f"{where}.parity")
        parity_out[split] = {
            "max_abs_diff": _number(_entry(split_block, "max_abs_diff", split), f"{where}.parity.{split}.max_abs_diff"),
            "mean_abs_diff": _number(
                _entry(split_block, "mean_abs_diff", split), f"{where}.parity.{split}.mean_abs_diff"
            ),
            "rows": int(_number(_entry(split_block, "rows", split), f"{where}.parity.{split}.rows")),
            "nan_rows": int(_number(_entry(split_block, "nan_rows", split), f"{where}.parity.{split}.nan_rows")),
            "rows_over_tolerance": int(
                _number(
                    _entry(split_block, "rows_over_tolerance", split),
                    f"{where}.parity.{split}.rows_over_tolerance",
                )
            ),
        }
    result: dict[str, Any] = {
        "file": _text(_entry(block, "file", where), f"{where}.file"),
        "sha256": _text(_entry(block, "sha256", where), f"{where}.sha256"),
        "parity": parity_out,
    }
    if "merge_proof" in block:
        result["merge_proof"] = _flat(
            _mapping(_entry(block, "merge_proof", where), f"{where}.merge_proof"), f"{where}.merge_proof"
        )
    return result


def _account_leakage(leakage_document: dict[str, Any]) -> dict[str, Any]:
    """Copy the seven published leakage outcomes in their committed order plus the trivial-world check."""
    checks = _entry(leakage_document, "checks", LEAKAGE)
    if not isinstance(checks, list):
        raise ValueError(f"expected a list at {LEAKAGE}:checks, got {type(checks).__name__}")
    published = []
    for index, item in enumerate(checks):
        where = f"{LEAKAGE}:checks[{index}]"
        block = _mapping(item, where)
        published.append(
            {
                "check": _text(_entry(block, "check", where), f"{where}.check"),
                "status": _text(_entry(block, "status", where), f"{where}.status"),
            }
        )
    trivial = _mapping(_entry(leakage_document, "trivial_world", LEAKAGE), f"{LEAKAGE}:trivial_world")
    return {
        "checks": published,
        "trivial_world": {"status": _text(_entry(trivial, "status", "trivial_world"), "trivial_world.status")},
    }


def _payment_leakage(payment: dict[str, Any]) -> dict[str, Any]:
    """Copy the five payment leakage item outcomes, the Run A check 6b record and the overall status."""
    leakage = _mapping(_entry(payment, "leakage", PAYMENT_RESULTS), f"{PAYMENT_RESULTS}:leakage")
    items = _mapping(_entry(leakage, "item_status", PAYMENT_RESULTS), f"{PAYMENT_RESULTS}:leakage.item_status")
    item_status = {key: _text(value, f"leakage.item_status.{key}") for key, value in items.items()}
    six_b = _flat(
        _mapping(_entry(leakage, "run_a_check_6b", PAYMENT_RESULTS), f"{PAYMENT_RESULTS}:leakage.run_a_check_6b"),
        "leakage.run_a_check_6b",
    )
    return {
        "item_status": item_status,
        "run_a_check_6b": six_b,
        "status": _text(_entry(leakage, "status", PAYMENT_RESULTS), f"{PAYMENT_RESULTS}:leakage.status"),
    }


def _comparison_xgboost(comparison: dict[str, Any]) -> tuple[dict[str, Any], float]:
    """Return the Run A test xgboost block and the published decision cut."""
    test = _mapping(_entry(comparison, "test", COMPARISON), f"{COMPARISON}:test")
    xgboost = _mapping(_entry(test, "xgboost", COMPARISON), f"{COMPARISON}:test.xgboost")
    cut = _mapping(_entry(comparison, "decision_cut", COMPARISON), f"{COMPARISON}:decision_cut")
    decision_cut = _number(_entry(cut, "value", "decision_cut"), "decision_cut.value")
    return xgboost, decision_cut


def _payment_splits(payment: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Return each payment split's aggregate block and the split block it came from."""
    splits = _mapping(_entry(payment, "splits", PAYMENT_RESULTS), f"{PAYMENT_RESULTS}:splits")
    result: dict[str, dict[str, Any]] = {}
    for split in ("test", "stress"):
        where = f"{PAYMENT_RESULTS}:splits.{split}"
        block = _mapping(_entry(splits, split, f"{PAYMENT_RESULTS}:splits"), where)
        result[split] = {
            "raw": block,
            "aggregate": _mapping(_entry(block, "aggregate", where), f"{where}.aggregate"),
        }
    return result


def _account_limitations(leakage: dict[str, Any]) -> list[str]:
    """State the limitations that apply to the account model, using only published console values."""
    six_b = next(
        (check for check in leakage["checks"] if check["check"] == "check_6b"),
        None,
    )
    if six_b is None:
        raise ValueError("the published leakage checks do not include check_6b")
    if six_b["status"] != "fail":
        raise ValueError(f"expected check_6b to stay failed as published, got {six_b['status']!r}")
    return [
        (
            "Simulator new-account cleanness: the simulator opens accounts itself and records the opening "
            "minute (world/bank_connectors.py:344-369), so a new account here has no prior history; a mule's "
            "fresh account opened with a borrowed identity is not modelled, so new-account detection is "
            "cleaner than reality."
        ),
        (
            "Capacity rounds to one per bank-day: recall at capacity uses "
            "max(1, ceil(active_accounts * capacity_pct / 100)) per bank-day (evaluation/level17_scoring.py:527), "
            "so on a bank-day with few active accounts a single review decision is the whole capacity."
        ),
        (
            "Leakage check 6b is recorded as failed in leakage.json: the account model's validation ranking "
            "exceeds the A-057 average-precision-over-base-rate limit, so these results are not production "
            "evidence."
        ),
    ]


def _payment_limitations(metrics: dict[str, Any], hold_share: float) -> list[str]:
    """State the five payment limitations plainly, embedding the published console numbers."""
    payee = metrics["payee_precision"]
    truth_ap = metrics["truth_ap"]
    feature_ap = metrics["feature_truth_ap_payee_risk"]
    nan_share = metrics["nan_payee_risk_share"]["test"]
    other_bank_count = metrics["nan_reason_counts"]["test"]["receiver_at_other_bank"]
    return [
        (
            f"Per-payee precision is {_fmt(payee['payee_precision'])}, not the reported per-world median "
            f"{_fmt(metrics['precision']['test']['median'])}: {payee['distinct_payees']} distinct payees "
            f"and {payee['held_payments']} held payments, so the per-world number overstates precision at the "
            "payee level."
        ),
        (
            f"Payment ranking collapses under stress: test truth AP median {_fmt(truth_ap['test']['median'])} "
            f"falls to {_fmt(truth_ap['stress']['median'])}, while the payee-risk feature alone keeps "
            f"{_fmt(feature_ap['stress']['median'])} under stress against {_fmt(feature_ap['test']['median'])} "
            f"on test, and the stress hold rate {_fmt(metrics['hold_rate']['stress']['median'])} overshoots the "
            f"{_fmt(hold_share)} target."
        ),
        (
            f"Payees at another bank have no payee score: {_fmt(nan_share)} of test payments carry a NaN payee "
            f"risk, and receiver_at_other_bank accounts for {other_bank_count} of the NaN rows."
        ),
        (
            "Simulator new-account cleanness: the simulator opens accounts itself and records the opening "
            "minute (world/bank_connectors.py:344-369), so a new account here has no prior history; a mule's "
            "fresh account opened with a borrowed identity is not modelled, so new-account detection is "
            "cleaner than reality."
        ),
        (
            "Capacity rounds to one per bank-day: recall at capacity uses "
            "max(1, ceil(active_accounts * capacity_pct / 100)) per bank-day (evaluation/level17_scoring.py:527), "
            "so on a bank-day with few active accounts a single review decision is the whole capacity."
        ),
    ]


def _account_card(documents: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Build the account card's data block from the console documents."""
    xgboost, decision_cut = _comparison_xgboost(documents[COMPARISON])
    metrics = {
        "recall_at_capacity": _spread(
            _mapping(_entry(xgboost, "recall_at_capacity", "test.xgboost"), "test.xgboost"),
            "test.xgboost.recall_at_capacity",
        ),
        "genuine_restriction_rate": _spread(
            _mapping(_entry(xgboost, "genuine_restriction_rate", "test.xgboost"), "test.xgboost"),
            "test.xgboost.genuine_restriction_rate",
        ),
        "median_time_to_detection_hours": _spread(
            _mapping(_entry(xgboost, "median_time_to_detection_hours", "test.xgboost"), "test.xgboost"),
            "test.xgboost.median_time_to_detection_hours",
        ),
        "yield_measured": _spread(
            _mapping(_entry(xgboost, "yield_measured", "test.xgboost"), "test.xgboost"),
            "test.xgboost.yield_measured",
        ),
        "yield_reweighted": _spread(
            _mapping(_entry(xgboost, "yield_reweighted", "test.xgboost"), "test.xgboost"),
            "test.xgboost.yield_reweighted",
        ),
        "recall_by_depth": _depth_metrics(documents[HIDDEN_DEPTH]),
        "held_out_families": _family_metrics(documents[HELD_OUT_FAMILIES]),
    }
    leakage = _account_leakage(documents[LEAKAGE])
    return {
        "model": ACCOUNT_MODEL,
        "intended_use": (
            "Rank account-days for human review within a bank's daily review capacity, using the frozen Run A "
            f"xgboost ensemble with the committed decision cut {decision_cut!r} (comparison.json decision_cut)."
        ),
        "not_intended_for": [
            "Any automatic block, restriction or cash-out without a human reviewer",
            "Production evidence while leakage check 6b is recorded as failed",
            "Accounts or behaviours outside the ashFALL simulator's generation rules",
            "Scoring without the frozen Run A feature order",
        ],
        "training_data": {
            "label": "SIMULATED",
            "description": (
                "Account-day rows generated by the ashFALL interactive simulator with the simulator's hidden "
                "roles as labels, not real complaints. The frozen Run A ensemble is scored without re-training."
            ),
        },
        "metrics": metrics,
        "calibration": _calibration(documents[CALIBRATION], "account"),
        "leakage": leakage,
        "onnx": _onnx(documents[ONNX_PARITY], "account"),
        "known_limitations": _account_limitations(leakage),
    }


def _payment_card(documents: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Build the payment card's data block from the console documents."""
    payment = documents[PAYMENT_RESULTS]
    splits = _payment_splits(payment)
    feature_truth_ap = _mapping(
        _entry(payment, "feature_truth_ap", PAYMENT_RESULTS), f"{PAYMENT_RESULTS}:feature_truth_ap"
    )
    payee_precision = _flat(
        _mapping(_entry(payment, "payee_precision", PAYMENT_RESULTS), f"{PAYMENT_RESULTS}:payee_precision"),
        "payee_precision",
    )
    nan_shares = {}
    nan_counts = {}
    for split in ("test", "stress"):
        block = splits[split]["raw"]
        nan_shares[split] = _number(
            _entry(block, "nan_payee_risk_share", f"splits.{split}"), f"splits.{split}.nan_payee_risk_share"
        )
        reasons = _mapping(_entry(block, "nan_reason_counts", f"splits.{split}"), f"splits.{split}.nan_reason_counts")
        nan_counts[split] = {
            key: int(_number(value, f"splits.{split}.nan_reason_counts.{key}")) for key, value in reasons.items()
        }
    aggregates = {split: splits[split]["aggregate"] for split in splits}
    other_bank = _spread(aggregates["test"]["other_bank_truth_ap"], "splits.test.other_bank_truth_ap")
    other_bank["stress"] = _spread(aggregates["stress"]["other_bank_truth_ap"], "splits.stress.other_bank_truth_ap")
    metrics = {
        "truth_ap": {split: _spread(aggregates[split]["truth_ap"], f"splits.{split}.truth_ap") for split in aggregates},
        "precision": {
            split: _spread(aggregates[split]["precision"], f"splits.{split}.precision") for split in aggregates
        },
        "recall": {split: _spread(aggregates[split]["recall"], f"splits.{split}.recall") for split in aggregates},
        "hold_rate": {
            split: _spread(aggregates[split]["hold_rate"], f"splits.{split}.hold_rate") for split in aggregates
        },
        "same_bank_truth_ap": {
            split: _spread(aggregates[split]["same_bank_truth_ap"], f"splits.{split}.same_bank_truth_ap")
            for split in aggregates
        },
        "other_bank_truth_ap": other_bank,
        "payee_precision": payee_precision,
        "feature_truth_ap_payee_risk": {
            split: _spread(feature_truth_ap[split], f"feature_truth_ap.{split}") for split in ("test", "stress")
        },
        "nan_payee_risk_share": nan_shares,
        "nan_reason_counts": nan_counts,
    }
    frozen = _flat(
        _mapping(_entry(payment, "frozen_model", PAYMENT_RESULTS), f"{PAYMENT_RESULTS}:frozen_model"),
        "frozen_model",
    )
    hold_share = _number(_entry(payment, "hold_share", PAYMENT_RESULTS), f"{PAYMENT_RESULTS}:hold_share")
    previous = _mapping(_entry(payment, "previous_version", PAYMENT_RESULTS), f"{PAYMENT_RESULTS}:previous_version")
    previous_splits = _mapping(_entry(previous, "splits", "previous_version"), "previous_version.splits")
    previous_aggregates = {
        split: _mapping(
            _entry(
                _mapping(_entry(previous_splits, split, "previous_version.splits"), f"previous_version.splits.{split}"),
                "aggregate",
                f"previous_version.splits.{split}",
            ),
            f"previous_version.splits.{split}.aggregate",
        )
        for split in ("test", "stress")
    }
    previous_metrics = {
        split: {
            name: _spread(
                _mapping(
                    _entry(previous_aggregates[split], name, f"previous_version.splits.{split}.aggregate"),
                    f"previous_version.splits.{split}.aggregate.{name}",
                ),
                f"previous_version.splits.{split}.aggregate.{name}",
            )
            for name in PAYMENT_AGGREGATES
        }
        for split in ("test", "stress")
    }
    previous_version = {
        "version": _text(_entry(previous, "version", "previous_version"), "previous_version.version"),
        "git_hash": _text(_entry(previous, "git_hash", "previous_version"), "previous_version.git_hash"),
        "threshold": _number(_entry(previous, "threshold", "previous_version"), "previous_version.threshold"),
        "hold_share": _number(_entry(previous, "hold_share", "previous_version"), "previous_version.hold_share"),
        "frozen_model": _flat(
            _mapping(_entry(previous, "frozen_model", "previous_version"), "previous_version.frozen_model"),
            "previous_version.frozen_model",
        ),
        "metrics": previous_metrics,
        "payee_precision": _flat(
            _mapping(_entry(previous, "payee_precision", "previous_version"), "previous_version.payee_precision"),
            "previous_version.payee_precision",
        ),
    }
    return {
        "model": PAYMENT_MODEL,
        "version": _text(_entry(payment, "version", PAYMENT_RESULTS), f"{PAYMENT_RESULTS}:version"),
        "intended_use": (
            "Rank same-bank incoming payments for human review at the frozen hold share "
            f"{hold_share!r} and the committed threshold, using the frozen xgboost payment model recorded in "
            f"payment_results.json ({frozen['rounds']} boosted rounds, seed {frozen['seed']})."
        ),
        "not_intended_for": [
            "Blocking payments automatically or any decision without a human reviewer",
            "Ranking payments whose receiver is at another bank, which carry no payee-risk score",
            "Production evidence while leakage checks id_leak and 6b are recorded as failed",
            "Payments from banks or behaviours outside the simulator's rules",
        ],
        "training_data": {
            "label": "SIMULATED",
            "description": (
                "Payment rows generated by the ashFALL interactive simulator. The frozen payment model is scored "
                "without re-training; the test check uses the hidden receiver role as truth (A-060a)."
            ),
        },
        "threshold": _number(_entry(payment, "threshold", PAYMENT_RESULTS), f"{PAYMENT_RESULTS}:threshold"),
        "hold_share": hold_share,
        "frozen_model": frozen,
        "previous_version": previous_version,
        "metrics": metrics,
        "calibration": _calibration(documents[CALIBRATION], "payment"),
        "leakage": _payment_leakage(payment),
        "onnx": _onnx(documents[ONNX_PARITY], "payment"),
        "known_limitations": _payment_limitations(metrics, hold_share),
    }


def _fmt(value: Any) -> str:
    """Render one card value: booleans as true/false, integers plain, floats to six decimals, strings as-is."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def _fmt_parity(value: float) -> str:
    """Render one ONNX difference as six decimals plus its exact scientific value, so a tiny nonzero diff is visible."""
    return f"{value:.6f} ({value:.6e})"


def _spread_rows(title: str, spreads: dict[str, Any]) -> list[str]:
    """Render one markdown table of median/p10/p90/world-count rows in the given order."""
    lines = [f"| {title} | median | p10 | p90 | worlds |", "| --- | --- | --- | --- | --- |"]
    for name, spread in spreads.items():
        lines.append(
            f"| {name} | {_fmt(spread['median'])} | {_fmt(spread['p10'])} | {_fmt(spread['p90'])} | "
            f"{spread['worlds']} |"
        )
    return lines


def _manifest_block(git_hash: str, sources: tuple[str, ...], root: Path) -> list[str]:
    """Render the card's manifest footer with the generator, commit and every source digest."""
    lines = ["## Manifest", ""]
    lines.append(
        f"Generated by `{GENERATOR}` from the committed Level 18 console files at git commit `{git_hash}`. "
        "Every value is SIMULATED."
    )
    lines.extend(["", "| source | sha256 |", "| --- | --- |"])
    for rel_path in sorted(sources):
        lines.append(f"| `{rel_path}` | `{_sha256_of_file(root / rel_path)}` |")
    return lines


def _account_markdown(data: dict[str, Any], git_hash: str, root: Path) -> str:
    """Render the account card as markdown from the same data block written to JSON."""
    metrics = data["metrics"]
    calibration = data["calibration"]
    onnx = data["onnx"]
    lines = [
        f"# Model card: {data['model']}",
        "",
        "SIMULATED. Every number below is copied from the committed Level 18 console files and describes "
        "simulated data, not real customers.",
        "",
        "## Intended use",
        "",
        data["intended_use"],
        "",
        "Not intended for:",
        "",
    ]
    lines.extend(f"- {item}" for item in data["not_intended_for"])
    lines.extend(
        [
            "",
            "## Training data",
            "",
            f"Label: **{data['training_data']['label']}**. {data['training_data']['description']}",
            "",
            "## Metrics (test split, across worlds)",
            "",
        ]
    )
    lines.extend(
        _spread_rows(
            "metric",
            {
                "recall at capacity": metrics["recall_at_capacity"],
                "genuine restriction rate": metrics["genuine_restriction_rate"],
                "median time to detection (hours)": metrics["median_time_to_detection_hours"],
                "yield measured": metrics["yield_measured"],
                "yield re-weighted (E1)": metrics["yield_reweighted"],
            },
        )
    )
    lines.extend(
        [
            "",
            "### Recall by hidden mule depth",
            "",
            "| depth | median | p10 | p90 | worlds | accounts (median) | accounts (total) |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for depth, spread in metrics["recall_by_depth"].items():
        lines.append(
            f"| {depth} | {_fmt(spread['median'])} | {_fmt(spread['p10'])} | {_fmt(spread['p90'])} | "
            f"{spread['worlds']} | {_fmt(spread['accounts_median'])} | {spread['accounts_total']} |"
        )
    lines.extend(["", "### Held-out typology families", ""])
    lines.extend(
        [
            "| family | recall at cut (median) | p10 | p90 | recall at capacity (median) | p10 | p90 | "
            "accounts | worlds used | worlds absent |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
    )
    for name, block in metrics["held_out_families"].items():
        cut = block["recall_at_cut"]
        capacity = block["recall_at_capacity"]
        lines.append(
            f"| {name} | {_fmt(cut['median'])} | {_fmt(cut['p10'])} | {_fmt(cut['p90'])} | "
            f"{_fmt(capacity['median'])} | {_fmt(capacity['p10'])} | {_fmt(capacity['p90'])} | "
            f"{block['accounts_total']} | {block['worlds_used']} | {block['worlds_absent']} |"
        )
    lines.extend(
        [
            "",
            "## Calibration (test split)",
            "",
            "| state | Brier | ECE | rows | positives |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for state in ("raw", "calibrated"):
        block = calibration[state]
        lines.append(
            f"| {state} | {_fmt(block['brier'])} | {_fmt(block['ece'])} | {block['rows']} | {block['positives']} |"
        )
    calibrator = calibration["calibrator"]
    calibrator_line = (
        f"Calibrator `{calibrator['file']}` (sha256 `{calibrator['sha256']}`), applied by {calibrator['application']}."
    )
    lines.extend(
        [
            "",
            calibrator_line,
            "",
            "## Leakage checks",
            "",
            "| check | status |",
            "| --- | --- |",
        ]
    )
    lines.extend(f"| {check['check']} | {check['status']} |" for check in data["leakage"]["checks"])
    lines.extend(["", f"Trivial-world check: {data['leakage']['trivial_world']['status']}.", "", "## ONNX parity", ""])
    lines.append(
        f"`{onnx['file']}` (sha256 `{onnx['sha256']}`). Maximum absolute difference against the native model: "
        f"{_fmt_parity(onnx['parity']['validation']['max_abs_diff'])} on validation and "
        f"{_fmt_parity(onnx['parity']['test']['max_abs_diff'])} on test, over "
        f"{onnx['parity']['test']['rows']} rows each; {onnx['parity']['test']['nan_rows']} of those rows carry "
        f"at least one NaN feature and {onnx['parity']['test']['rows_over_tolerance']} rows exceed the tolerance."
    )
    lines.extend(["", "Merge proof:", ""])
    lines.extend(f"- {key}: {_fmt(value)}" for key, value in onnx.get("merge_proof", {}).items())
    lines.extend(["", "## Known limitations", ""])
    lines.extend(f"- {item}" for item in data["known_limitations"])
    lines.extend([""])
    sources = ACCOUNT_SOURCES
    if "fallback_model" in data:
        lines.extend(_fallback_markdown(data["fallback_model"]))
        sources = (*ACCOUNT_SOURCES, FALLBACK)
    lines.extend(_manifest_block(git_hash, sources, root))
    return "\n".join(lines) + "\n"


def _fallback_markdown(fallback: dict[str, Any]) -> list[str]:
    """Render the fallback model section: the serving numbers, Run B beside Run A, and Run B's leakage checks."""
    serving = fallback["serving"]
    validation = serving["validation"]
    lines = [
        "## Fallback model for banks with no complaint history",
        "",
        "Run B, the behaviour-only account model, is not the served model. Run A is served "
        f"({serving['amendment']}): median validation recall at {_fmt(serving['capacity_pct'])}% review capacity "
        f"was {validation['run_a']['median_recall_at_capacity']:.4f} for Run A against "
        f"{validation['run_b']['median_recall_at_capacity']:.4f} for Run B, over "
        f"{validation['run_b']['defined_worlds']} worlds. Run B's decision cut is "
        f"{fallback['decision_cut']['value']!r}. Its success criterion status is "
        f"`{fallback['success_criterion_status']}`.",
        "",
        f"Test split, {fallback['test_world_count']} worlds, median (p10 to p90):",
        "",
        "| metric | Run B (fallback) | Run A (served) |",
        "| --- | --- | --- |",
    ]
    for metric, run_b in fallback["test"]["run_b"].items():
        cells = [
            f"{_fmt(block['median'])} ({_fmt(block['p10'])} to {_fmt(block['p90'])})"
            for block in (run_b, fallback["test"]["run_a"][metric])
        ]
        lines.append(f"| {metric} | {cells[0]} | {cells[1]} |")
    lines.extend(["", "Run B leakage checks:", "", "| check | status | numbers |", "| --- | --- | --- |"])
    for check in fallback["leakage"]:
        shown = check["shown_as"] if "shown_as" in check else check["status"]
        numbers = ", ".join(f"{key} {_fmt(value)}" for key, value in check["numbers"].items())
        lines.append(f"| {check['check']} | {shown} | {numbers} |")
    lines.append("")
    return lines


def _payment_markdown(data: dict[str, Any], git_hash: str, root: Path) -> str:
    """Render the payment card as markdown from the same data block written to JSON."""
    metrics = data["metrics"]
    calibration = data["calibration"]
    onnx = data["onnx"]
    leakage = data["leakage"]
    lines = [
        f"# Model card: {data['model']}",
        "",
        "SIMULATED. Every number below is copied from the committed Level 18 console files and describes "
        "simulated data, not real customers.",
        "",
        "## Intended use",
        "",
        data["intended_use"],
        "",
        "Not intended for:",
        "",
    ]
    lines.extend(f"- {item}" for item in data["not_intended_for"])
    lines.extend(
        [
            "",
            "## Training data",
            "",
            f"Label: **{data['training_data']['label']}**. {data['training_data']['description']}",
            "",
            "## Metrics (across worlds)",
            "",
            "### Test split",
            "",
        ]
    )
    lines.extend(
        _spread_rows(
            "metric",
            {
                "truth AP": metrics["truth_ap"]["test"],
                "precision": metrics["precision"]["test"],
                "recall": metrics["recall"]["test"],
                "hold rate": metrics["hold_rate"]["test"],
                "same-bank truth AP": metrics["same_bank_truth_ap"]["test"],
                "other-bank truth AP": metrics["other_bank_truth_ap"],
            },
        )
    )
    lines.extend(["", "### Stress split", ""])
    lines.extend(
        _spread_rows(
            "metric",
            {
                "truth AP": metrics["truth_ap"]["stress"],
                "precision": metrics["precision"]["stress"],
                "recall": metrics["recall"]["stress"],
                "hold rate": metrics["hold_rate"]["stress"],
                "same-bank truth AP": metrics["same_bank_truth_ap"]["stress"],
                "other-bank truth AP": metrics["other_bank_truth_ap"]["stress"],
            },
        )
    )
    payee = metrics["payee_precision"]
    lines.extend(["", "### Payee precision", "", "| measure | value |", "| --- | --- |"])
    lines.extend(f"| {key} | {_fmt(value)} |" for key, value in payee.items())
    lines.extend(
        [
            "",
            "### Payee-risk feature alone",
            "",
        ]
    )
    lines.extend(
        _spread_rows(
            "split",
            {
                "test": metrics["feature_truth_ap_payee_risk"]["test"],
                "stress": metrics["feature_truth_ap_payee_risk"]["stress"],
            },
        )
    )
    lines.extend(["", "### Payments without a payee score", "", "| split | NaN payee-risk share |", "| --- | --- |"])
    lines.extend(f"| {split} | {_fmt(share)} |" for split, share in metrics["nan_payee_risk_share"].items())
    lines.extend(["", "| split | reason | rows |", "| --- | --- | --- |"])
    for split, reasons in metrics["nan_reason_counts"].items():
        for reason, count in reasons.items():
            lines.append(f"| {split} | {reason} | {count} |")
    previous = data["previous_version"]
    lines.extend(
        [
            "",
            "## Previous version (v1)",
            "",
            f"Version {previous['version']} (git commit `{previous['git_hash']}`): threshold "
            f"{previous['threshold']!r}, hold share {previous['hold_share']!r}, "
            f"{previous['frozen_model']['rounds']} boosted rounds, seed {previous['frozen_model']['seed']}.",
            "",
            "### Test split",
            "",
        ]
    )
    lines.extend(
        _spread_rows(
            "metric",
            {
                "truth AP": previous["metrics"]["test"]["truth_ap"],
                "precision": previous["metrics"]["test"]["precision"],
                "recall": previous["metrics"]["test"]["recall"],
                "hold rate": previous["metrics"]["test"]["hold_rate"],
                "same-bank truth AP": previous["metrics"]["test"]["same_bank_truth_ap"],
                "other-bank truth AP": previous["metrics"]["test"]["other_bank_truth_ap"],
            },
        )
    )
    lines.extend(["", "### Stress split", ""])
    lines.extend(
        _spread_rows(
            "metric",
            {
                "truth AP": previous["metrics"]["stress"]["truth_ap"],
                "precision": previous["metrics"]["stress"]["precision"],
                "recall": previous["metrics"]["stress"]["recall"],
                "hold rate": previous["metrics"]["stress"]["hold_rate"],
                "same-bank truth AP": previous["metrics"]["stress"]["same_bank_truth_ap"],
                "other-bank truth AP": previous["metrics"]["stress"]["other_bank_truth_ap"],
            },
        )
    )
    lines.extend(["", "### Payee precision", "", "| measure | value |", "| --- | --- |"])
    lines.extend(f"| {key} | {_fmt(value)} |" for key, value in previous["payee_precision"].items())
    lines.extend(
        [
            "",
            "## Calibration (test split)",
            "",
            "| state | Brier | ECE | rows | positives |",
            "| --- | --- | --- | --- | --- |",
        ]
    )
    for state in ("raw", "calibrated"):
        block = calibration[state]
        lines.append(
            f"| {state} | {_fmt(block['brier'])} | {_fmt(block['ece'])} | {block['rows']} | {block['positives']} |"
        )
    calibrator = calibration["calibrator"]
    calibrator_line = (
        f"Calibrator `{calibrator['file']}` (sha256 `{calibrator['sha256']}`), applied by {calibrator['application']}."
    )
    lines.extend(
        [
            "",
            calibrator_line,
            "",
            "## Leakage checks",
            "",
            "| item | status |",
            "| --- | --- |",
        ]
    )
    lines.extend(f"| {item} | {status} |" for item, status in leakage["item_status"].items())
    six_b = leakage["run_a_check_6b"]
    lines.extend(
        [
            "",
            "Run A check 6b: status "
            f"{six_b['status']}, average precision over base rate {_fmt(six_b['average_precision_over_base_rate'])} "
            f"against the limit {_fmt(six_b['max_ap_over_base_rate'])} (validation AP "
            f"{_fmt(six_b['validation_average_precision'])} over base rate "
            f"{_fmt(six_b['validation_base_rate'])}).",
            "",
            f"Overall leakage status: {leakage['status']}.",
            "",
            "## ONNX parity",
            "",
            f"`{onnx['file']}` (sha256 `{onnx['sha256']}`). Maximum absolute difference against the native model: "
            f"{_fmt_parity(onnx['parity']['validation']['max_abs_diff'])} on validation and "
            f"{_fmt_parity(onnx['parity']['test']['max_abs_diff'])} on test, over "
            f"{onnx['parity']['test']['rows']} rows each; {onnx['parity']['test']['nan_rows']} of those rows carry "
            f"at least one NaN feature and {onnx['parity']['test']['rows_over_tolerance']} rows exceed the "
            "tolerance.",
            "",
            "## Known limitations",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in data["known_limitations"])
    lines.extend([""])
    lines.extend(_manifest_block(git_hash, PAYMENT_SOURCES, root))
    return "\n".join(lines) + "\n"


def _write_pair(
    root: Path,
    output_dir: Path,
    model: str,
    data: dict[str, Any],
    sources: tuple[str, ...],
    git_hash: str,
    renderer: Any,
) -> list[Path]:
    """Write one card's JSON envelope and its markdown twin, both with LF line endings."""
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
    json_path = output_dir / f"{model}.json"
    json_path.write_bytes(orjson.dumps(document, option=orjson.OPT_SORT_KEYS | orjson.OPT_INDENT_2) + b"\n")
    markdown_path = output_dir / f"{model}.md"
    markdown_path.write_bytes(renderer(data, git_hash, root).encode("utf-8"))
    return [json_path, markdown_path]


def build(root: Path = REPO_ROOT, output_dir: Path | None = None) -> list[Path]:
    """Read the console files under root and write the two cards."""
    documents = {rel_path: _console(root, rel_path) for rel_path in INPUT_PATHS}
    account = _account_card(documents)
    payment = _payment_card(documents)
    git_hash = _git_head()
    target = output_dir if output_dir is not None else root / OUT_DIR
    return [
        *_write_pair(root, target, ACCOUNT_MODEL, account, ACCOUNT_SOURCES, git_hash, _account_markdown),
        *_write_pair(root, target, PAYMENT_MODEL, payment, PAYMENT_SOURCES, git_hash, _payment_markdown),
    ]


def build_with_fallback(root: Path = REPO_ROOT, output_dir: Path | None = None) -> list[Path]:
    """Write the two cards, then rewrite the account card with the fallback model added."""
    paths = build(root, output_dir)
    documents = {rel_path: _console(root, rel_path) for rel_path in INPUT_PATHS}
    account = _account_card(documents)
    account["fallback_model"] = _console(root, FALLBACK)
    target = output_dir if output_dir is not None else root / OUT_DIR
    _write_pair(root, target, ACCOUNT_MODEL, account, (*ACCOUNT_SOURCES, FALLBACK), _git_head(), _account_markdown)
    return paths


def main() -> int:
    """Write the cards and print each path relative to the repository root."""
    for path in sorted(build_with_fallback()):
        print(f"wrote {path.relative_to(REPO_ROOT).as_posix()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
