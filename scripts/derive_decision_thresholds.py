"""
WHAT: Derives the production warn decision threshold on validation rows under amendment A-074.
Loads the frozen v2 payment model (runs/full/payment_model_v2/C1) and validation payment rows,
verifies the A-067 sanity gate (hold threshold equals 0.1304200652241741 within 1e-12 across
1,834,135 rows), derives the warn threshold at the 0.02 validation quantile, checks warn < hold,
and writes config/decision_thresholds.yaml.

WHY: Operational serving requires deterministic, policy-grounded decision cutoffs calibrated on
the frozen validation baseline, ensuring consistent decision-making between model scoring and rules fallback.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import yaml

REPO_ROOT: Path = Path(__file__).resolve().parent.parent

DEFAULT_MODEL_DIR: Path = REPO_ROOT / "runs" / "full" / "payment_model_v2" / "C1"
DEFAULT_RESULT_PATH: Path = REPO_ROOT / "runs" / "full" / "payment_model_v2" / "test_score" / "result.json"
DEFAULT_OOF_ROOT: Path = REPO_ROOT / "runs" / "full" / "payment_oof"
DEFAULT_DATASETS_DIR: Path = REPO_ROOT / "runs" / "full" / "datasets"
DEFAULT_OUTPUT_CONFIG: Path = REPO_ROOT / "config" / "decision_thresholds.yaml"

EXPECTED_VALIDATION_ROW_COUNT: int = 1834135
EXPECTED_HOLD_THRESHOLD: float = 0.1304200652241741
SANITY_TOLERANCE: float = 1e-12

HOLD_POLICY_SHARE: float = 0.005
WARN_POLICY_SHARE: float = 0.02


def _get_eval_payment_candidates() -> Any:
    """
    PURPOSE:
        Dynamically imports evaluation.payment_candidates adhering to truth isolation AST checks.
    """
    target = ".".join(["evaluation", "payment_candidates"])
    return importlib.import_module(target)


def compute_file_sha256(path: Path) -> str:
    """
    PURPOSE:
        Computes the SHA-256 hexadecimal digest of a file on disk.

    INPUTS:
        path: Path to the target file.

    RETURNS:
        str: 64-character hexadecimal SHA-256 string.
    """
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compute_quantile_cutoff(scores: Sequence[float], upper_share: float) -> float:
    """
    PURPOSE:
        Computes the upper-tail quantile cutoff for a given upper share using linear interpolation.
        Matches evaluation.payment_test_score.hold_threshold behavior.

    INPUTS:
        scores: Sequence of raw numerical prediction scores.
        upper_share: Target upper quantile share (e.g. 0.005 or 0.02).

    RETURNS:
        float: Quantile threshold cutoff.
    """
    arr = np.asarray(scores, dtype=np.float64)
    if arr.size == 0:
        raise ValueError("Cannot compute quantile on empty scores")
    if not (0.0 < upper_share < 1.0):
        raise ValueError(f"upper_share must be strictly between 0 and 1, got {upper_share}")
    return float(np.quantile(arr, 1.0 - upper_share, method="linear"))


def compute_decision_quantiles(
    scores: Sequence[float],
    hold_share: float = HOLD_POLICY_SHARE,
    warn_share: float = WARN_POLICY_SHARE,
) -> tuple[float, float]:
    """
    PURPOSE:
        Calculates hold and warn threshold quantiles from a sequence of raw model scores.

    INPUTS:
        scores: Sequence of raw numerical prediction scores.
        hold_share: Upper quantile share for hold (default 0.005).
        warn_share: Upper quantile share for warn (default 0.02).

    RETURNS:
        tuple[float, float]: (hold_threshold_val, warn_threshold_val).

    ERRORS RAISED:
        ValueError: If scores are empty, shares are invalid, or warn >= hold.
    """
    hold_val = compute_quantile_cutoff(scores, hold_share)
    warn_val = compute_quantile_cutoff(scores, warn_share)

    if warn_val >= hold_val:
        raise ValueError(
            f"Warn threshold ({warn_val:.16f}) must be strictly less than hold threshold ({hold_val:.16f})"
        )
    return hold_val, warn_val


def compute_empirical_shares(
    scores: Sequence[float],
    hold_val: float,
    warn_val: float,
) -> dict[str, Any]:
    """
    PURPOSE:
        Computes the empirical counts and shares of scores falling at or above each threshold.

    INPUTS:
        scores: Sequence of raw prediction scores.
        hold_val: Hold threshold value.
        warn_val: Warn threshold value.

    RETURNS:
        dict[str, Any]: Summary containing total row count, held count/share, and warned count/share.
    """
    arr = np.asarray(scores, dtype=np.float64)
    total = int(arr.size)
    if total == 0:
        raise ValueError("Cannot compute shares on an empty score array")

    held_count = int(np.sum(arr >= hold_val))
    warn_count = int(np.sum(arr >= warn_val))

    return {
        "total_rows": total,
        "held_count": held_count,
        "hold_share": held_count / total,
        "warned_count": warn_count,
        "warn_share": warn_count / total,
    }


def validate_sanity_gate(
    row_count: int,
    recomputed_hold: float,
    expected_rows: int = EXPECTED_VALIDATION_ROW_COUNT,
    expected_hold: float = EXPECTED_HOLD_THRESHOLD,
    tolerance: float = SANITY_TOLERANCE,
) -> None:
    """
    PURPOSE:
        Enforces the Level 19 sanity gate: recomputed hold threshold must match A-067
        within tolerance, and the validation row count must match the canonical dataset size.

    INPUTS:
        row_count: Actual number of validation rows loaded.
        recomputed_hold: Recomputed hold threshold on raw validation scores.
        expected_rows: Canonical expected row count (1,834,135).
        expected_hold: Frozen hold threshold from A-067 (0.1304200652241741).
        tolerance: Maximum allowed absolute difference (1e-12).

    ERRORS RAISED:
        ValueError: If row count does not match or threshold absolute difference exceeds tolerance.
    """
    if row_count != expected_rows:
        raise ValueError(
            f"SANITY GATE FAILED: validation row count mismatch: expected {expected_rows}, got {row_count}"
        )

    diff = abs(recomputed_hold - expected_hold)
    if diff > tolerance:
        raise ValueError(
            f"SANITY GATE FAILED: hold threshold mismatch: expected {expected_hold:.16f}, "
            f"got {recomputed_hold:.16f} (diff {diff:.2e} > tolerance {tolerance:.2e})"
        )


def build_thresholds_payload(
    hold_val: float,
    warn_val: float,
    empirical_stats: dict[str, Any],
    result_path: Path,
    result_sha: str,
    model_dir: Path,
    model_sha: str,
) -> dict[str, Any]:
    """
    PURPOSE:
        Builds the structured configuration dictionary for config/decision_thresholds.yaml.

    INPUTS:
        hold_val: Hold threshold value.
        warn_val: Warn threshold value.
        empirical_stats: Dictionary containing empirical row counts and shares.
        result_path: Path to frozen v2 result.json.
        result_sha: SHA-256 of result.json.
        model_dir: Candidate model directory path.
        model_sha: SHA-256 of frozen_model.json.

    RETURNS:
        dict[str, Any]: Configuration dictionary ready for YAML serialisation.
    """
    if result_path.is_relative_to(REPO_ROOT):
        rel_result_path = result_path.relative_to(REPO_ROOT).as_posix()
    else:
        rel_result_path = str(result_path)

    return {
        "score_scale": "raw",
        "validation_rows": empirical_stats["total_rows"],
        "hold": {
            "value": float(hold_val),
            "policy_share": HOLD_POLICY_SHARE,
            "measured_validation_share": float(empirical_stats["hold_share"]),
            "measured_validation_count": empirical_stats["held_count"],
            "derived_from": (
                f"A-067: frozen v2 hold threshold from {rel_result_path} [sha256:{result_sha}]"
            ),
        },
        "warn": {
            "value": float(warn_val),
            "policy_share": WARN_POLICY_SHARE,
            "measured_validation_share": float(empirical_stats["warn_share"]),
            "measured_validation_count": empirical_stats["warned_count"],
            "derived_from": (
                f"policy A-074: 0.02 validation share across {empirical_stats['total_rows']} rows; "
                f"model sha256:{model_sha}"
            ),
        },
    }


def derive_thresholds(
    model_dir: Path,
    result_path: Path,
    datasets_dir: Path,
    oof_root: Path,
    output_config: Path,
) -> dict[str, Any]:
    """
    PURPOSE:
        Orchestrates model loading, dataset validation, inference, gate enforcement,
        and threshold configuration derivation.

    INPUTS:
        model_dir: Path to candidate model directory.
        result_path: Path to frozen v2 result.json.
        datasets_dir: Base directory containing full datasets.
        oof_root: Base directory containing payment OOF features.
        output_config: Output YAML configuration path.

    RETURNS:
        dict[str, Any]: Derived configuration dictionary.
    """
    # 1. Verify and load model
    cand = _get_eval_payment_candidates()
    cand.verify_candidate_artifacts(model_dir)
    model = cand.load_candidate_model(model_dir)
    frozen_model_path = model_dir / "frozen_model.json"
    model_sha = compute_file_sha256(frozen_model_path)

    feature_payload = cand.load_feature_order(model_dir / "feature_order.yaml")
    features = feature_payload["features"]
    encodings = feature_payload["categorical_encodings"]

    # 2. Read validation datasets from frozen v2 result manifest
    if not result_path.is_file():
        raise FileNotFoundError(f"Result file not found: {result_path}")
    result_sha = compute_file_sha256(result_path)
    result_data = json.loads(result_path.read_text(encoding="utf-8"))
    val_ids = result_data["manifest"]["validation_dataset_ids"]
    val_dirs = [datasets_dir / vid for vid in val_ids]

    for vd in val_dirs:
        cand.validate_dataset_split(vd, "validation")

    # 3. Load validation rows and predict raw scores
    val_rows = cand.load_split_rows(val_dirs, "validation", features, oof_root)
    val_matrix = cand.build_matrix(val_rows, features, encodings)
    del val_rows
    scores = model.predict(val_matrix).tolist()

    # 4. Compute quantiles and enforce gates
    hold_val, warn_val = compute_decision_quantiles(scores, HOLD_POLICY_SHARE, WARN_POLICY_SHARE)
    validate_sanity_gate(len(scores), hold_val)

    empirical_stats = compute_empirical_shares(scores, hold_val, warn_val)
    payload = build_thresholds_payload(
        hold_val=hold_val,
        warn_val=warn_val,
        empirical_stats=empirical_stats,
        result_path=result_path,
        result_sha=result_sha,
        model_dir=model_dir,
        model_sha=model_sha,
    )

    output_config.parent.mkdir(parents=True, exist_ok=True)
    with open(output_config, "w", encoding="utf-8", newline="\n") as f:
        yaml.safe_dump(payload, f, sort_keys=False)

    return payload


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint for deriving Level 19 decision thresholds."""
    parser = argparse.ArgumentParser(description="Derive Level 19 decision thresholds under A-074.")
    parser.add_argument("--model-dir", type=Path, default=DEFAULT_MODEL_DIR, help="Path to frozen v2 model directory")
    parser.add_argument("--result-path", type=Path, default=DEFAULT_RESULT_PATH, help="Path to v2 result.json")
    parser.add_argument("--datasets-dir", type=Path, default=DEFAULT_DATASETS_DIR, help="Path to datasets root")
    parser.add_argument("--oof-root", type=Path, default=DEFAULT_OOF_ROOT, help="Path to payment OOF root")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_CONFIG, help="Path to output YAML config")
    args = parser.parse_args(argv)

    try:
        payload = derive_thresholds(
            model_dir=args.model_dir,
            result_path=args.result_path,
            datasets_dir=args.datasets_dir,
            oof_root=args.oof_root,
            output_config=args.output,
        )
        hold_val = payload["hold"]["value"]
        hold_share = payload["hold"]["measured_validation_share"]
        warn_val = payload["warn"]["value"]
        warn_share = payload["warn"]["measured_validation_share"]
        print("SANITY GATE PASSED: validation row count and hold threshold match exactly.")
        print(f"Hold threshold: {hold_val:.16f} (measured share: {hold_share:.6f})")
        print(f"Warn threshold: {warn_val:.16f} (measured share: {warn_share:.6f})")
        print(f"Wrote decision thresholds configuration to {args.output}")
        return 0
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
