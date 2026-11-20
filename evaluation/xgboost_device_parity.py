"""
WHAT: Trains the plain account model on the CPU, then on the GPU twice, and writes whether the GPU
may be used for the full run.

WHY: XGBoost's GPU sums gradients in a different order from the CPU, so the same seed can give a
slightly different model. The GPU is only safe to switch on if the validation truth AP stays within
the configured tolerance of the CPU's and the GPU is repeatable with itself; this script measures
both on this laptop, once, before anyone sets xgboost_device to "cuda".

Run by hand, not as a CI test (CI has no GPU):
    py -3.12 -m evaluation.xgboost_device_parity --training <dir> --validation <dir>
        --truth-root <dir> --out <file.json>

The model is the supervised account model with weighting off, the seed the full driver uses and
the probe's boosting rounds (LEARNING_CURVE_N_ESTIMATORS) with train_account_model's own
hyperparameter defaults.
"""

import argparse
import time
import warnings
from pathlib import Path
from typing import Any, Sequence

import numpy
import orjson
import xgboost

from decisions.anomaly_config import load_anomaly_detector_config
from decisions.mule_account_model import (
    load_mule_model_config,
    predict_account_days,
    train_account_model,
)
from decisions.rules_config import load_rules_config, load_rules_config_with_labels
from evaluation.compact_rows import load_compact_split
from evaluation.hyperparameter_search import validation_truth_average_precision
from evaluation.level17_scoring import ScoredDay
from evaluation.run_level_18_full import LEARNING_CURVE_N_ESTIMATORS
from evaluation.run_level_18_pilot import _compact_row_keys, _validation_scored_in_model_row_order

__all__ = ["judge_parity", "main"]


def judge_parity(
    cpu_probabilities: Sequence[float],
    cuda_probabilities: Sequence[float],
    cuda_second_probabilities: Sequence[float],
    cpu_ap: float,
    cuda_ap: float,
    tolerance: float,
) -> dict[str, Any]:
    """
    WHAT: Compares the three fits and applies the pass rule.

    WHY: The pass rule is written here and into the output file, so the verdict can be re-derived
    from the numbers beside it: |AP_cuda - AP_cpu| <= tolerance AND the two cuda runs are
    bit-identical.

    Args:
        cpu_probabilities: Validation probabilities from the cpu fit.
        cuda_probabilities: Validation probabilities from the first cuda fit.
        cuda_second_probabilities: Validation probabilities from the second cuda fit.
        cpu_ap: Validation truth AP of the cpu fit.
        cuda_ap: Validation truth AP of the first cuda fit.
        tolerance: Largest accepted |AP_cuda - AP_cpu|.

    Returns:
        ap_difference, max_probability_difference_cpu_vs_cuda, cuda_runs_bit_identical, the rule text
        and passed.

    Raises:
        ValueError: If the three probability lists differ in length or the tolerance is not positive.
    """
    if tolerance <= 0.0:
        raise ValueError(f"parity tolerance must be positive; got {tolerance}")
    cpu = numpy.asarray(cpu_probabilities, dtype=numpy.float64)
    cuda = numpy.asarray(cuda_probabilities, dtype=numpy.float64)
    cuda_second = numpy.asarray(cuda_second_probabilities, dtype=numpy.float64)
    if not cpu.shape == cuda.shape == cuda_second.shape:
        raise ValueError(f"probability lists differ in shape: {cpu.shape}, {cuda.shape}, {cuda_second.shape}")
    ap_difference = abs(cuda_ap - cpu_ap)
    bit_identical = bool(numpy.array_equal(cuda, cuda_second))
    return {
        "ap_difference": float(ap_difference),
        "max_probability_difference_cpu_vs_cuda": float(numpy.max(numpy.abs(cpu - cuda))),
        "cuda_runs_bit_identical": bit_identical,
        "pass_rule": (
            f"abs(AP_cuda - AP_cpu) <= parity_ap_tolerance ({tolerance}) "
            "AND the two cuda runs' predictions are bit-identical"
        ),
        "passed": bool(ap_difference <= tolerance and bit_identical),
    }


def _fit_and_score(
    device: str,
    matrix: numpy.ndarray,
    targets: list[int],
    seed: int,
    validation_features: numpy.ndarray,
    truth_scored: Sequence[ScoredDay],
) -> dict[str, Any]:
    """
    WHAT: Fits the plain model on one device and returns its probabilities, truth AP, fit seconds
    and any warnings XGBoost raised.

    WHY: Warnings (for example about data living on another device) are part of the evidence for or
    against using the GPU, so they are recorded rather than hidden.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        started = time.perf_counter()
        model, _ = train_account_model(
            matrix, targets, seed, LEARNING_CURVE_N_ESTIMATORS, sample_weight=None, device=device
        )
        fit_seconds = time.perf_counter() - started
        probabilities = predict_account_days(model, validation_features)
    return {
        "device": device,
        "probabilities": probabilities,
        "validation_truth_average_precision": float(validation_truth_average_precision(probabilities, truth_scored)),
        "fit_seconds": fit_seconds,
        "warnings": sorted({str(item.message) for item in caught}),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """
    WHAT: Loads the splits, runs the cpu, cuda and cuda-again fits and writes the parity JSON.

    Raises:
        ValueError: If xgboost has no CUDA build or a split cannot be loaded.
    """
    parser = argparse.ArgumentParser(description="Check that GPU training matches CPU training")
    parser.add_argument("--training", type=Path, action="append", required=True)
    parser.add_argument("--validation", type=Path, action="append", required=True)
    parser.add_argument("--truth-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)

    mule_config = load_mule_model_config()
    base_rules_config = (
        load_rules_config_with_labels() if mule_config.rules_plus_labels_and_registry is True else load_rules_config()
    )
    seed = load_anomaly_detector_config().random_state

    training = load_compact_split(args.training)
    validation = load_compact_split(args.validation, args.truth_root, include_days=True, rules_config=base_rules_config)
    if validation.days is None:
        raise ValueError("compact validation load did not return account-day arrays")
    truth_scored = _validation_scored_in_model_row_order(
        "compact",
        validation.days,
        validation.days.aligned_features,
        _compact_row_keys(validation.rows),
        base_rules_config,
        None,
    )
    targets = [int(target) for target in training.rows.targets]
    features = validation.days.aligned_features

    runs = [
        _fit_and_score(device, training.rows.matrix, targets, seed, features, truth_scored)
        for device in ("cpu", "cuda", "cuda")
    ]
    cpu_run, cuda_run, cuda_second_run = runs
    verdict = judge_parity(
        cpu_run["probabilities"],
        cuda_run["probabilities"],
        cuda_second_run["probabilities"],
        cpu_run["validation_truth_average_precision"],
        cuda_run["validation_truth_average_precision"],
        mule_config.parity_ap_tolerance,
    )
    result = {
        "xgboost_version": xgboost.__version__,
        "seed": seed,
        "n_estimators": LEARNING_CURVE_N_ESTIMATORS,
        "training_row_count": len(targets),
        "validation_row_count": len(truth_scored),
        "parity_ap_tolerance": mule_config.parity_ap_tolerance,
        "runs": [{key: value for key, value in run.items() if key != "probabilities"} for run in runs],
        **verdict,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(orjson.dumps(result, option=orjson.OPT_INDENT_2) + b"\n")
    print(f"parity {'PASSED' if verdict['passed'] else 'FAILED'}: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
