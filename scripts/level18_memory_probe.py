# File: scripts/level18_memory_probe.py
# Purpose: Measures the compact data path's resident memory and stage timings on a bounded number
#          of full-scale worlds, without scoring any test world and without writing result files.
"""
WHAT:
    Runs the compact data path on the first N worlds of each split, in sorted world order: load the
    training worlds into float32 arrays, load the validation worlds with their truth columns and
    account-day rows, train one supervised account model with 100 boosting rounds, tune the rules
    grid on the validation account-days and load the test worlds, sampling the process's resident
    memory at the end of every stage.

WHY:
    The full-scale pilot must fit a 16 GB laptop. The pilot itself prints nothing about memory and
    takes hours, so a small probe is the only cheap way to answer "which stage peaks, and at how
    many gigabytes" before committing to a full run.

TERM: probe: a measurement run. Its numbers are memory and time, never model quality.

The probe never scores a test world and never writes under results/. Every dataset folder, truth
folder and config file it reads is required; a missing or malformed input raises a plain ValueError.
"""

import argparse
import time
from pathlib import Path
from typing import Any, Sequence

import orjson
import psutil

from decisions.anomaly_config import load_anomaly_detector_config
from decisions.mule_account_model import load_mule_model_config, train_account_model
from decisions.rules_config import load_rules_config, load_rules_config_with_labels
from evaluation.compact_rows import _world_sources, load_compact_split
from evaluation.rules_tuning import (
    build_rules_day_matrix,
    load_labels_tuning_config,
    load_tuning_config,
    tune_rules_matrix,
)
from evaluation.run_level_17 import load_evaluation_settings

__all__ = ["run_probe"]

#: Boosting rounds for the probe's single training stage, as the round brief fixes them.
PROBE_N_ESTIMATORS = 100


def _record_stage(
    name: str,
    started: float,
    stage_seconds: dict[str, float],
    stage_peak_rss_mb: dict[str, float],
) -> None:
    """
    WHAT: Records one stage's elapsed seconds and the process's resident memory at the stage's end.

    WHY: One convention for every stage keeps the probe's numbers comparable with the pilot
    manifest's stage profile.

    Args:
        name: Stage label, such as "load_training" or "tune".
        started: time.perf_counter() value taken when the stage began.
        stage_seconds: Destination mapping stage label -> seconds.
        stage_peak_rss_mb: Destination mapping stage label -> resident memory in MiB.
    """
    stage_seconds[name] = time.perf_counter() - started
    stage_peak_rss_mb[name] = psutil.Process().memory_info().rss / (1024.0 * 1024.0)


def _first_worlds(dataset_dirs: Sequence[Path], count: int) -> list[str]:
    """
    WHAT: Returns the first count world ids of the given dataset folders in sorted order.

    WHY: A bounded probe must read the same prefix of worlds the brief names, not whichever world
    happens to sit first in a parquet file.

    Args:
        dataset_dirs: Dataset folders of one split.
        count: Maximum number of world ids to return.

    Returns:
        The first count ids of _world_sources, which is already sorted by world_id.

    Raises:
        ValueError: If count is not positive or a dataset folder has no usable manifest.
    """
    if count <= 0:
        raise ValueError(f"worlds per split must be positive; got {count}")
    world_ids = [world_id for world_id, _ in _world_sources(dataset_dirs)]
    return world_ids[:count]


def run_probe(
    training_dirs: Sequence[Path],
    validation_dirs: Sequence[Path],
    test_dirs: Sequence[Path],
    truth_root: Path,
    worlds_per_split: int,
    output_path: Path,
) -> dict[str, Any]:
    """
    WHAT: Runs the compact data path on the first worlds_per_split worlds of each split and writes
    the stage seconds and resident memory to output_path as JSON.

    WHY: See the module docstring: this is the cheap measurement that decides whether the full-scale
    pilot can run on a 16 GB laptop.

    Args:
        training_dirs: Full-scale training dataset folders.
        validation_dirs: Full-scale validation dataset folders.
        test_dirs: Full-scale test dataset folders.
        truth_root: Root folder holding one truth folder per run_id.
        worlds_per_split: Number of leading worlds to use from each split.
        output_path: JSON destination; nothing is written under results/.

    Returns:
        The payload that was written.

    Key logic:
        Stage order matches the pilot: load_training, load_validation, train, tune, load_test. The
        rules table and tuning grid follow the committed mule model config's rules switch, and the
        budget is the committed evaluation setting, so the tuning stage measures the real grid.

    Raises:
        ValueError: If a config, dataset or truth input is missing or malformed.
    """
    mule_model_config = load_mule_model_config()
    if mule_model_config.rules_plus_labels_and_registry is True:
        rules_config = load_rules_config_with_labels()
        tuning = load_labels_tuning_config()
    else:
        rules_config = load_rules_config()
        tuning = load_tuning_config()
    budget_pct = load_evaluation_settings().genuine_restriction_budget_pct
    seed = load_anomaly_detector_config().random_state

    stage_seconds: dict[str, float] = {}
    stage_peak_rss_mb: dict[str, float] = {}
    probe_started = time.perf_counter()

    started = time.perf_counter()
    training_worlds = _first_worlds(training_dirs, worlds_per_split)
    training_split = load_compact_split(training_dirs, world_ids=training_worlds)
    _record_stage("load_training", started, stage_seconds, stage_peak_rss_mb)

    started = time.perf_counter()
    validation_worlds = _first_worlds(validation_dirs, worlds_per_split)
    validation_split = load_compact_split(validation_dirs, truth_root, include_days=True, world_ids=validation_worlds)
    if validation_split.day_rows is None:
        raise ValueError("compact validation load did not return account-day rows")
    _record_stage("load_validation", started, stage_seconds, stage_peak_rss_mb)

    started = time.perf_counter()
    train_account_model(
        training_split.rows.matrix,
        [int(target) for target in training_split.rows.targets],
        seed,
        n_estimators=PROBE_N_ESTIMATORS,
    )
    _record_stage("train", started, stage_seconds, stage_peak_rss_mb)

    started = time.perf_counter()
    rules_matrix = build_rules_day_matrix(validation_split.day_rows, rules_config)
    rules_result = tune_rules_matrix(
        rules_matrix,
        rules_config,
        tuning,
        "validation",
        budget_pct,
        collect_grid_points=False,
    )
    _record_stage("tune", started, stage_seconds, stage_peak_rss_mb)

    started = time.perf_counter()
    test_worlds = _first_worlds(test_dirs, worlds_per_split)
    load_compact_split(test_dirs, truth_root, include_days=True, world_ids=test_worlds)
    _record_stage("load_test", started, stage_seconds, stage_peak_rss_mb)

    payload: dict[str, Any] = {
        "simulated": True,
        "worlds_per_split": worlds_per_split,
        "world_counts": {
            "training": len(training_worlds),
            "validation": len(validation_worlds),
            "test": len(test_worlds),
        },
        "training_rows": int(len(training_split.rows.targets)),
        "validation_days": int(len(validation_split.day_rows)),
        "rules_combinations_evaluated": rules_result.combinations_evaluated,
        "stage_seconds": stage_seconds,
        "stage_peak_rss_mb": stage_peak_rss_mb,
        "peak_rss_mb": max(stage_peak_rss_mb.values()),
        "total_seconds": time.perf_counter() - probe_started,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_text = orjson.dumps(payload, option=orjson.OPT_SORT_KEYS | orjson.OPT_INDENT_2).decode("utf-8")
    output_path.write_text(output_text + "\n", encoding="utf-8")
    print(output_text)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    """
    WHAT: Command-line entry point for the compact data path memory probe.

    Args:
        argv: Optional argument list; defaults to sys.argv.

    Returns:
        Process exit code 0.
    """
    parser = argparse.ArgumentParser(description="Measure the compact data path's stage seconds and resident memory")
    parser.add_argument(
        "--training", type=Path, action="append", required=True, help="training dataset folder; repeatable"
    )
    parser.add_argument(
        "--validation", type=Path, action="append", required=True, help="validation dataset folder; repeatable"
    )
    parser.add_argument("--test", type=Path, action="append", required=True, help="test dataset folder; repeatable")
    parser.add_argument(
        "--truth-root", type=Path, required=True, help="root folder holding one truth folder per run_id"
    )
    parser.add_argument(
        "--worlds-per-split",
        type=int,
        required=True,
        help="number of leading worlds, in sorted order, to use from each split",
    )
    parser.add_argument("--output", type=Path, required=True, help="JSON destination for the probe result")
    args = parser.parse_args(argv)
    run_probe(
        tuple(args.training),
        tuple(args.validation),
        tuple(args.test),
        args.truth_root,
        args.worlds_per_split,
        args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
