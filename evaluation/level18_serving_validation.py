"""
WHAT: Scores Run A's frozen account model and Run B's final model on the same validation worlds and
applies issue #154's serving rule to their recall at capacity.

WHY: No stored Run A number is review-capacity recall on validation, so both models are scored here
by identical code (amendment A-070a, ruling 2). It reads no test or stress dataset.

    py -3.12 -m evaluation.level18_serving_validation --run-a <stage5 out> --run-b <stage5 out>
        --validation <dir> --validation <dir> --truth-root <dir> --out <file.json>

Each --run-* folder holds final/<key>/checkpoint/ and stages/final_mule_model.yaml.
"""

import argparse
import statistics
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import orjson
import xgboost

from decisions.mule_account_model import load_model, load_mule_model_config, set_mule_model_config_path
from decisions.pu_learning import PUBaggedAccountModel
from evaluation.dataset_truth_join import join_truth
from evaluation.level17_scoring import ScoredDay, daily_escalations, ground_sets
from evaluation.level18_extra_reporting import (
    _checkpoint_directory,
    _frozen_rules_config,
    _read_fit_record,
    _read_thresholds,
    _require_dataset_ids,
    _sha256_file,
    _world_evidence,
    require_mule_model_config_sha,
)
from evaluation.level18_sanity import load_escalation_repeat_policy
from evaluation.rules_metrics import recall_at_capacity
from evaluation.run_level_17 import load_split_datasets
from showcase.framework import get_code_version

__all__ = ["main", "median_defined_recall", "serving_decision", "world_recall"]

#: Review capacity, percent of a bank's active accounts per day, that A-070 ruling 7 and A-070a
#: ruling 3 fix for this one comparison. It is a pre-registered ruling, not a config value.
CAPACITY_PCT = 0.5


def world_recall(scored: list[ScoredDay], xgboost_cut: float, repeat_policy: str) -> float | None:
    """
    WHAT: One world's recall at capacity, or None when the world has no mule.

    Args:
        scored: The world's account-days with the model probability in anomaly_score.
        xgboost_cut: The model's own frozen inclusive cut.
        repeat_policy: Escalation repeat policy.
    """
    mules = ground_sets(scored)[0]
    if not mules:
        return None
    escalations = daily_escalations(scored, "anomaly", xgboost_cut, CAPACITY_PCT, repeat_policy)
    return recall_at_capacity(escalations, mules)


def median_defined_recall(recalls: Mapping[str, float | None]) -> float:
    """
    WHAT: Median recall over the worlds where it is defined.

    Raises:
        ValueError: If no world has a defined recall.
    """
    defined = [value for value in recalls.values() if value is not None]
    if not defined:
        raise ValueError(f"No world has a defined recall at capacity: {sorted(recalls)}")
    return float(statistics.median(defined))


def serving_decision(run_a_recall: float, run_b_recall: float) -> str:
    """
    WHAT: Issue #154's rule: "run_a" only when it beats Run B; otherwise, ties included, "run_b".
    """
    return "run_a" if run_a_recall > run_b_recall else "run_b"


def _score_run(run_out: Path, validation_dirs: Sequence[Path], truth_root: Path) -> dict[str, Any]:
    """
    WHAT: Loads one run's checkpoint and scores every validation world with it.

    Raises:
        ValueError: If the checkpoint, config or datasets do not match, or a world repeats.
    """
    checkpoint = _checkpoint_directory(run_out)
    record = _read_fit_record(checkpoint / "fit_record.json")
    fingerprint = record["fingerprint"]
    config_path = run_out / "stages" / "final_mule_model.yaml"
    config_sha256 = require_mule_model_config_sha(config_path, fingerprint["mule_model_config_sha256"])
    _require_dataset_ids(fingerprint["validation_dataset_ids"], validation_dirs, "validation")
    rules_thresholds, xgboost_cut = _read_thresholds(checkpoint / "thresholds.json")
    repeat_policy = load_escalation_repeat_policy()
    set_mule_model_config_path(config_path)
    try:
        model: xgboost.XGBClassifier | PUBaggedAccountModel
        # The same choice the pilot's checkpoint loader makes between the two saved formats.
        if record["pu_method"] == "none" or (checkpoint / "model.ubj").is_file():
            model = load_model(checkpoint / "model.ubj")
        else:
            model = PUBaggedAccountModel.load(checkpoint / "model_bags")
        rules_config = _frozen_rules_config(load_mule_model_config(), rules_thresholds)
        recalls: dict[str, float | None] = {}
        for directory, dataset in zip(validation_dirs, load_split_datasets("validation", validation_dirs), strict=True):
            _, snapshots = join_truth(directory, truth_root)
            for world in dataset.manifest["worlds"]:
                world_id = str(world["world_id"])
                if world_id in recalls:
                    raise ValueError(f"World {world_id!r} repeats across the validation datasets")
                evidence = _world_evidence(
                    world_id, str(world["run_id"]), snapshots, dataset.labels, model, rules_config, xgboost_cut
                )
                recalls[world_id] = world_recall(evidence.scored, xgboost_cut, repeat_policy)
    finally:
        set_mule_model_config_path(None)
    return {
        "run_out": run_out.as_posix(),
        "checkpoint_code_version": fingerprint["code_version"],
        "pu_method": record["pu_method"],
        "mule_model_config_sha256": config_sha256,
        "xgboost_cut": xgboost_cut,
        "repeat_policy": repeat_policy,
        "checkpoint_sha256": {
            path.relative_to(checkpoint).as_posix(): _sha256_file(path)
            for path in sorted(checkpoint.rglob("*"))
            if path.is_file()
        },
        "per_world_recall_at_capacity": recalls,
        "defined_worlds": sum(value is not None for value in recalls.values()),
        "median_recall_at_capacity": median_defined_recall(recalls),
    }


def main(argv: Sequence[str] | None = None) -> int:
    """
    WHAT: Scores both runs, writes the comparison as JSON and prints the two medians and decision.

    Raises:
        ValueError: If --out lies inside a Run A folder, or any scoring check fails.
    """
    parser = argparse.ArgumentParser(description="Validation recall at capacity for Run A and Run B")
    parser.add_argument("--run-a", type=Path, required=True)
    parser.add_argument("--run-b", type=Path, required=True)
    parser.add_argument("--validation", type=Path, action="append", required=True)
    parser.add_argument("--truth-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if any(part.startswith("run_a320") for part in args.out.resolve().parts):
        raise ValueError(f"--out must not be inside a Run A folder: {args.out}")

    run_a = _score_run(args.run_a, args.validation, args.truth_root)
    run_b = _score_run(args.run_b, args.validation, args.truth_root)
    decision = serving_decision(run_a["median_recall_at_capacity"], run_b["median_recall_at_capacity"])
    report = {
        "simulated": True,
        "code_version": get_code_version(),
        "capacity_pct": CAPACITY_PCT,
        "validation_dataset_ids": [directory.name for directory in args.validation],
        "run_a": run_a,
        "run_b": run_b,
        "serve": decision,
    }
    args.out.write_bytes(orjson.dumps(report, option=orjson.OPT_SORT_KEYS | orjson.OPT_INDENT_2) + b"\n")
    for name, block in (("run_a", run_a), ("run_b", run_b)):
        print(f"{name} median_recall_at_capacity={block['median_recall_at_capacity']!r}")
    print(f"serve={decision}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
