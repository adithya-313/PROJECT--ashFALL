# File: evaluation/payment_test_score.py
# Purpose: Score the frozen Level 18 payment model exactly once on the test and stress datasets.
"""
WHAT:
    Scores every test and stress payment with the frozen payment model and reports, per world and
    across worlds, the payment truth average precision, recall and precision at a hold threshold
    chosen on validation, the hold rate, and the same-bank and other-bank average precision.

WHY:
    Task 4 ends by measuring the frozen model on hidden test and stress truth. A-058 fixes the model
    artifacts, the `payee_risk` rule, the hold threshold and the metric definitions before the first
    test or stress score exists, so the reported numbers are a held-out measurement rather than a
    choice made after seeing them.

TERM: hold threshold: the validation score such that HOLD_SHARE of validation payments score at or
      above it, computed from validation scores only and applied unchanged to test and stress.
TERM: same-bank rows: payments whose `payee_risk` is finite; other-bank rows: payments where the
      receiver could not be scored under the same-bank rule and `payee_risk` is NaN.

The entry point accepts only test and stress manifests for the scored folders and only validation
manifests for the threshold; any other split raises.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import xgboost

from decisions.mule_account_model import mule_model_config_override, set_mule_model_config_path
from decisions.pu_learning import PUBaggedAccountModel
from evaluation.dataset_truth_join import join_truth
from evaluation.level17_scoring import MULE_ROLES
from evaluation.payment_model import (
    KEY_COLUMNS,
    PAYEE_RISK,
    build_matrix,
    load_feature_order,
    load_split_rows,
    predict_scores,
)
from evaluation.payment_oof_risk import (
    _assert_score_or_reason,
    _git_head,
    _read_world_snapshots,
    _require_manifest,
    build_world_payee_risk,
    load_frozen_recipe,
    score_world_account_days,
    validate_dataset_split,
)
from evaluation.trivial_world import average_precision

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = REPO_ROOT / "runs" / "full" / "payment_model_v1"
DEFAULT_OUT_ROOT = DEFAULT_MODEL_DIR / "test_score"
DEFAULT_OOF_ROOT = REPO_ROOT / "runs" / "full" / "payment_oof"
DEFAULT_TRUTH_ROOT = REPO_ROOT / "runs" / "full" / "truth"

#: A-058 fixes the validation hold share at one half of one percent.
HOLD_SHARE = 0.005

#: A-058 ruling 1 records the frozen artifact hashes; any mismatch is a stop, not a fix.
FROZEN_ARTIFACT_SHA256 = {
    "frozen_model.json": "98dffe1fda74342b36d55651deb2168fa144c4886a87cdb10715ff90d3a7864e",
    "feature_order.yaml": "6da1c373ac3d59d22f11808f033ae3462e9d1222d6c2e6a82d767b931f54211a",
}

SCORING_SPLITS = ("test", "stress")
METRIC_KEYS = (
    "truth_ap",
    "recall",
    "precision",
    "hold_rate",
    "same_bank_truth_ap",
    "other_bank_truth_ap",
)


@dataclass(frozen=True)
class WorldScores:
    """One world's frozen-model scores, hidden-truth labels and payee-risk values, row aligned."""

    split: str
    dataset: str
    world_id: str
    scores: Sequence[float]
    labels: Sequence[int]
    payee_risk: Sequence[float]


def validate_scoring_split(dataset_dir: Path, expected_split: str) -> dict[str, Any]:
    """
    WHAT: Returns a scored folder's manifest after proving its split is test or stress and equals
    the split the caller passes.

    WHY: The final scoring must read test and stress and nothing else; this is the inverse of the
    Phase 0-3 guard, so a training or validation folder fails loudly instead of being scored.

    Raises:
        ValueError: If expected_split is not test/stress, the manifest split is not test/stress, or
            the split differs from expected_split.
    """
    if expected_split not in SCORING_SPLITS:
        raise ValueError(f"expected_split must be 'test' or 'stress'; got {expected_split!r}")
    manifest = _require_manifest(Path(dataset_dir))
    split = manifest.get("split")
    if split not in SCORING_SPLITS:
        raise ValueError(
            f"{Path(dataset_dir).name} has manifest split {split!r}; final scoring accepts only "
            "test and stress datasets"
        )
    if split != expected_split:
        raise ValueError(f"{Path(dataset_dir).name} has manifest split {split!r} but was passed as {expected_split!r}")
    return manifest


def hold_threshold(scores: Sequence[float], hold_share: float = HOLD_SHARE) -> float:
    """
    WHAT: Returns the score at the (1 - hold_share) linear quantile of the given scores.

    WHY: A-058 fixes the validation hold share before any test score exists; taking only the score
    sequence keeps the caller from mixing validation and test rows into one threshold.

    Raises:
        ValueError: If the scores are empty or the hold share is not strictly inside (0, 1).
    """
    values = np.asarray(scores, dtype=np.float64)
    if values.size == 0:
        raise ValueError("hold threshold needs at least one validation score")
    if not 0.0 < hold_share < 1.0:
        raise ValueError(f"hold_share must lie strictly inside (0, 1); got {hold_share!r}")
    return float(np.quantile(values, 1.0 - hold_share, method="linear"))


def verify_frozen_artifacts(model_dir: Path) -> tuple[str, str]:
    """
    WHAT: Returns the sha256 of the frozen model and the feature order after proving both equal the
    hashes A-058 records.

    WHY: A-058 fixes the scored artifacts by hash; a changed file must stop the run instead of being
    scored as if it were the frozen model.

    Raises:
        ValueError: If either file's sha256 differs, naming every file that differs.
    """
    digests: dict[str, str] = {}
    mismatches: list[str] = []
    for name, expected in FROZEN_ARTIFACT_SHA256.items():
        digest = hashlib.sha256((Path(model_dir) / name).read_bytes()).hexdigest()
        digests[name] = digest
        if digest != expected:
            mismatches.append(f"{name} sha256 {digest} != A-058 {expected}")
    if mismatches:
        raise ValueError("frozen artifacts do not match A-058: " + "; ".join(mismatches))
    return digests["frozen_model.json"], digests["feature_order.yaml"]


def verify_booster_rounds(booster: xgboost.Booster, metadata: Mapping[str, Any]) -> int:
    """
    WHAT: Returns the booster's round count after proving it equals the metadata's chosen
    `n_estimators`.

    WHY: A-058 ruling 1 fixes the frozen model's round count; a booster with another count is a
    different model even when its feature count matches.

    Raises:
        ValueError: If the booster's round count differs from `best_params.n_estimators`.
    """
    rounds = booster.num_boosted_rounds()
    expected = metadata["best_params"]["n_estimators"]
    if rounds != expected:
        raise ValueError(f"frozen model has {rounds} boosted rounds but the metadata records {expected}")
    return rounds


def _average_precision_or_none(scores: Sequence[float], labels: Sequence[int]) -> float | None:
    """
    WHAT: Returns the discrete average precision, or None when the subset lacks a positive or a
    negative.

    WHY: A-058 pre-registers that a world or subset without both classes has no defined AP; a null
    keeps it out of the aggregates instead of substituting a value.
    """
    positives = int(sum(1 for label in labels if label))
    if positives == 0 or len(labels) - positives == 0:
        return None
    return average_precision(list(scores), list(labels))


def world_metrics(world: WorldScores, threshold: float) -> dict[str, Any]:
    """
    WHAT: Computes one world's truth AP, recall, precision, hold rate and subset APs at `threshold`.

    WHY: A-058 names these metrics per world before any score exists; keeping them in one function
    makes the same-bank split and the held mask use the one shared threshold.

    Raises:
        ValueError: If the world holds no rows or its three arrays differ in length.
    """
    scores = np.asarray(world.scores, dtype=np.float64)
    labels = np.asarray(world.labels, dtype=np.int8)
    risk = np.asarray(world.payee_risk, dtype=np.float64)
    if labels.size == 0:
        raise ValueError(f"world {world.world_id!r} holds no payment rows")
    if scores.size != labels.size or risk.size != labels.size:
        raise ValueError(f"world {world.world_id!r} holds arrays of different lengths")

    rows = int(labels.size)
    positives = int(labels.sum())
    held_mask = scores >= threshold
    held = int(held_mask.sum())
    true_positives = int(labels[held_mask].sum())
    same_bank = np.isfinite(risk)
    other_bank = ~same_bank

    same_bank_ap = _average_precision_or_none(scores[same_bank], labels[same_bank])
    other_bank_ap = _average_precision_or_none(scores[other_bank], labels[other_bank])
    return {
        "split": world.split,
        "dataset": world.dataset,
        "world_id": world.world_id,
        "rows": rows,
        "positives": positives,
        "held": held,
        "true_positives": true_positives,
        "truth_ap": _average_precision_or_none(scores, labels),
        "recall": true_positives / positives if positives else None,
        "precision": true_positives / held if held else None,
        "hold_rate": held / rows,
        "same_bank_truth_ap": same_bank_ap,
        "other_bank_truth_ap": other_bank_ap,
        "same_bank": {
            "rows": int(same_bank.sum()),
            "positives": int(labels[same_bank].sum()),
            "truth_ap": same_bank_ap,
        },
        "other_bank": {
            "rows": int(other_bank.sum()),
            "positives": int(labels[other_bank].sum()),
            "truth_ap": other_bank_ap,
        },
        "nan_payee_risk_share": float(other_bank.mean()),
    }


def aggregate(values: Sequence[float | None]) -> dict[str, Any]:
    """
    WHAT: Returns the median, p10 and p90 of the non-null values and how many values were used.

    WHY: A-058 fixes these cross-world aggregates; leaving null worlds out keeps a world without a
    defined metric from silently entering as zero.
    """
    kept = np.asarray([value for value in values if value is not None], dtype=np.float64)
    if kept.size == 0:
        return {"median": None, "p10": None, "p90": None, "worlds_used": 0}
    return {
        "median": float(np.median(kept)),
        "p10": float(np.percentile(kept, 10, method="linear")),
        "p90": float(np.percentile(kept, 90, method="linear")),
        "worlds_used": int(kept.size),
    }


def evaluate_splits(
    validation_scores: Sequence[float],
    splits: Mapping[str, Sequence[WorldScores]],
    hold_share: float = HOLD_SHARE,
) -> dict[str, Any]:
    """
    WHAT: Builds the whole result: the validation threshold, the validation hold rate, and each
    split's per-world metrics and cross-world aggregates.

    WHY: The threshold is computed here, from the validation scores alone, before any split metric
    is read; the split sequences are passed whole so the rule is testable, but nothing from them
    enters the threshold calculation (A-058).

    Raises:
        ValueError: If the validation scores are empty or a split holds no worlds.
    """
    threshold = hold_threshold(validation_scores, hold_share)
    validation = np.asarray(validation_scores, dtype=np.float64)
    held = int((validation >= threshold).sum())
    result: dict[str, Any] = {
        "hold_share": hold_share,
        "threshold": threshold,
        "validation": {
            "rows": int(validation.size),
            "held": held,
            "hold_rate": held / int(validation.size),
        },
        "splits": {},
    }
    for split, worlds in splits.items():
        if not worlds:
            raise ValueError(f"split {split!r} holds no scored worlds")
        rows = [world_metrics(world, threshold) for world in worlds]
        result["splits"][split] = {
            "worlds": rows,
            "aggregate": {name: aggregate([row[name] for row in rows]) for name in METRIC_KEYS},
        }
    return result


def attach_payee_risk(table: pa.Table, risk_rows: Sequence[Mapping[str, Any]]) -> pa.Table:
    """
    WHAT: Appends the computed `payee_risk` and `nan_reason` columns to a world's payment table.

    WHY: The rows are produced from the same sorted payment table, so the key check proves the two
    never drift; a misaligned attach would silently score the wrong payment.

    Raises:
        ValueError: If the row counts differ or a (world_id, payment_id) key disagrees in order.
    """
    if len(risk_rows) != table.num_rows:
        raise ValueError(f"payee-risk rows {len(risk_rows)} do not match the payment rows {table.num_rows}")
    keys = zip(table["world_id"].to_pylist(), table["payment_id"].to_pylist(), strict=True)
    for left, row in zip(keys, risk_rows, strict=True):
        right = (row["world_id"], row["payment_id"])
        if left != right:
            raise ValueError(f"payee-risk row key {right!r} does not match the payment row key {left!r}")
    values = pa.array([float(row["payee_risk"]) for row in risk_rows], pa.float64())
    reasons = pa.array([str(row["nan_reason"]) for row in risk_rows], pa.string())
    return table.append_column(PAYEE_RISK, values).append_column("nan_reason", reasons)


def score_split(
    dataset_dirs: Sequence[Path],
    split: str,
    features: Sequence[str],
    encodings: Mapping[str, Sequence[str]],
    model: xgboost.XGBClassifier,
    truth_root: Path,
    account_model: PUBaggedAccountModel,
    account_config: Path,
) -> tuple[list[WorldScores], dict[str, int], list[dict[str, Any]]]:
    """
    WHAT: Scores every payment of one split's worlds with the frozen payment model, attaching the
    frozen Run A payee risk and the hidden receiver role, and returns per-world arrays plus the
    NaN-reason counts and dataset records.

    WHY: Test and stress are scored from their own snapshots and payments with the frozen recipes;
    the account config is active while receiver scores are produced, exactly as for validation.

    Raises:
        ValueError: If a folder is not the expected split, truth is missing, roles or keys disagree,
            or a payment has neither a score nor a recorded reason.
    """
    worlds: list[WorldScores] = []
    reason_counts: dict[str, int] = {}
    dataset_records: list[dict[str, Any]] = []
    needed = [*KEY_COLUMNS, "bank_id", "decision_time", "receiver_account_id"]
    needed.extend(name for name in features if name != PAYEE_RISK and name not in needed)

    previous = mule_model_config_override()
    set_mule_model_config_path(account_config)
    try:
        for dataset_dir in dataset_dirs:
            manifest = validate_scoring_split(dataset_dir, split)
            truth_payments, truth_snapshots = join_truth(dataset_dir, truth_root)
            roles = truth_payments.select([*KEY_COLUMNS, "truth_role_receiver"]).sort_by(
                [("world_id", "ascending"), ("payment_id", "ascending")]
            )
            del truth_snapshots, truth_payments
            dataset_rows = 0
            for world in manifest["worlds"]:
                world_id = world.get("world_id")
                if not isinstance(world_id, str) or not world_id:
                    raise ValueError(f"{dataset_dir.name} has a world entry without world_id")
                snapshots = _read_world_snapshots(dataset_dir, world_id)
                scores_by_account, bank_by_account = score_world_account_days(account_model, snapshots)
                table = pq.read_table(
                    dataset_dir / "payments.parquet",
                    columns=needed,
                    filters=[("world_id", "=", world_id)],
                ).sort_by([("world_id", "ascending"), ("payment_id", "ascending")])
                if table.num_rows == 0:
                    raise ValueError(f"world {world_id!r} has no payment rows in {dataset_dir.name}")
                risk_rows = build_world_payee_risk(
                    table.to_pylist(),
                    scores_by_account,
                    bank_by_account,
                    world_id,
                    split,
                    -1,
                )
                _assert_score_or_reason(risk_rows)
                for row in risk_rows:
                    reason = str(row["nan_reason"]) or "scored"
                    reason_counts[reason] = reason_counts.get(reason, 0) + 1
                table = attach_payee_risk(table, risk_rows)

                world_roles = roles.filter(pc.equal(roles["world_id"], world_id))
                if world_roles.num_rows != table.num_rows:
                    raise ValueError(
                        f"world {world_id!r} has {table.num_rows} payments but {world_roles.num_rows} truth rows"
                    )
                if not pc.all(pc.equal(world_roles["payment_id"], table["payment_id"])).as_py():
                    raise ValueError(f"world {world_id!r} truth rows do not align with its payment rows")
                role_values = world_roles["truth_role_receiver"].to_pylist()
                for role in role_values:
                    if not isinstance(role, str) or not role:
                        raise ValueError(f"world {world_id!r} holds a payment without a receiver role")
                labels = [1 if role in MULE_ROLES else 0 for role in role_values]

                matrix = build_matrix(table, features, encodings)
                scores = predict_scores(model, matrix)
                worlds.append(
                    WorldScores(
                        split=split,
                        dataset=dataset_dir.name,
                        world_id=world_id,
                        scores=scores,
                        labels=labels,
                        payee_risk=table[PAYEE_RISK].to_numpy(zero_copy_only=False),
                    )
                )
                dataset_rows += table.num_rows
            dataset_records.append(
                {
                    "folder": dataset_dir.name,
                    "content_hash": manifest.get("content_hash"),
                    "rows": dataset_rows,
                    "worlds": len(manifest["worlds"]),
                }
            )
    finally:
        set_mule_model_config_path(previous)
    return worlds, reason_counts, dataset_records


def run(
    validation_dirs: Sequence[Path],
    test_dirs: Sequence[Path],
    stress_dirs: Sequence[Path],
    truth_root: Path,
    oof_root: Path,
    model_dir: Path,
    out_root: Path,
) -> dict[str, Any]:
    """
    WHAT: Verifies the frozen artifacts, scores validation for the threshold, scores test and stress
    once, and writes `result.json` with its manifest.

    WHY: One entry point keeps the threshold, the split guards and the artifacts that produced the
    numbers in one place, so the run manifest answers what was scored and with what.

    Raises:
        ValueError: If either scoring split is empty, the feature order disagrees with the model
            metadata, the booster shape differs, or any split guard fails.
    """
    if not test_dirs:
        raise ValueError("final scoring needs at least one test dataset")
    if not stress_dirs:
        raise ValueError("final scoring needs at least one stress dataset")
    started = time.perf_counter()

    payload = load_feature_order(Path(model_dir) / "feature_order.yaml")
    features = payload["features"]
    encodings = payload["categorical_encodings"]
    if not isinstance(features, list) or not isinstance(encodings, dict):
        raise ValueError(f"{model_dir} holds malformed feature order data")
    metadata = json.loads((Path(model_dir) / "model_metadata.json").read_text(encoding="utf-8"))
    if payload["feature_sha256"] != metadata["feature_sha256"] or len(features) != metadata["feature_count"]:
        raise ValueError("feature order does not match model_metadata.json; the frozen model cannot be scored")
    model_sha256, feature_file_sha256 = verify_frozen_artifacts(Path(model_dir))

    model_path = Path(model_dir) / "frozen_model.json"
    model = xgboost.XGBClassifier()
    model.load_model(str(model_path))
    booster = model.get_booster()
    if booster.num_features() != len(features):
        raise ValueError(f"frozen model takes {booster.num_features()} features but the order holds {len(features)}")
    rounds = verify_booster_rounds(booster, metadata)

    for dataset_dir in validation_dirs:
        validate_dataset_split(Path(dataset_dir), "validation")
    validation_rows = load_split_rows(validation_dirs, "validation", features, oof_root)
    validation_scores = np.asarray(predict_scores(model, build_matrix(validation_rows, features, encodings)))
    del validation_rows

    recipe = load_frozen_recipe()
    account_model = PUBaggedAccountModel.load(recipe.checkpoint_dir)
    scored: dict[str, list[WorldScores]] = {}
    extras: dict[str, dict[str, Any]] = {}
    for split, dataset_dirs in (("test", test_dirs), ("stress", stress_dirs)):
        worlds, reason_counts, dataset_records = score_split(
            dataset_dirs,
            split,
            features,
            encodings,
            model,
            truth_root,
            account_model,
            recipe.config_path,
        )
        scored[split] = worlds
        extras[split] = {"reason_counts": reason_counts, "datasets": dataset_records}

    result = evaluate_splits(validation_scores, scored)
    for split in SCORING_SPLITS:
        reason_counts = extras[split]["reason_counts"]
        total = sum(reason_counts.values())
        result["splits"][split]["nan_reason_counts"] = reason_counts
        result["splits"][split]["nan_payee_risk_share"] = (total - reason_counts.get("scored", 0)) / total
        result["splits"][split]["datasets"] = extras[split]["datasets"]

    result["manifest"] = {
        "git_hash": _git_head(),
        "model": {
            "path": str(model_path),
            "sha256": model_sha256,
            "feature_sha256": payload["feature_sha256"],
            "feature_count": len(features),
            "rounds": rounds,
            "seed": metadata["seed"],
            "device": metadata["device"],
        },
        "feature_order": {
            "path": str(Path(model_dir) / "feature_order.yaml"),
            "file_sha256": feature_file_sha256,
            "feature_sha256": payload["feature_sha256"],
        },
        "account_recipe": {
            "config": str(recipe.config_path),
            "config_sha256": recipe.config_sha256,
            "checkpoint": str(recipe.checkpoint_dir),
            "seed": recipe.seed,
            "n_bags": recipe.n_bags,
        },
        "validation_dataset_ids": [Path(path).name for path in validation_dirs],
        "scored_dataset_ids": {
            split: [Path(path).name for path in dataset_dirs]
            for split, dataset_dirs in (
                ("test", test_dirs),
                ("stress", stress_dirs),
            )
        },
        "threshold": result["threshold"],
        "hold_share": HOLD_SHARE,
        "wall_seconds": time.perf_counter() - started,
    }
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "threshold": result["threshold"],
                "validation_hold_rate": result["validation"]["hold_rate"],
                "test": result["splits"]["test"]["aggregate"],
                "stress": result["splits"]["stress"]["aggregate"],
                "test_nan_payee_risk_share": result["splits"]["test"]["nan_payee_risk_share"],
                "stress_nan_payee_risk_share": result["splits"]["stress"]["nan_payee_risk_share"],
            },
            indent=2,
        ),
        flush=True,
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    """Runs the final test and stress scoring from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run", help="Score test and stress once")
    run_parser.add_argument(
        "--validation", action="append", required=True, help="Validation dataset folder (repeatable)"
    )
    run_parser.add_argument("--test", action="append", required=True, help="Test dataset folder (repeatable)")
    run_parser.add_argument("--stress", action="append", required=True, help="Stress dataset folder (repeatable)")
    run_parser.add_argument("--truth-root", default=str(DEFAULT_TRUTH_ROOT), help="Hidden truth root")
    run_parser.add_argument("--oof-root", default=str(DEFAULT_OOF_ROOT), help="Phase 1 payee-risk root")
    run_parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR), help="Frozen payment-model folder")
    run_parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT), help="Output folder for result.json")
    args = parser.parse_args(argv)
    run(
        [Path(path) for path in args.validation],
        [Path(path) for path in args.test],
        [Path(path) for path in args.stress],
        Path(args.truth_root),
        Path(args.oof_root),
        Path(args.model_dir),
        Path(args.out_root),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
