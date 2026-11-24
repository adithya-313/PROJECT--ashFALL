# File: evaluation/payment_diagnosis.py
# Purpose: Diagnostic only. Explains the frozen Level 18 payment model's test and stress numbers.
"""
WHAT:
    Recomputes, without changing anything, the evidence behind the frozen payment model's test and
    stress scores: the leakage checklist over its 54 inputs, per-payee precision on test, single
    feature truth average precision, the oracle ceiling at the 0.5% hold budget, base rates and
    score quantiles.

WHY:
    Test truth AP 0.632 and precision 0.994 are high enough to ask whether the inputs or the
    measurement leak. This module answers that question with code citations, static scans and the
    data itself. It is a diagnosis, not a model choice: it fits nothing, tunes nothing, changes no
    threshold, and writes only under `runs/full/payment_model_v1/diagnosis/`.

TERM: diagnostic: a read-only measurement; no artifact it reports may be used to pick a model.
TERM: held payment: a payment whose frozen score is at or above the pre-registered threshold.
TERM: oracle recall: the recall of a perfect ranking that may hold only floor(hold_share * rows)
      payments per world.
"""

from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict
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
from evaluation.leakage_checks_5_6 import (
    FORBIDDEN_IDENTIFIER_NAMES,
    _default_truth_roles_path,
    _display_path,
    _feature_source_files,
    bank_visible_fields,
    event_record_fields,
    scan_identifier_derivations,
    scan_simulator_field_usage,
    simulator_only_fields,
    truth_file_columns,
)
from evaluation.level17_scoring import MULE_ROLES
from evaluation.payment_model import KEY_COLUMNS, PAYEE_RISK, build_matrix, load_feature_order, predict_scores
from evaluation.payment_oof_risk import (
    _assert_score_or_reason,
    _read_world_snapshots,
    build_world_payee_risk,
    load_frozen_recipe,
    score_world_account_days,
)
from evaluation.payment_test_score import (
    HOLD_SHARE,
    aggregate,
    attach_payee_risk,
    validate_scoring_split,
    verify_booster_rounds,
    verify_frozen_artifacts,
)
from evaluation.trivial_world import average_precision

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_DIR = REPO_ROOT / "runs" / "full" / "payment_model_v1"
DEFAULT_OUT_ROOT = DEFAULT_MODEL_DIR / "diagnosis"
DEFAULT_SCORE_RESULT = DEFAULT_MODEL_DIR / "test_score" / "result.json"
DEFAULT_TRUTH_ROOT = REPO_ROOT / "runs" / "full" / "truth"
DEFAULT_RUN_A_LEAKAGE = REPO_ROOT / "runs" / "full" / "run_a320_stage5c" / "leakage_checks_5_6.json"

DIAGNOSIS_LABEL = "diagnostic, not a model choice"

FEATURES_OF_INTEREST: tuple[str, ...] = (
    PAYEE_RISK,
    "account_age_days",
    "history_days_available",
    "days_since_device_change",
    "payee_registry_match",
    "first_payment_to_payee",
    "amount_vs_payer_median_ratio",
    "amount_vs_payer_max_ratio",
    "payee_same_bank_risk",
    "hour_ist",
    "payee_type",
    "customer_failed_attempts_24h",
)

ID_COLUMN_NAMES: frozenset[str] = frozenset(
    {"payment_id", "account_id", "receiver_account_id", "payer_account_id", "bank_id", "world_id", "run_id"}
)
PAYMENT_ID_PATTERN = r"^[0-9a-f]{16}-[0-9]{6}$"
ACCOUNT_ID_PATTERN = r"^acc_[a-z]+_[0-9]+$"


def load_threshold(result_path: Path) -> float:
    """
    WHAT: Returns the frozen scoring run's validation-only hold threshold.

    WHY: The diagnosis must use the threshold that produced the reported scores, never a new one.

    Raises:
        ValueError: If the result file lacks a usable threshold.
    """
    result = json.loads(Path(result_path).read_text(encoding="utf-8"))
    threshold = result.get("threshold")
    if not isinstance(threshold, (int, float)) or not 0.0 < float(threshold) < 1.0:
        raise ValueError(f"{result_path} holds no usable threshold; got {threshold!r}")
    return float(threshold)


def feature_truth_ap(values: np.ndarray, labels: np.ndarray) -> tuple[float | None, int]:
    """
    WHAT: Returns the truth average precision of one feature as a score, using only rows where the
    feature is finite, plus the count of excluded non-finite rows.

    WHY: A feature with a missing value has no ranking there; excluding those rows is recorded rather
    than silently ranked.

    Raises:
        ValueError: If the arrays differ in length.
    """
    if values.size != labels.size:
        raise ValueError(f"feature values {values.size} do not match labels {labels.size}")
    finite = np.isfinite(values)
    y = labels[finite]
    missing = int(values.size - y.size)
    if y.size == 0 or int(y.sum()) == 0 or int(y.sum()) == y.size:
        return None, missing
    return average_precision(values[finite].tolist(), y.tolist()), missing


def oracle_recall(labels: np.ndarray, hold_share: float = HOLD_SHARE) -> float | None:
    """
    WHAT: Returns the best recall possible at floor(hold_share * rows) held payments.

    WHY: A perfect ranking can hold exactly the budget's worth of payments and would take every true
    positive it can afford; the ratio bounds what any reranking of these rows could reach.

    Raises:
        ValueError: If the hold share is not strictly inside (0, 1) or the labels are empty.
    """
    if labels.size == 0:
        raise ValueError("oracle recall needs at least one row")
    if not 0.0 < hold_share < 1.0:
        raise ValueError(f"hold_share must lie strictly inside (0, 1); got {hold_share!r}")
    positives = int(labels.sum())
    if positives == 0:
        return None
    budget = int(np.floor(hold_share * labels.size))
    return min(budget, positives) / positives


def dedupe_payees(held_rows: Sequence[tuple[str, str, int]]) -> dict[str, Any]:
    """
    WHAT: Collapses held payments to one row per (world, receiver) and reports payee precision and
    how many payees account for half of the held payments.

    WHY: Per-payment precision can be inflated by one payee taking many held rows; the payee view
    shows whether the model is right about recipients or merely busy on a few.

    Raises:
        ValueError: If there are no held payments.
    """
    if not held_rows:
        raise ValueError("dedupe_payees needs at least one held payment")
    counts: Counter[tuple[str, str]] = Counter()
    positive: dict[tuple[str, str], bool] = {}
    for world_id, receiver, label in held_rows:
        key = (world_id, receiver)
        counts[key] += 1
        positive[key] = positive.get(key, False) or bool(label)
    payees = len(counts)
    positive_payees = sum(1 for value in positive.values() if value)
    half = sum(counts.values()) / 2.0
    cumulative = 0
    payees_for_half = 0
    for count in sorted(counts.values(), reverse=True):
        cumulative += count
        payees_for_half += 1
        if cumulative >= half:
            break
    return {
        "held_payments": sum(counts.values()),
        "distinct_payees": payees,
        "positive_payees": positive_payees,
        "payee_precision": positive_payees / payees,
        "payees_for_half": payees_for_half,
        "payees_for_half_share": payees_for_half / payees,
    }


def _id_format_report(dataset_dir: Path) -> dict[str, Any]:
    """
    WHAT: Returns how many of a dataset's payment and receiver ids fail the recorded formats.

    WHY: An id whose text encodes the label or the split would be a direct leak; counting every row
    against the two formats shows the model is fed ids and never sees them.
    """
    table = pq.read_table(Path(dataset_dir) / "payments.parquet", columns=["payment_id", "receiver_account_id"])
    payment_ok = pc.fill_null(pc.match_substring_regex(table["payment_id"], PAYMENT_ID_PATTERN), False)
    account_ok = pc.fill_null(pc.match_substring_regex(table["receiver_account_id"], ACCOUNT_ID_PATTERN), False)
    rows = int(table.num_rows)
    payment_good = int(pc.sum(pc.cast(payment_ok, pa.int64())).as_py())
    account_good = int(pc.sum(pc.cast(account_ok, pa.int64())).as_py())
    return {
        "dataset": Path(dataset_dir).name,
        "payment_ids_checked": rows,
        "bad_payment_ids": rows - payment_good,
        "account_ids_checked": rows,
        "bad_account_ids": rows - account_good,
    }


def _manifest_worlds(dataset_dir: Path) -> set[str]:
    """Returns the world ids a dataset manifest lists."""
    manifest = json.loads((Path(dataset_dir) / "manifest.json").read_text(encoding="utf-8"))
    worlds = manifest.get("worlds")
    if not isinstance(worlds, list) or not worlds:
        raise ValueError(f"{dataset_dir} manifest holds no worlds")
    ids = {world.get("world_id") for world in worlds if isinstance(world, dict)}
    if not all(isinstance(world_id, str) and world_id for world_id in ids):
        raise ValueError(f"{dataset_dir} manifest holds a world without world_id")
    return {str(world_id) for world_id in ids}


def _citation(relative: str, needle: str) -> str:
    """Returns the repository path and line of a source line that must exist, or raises."""
    path = REPO_ROOT / relative
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if needle in line:
            return f"{relative}:{number}"
    raise ValueError(f"{needle!r} was not found in {relative}")


def leakage_checks(
    training_dirs: Sequence[Path],
    validation_dirs: Sequence[Path],
    test_dirs: Sequence[Path],
    stress_dirs: Sequence[Path],
    truth_root: Path,
    features: Sequence[str],
    run_a_leakage_path: Path,
    score_result: Path,
) -> dict[str, Any]:
    """
    WHAT: Runs the Section 23 leakage checklist for the payment model's inputs and returns each
    item's PASS/FAIL with evidence.

    WHY: The checklist is the pre-existing Run A logic, pointed at the payment model's inputs and
    datasets; test and stress folders are read only as schemas, ids and world lists.

    Raises:
        ValueError: If a manifest, truth file, result file or source line the checks rely on is missing.
    """
    training_paths = [Path(path) for path in training_dirs]
    validation_paths = [Path(path) for path in validation_dirs]
    test_paths = [Path(path) for path in test_dirs]
    stress_paths = [Path(path) for path in stress_dirs]
    if not training_paths or not validation_paths:
        raise ValueError("leakage_checks needs at least one training and one validation dataset")
    truth_roles_path = _default_truth_roles_path(training_paths[0], Path(truth_root))
    world_fields = event_record_fields() | truth_file_columns(truth_roles_path)
    bank_visible = bank_visible_fields()
    simulator_only = simulator_only_fields(world_fields, bank_visible)
    source_files = _feature_source_files()
    field_hits, field_excluded = scan_simulator_field_usage(source_files, simulator_only)
    id_hits, id_excluded = scan_identifier_derivations(source_files, FORBIDDEN_IDENTIFIER_NAMES)

    payment_inputs = [name for name in features if name != PAYEE_RISK]
    datasets = [*training_paths, *validation_paths, *test_paths, *stress_paths]
    column_violations: list[dict[str, Any]] = []
    id_formats: list[dict[str, Any]] = []
    for dataset_dir in datasets:
        columns = set(pq.read_schema(Path(dataset_dir) / "payments.parquet").names)
        violations = sorted(columns & simulator_only)
        missing = sorted(name for name in payment_inputs if name not in columns)
        column_violations.append({"dataset": Path(dataset_dir).name, "violations": violations, "missing": missing})
        id_formats.append(_id_format_report(dataset_dir))
    column_bad = [row for row in column_violations if row["violations"] or row["missing"]]
    id_format_bad = [row for row in id_formats if row["bad_payment_ids"] or row["bad_account_ids"]]

    training_worlds = set().union(*(_manifest_worlds(path) for path in training_paths))
    validation_worlds = set().union(*(_manifest_worlds(path) for path in validation_paths))
    scored_worlds = set().union(*(_manifest_worlds(path) for path in [*test_paths, *stress_paths]))
    split_overlap = sorted((training_worlds | validation_worlds) & scored_worlds)

    id_inputs = sorted(set(features) & ID_COLUMN_NAMES)
    run_a = json.loads(Path(run_a_leakage_path).read_text(encoding="utf-8"))
    run_a_6b = run_a["check_6b"]
    scored_result = json.loads(Path(score_result).read_text(encoding="utf-8"))
    validation_scores_result = scored_result["validation"]

    temporal = {
        "payee_risk_prior_snapshot": _citation(
            "evaluation/payment_oof_risk.py", "if day.snapshot_time < decision_time:"
        ),
        "payment_decision_time": _citation("features/m1_payment.py", "decision_time = proposed.event_time"),
        "payment_feature_call": _citation("features/m1_payment.py", "arrival_cutoff=arrival_cutoff"),
        "label_arrival_cutoff": _citation(
            "features/m1_payment.py",
            "labelled_accounts_visible(client, run_id, bank_id, event_time, arrival_cutoff)",
        ),
        "label_visible_rule": _citation("features/account_labels.py", "cutoff = min(decision_time, arrival_cutoff)"),
        "payment_features": {
            "payee_registry_match": _citation(
                "features/m1_payment.py", 'outcome = screener.screen(payee_hash, "account", as_of=event_time)'
            ),
            "first_payment_to_payee": _citation(
                "features/m1_payment.py", "first_payment_to_payee = not (has_closed_history or has_raw_history)"
            ),
            "amount_vs_payer_median_ratio": _citation(
                "features/m1_payment.py", "amount_vs_payer_median_ratio = float(amount_paise / median_debit)"
            ),
            "amount_vs_payer_max_ratio": _citation(
                "features/m1_payment.py", "amount_vs_payer_max_ratio = float(amount_paise / max_debit)"
            ),
            "payee_same_bank_risk": _citation(
                "features/m1_payment.py",
                "labelled = labelled_accounts_visible(client, run_id, bank_id, event_time, arrival_cutoff)",
            ),
            "hour_ist": _citation("features/m1_payment.py", "ist_dt = event_time.astimezone(IST)"),
            "payee_type": _citation("features/m1_payment.py", 'payee_type": proposed.payee_type'),
            "customer_failed_attempts_24h": _citation(
                "features/m1_payment.py", "cutoff_24h < r.event_time <= event_time"
            ),
        },
    }

    items: dict[str, dict[str, Any]] = {
        "truth_and_simulator_only_fields": {
            "status": "pass" if not field_hits and not column_bad else "fail",
            "simulator_only_field_count": len(simulator_only),
            "code_hits": [asdict(hit) for hit in field_hits],
            "excluded_method_calls": [asdict(hit) for hit in field_excluded],
            "payment_column_violations": column_bad,
            "datasets_checked": len(datasets),
            "files_scanned": [_display_path(path) for path in source_files],
        },
        "temporal_cutoff": {"status": "pass", "citations": temporal},
        "split_isolation": {
            "status": "pass" if not split_overlap else "fail",
            "training_worlds": len(training_worlds),
            "validation_worlds": len(validation_worlds),
            "scored_worlds": len(scored_worlds),
            "overlap": split_overlap,
            "threshold": scored_result["threshold"],
            "hold_share": scored_result["hold_share"],
            "threshold_source": {
                "split": "validation",
                "validation_rows": validation_scores_result["rows"],
                "validation_held": validation_scores_result["held"],
                "code": _citation(
                    "evaluation/payment_test_score.py", "threshold = hold_threshold(validation_scores, hold_share)"
                ),
            },
        },
        "id_leak": {
            "status": (
                "pass"
                if not id_hits and not id_inputs and not id_format_bad and run_a_6b["status"] == "pass"
                else "fail"
            ),
            "identifier_derivations": [asdict(hit) for hit in id_hits],
            "excluded_contexts": [asdict(hit) for hit in id_excluded],
            "id_inputs": id_inputs,
            "payment_id_pattern": PAYMENT_ID_PATTERN,
            "account_id_pattern": ACCOUNT_ID_PATTERN,
            "id_format_checks": id_formats,
            "id_format_failures": id_format_bad,
            "run_a_check_6b": {
                "status": run_a_6b["status"],
                "validation_average_precision": run_a_6b["validation_average_precision"],
                "validation_base_rate": run_a_6b["validation_base_rate"],
                "average_precision_over_base_rate": run_a_6b["average_precision_over_base_rate"],
                "max_ap_over_base_rate": run_a_6b["max_ap_over_base_rate"],
            },
        },
        "label_dependent_inputs_by_arrival": {
            "status": "pass",
            "citations": {
                "payee_same_bank_risk_reads_labels": _citation(
                    "features/m1_payment.py",
                    "labelled_accounts_visible(client, run_id, bank_id, event_time, arrival_cutoff)",
                ),
                "labelled_visibility": _citation(
                    "features/account_labels.py", "cutoff = min(decision_time, arrival_cutoff)"
                ),
                "complained_visibility": _citation(
                    "features/m1_payment.py",
                    "complained_accounts_visible(client, run_id, bank_id, event_time, arrival_cutoff)",
                ),
            },
        },
    }
    statuses = {name: item["status"] for name, item in items.items()}
    return {
        "status": "pass" if all(status == "pass" for status in statuses.values()) else "fail",
        "items": items,
        "item_status": statuses,
    }


def diagnose_split(
    dataset_dirs: Sequence[Path],
    split: str,
    features: Sequence[str],
    encodings: Mapping[str, Sequence[str]],
    model: xgboost.XGBClassifier,
    truth_root: Path,
    account_model: PUBaggedAccountModel,
    account_config: Path,
    threshold: float,
) -> dict[str, Any]:
    """
    WHAT: Scores one split's worlds with the frozen model, then measures base rate, single-feature
    truth AP, oracle recall, score quantiles and, on test, per-payee precision.

    WHY: Every number the diagnosis reports comes from the same frozen scores, feature matrix and
    hidden-truth join used by the frozen scoring run.

    Raises:
        ValueError: If a folder is not the expected split, truth is missing, roles or keys disagree,
            or a payment has neither a score nor a recorded reason.
    """
    base_rates: list[float] = []
    oracles: list[float | None] = []
    feature_aps: dict[str, list[float | None]] = {name: [] for name in FEATURES_OF_INTEREST}
    feature_missing: Counter[str] = Counter()
    pooled_scores: list[np.ndarray] = []
    held_rows: list[tuple[str, str, int]] = []
    rows_total = positives_total = 0
    world_count = 0
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
                    table.to_pylist(), scores_by_account, bank_by_account, world_id, split, -1
                )
                _assert_score_or_reason(risk_rows)
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
                labels = np.asarray([1 if role in MULE_ROLES else 0 for role in role_values], dtype=np.int8)

                matrix = build_matrix(table, features, encodings)
                scores = np.asarray(predict_scores(model, matrix), dtype=np.float64)
                rows = int(labels.size)
                positives = int(labels.sum())
                rows_total += rows
                positives_total += positives
                world_count += 1
                base_rates.append(positives / rows)
                oracles.append(oracle_recall(labels, HOLD_SHARE))
                for name in FEATURES_OF_INTEREST:
                    column = features.index(name)
                    ap, missing = feature_truth_ap(matrix[:, column].astype(np.float64), labels)
                    feature_aps[name].append(ap)
                    feature_missing[name] += missing
                pooled_scores.append(scores.astype(np.float32))
                if split == "test":
                    held = scores >= threshold
                    receivers = np.asarray(table["receiver_account_id"].to_pylist(), dtype=object)
                    for receiver, label in zip(receivers[held], labels[held], strict=True):
                        held_rows.append((world_id, str(receiver), int(label)))
                del matrix, table
    finally:
        set_mule_model_config_path(previous)

    if not pooled_scores:
        raise ValueError(f"split {split!r} holds no scored worlds")
    if positives_total == 0:
        raise ValueError(f"split {split!r} holds no mule-receiver payments; its oracle ceiling is undefined")
    scores_all = np.concatenate(pooled_scores)
    quantiles = np.asarray(np.percentile(scores_all, [50.0, 90.0, 99.0], method="linear"), dtype=np.float64)
    pooled_budget = min(int(np.floor(HOLD_SHARE * rows_total)), positives_total)
    result: dict[str, Any] = {
        "worlds": world_count,
        "rows": rows_total,
        "positives": positives_total,
        "base_rate": {
            "pooled": positives_total / rows_total,
            "median": aggregate(base_rates)["median"],
            "p10": aggregate(base_rates)["p10"],
            "p90": aggregate(base_rates)["p90"],
        },
        "feature_truth_ap": {
            name: {
                **aggregate(feature_aps[name]),
                "nan_rows": int(feature_missing[name]),
            }
            for name in FEATURES_OF_INTEREST
        },
        "oracle_recall_at_half_percent": {
            **aggregate(oracles),
            "pooled_all_rows": pooled_budget / positives_total,
        },
        "score_quantiles": {
            "p50": float(quantiles[0]),
            "p90": float(quantiles[1]),
            "p99": float(quantiles[2]),
        },
    }
    if split == "test":
        result["payee_precision"] = dedupe_payees(held_rows)
        result["payee_precision"]["per_payment_precision"] = sum(label for _, _, label in held_rows) / len(held_rows)
    return result


def run(
    training_dirs: Sequence[Path],
    validation_dirs: Sequence[Path],
    test_dirs: Sequence[Path],
    stress_dirs: Sequence[Path],
    truth_root: Path,
    model_dir: Path,
    score_result: Path,
    out_root: Path,
    run_a_leakage_path: Path,
) -> dict[str, Any]:
    """
    WHAT: Runs the leakage checklist and both split diagnoses, then writes `diagnosis.json`.

    WHY: One entry point keeps the frozen threshold, the verified artifacts and every diagnostic
    number in one labelled record.

    Raises:
        ValueError: If a split is empty, the artifacts differ from A-058, or any check input is missing.
    """
    if not test_dirs or not stress_dirs:
        raise ValueError("the diagnosis needs at least one test and one stress dataset")
    started = time.perf_counter()
    payload = load_feature_order(Path(model_dir) / "feature_order.yaml")
    features = payload["features"]
    encodings = payload["categorical_encodings"]
    metadata = json.loads((Path(model_dir) / "model_metadata.json").read_text(encoding="utf-8"))
    if payload["feature_sha256"] != metadata["feature_sha256"] or len(features) != metadata["feature_count"]:
        raise ValueError("feature order does not match model_metadata.json")
    model_sha, feature_sha = verify_frozen_artifacts(Path(model_dir))
    model = xgboost.XGBClassifier()
    model.load_model(str(Path(model_dir) / "frozen_model.json"))
    rounds = verify_booster_rounds(model.get_booster(), metadata)
    threshold = load_threshold(Path(score_result))

    leakage = leakage_checks(
        training_dirs,
        validation_dirs,
        test_dirs,
        stress_dirs,
        truth_root,
        features,
        run_a_leakage_path,
        score_result,
    )
    recipe = load_frozen_recipe()
    account_model = PUBaggedAccountModel.load(recipe.checkpoint_dir)
    splits: dict[str, Any] = {}
    for split, dataset_dirs in (("test", test_dirs), ("stress", stress_dirs)):
        splits[split] = diagnose_split(
            dataset_dirs, split, features, encodings, model, truth_root, account_model, recipe.config_path, threshold
        )
    scored = json.loads(Path(score_result).read_text(encoding="utf-8"))
    splits["test"]["payee_precision"]["reported_per_world_precision_median"] = scored["splits"]["test"]["aggregate"][
        "precision"
    ]["median"]

    result = {
        "label": DIAGNOSIS_LABEL,
        "label_note": "read-only diagnosis; no model, threshold, feature or score was chosen from these numbers",
        "threshold": threshold,
        "hold_share": HOLD_SHARE,
        "leakage": leakage,
        "splits": splits,
        "manifest": {
            "model_sha256": model_sha,
            "feature_order_sha256": feature_sha,
            "feature_count": len(features),
            "rounds": rounds,
            "seed": metadata["seed"],
            "device": metadata["device"],
            "account_recipe_seed": recipe.seed,
            "score_result": str(Path(score_result)),
            "wall_seconds": time.perf_counter() - started,
        },
    }
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "diagnosis.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "label": DIAGNOSIS_LABEL,
                "leakage": leakage["item_status"],
                "threshold": threshold,
                "test": splits["test"],
                "stress": splits["stress"],
            },
            indent=2,
        ),
        flush=True,
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    """Runs the read-only diagnosis from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run", help="Diagnose the frozen model's test and stress scores")
    run_parser.add_argument("--training", action="append", required=True)
    run_parser.add_argument("--validation", action="append", required=True)
    run_parser.add_argument("--test", action="append", required=True)
    run_parser.add_argument("--stress", action="append", required=True)
    run_parser.add_argument("--truth-root", default=str(DEFAULT_TRUTH_ROOT))
    run_parser.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    run_parser.add_argument("--score-result", default=str(DEFAULT_SCORE_RESULT))
    run_parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    run_parser.add_argument("--run-a-leakage", default=str(DEFAULT_RUN_A_LEAKAGE))
    args = parser.parse_args(argv)
    run(
        [Path(path) for path in args.training],
        [Path(path) for path in args.validation],
        [Path(path) for path in args.test],
        [Path(path) for path in args.stress],
        Path(args.truth_root),
        Path(args.model_dir),
        Path(args.score_result),
        Path(args.out_root),
        Path(args.run_a_leakage),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
