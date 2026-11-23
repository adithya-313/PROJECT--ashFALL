"""
WHAT: Trains the Level 18 payment model `mule_payment_xgb` on payment rows, one supervised XGBoost
classifier per A-056, chooses the device with the pre-registered rule, searches its hyperparameters on
validation truth average precision, and freezes the best model.

WHY: The payment model consumes the out-of-fold payee risk built in Phase 1 and must be chosen on
hidden truth rather than on the bank-visible label it is trained on. The feature order, the target
and the device rule are frozen in writing (A-055, A-056) before any fit, so every reported number is
a held-out measurement rather than a choice tuned on the result.

TERM: payer account features: the M1 feature groups computed from the paying customer's account
snapshot and present in `payments.parquet`; the payment-level group describes the payment and its
payee. `payee_risk` is the Phase 1 receiver score.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
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
import yaml

from decisions.mule_account_model import load_mule_model_config
from evaluation.dataset_truth_join import join_truth
from evaluation.hyperparameter_search import OptunaSearchResult, run_optuna_search
from evaluation.level17_scoring import MULE_ROLES
from evaluation.payment_oof_risk import validate_dataset_split
from evaluation.trivial_world import average_precision

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCHEMA_PATH = REPO_ROOT / "data" / "schemas" / "feature_set_m1.yaml"
DEFAULT_FEATURE_ORDER_PATH = REPO_ROOT / "config" / "mule_payment_feature_order.yaml"
DEFAULT_OOF_ROOT = REPO_ROOT / "runs" / "full" / "payment_oof"
DEFAULT_OUT_ROOT = REPO_ROOT / "runs" / "full" / "payment_model_v1"
DEFAULT_TRUTH_ROOT = REPO_ROOT / "runs" / "truth"
DEFAULT_MULE_CONFIG_PATH = REPO_ROOT / "config" / "mule_model.yaml"

TARGET = "label_receiver_reported_by_cutoff"
PAYEE_RISK = "payee_risk"
PAYEE_SAME_BANK_RISK = "payee_same_bank_risk"
PAYMENT_LEVEL_GROUP = "payment_level"
CATEGORICAL_FEATURES = ("account_type", "payee_type")
KEY_COLUMNS = ("world_id", "payment_id")
OBJECTIVE = "binary:logistic"
TREE_METHOD = "hist"
DEVICE_AP_TOLERANCE = 0.005
NO_PU_REASON = (
    "no PU bagging: positives are about 45% of training rows (2,523,136 of 4,911,546 payment rows), "
    "so unlabelled rows need no maybe-positive handling"
)


@dataclass(frozen=True)
class PaymentScored:
    """
    WHAT: One validation payment's hidden receiver role, the only field the search objective reads.

    WHY: The shared objective (`validation_truth_average_precision`) is typed for account-days but
    reads only `.role`; carrying just the role avoids building millions of account-day records for
    payment rows that have no account-day identity of their own.
    """

    role: str


def _sha256_text(text: str) -> str:
    """Returns the sha256 hex digest of a UTF-8 string."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def feature_list_sha256(features: Sequence[str]) -> str:
    """Returns the sha256 of the newline-joined feature names, the same rule the account model uses."""
    return _sha256_text("\n".join(features))


def load_schema_features(schema_path: Path) -> list[dict[str, object]]:
    """
    WHAT: Returns the M1 feature catalog (`name`, `group`, ...) in schema order.

    Raises:
        ValueError: If the schema holds no features, or a feature has no name or group.
    """
    document = yaml.safe_load(schema_path.read_text(encoding="utf-8"))
    if not isinstance(document, dict) or not isinstance(document.get("features"), list):
        raise ValueError(f"schema {schema_path} holds no 'features' list")
    features: list[dict[str, object]] = []
    for record in document["features"]:
        if not isinstance(record, dict):
            raise ValueError(f"schema {schema_path} holds a feature that is not a mapping")
        name = record.get("name")
        group = record.get("group")
        if not isinstance(name, str) or not name:
            raise ValueError(f"schema {schema_path} holds a feature without a name")
        if not isinstance(group, str) or not group:
            raise ValueError(f"schema {schema_path} feature {name!r} has no group")
        features.append(record)
    return features


def build_feature_order(
    schema_path: Path,
    categorical_encodings: Mapping[str, Sequence[str]],
) -> dict[str, Any]:
    """
    WHAT: Builds the frozen payment-model feature order: `payee_risk`, then the eight
    `payment_level` features, then the remaining payer account features, each group in schema order.

    WHY: A single stored order is what makes the frozen model reproducible; deriving it from the
    schema keeps the order tied to the catalog instead of a hand-typed list.

    Raises:
        ValueError: If the schema has no payment-level group, a categorical feature is missing from
            the encodings, or an encoding is empty or not sorted/unique.
    """
    records = load_schema_features(schema_path)
    names = [str(record["name"]) for record in records]
    payment_level = [str(r["name"]) for r in records if r["group"] == PAYMENT_LEVEL_GROUP]
    if not payment_level:
        raise ValueError(f"schema {schema_path} holds no {PAYMENT_LEVEL_GROUP} features")
    payer = [name for name in names if name not in set(payment_level)]
    features = [PAYEE_RISK, *payment_level, *payer]
    if len(features) != len(set(features)):
        raise ValueError("feature order repeats a name")
    for name in CATEGORICAL_FEATURES:
        values = list(categorical_encodings[name]) if name in categorical_encodings else []
        if not values:
            raise ValueError(f"categorical feature {name!r} has no frozen encoding")
        if values != sorted(set(values)):
            raise ValueError(f"categorical encoding for {name!r} is not sorted and unique")
    return {
        "version": 1,
        "target": TARGET,
        "payee_risk_source": "runs/full/payment_oof/per_world",
        "feature_sha256": feature_list_sha256(features),
        "features": features,
        "groups": {
            "payee_risk": [PAYEE_RISK],
            PAYMENT_LEVEL_GROUP: payment_level,
            "payer_account": payer,
        },
        "categorical_encodings": {name: list(values) for name, values in categorical_encodings.items()},
    }


def _categorical_vocabularies(training_dirs: Sequence[Path]) -> dict[str, list[str]]:
    """
    WHAT: Returns the sorted values each categorical feature takes across the training payment rows.

    WHY: The encoding is frozen with the feature order, so it is built from training data only and
    never learns a category from validation rows.
    """
    values: dict[str, set[str]] = {name: set() for name in CATEGORICAL_FEATURES}
    for dataset_dir in training_dirs:
        validate_dataset_split(dataset_dir, "training")
        table = pq.read_table(dataset_dir / "payments.parquet", columns=list(CATEGORICAL_FEATURES))
        for name in CATEGORICAL_FEATURES:
            for value in pc.unique(table[name]).to_pylist():
                if value is not None:
                    values[name].add(str(value))
    return {name: sorted(found) for name, found in values.items()}


def write_feature_order(
    training_dirs: Sequence[Path],
    feature_order_path: Path,
    schema_path: Path,
) -> dict[str, object]:
    """
    WHAT: Writes the frozen feature order and categorical encodings to `feature_order_path`.

    Raises:
        ValueError: If a training dataset lacks a schema feature column.
    """
    vocabularies = _categorical_vocabularies(training_dirs)
    payload = build_feature_order(schema_path, vocabularies)
    features = payload["features"]
    if not isinstance(features, list):
        raise ValueError("built feature order holds no list")
    for dataset_dir in training_dirs:
        columns = set(pq.read_schema(dataset_dir / "payments.parquet").names)
        missing = sorted(str(name) for name in features if name != PAYEE_RISK and name not in columns)
        if missing:
            raise ValueError(f"{dataset_dir.name} lacks feature columns {missing}")
    feature_order_path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return payload


def load_feature_order(feature_order_path: Path) -> dict[str, Any]:
    """
    WHAT: Loads the frozen feature order and proves its sha256 is the recorded one.

    Raises:
        ValueError: If the file is malformed, its feature list is empty or repeats a name, or its
            sha256 disagrees with the list.
    """
    payload = yaml.safe_load(feature_order_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{feature_order_path} is not a mapping")
    features = payload.get("features")
    recorded = payload.get("feature_sha256")
    if not isinstance(features, list) or not features or not all(isinstance(f, str) for f in features):
        raise ValueError(f"{feature_order_path} holds no feature list")
    if len(features) != len(set(features)):
        raise ValueError(f"{feature_order_path} repeats a feature name")
    if not isinstance(recorded, str) or recorded != feature_list_sha256(features):
        raise ValueError(f"{feature_order_path} feature sha256 does not match its feature list")
    encodings = payload.get("categorical_encodings")
    if not isinstance(encodings, dict):
        raise ValueError(f"{feature_order_path} holds no categorical encodings")
    return payload


def _read_oof_rows(oof_root: Path, split: str) -> pa.Table:
    """Returns the per-world payee-risk rows for one split, with key and score columns only."""
    tables: list[pa.Table] = []
    for path in sorted((oof_root / "per_world").glob("*.parquet")):
        table = pq.read_table(path, columns=["world_id", "payment_id", "split", "same_bank", PAYEE_RISK])
        tables.append(table.filter(pc.equal(table["split"], split)))
    if not tables:
        raise ValueError(f"no per-world payee-risk files under {oof_root / 'per_world'}")
    combined = pa.concat_tables(tables)
    if combined.num_rows == 0:
        raise ValueError(f"no payee-risk rows for split {split!r} under {oof_root}")
    return combined.drop_columns(["split"])


def _assert_same_keys(left: pa.Table, right: pa.Table, left_name: str, right_name: str) -> None:
    """Raises if two tables do not hold the same (world_id, payment_id) keys in the same order."""
    if left.num_rows != right.num_rows:
        raise ValueError(f"{left_name} has {left.num_rows} rows but {right_name} has {right.num_rows}")
    for name in KEY_COLUMNS:
        if not pc.all(pc.equal(left[name], right[name])).as_py():
            raise ValueError(f"{left_name} and {right_name} disagree on {name!r} at the same row")


def load_split_rows(
    dataset_dirs: Sequence[Path],
    split: str,
    features: Sequence[str],
    oof_root: Path,
) -> pa.Table:
    """
    WHAT: Loads one split's payment rows with `payee_risk` and `same_bank` attached, sorted by
    (world_id, payment_id).

    WHY: Payments and the per-world payee-risk files are written in different file orders, so both
    are sorted on the same keys and their keys are proven equal before any row is used.

    Raises:
        ValueError: If a dataset's manifest split is wrong, a dataset lacks a needed column, or the
            payment and payee-risk keys disagree.
    """
    tables: list[pa.Table] = []
    needed = [*KEY_COLUMNS, "bank_id", "decision_time", "receiver_account_id", TARGET]
    needed.extend(name for name in features if name != PAYEE_RISK and name not in needed)
    for dataset_dir in dataset_dirs:
        validate_dataset_split(dataset_dir, split)
        tables.append(pq.read_table(dataset_dir / "payments.parquet", columns=needed))
    payments = pa.concat_tables(tables)
    if payments[TARGET].null_count:
        raise ValueError(f"{split} payments hold {payments[TARGET].null_count} null targets")
    oof = _read_oof_rows(oof_root, split)
    payments = payments.sort_by([("world_id", "ascending"), ("payment_id", "ascending")])
    oof = oof.sort_by([("world_id", "ascending"), ("payment_id", "ascending")])
    _assert_same_keys(payments, oof, f"{split} payments", "payee-risk rows")
    payments = payments.append_column("same_bank", oof["same_bank"])
    return payments.append_column(PAYEE_RISK, oof[PAYEE_RISK])


def load_validation_roles(
    dataset_dirs: Sequence[Path],
    truth_root: Path,
    validation_keys: pa.Table,
) -> list[str]:
    """
    WHAT: Returns the receiver's hidden role at each validation payment's decision minute, aligned to
    the validation rows.

    WHY: Truth is read only here, inside evaluation, and only for validation worlds; the role is the
    search objective's label.

    Raises:
        ValueError: If a dataset is not validation, a truth role is missing, or the truth rows do not
            align with the payment rows.
    """
    tables: list[pa.Table] = []
    for dataset_dir in dataset_dirs:
        validate_dataset_split(dataset_dir, "validation")
        payments, _ = join_truth(dataset_dir, truth_root)
        tables.append(payments.select([*KEY_COLUMNS, "truth_role_receiver"]))
    truth = pa.concat_tables(tables).sort_by([("world_id", "ascending"), ("payment_id", "ascending")])
    _assert_same_keys(validation_keys, truth, "validation payments", "validation truth")
    roles = truth["truth_role_receiver"].to_pylist()
    for role in roles:
        if not isinstance(role, str) or not role:
            raise ValueError("validation truth holds a payment without a receiver role")
    return roles


def build_matrix(
    table: pa.Table,
    features: Sequence[str],
    categorical_encodings: Mapping[str, Sequence[str]],
) -> np.ndarray:
    """
    WHAT: Builds the float32 feature matrix in the frozen feature order.

    WHY: XGBoost treats NaN as missing, so a missing value stays NaN rather than being imputed to a
    number that claims to be observed.

    Raises:
        ValueError: If a value of a categorical feature is not in the frozen encoding.
    """
    row_count = table.num_rows
    matrix = np.empty((row_count, len(features)), dtype=np.float32)
    encodings = {
        name: {value: code for code, value in enumerate(values)} for name, values in categorical_encodings.items()
    }
    for column_index, name in enumerate(features):
        if name in encodings:
            mapping = encodings[name]
            encoded = np.full(row_count, np.nan, dtype=np.float64)
            for row_index, value in enumerate(table[name].to_pylist()):
                if value is None:
                    continue
                code = mapping.get(str(value))
                if code is None:
                    raise ValueError(f"categorical value {value!r} in {name!r} is not in the frozen encoding")
                encoded[row_index] = code
            matrix[:, column_index] = encoded.astype(np.float32)
            continue
        values = table[name].cast(pa.float64()).to_numpy(zero_copy_only=False)
        matrix[:, column_index] = values.astype(np.float32)
    return matrix


def target_labels(table: pa.Table) -> np.ndarray:
    """Returns the bank-visible training target as an int8 array."""
    return table[TARGET].cast(pa.int8()).to_numpy(zero_copy_only=False)


def truth_labels(roles: Sequence[str]) -> list[int]:
    """Returns 1 for a hidden role in MULE_ROLES and 0 otherwise."""
    return [1 if role in MULE_ROLES else 0 for role in roles]


def payment_scored(roles: Sequence[str]) -> list[PaymentScored]:
    """Wraps validation roles for the search objective, one per validation payment row."""
    return [PaymentScored(role=role) for role in roles]


def fit_payment_model(
    matrix: np.ndarray,
    labels: np.ndarray,
    params: Mapping[str, int | float],
    device: str,
    seed: int,
) -> xgboost.XGBClassifier:
    """
    WHAT: Fits the one supervised XGBoost classifier the payment model is.

    WHY: A-056 fixes binary:logistic, hist and the sampled or frozen parameters; the positive share
    makes PU bagging's maybe-positive treatment unnecessary.
    """
    model = xgboost.XGBClassifier(
        objective=OBJECTIVE,
        tree_method=TREE_METHOD,
        device=device,
        random_state=seed,
        **dict(params),
    )
    model.fit(matrix, labels)
    return model


def predict_scores(model: xgboost.XGBClassifier, matrix: np.ndarray) -> list[float]:
    """Returns the model's probability of the positive class for every row."""
    return model.predict_proba(matrix)[:, 1].tolist()


def choose_device(
    cpu_ap: float,
    cpu_seconds: float,
    cuda_ap: float | None,
    cuda_seconds: float | None,
    cuda_error: str | None,
) -> tuple[str, str]:
    """
    WHAT: Applies A-055's device rule: cuda only when it ran, its truth AP is within 0.005 of cpu,
    and it was faster.

    Returns:
        (device, reason).
    """
    if cuda_error is not None:
        return "cpu", f"cuda failed and cpu is used: {cuda_error}"
    if cuda_ap is None or cuda_seconds is None:
        raise ValueError("cuda has no error but also no AP or wall time")
    difference = abs(cuda_ap - cpu_ap)
    if difference > DEVICE_AP_TOLERANCE:
        return "cpu", f"cuda truth AP differs by {difference:.6f}, above {DEVICE_AP_TOLERANCE}"
    if cuda_seconds >= cpu_seconds:
        return "cpu", f"cuda wall time {cuda_seconds:.3f}s is not lower than cpu {cpu_seconds:.3f}s"
    return "cuda", (
        f"cuda truth AP differs by {difference:.6f} and is faster ({cuda_seconds:.3f}s vs {cpu_seconds:.3f}s)"
    )


def run_device_check(
    training_matrix: np.ndarray,
    training_labels: np.ndarray,
    validation_matrix: np.ndarray,
    validation_roles: Sequence[str],
    fixed_params: Mapping[str, int | float],
    seed: int,
) -> dict[str, object]:
    """
    WHAT: Fits the fixed-parameter model once on cpu and once on cuda, records both truth APs and
    wall times, and applies the device rule.

    WHY: The device is chosen before the search so the search cannot rerun on a device picked after
    seeing its scores.

    Raises:
        ValueError: If a fit produces a non-finite truth average precision.
    """
    labels = truth_labels(validation_roles)
    started = time.perf_counter()
    cpu_model = fit_payment_model(training_matrix, training_labels, fixed_params, "cpu", seed)
    cpu_seconds = time.perf_counter() - started
    cpu_ap = average_precision(predict_scores(cpu_model, validation_matrix), labels)

    cuda_error: str | None = None
    cuda_ap: float | None = None
    cuda_seconds: float | None = None
    started = time.perf_counter()
    try:
        cuda_model = fit_payment_model(training_matrix, training_labels, fixed_params, "cuda", seed)
    except (xgboost.core.XGBoostError, RuntimeError, OSError) as error:
        # A-055/A-056 require recording an unusable cuda and falling back, not aborting the run.
        cuda_error = f"{type(error).__name__}: {error}"
        cuda_seconds = time.perf_counter() - started
    else:
        cuda_seconds = time.perf_counter() - started
        cuda_ap = average_precision(predict_scores(cuda_model, validation_matrix), labels)

    if not math.isfinite(cpu_ap):
        raise ValueError(f"cpu truth average precision is not finite: {cpu_ap!r}")
    if cuda_ap is not None and not math.isfinite(cuda_ap):
        raise ValueError(f"cuda truth average precision is not finite: {cuda_ap!r}")
    device, reason = choose_device(cpu_ap, cpu_seconds, cuda_ap, cuda_seconds, cuda_error)
    return {
        "fixed_params": dict(fixed_params),
        "seed": seed,
        "objective": OBJECTIVE,
        "tree_method": TREE_METHOD,
        "cpu": {"truth_ap": cpu_ap, "seconds": cpu_seconds},
        "cuda": {"truth_ap": cuda_ap, "seconds": cuda_seconds, "error": cuda_error},
        "ap_tolerance": DEVICE_AP_TOLERANCE,
        "chosen_device": device,
        "reason": reason,
    }


def _subset_ap(scores: Sequence[float], labels: Sequence[int], mask: np.ndarray, name: str) -> dict[str, object]:
    """Returns one subset's truth average precision and row/positive counts."""
    selected_scores = [score for score, keep in zip(scores, mask.tolist(), strict=True) if keep]
    selected_labels = [label for label, keep in zip(labels, mask.tolist(), strict=True) if keep]
    if not selected_scores:
        raise ValueError(f"{name} subset holds no rows")
    return {
        "rows": len(selected_scores),
        "positives": sum(selected_labels),
        "truth_ap": average_precision(selected_scores, selected_labels),
    }


def freeze_best_model(
    out_root: Path,
    training_matrix: np.ndarray,
    training_labels: np.ndarray,
    validation_matrix: np.ndarray,
    validation_roles: Sequence[str],
    validation_payee_risk: np.ndarray,
    features: Sequence[str],
    feature_order_path: Path,
    search: OptunaSearchResult,
    device: str,
    seed: int,
) -> dict[str, object]:
    """
    WHAT: Refits the best search parameters, scores the ablation and the same-bank split, and writes
    the frozen model, feature order, trials, importances and metadata.

    WHY: The frozen artifacts are the deliverable; the ablation and the split are reported numbers the
    architect asked for and are computed from the frozen model without changing any choice.

    Raises:
        ValueError: If the ablation column is absent, or a reported average precision is not finite.
    """
    labels = truth_labels(validation_roles)
    best_params = dict(search.best_params)
    best_model = fit_payment_model(training_matrix, training_labels, best_params, device, seed)
    scores = predict_scores(best_model, validation_matrix)
    validation_ap = average_precision(scores, labels)
    if not math.isfinite(validation_ap):
        raise ValueError(f"best-model validation truth average precision is not finite: {validation_ap!r}")

    same_bank_mask = np.isfinite(validation_payee_risk)
    other_bank_mask = np.isnan(validation_payee_risk)
    same_bank = _subset_ap(scores, labels, same_bank_mask, "same-bank")
    other_bank = _subset_ap(scores, labels, other_bank_mask, "other-bank")

    if PAYEE_SAME_BANK_RISK not in features:
        raise ValueError(f"ablation needs {PAYEE_SAME_BANK_RISK!r} in the feature order")
    ablation_index = list(features).index(PAYEE_SAME_BANK_RISK)
    training_matrix[:, ablation_index] = np.nan
    validation_matrix[:, ablation_index] = np.nan
    ablation_model = fit_payment_model(training_matrix, training_labels, best_params, device, seed)
    ablation_ap = average_precision(predict_scores(ablation_model, validation_matrix), labels)
    if not math.isfinite(ablation_ap):
        raise ValueError(f"ablation validation truth average precision is not finite: {ablation_ap!r}")

    out_root.mkdir(parents=True, exist_ok=True)
    best_model.save_model(out_root / "frozen_model.json")
    feature_order = load_feature_order(feature_order_path)
    (out_root / "feature_order.yaml").write_text(
        yaml.safe_dump(feature_order, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    (out_root / "search_trials.json").write_text(
        json.dumps(
            {
                "trials_run": search.trials_run,
                "stopped_early": search.stopped_early,
                "stop_reason": search.stop_reason,
                "best_params": best_params,
                "best_score": search.best_score,
                "trials": list(search.trials),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    importances = search.param_importances
    (out_root / "search_importances.json").write_text(json.dumps(importances, indent=2), encoding="utf-8")

    metadata = {
        "model": "mule_payment_xgb",
        "target": TARGET,
        "objective": OBJECTIVE,
        "tree_method": TREE_METHOD,
        "no_pu_reason": NO_PU_REASON,
        "seed": seed,
        "device": device,
        "best_params": best_params,
        "feature_sha256": feature_list_sha256(features),
        "feature_count": len(features),
        "trials_run": search.trials_run,
        "stopped_early": search.stopped_early,
        "stop_reason": search.stop_reason,
        "search_best_score": search.best_score,
        "validation_truth_ap": validation_ap,
        "validation_rows": len(labels),
        "validation_positives": sum(labels),
        "ablation_without_payee_same_bank_risk": {
            "validation_truth_ap": ablation_ap,
            "rows": len(labels),
            "positives": sum(labels),
        },
        "validation_truth_ap_by_receiver_bank": {
            "same_bank": same_bank,
            "other_bank": other_bank,
        },
        "param_importances": importances,
    }
    (out_root / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def run_search_and_freeze(
    out_root: Path,
    training_matrix: np.ndarray,
    training_labels: np.ndarray,
    validation_matrix: np.ndarray,
    validation_roles: Sequence[str],
    validation_payee_risk: np.ndarray,
    features: Sequence[str],
    device: str,
    mule_config_path: Path,
    feature_order_path: Path,
) -> dict[str, object]:
    """
    WHAT: Runs the pre-registered Optuna search on validation truth AP, then freezes the best model.

    WHY: The search must be the shared one (`run_optuna_search`) with the config's space, seed and
    stall rule, so the payment model's numbers are comparable with every other searched model.

    Raises:
        ValueError: If the config lacks the search space, seed or stall rule.
    """
    config = load_mule_model_config(mule_config_path)
    space = config.hyperparameter_space
    trials = config.optuna_trials
    seed = config.optuna_seed
    if space is None or trials is None or seed is None:
        raise ValueError(f"{mule_config_path} lacks hyperparameter_space, optuna_trials or optuna_seed")

    def train_trial(params: dict[str, int | float]) -> Sequence[float]:
        """Trains one candidate and returns its validation probabilities."""
        model = fit_payment_model(training_matrix, training_labels, params, device, seed)
        return predict_scores(model, validation_matrix)

    started = time.perf_counter()
    search = run_optuna_search(
        train_trial,
        payment_scored(validation_roles),
        space,
        trials,
        seed,
        min_trials_before_stop=config.optuna_min_trials_before_stop,
        stall_patience_trials=config.optuna_stall_patience_trials,
        min_improvement=config.optuna_min_improvement,
    )
    search_seconds = time.perf_counter() - started
    metadata = freeze_best_model(
        out_root,
        training_matrix,
        training_labels,
        validation_matrix,
        validation_roles,
        validation_payee_risk,
        features,
        feature_order_path,
        search,
        device,
        seed,
    )
    metadata["search_seconds"] = search_seconds
    (out_root / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def run(
    training_dirs: Sequence[Path],
    validation_dirs: Sequence[Path],
    truth_root: Path,
    oof_root: Path,
    out_root: Path,
    feature_order_path: Path,
) -> dict[str, object]:
    """
    WHAT: Loads the rows, performs the device check, runs the search and freezes the best model.

    WHY: One entry point keeps every phase on the same frozen feature order, dataset splits and seed.

    Raises:
        ValueError: If the feature order is missing or a dataset is not training/validation.
    """
    payload = load_feature_order(feature_order_path)
    features = payload["features"]
    encodings = payload["categorical_encodings"]
    if not isinstance(features, list) or not isinstance(encodings, dict):
        raise ValueError(f"{feature_order_path} holds malformed feature order data")

    print(f"loading {len(training_dirs)} training datasets", flush=True)
    training_rows = load_split_rows(training_dirs, "training", features, oof_root)
    print(f"training rows {training_rows.num_rows}", flush=True)
    training_matrix = build_matrix(training_rows, features, encodings)
    training_labels = target_labels(training_rows)

    print(f"loading {len(validation_dirs)} validation datasets", flush=True)
    validation_rows = load_split_rows(validation_dirs, "validation", features, oof_root)
    print(f"validation rows {validation_rows.num_rows}", flush=True)
    validation_matrix = build_matrix(validation_rows, features, encodings)
    validation_roles = load_validation_roles(validation_dirs, truth_root, validation_rows)
    validation_payee_risk = validation_rows[PAYEE_RISK].to_numpy(zero_copy_only=False)

    mule_config = load_mule_model_config(DEFAULT_MULE_CONFIG_PATH)
    space = mule_config.hyperparameter_space or {}
    fixed_params = _fixed_device_params(space)
    seed = mule_config.optuna_seed
    if seed is None:
        raise ValueError(f"{DEFAULT_MULE_CONFIG_PATH} holds no optuna_seed")
    root = Path(out_root)
    root.mkdir(parents=True, exist_ok=True)
    print("device check", flush=True)
    device_record = run_device_check(
        training_matrix,
        training_labels,
        validation_matrix,
        validation_roles,
        fixed_params,
        seed,
    )
    (root / "device_check.json").write_text(json.dumps(device_record, indent=2), encoding="utf-8")
    print(f"device check chose {device_record['chosen_device']}: {device_record['reason']}", flush=True)

    print("search", flush=True)
    metadata = run_search_and_freeze(
        root,
        training_matrix,
        training_labels,
        validation_matrix,
        validation_roles,
        validation_payee_risk,
        features,
        str(device_record["chosen_device"]),
        DEFAULT_MULE_CONFIG_PATH,
        feature_order_path,
    )
    print(
        json.dumps(
            {
                "validation_truth_ap": metadata["validation_truth_ap"],
                "ablation_truth_ap": metadata["ablation_without_payee_same_bank_risk"],
                "same_bank": metadata["validation_truth_ap_by_receiver_bank"],
                "trials_run": metadata["trials_run"],
                "stop_reason": metadata["stop_reason"],
            },
            indent=2,
        ),
        flush=True,
    )
    return metadata


def _fixed_device_params(space: Mapping[str, object]) -> dict[str, int | float]:
    """
    WHAT: Returns the A-056 fixed device-check parameters.

    WHY: A-056 fixes them to the Run A frozen account-model best params; taking the names from the
    search space proves the fixed set is the same nine parameters the search varies.

    Raises:
        ValueError: If the search space lacks one of the nine fixed parameters.
    """
    fixed: dict[str, int | float] = {
        "max_depth": 8,
        "learning_rate": 0.14438948595626225,
        "n_estimators": 118,
        "min_child_weight": 3.2610095175519205,
        "subsample": 0.9690047660778822,
        "colsample_bytree": 0.6934144991295391,
        "gamma": 3.010102041099164,
        "reg_alpha": 0.6145078263840675,
        "reg_lambda": 0.1514174707354069,
    }
    missing = sorted(name for name in fixed if name not in space)
    if missing:
        raise ValueError(f"search space lacks fixed device-check parameters {missing}")
    return fixed


def main(argv: Sequence[str] | None = None) -> int:
    """Runs the payment-model CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    order = commands.add_parser("write-feature-order", help="Write the frozen feature order config")
    order.add_argument("--training", action="append", required=True, help="Training dataset folder (repeatable)")
    order.add_argument("--out", default=str(DEFAULT_FEATURE_ORDER_PATH), help="Feature order file")
    order.add_argument("--schema", default=str(DEFAULT_SCHEMA_PATH), help="M1 feature schema")

    run_parser = commands.add_parser("run", help="Device check, search and freeze")
    run_parser.add_argument("--training", action="append", required=True, help="Training dataset folder (repeatable)")
    run_parser.add_argument(
        "--validation", action="append", required=True, help="Validation dataset folder (repeatable)"
    )
    run_parser.add_argument("--truth-root", default=str(DEFAULT_TRUTH_ROOT), help="Hidden truth root")
    run_parser.add_argument("--oof-root", default=str(DEFAULT_OOF_ROOT), help="Phase 1 payee-risk root")
    run_parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT), help="Output folder")
    run_parser.add_argument("--feature-order", default=str(DEFAULT_FEATURE_ORDER_PATH), help="Frozen feature order")

    args = parser.parse_args(argv)
    if args.command == "write-feature-order":
        write_feature_order([Path(path) for path in args.training], Path(args.out), Path(args.schema))
        return 0
    run(
        [Path(path) for path in args.training],
        [Path(path) for path in args.validation],
        Path(args.truth_root),
        Path(args.oof_root),
        Path(args.out_root),
        Path(args.feature_order),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
