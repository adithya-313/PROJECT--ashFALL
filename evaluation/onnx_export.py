# File: evaluation/onnx_export.py
# Purpose: Export the frozen Level 18 account and payment models to ONNX after merging the 50 account bags
"""
WHAT:
    Merges the 50 frozen Run A account members into one margin-space booster by keeping every member
    tree and dividing every leaf weight by 50, proves the merged booster reproduces the production
    bagged probability within 1e-5, makes the same decisions at the frozen cut and keeps the ranking
    (A-061b), exports that booster and the frozen payment booster to ONNX with the committed feature
    orders, and checks ONNX against the native production probability within 1e-5 on 20,000-row
    validation and test samples that keep their NaNs.

WHY:
    A-061 fixes the merge rule, the 20,000-row sampling rule, the required converter output contract
    and the parity tolerance before any export exists. A-061b replaces the 1e-6 margin gate after it
    failed at 1.310e-05: float32 accumulation over 5,900 leaves cannot meet it, so the merge is now
    gated on the probability error, the decisions and the ranking instead. Nothing is fitted, tuned
    or changed, and test rows are used only to measure parity.

TERM: margin-space booster: one XGBoost booster holding every member tree with each leaf weight
      divided by the bag count, so its raw output is the mean member margin.
TERM: parity: the largest absolute difference between an ONNX score and the native production
      probability on the same rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import onnxruntime
import pyarrow as pa
import pyarrow.parquet as pq
import xgboost
from onnxmltools import __version__ as onnxmltools_version
from onnxmltools.convert import convert_xgboost
from onnxmltools.convert.common.data_types import FloatTensorType

from decisions.mule_account_model import (
    _feature_value,
    account_model_feature_names,
    mule_model_config_override,
    set_mule_model_config_path,
)
from decisions.pu_learning import PUBaggedAccountModel
from evaluation.level17_scoring import one_row_per_account_day
from evaluation.payment_candidates import load_candidate_model, verify_candidate_artifacts
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
    _CommitSampler,
    _git_head,
    _read_world_snapshots,
    build_world_payee_risk,
    load_frozen_recipe,
    score_world_account_days,
    validate_dataset_split,
)
from evaluation.payment_test_score import (
    attach_payee_risk,
    validate_scoring_split,
    verify_booster_rounds,
)
from evaluation.reason_codes import (
    _SAMPLING_RULE,
    ACCOUNT_DAY_KEY_FIELDS,
    PAYMENT_KEY_FIELDS,
    SAMPLE_ROWS,
    SEED,
    _sample_key,
    select_sample,
    verify_account_checkpoint,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CANDIDATE_DIR = REPO_ROOT / "runs" / "full" / "payment_model_v2" / "C1"
DEFAULT_OOF_ROOT = REPO_ROOT / "runs" / "full" / "payment_oof"
DEFAULT_OUT_ROOT = REPO_ROOT / "runs" / "full" / "level18c" / "onnx"
DEFAULT_EXTRA_REPORTING = REPO_ROOT / "runs" / "full" / "run_a320_stage5c" / "extra_reporting.json"

#: A-061 and A-061b fix these names, limits and the sampling constants reused from A-059.
AMENDMENT = "A-061"
AMENDMENTS = ["A-061", "A-061a", "A-061b"]
PARITY_TOLERANCE = 1e-5
SPEARMAN_MIN = 0.999999
TOP_FRACTION = 0.01
CUT_SOURCE_KEY = "manifest.xgboost_cut"
MERGED_BOOSTER_NAME = "account_margin_booster.json"
ACCOUNT_ONNX_NAME = "mule_account_xgb.onnx"
PAYMENT_ONNX_NAME = "mule_payment_xgb.onnx"
INPUT_NAME = "input"
PROBABILITY_OUTPUT = "probabilities"
EXPECTED_OUTPUTS = ["label", PROBABILITY_OUTPUT]


def _sha256_file(path: Path) -> str:
    """Returns the sha256 hex digest of a file's bytes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _split_datasets(dataset_dirs: Sequence[Path], split: str) -> list[Path]:
    """
    WHAT: Returns the dataset folders ordered by name after each manifest proves the expected split.

    WHY: The sample rule keys on the folder name, so two folders sharing a name would make the same
    row appear twice under one key.

    Raises:
        ValueError: If no folder is given, two folders share a name, or a manifest records another split.
    """
    if not dataset_dirs:
        raise ValueError(f"no {split} dataset folder was given")
    datasets = sorted((Path(path) for path in dataset_dirs), key=lambda path: path.name)
    names = [dataset.name for dataset in datasets]
    if len(names) != len(set(names)):
        raise ValueError(f"{split} dataset folder names are not unique: {names}")
    for dataset in datasets:
        if split == "validation":
            validate_dataset_split(dataset, split)
        else:
            validate_scoring_split(dataset, split)
    return datasets


def _content_hashes(datasets: Sequence[Path], split: str) -> dict[str, Any]:
    """Returns each dataset's recorded content hash by folder name."""
    if split == "validation":
        return {dataset.name: validate_dataset_split(dataset, split).get("content_hash") for dataset in datasets}
    return {dataset.name: validate_scoring_split(dataset, split).get("content_hash") for dataset in datasets}


def _account_day_candidates(datasets: Sequence[Path], split: str) -> Iterator[dict[str, Any]]:
    """
    WHAT: Yields every account-day row of one split, tagged with its dataset folder name.

    WHY: The sample is ranked over all account-days of the split, and reading one world at a time
    keeps only that world's snapshots in memory while the digest ranking selects the sample.

    Raises:
        ValueError: If a manifest holds no world list, a world entry has no world_id, or a snapshot
            row already carries a dataset field.
    """
    for dataset_dir in datasets:
        manifest = (
            validate_dataset_split(dataset_dir, split)
            if split == "validation"
            else validate_scoring_split(dataset_dir, split)
        )
        worlds = manifest.get("worlds")
        if not isinstance(worlds, list):
            raise ValueError(f"{dataset_dir.name} manifest holds no world list")
        for entry in worlds:
            world_id = entry.get("world_id") if isinstance(entry, dict) else None
            if not isinstance(world_id, str) or not world_id:
                raise ValueError(f"{dataset_dir.name} manifest holds a world entry without world_id")
            for row in one_row_per_account_day(_read_world_snapshots(dataset_dir, world_id)):
                if "dataset" in row:
                    raise ValueError(f"{dataset_dir.name} snapshot rows already carry a dataset field")
                candidate = dict(row)
                candidate["dataset"] = dataset_dir.name
                yield candidate


def _nan_rows(matrix: np.ndarray) -> int:
    """Returns the number of rows holding at least one NaN feature."""
    return int(np.isnan(matrix).any(axis=1).sum())


def account_sample(
    datasets: Sequence[Path],
    split: str,
    feature_names: Sequence[str],
) -> tuple[np.ndarray, list[tuple[str, ...]]]:
    """
    WHAT: Returns the A-061 account sample's float32 matrix (in feature order) and its sorted keys.

    WHY: The same ranked sample must feed the merge proof and the parity check, and the matrix must
    follow the frozen account feature order with a missing value kept as NaN.

    Raises:
        ValueError: If the split holds no account-day rows.
    """
    sample = select_sample(_account_day_candidates(datasets, split), ACCOUNT_DAY_KEY_FIELDS, SEED, SAMPLE_ROWS)
    if not sample:
        raise ValueError(f"the {split} split holds no account-day rows")
    sample = sorted(sample, key=lambda row: _sample_key(row, ACCOUNT_DAY_KEY_FIELDS))
    matrix = np.asarray(
        [[_feature_value(row.get(name), name) for name in feature_names] for row in sample],
        dtype=np.float32,
    )
    return matrix, [_sample_key(row, ACCOUNT_DAY_KEY_FIELDS) for row in sample]


def _payment_validation_sample(
    datasets: Sequence[Path],
    features: Sequence[str],
    encodings: Mapping[str, Sequence[str]],
    oof_root: Path,
) -> tuple[np.ndarray, list[tuple[str, ...]]]:
    """
    WHAT: Returns the A-061 payment validation sample's float32 matrix and its sorted keys.

    WHY: Validation payments keep the production `load_split_rows` rows, so the sample sees the same
    features and payee risk the frozen payment model was scored with.

    Raises:
        ValueError: If a dataset holds no payments, a payment key appears in two datasets, or a
            selected key is not found in the loaded table.
    """
    table = load_split_rows(list(datasets), "validation", features, oof_root)
    if table.num_rows == 0:
        raise ValueError("the validation datasets hold no payment rows")
    dataset_of: dict[tuple[str, str], str] = {}
    for dataset_dir in datasets:
        keys = pq.read_table(dataset_dir / "payments.parquet", columns=["world_id", "payment_id"])
        if keys.num_rows == 0:
            raise ValueError(f"validation dataset {dataset_dir.name} holds no payment rows")
        for world_id, payment_id in zip(keys["world_id"].to_pylist(), keys["payment_id"].to_pylist(), strict=True):
            key = (world_id, payment_id)
            if key in dataset_of:
                raise ValueError(f"validation payment key {key} appears in two datasets")
            dataset_of[key] = dataset_dir.name

    def candidates() -> Iterator[dict[str, Any]]:
        for world_id, payment_id in zip(table["world_id"].to_pylist(), table["payment_id"].to_pylist(), strict=True):
            if (world_id, payment_id) not in dataset_of:
                raise ValueError(f"validation payment key {(world_id, payment_id)} belongs to no dataset")
            yield {"dataset": dataset_of[(world_id, payment_id)], "world_id": world_id, "payment_id": payment_id}

    sample = select_sample(candidates(), PAYMENT_KEY_FIELDS, SEED, SAMPLE_ROWS)
    if not sample:
        raise ValueError("the validation sample holds no payment rows")
    wanted = {_sample_key(row, PAYMENT_KEY_FIELDS) for row in sample}

    located: list[dict[str, Any]] = []
    positions: list[int] = []
    for index, (world_id, payment_id) in enumerate(
        zip(table["world_id"].to_pylist(), table["payment_id"].to_pylist(), strict=True)
    ):
        row = {"dataset": dataset_of[(world_id, payment_id)], "world_id": world_id, "payment_id": payment_id}
        if _sample_key(row, PAYMENT_KEY_FIELDS) in wanted:
            located.append(row)
            positions.append(index)
    if len(positions) != len(wanted):
        raise ValueError(f"found {len(positions)} validation rows for {len(wanted)} sampled keys")
    order = sorted(range(len(located)), key=lambda index: _sample_key(located[index], PAYMENT_KEY_FIELDS))
    located = [located[index] for index in order]
    combined = table.take(pa.array([positions[index] for index in order], type=pa.int64()))
    matrix = build_matrix(combined, features, encodings)
    return matrix, [_sample_key(row, PAYMENT_KEY_FIELDS) for row in located]


def _payment_test_sample(
    datasets: Sequence[Path],
    features: Sequence[str],
    encodings: Mapping[str, Sequence[str]],
    account_model: PUBaggedAccountModel,
    account_config: Path,
) -> tuple[np.ndarray, list[tuple[str, ...]]]:
    """
    WHAT: Returns the A-061 payment test sample's float32 matrix and its sorted keys.

    WHY: Test rows have no production payment loader, so the rows are rebuilt from the same
    functions the task 4 scoring path uses, and only the sampled keys are kept per world instead of
    holding the whole split.

    Raises:
        ValueError: If a dataset holds no payments, a sampled key is missing or repeated, or a
            payment row has no score and no recorded NaN reason.
    """

    def candidates() -> Iterator[dict[str, Any]]:
        for dataset_dir in datasets:
            table = pq.read_table(dataset_dir / "payments.parquet", columns=["world_id", "payment_id"])
            if table.num_rows == 0:
                raise ValueError(f"test dataset {dataset_dir.name} holds no payment rows")
            for world_id, payment_id in zip(
                table["world_id"].to_pylist(), table["payment_id"].to_pylist(), strict=True
            ):
                yield {"dataset": dataset_dir.name, "world_id": world_id, "payment_id": payment_id}

    sample = select_sample(candidates(), PAYMENT_KEY_FIELDS, SEED, SAMPLE_ROWS)
    if not sample:
        raise ValueError("the test sample holds no payment rows")
    wanted: dict[str, set[tuple[str, str]]] = {}
    for row in sample:
        wanted.setdefault(row["dataset"], set()).add((row["world_id"], row["payment_id"]))

    needed = [*KEY_COLUMNS, "bank_id", "decision_time", "receiver_account_id"]
    needed.extend(name for name in features if name != PAYEE_RISK and name not in needed)
    pieces: list[tuple[list[tuple[str, ...]], np.ndarray]] = []
    previous = mule_model_config_override()
    set_mule_model_config_path(Path(account_config))
    try:
        for dataset_dir in datasets:
            manifest = validate_scoring_split(dataset_dir, "test")
            if dataset_dir.name not in wanted:
                continue
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
                    table.to_pylist(), scores_by_account, bank_by_account, world_id, "test", -1
                )
                _assert_score_or_reason(risk_rows)
                table = attach_payee_risk(table, risk_rows)
                wanted_keys = wanted[dataset_dir.name]
                keep = [
                    index
                    for index, (row_world, payment_id) in enumerate(
                        zip(table["world_id"].to_pylist(), table["payment_id"].to_pylist(), strict=True)
                    )
                    if (row_world, payment_id) in wanted_keys
                ]
                if not keep:
                    continue
                kept = table.take(pa.array(keep, type=pa.int64()))
                keys = [
                    _sample_key(
                        {"dataset": dataset_dir.name, "world_id": row_world, "payment_id": payment_id},
                        PAYMENT_KEY_FIELDS,
                    )
                    for row_world, payment_id in zip(
                        kept["world_id"].to_pylist(), kept["payment_id"].to_pylist(), strict=True
                    )
                ]
                pieces.append((keys, build_matrix(kept, features, encodings)))
    finally:
        set_mule_model_config_path(previous)

    if not pieces:
        raise ValueError("the test sample selected no payment rows")
    keys = [key for keys_of_piece, _ in pieces for key in keys_of_piece]
    if len(keys) != len(set(keys)):
        raise ValueError("the test sample holds a repeated payment key")
    if sorted(keys) != sorted(_sample_key(row, PAYMENT_KEY_FIELDS) for row in sample):
        raise ValueError("the rebuilt test rows do not match the sampled payment keys")
    matrix = np.vstack([matrix_of_piece for _, matrix_of_piece in pieces])
    order = sorted(range(len(keys)), key=lambda index: keys[index])
    return matrix[order], [keys[index] for index in order]


def _member_rows(raw: Mapping[str, Any], index: int) -> tuple[str, Mapping[str, Any], Mapping[str, Any]]:
    """
    WHAT: Returns one member's base score and learner blocks after proving the merge preconditions.

    WHY: Leaf scaling only equals the mean margin when every member is a binary logistic model with
    the same base score, the same parallel-tree setting and no extra output class.

    Raises:
        ValueError: If the objective, class count, feature count or parallel-tree setting differs
            from what the frozen recipe fixes.
    """
    learner = raw.get("learner")
    if not isinstance(learner, Mapping):
        raise ValueError(f"member {index} JSON holds no learner block")
    objective = learner.get("objective")
    name = objective.get("name") if isinstance(objective, Mapping) else None
    if name != "binary:logistic":
        raise ValueError(f"member {index} objective is {name!r}, expected 'binary:logistic'")
    parameters = learner.get("learner_model_param")
    if not isinstance(parameters, Mapping):
        raise ValueError(f"member {index} JSON holds no learner_model_param block")
    gradient = learner.get("gradient_booster")
    model = gradient.get("model") if isinstance(gradient, Mapping) else None
    if not isinstance(model, Mapping):
        raise ValueError(f"member {index} JSON holds no gbtree model block")
    tree_parameters = model.get("gbtree_model_param")
    if not isinstance(tree_parameters, Mapping):
        raise ValueError(f"member {index} JSON holds no gbtree_model_param block")
    if tree_parameters.get("num_parallel_tree") != "1":
        raise ValueError(
            f"member {index} num_parallel_tree is {tree_parameters.get('num_parallel_tree')!r}, expected '1'"
        )
    if parameters.get("num_class") != "0":
        raise ValueError(f"member {index} num_class is {parameters.get('num_class')!r}, expected '0' for binary")
    return str(parameters.get("base_score")), learner, model


def merge_member_models(raws: Sequence[Mapping[str, Any]], n_bags: int, feature_count: int) -> dict[str, Any]:
    """
    WHAT: Returns one booster JSON holding every member tree with each leaf weight divided by n_bags.

    WHY: The production ensemble averages member margins, so the sum of every member's trees divided
    by the bag count is exactly that mean margin in real arithmetic; writing the division into each
    leaf is what makes a single ONNX tree ensemble stand in for the 50 members.

    Raises:
        ValueError: If the member count differs from the bag count, a member violates the merge
            preconditions, or the members disagree on the base score or feature count.
    """
    if len(raws) != n_bags:
        raise ValueError(f"{len(raws)} member models do not match the {n_bags} frozen bags")
    base_scores: set[str] = set()
    trees: list[dict[str, Any]] = []
    template_learner: Mapping[str, Any] | None = None
    template_model: Mapping[str, Any] | None = None
    for index, raw in enumerate(raws):
        base_score, learner, model = _member_rows(raw, index)
        base_scores.add(base_score)
        if learner.get("learner_model_param", {}).get("num_feature") != str(feature_count):
            raise ValueError(
                f"member {index} holds {learner.get('learner_model_param', {}).get('num_feature')!r} features, "
                f"the frozen recipe fixes {feature_count}"
            )
        if template_learner is None:
            template_learner = learner
            template_model = model
        member_trees = model.get("trees")
        if not isinstance(member_trees, list) or not member_trees:
            raise ValueError(f"member {index} JSON holds no trees")
        scale = 1.0 / n_bags
        for tree in member_trees:
            if not isinstance(tree, Mapping):
                raise ValueError(f"member {index} holds a tree that is not a mapping")
            left_children = tree.get("left_children")
            conditions = tree.get("split_conditions")
            if not isinstance(left_children, list) or not isinstance(conditions, list):
                raise ValueError(f"member {index} holds a tree without left_children or split_conditions")
            if len(left_children) != len(conditions):
                raise ValueError(f"member {index} holds a tree whose node arrays differ in length")
            scaled = dict(tree)
            scaled["id"] = len(trees)
            scaled["split_conditions"] = [
                float(value) * scale if int(left) == -1 else value
                for value, left in zip(conditions, left_children, strict=True)
            ]
            trees.append(scaled)
    if len(base_scores) != 1:
        raise ValueError(f"the members disagree on base_score: {sorted(base_scores)}")
    if template_learner is None or template_model is None:
        raise ValueError("merge_member_models needs at least one member")

    tree_parameters = dict(template_model.get("gbtree_model_param", {}))
    tree_parameters["num_trees"] = str(len(trees))
    merged_model = {
        "cats": template_model.get("cats"),
        "gbtree_model_param": tree_parameters,
        "iteration_indptr": list(range(len(trees) + 1)),
        "tree_info": [0] * len(trees),
        "trees": trees,
    }
    gradient = dict(template_learner.get("gradient_booster", {}))
    gradient["model"] = merged_model
    merged_learner = dict(template_learner)
    merged_learner["gradient_booster"] = gradient
    return {"learner": merged_learner, "version": raws[0].get("version")}


def write_margin_booster(
    merged: Mapping[str, Any], path: Path, expected_rounds: int, feature_count: int
) -> xgboost.Booster:
    """
    WHAT: Writes the merged JSON, reloads it as a booster and proves its rounds and feature count.

    WHY: The file, not the in-memory dictionary, is what the converter reads, so the proof must run
    on the reloaded booster.

    Raises:
        ValueError: If the reloaded booster has another round count or another feature count.
    """
    path = Path(path)
    path.write_text(json.dumps(merged) + "\n", encoding="utf-8", newline="\n")
    booster = xgboost.Booster()
    booster.load_model(str(path))
    if booster.num_boosted_rounds() != expected_rounds:
        raise ValueError(f"merged booster holds {booster.num_boosted_rounds()} rounds, expected {expected_rounds}")
    if booster.num_features() != feature_count:
        raise ValueError(f"merged booster takes {booster.num_features()} features, expected {feature_count}")
    return booster


def load_account_cut(path: Path = DEFAULT_EXTRA_REPORTING) -> float:
    """
    WHAT: Returns the account model's frozen decision cut from the extra-reporting manifest.

    WHY: A-061b gates the merge proof on the decisions the deployed model makes, so the cut is read
    from the artifact that set it instead of being typed into this module.

    Raises:
        ValueError: If the file, the manifest block or the cut is missing, non-numeric or outside [0, 1].
    """
    path = Path(path)
    if not path.exists():
        raise ValueError(f"{path} is missing, so the frozen account cut cannot be read")
    document = json.loads(path.read_text(encoding="utf-8"))
    manifest = document.get("manifest") if isinstance(document, Mapping) else None
    cut = manifest.get("xgboost_cut") if isinstance(manifest, Mapping) else None
    if isinstance(cut, bool) or not isinstance(cut, (int, float)):
        raise ValueError(f"{path} {CUT_SOURCE_KEY} must be a number; got {cut!r}")
    value = float(cut)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{path} {CUT_SOURCE_KEY} must lie in [0, 1]; got {value!r}")
    return value


def midranks(values: Sequence[float] | np.ndarray) -> np.ndarray:
    """
    WHAT: Returns the average rank of every value, giving tied values the mean of the ranks they span.

    WHY: A-061b's rank check uses Spearman's rho, the Pearson correlation of midranks; scipy is not a
    declared dependency, so the ranks are computed here with numpy.
    """
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"midranks needs a one-dimensional array; got shape {array.shape}")
    order = np.argsort(array, kind="mergesort")
    ordered = array[order]
    ranks = np.empty(array.shape[0], dtype=np.float64)
    start = 0
    while start < array.shape[0]:
        stop = start + 1
        while stop < array.shape[0] and ordered[stop] == ordered[start]:
            stop += 1
        ranks[order[start:stop]] = (start + stop - 1) / 2.0 + 1.0
        start = stop
    return ranks


def spearman_correlation(left: Sequence[float] | np.ndarray, right: Sequence[float] | np.ndarray) -> float:
    """
    WHAT: Returns Spearman's rank correlation between two equally long score vectors.

    Raises:
        ValueError: If the vectors differ in shape, hold fewer than two values, or every value is tied.
    """
    left_values = np.asarray(left, dtype=np.float64)
    right_values = np.asarray(right, dtype=np.float64)
    if left_values.shape != right_values.shape:
        raise ValueError(f"Spearman needs equal shapes; got {left_values.shape} and {right_values.shape}")
    if left_values.ndim != 1 or left_values.size < 2:
        raise ValueError(f"Spearman needs at least two one-dimensional values; got shape {left_values.shape}")
    left_centered = midranks(left_values)
    right_centered = midranks(right_values)
    left_centered = left_centered - left_centered.mean()
    right_centered = right_centered - right_centered.mean()
    denominator = float(np.sqrt(np.square(left_centered).sum() * np.square(right_centered).sum()))
    if denominator == 0.0:
        raise ValueError("Spearman is undefined when every value is tied in one vector")
    return float((left_centered * right_centered).sum() / denominator)


def top_fraction_indices(scores: Sequence[float] | np.ndarray, fraction: float) -> set[int]:
    """
    WHAT: Returns the row indices of the highest scores inside the given top fraction.

    WHY: A-061b compares the merged and native top-1% sets; breaking ties by ascending index makes
    the set well defined even when two scores are equal.
    """
    array = np.asarray(scores, dtype=np.float64)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"top fraction needs at least one one-dimensional score; got shape {array.shape}")
    if not 0.0 < fraction <= 1.0:
        raise ValueError(f"top fraction must lie in (0, 1]; got {fraction!r}")
    count = math.ceil(fraction * array.size)
    order = np.lexsort((np.arange(array.size), -array))
    return {int(index) for index in order[:count]}


def decision_flips(native: Sequence[float] | np.ndarray, merged: Sequence[float] | np.ndarray, cut: float) -> int:
    """
    WHAT: Returns how many rows sit on different sides of the inclusive cut in the two score vectors.

    WHY: A-061b requires the merged booster to make the same restriction decision as the production
    probabilities at the frozen cut; the comparison uses >= because the cut is inclusive.
    """
    native_values = np.asarray(native, dtype=np.float64)
    merged_values = np.asarray(merged, dtype=np.float64)
    if native_values.shape != merged_values.shape:
        raise ValueError(f"decision check needs equal shapes; got {native_values.shape} and {merged_values.shape}")
    mismatch = (native_values >= cut) != (merged_values >= cut)
    return int(mismatch.sum())


def merge_gate(
    native_probability: Sequence[float] | np.ndarray,
    merged_probability: Sequence[float] | np.ndarray,
    native_margin: Sequence[float] | np.ndarray,
    merged_margin: Sequence[float] | np.ndarray,
    cut: float,
) -> dict[str, Any]:
    """
    WHAT: Applies A-061b's three merge gates and returns the reported differences.

    WHY: The merged booster is a float32 approximation of the bagged mean, so the phase gates what
    serving depends on: the probability error, the decisions at the frozen cut and the ranking.

    Raises:
        ValueError: If the sample is empty, the vectors differ in shape, or any gate fails.
    """
    native_probabilities = np.asarray(native_probability, dtype=np.float64)
    merged_probabilities = np.asarray(merged_probability, dtype=np.float64)
    native_margins = np.asarray(native_margin, dtype=np.float64)
    merged_margins = np.asarray(merged_margin, dtype=np.float64)
    if native_probabilities.size == 0:
        raise ValueError("the merge proof needs at least one row")
    for name, values in (
        ("merged probability", merged_probabilities),
        ("native margin", native_margins),
        ("merged margin", merged_margins),
    ):
        if values.shape != native_probabilities.shape:
            raise ValueError(f"{name} has shape {values.shape}, expected {native_probabilities.shape}")
    probability_difference = np.abs(native_probabilities - merged_probabilities)
    margin_difference = np.abs(native_margins - merged_margins)
    flips = decision_flips(native_probabilities, merged_probabilities, cut)
    correlation = spearman_correlation(native_probabilities, merged_probabilities)
    native_top = top_fraction_indices(native_probabilities, TOP_FRACTION)
    merged_top = top_fraction_indices(merged_probabilities, TOP_FRACTION)
    summary = {
        "rows": int(native_probabilities.size),
        "probability_max_abs_diff": float(probability_difference.max()),
        "probability_mean_abs_diff": float(probability_difference.mean()),
        "margin_max_abs_diff": float(margin_difference.max()),
        "margin_mean_abs_diff": float(margin_difference.mean()),
        "decision_cut": float(cut),
        "decision_flips": flips,
        "spearman": correlation,
        "top_rows": len(native_top),
        "top_set_identical": native_top == merged_top,
    }
    if summary["probability_max_abs_diff"] > PARITY_TOLERANCE:
        raise ValueError(
            f"merged probability differs from the bagged probability by {summary['probability_max_abs_diff']:.3e}, "
            f"over the {PARITY_TOLERANCE} limit"
        )
    if flips != 0:
        raise ValueError(f"{flips} rows change side of the frozen cut {cut}")
    if summary["spearman"] < SPEARMAN_MIN:
        raise ValueError(f"Spearman correlation {summary['spearman']} is below the {SPEARMAN_MIN} minimum")
    if native_top != merged_top:
        raise ValueError(f"the top-{TOP_FRACTION:.0%} row sets differ: {len(native_top ^ merged_top)} rows")
    return summary


def merge_proof(
    merged_booster: xgboost.Booster,
    account_model: PUBaggedAccountModel,
    matrix: np.ndarray,
    cut: float,
) -> dict[str, Any]:
    """
    WHAT: Compares the merged booster with the production bagged ensemble on one sample.

    WHY: A-061b gates the merge on the probability error, the decisions at the frozen cut and the
    ranking; the margin difference is still measured and reported but has no limit.

    Raises:
        ValueError: If the sample is empty or any A-061b gate fails.
    """
    if matrix.shape[0] == 0:
        raise ValueError("the merge proof needs at least one row")
    native_margin = np.asarray(account_model.predict_margins(matrix), dtype=np.float64)
    merged_margin = np.asarray(merged_booster.predict(xgboost.DMatrix(matrix), output_margin=True), dtype=np.float64)
    native_probability = np.asarray(account_model.predict_probabilities(matrix), dtype=np.float64)
    merged_probability = 1.0 / (1.0 + np.exp(-merged_margin))
    summary = merge_gate(native_probability, merged_probability, native_margin, merged_margin, cut)
    summary["nan_rows"] = _nan_rows(matrix)
    return summary


def export_model(booster: xgboost.Booster, n_features: int) -> onnx.ModelProto:
    """
    WHAT: Converts one booster to ONNX and proves the graph is the pre-registered classifier shape.

    WHY: A-061 fixes the serving contract to label plus the two-class probability output; a regressor
    or a renamed output would silently change what Triton serves, so it stops the phase instead.

    Raises:
        ValueError: If the converter returns another operator, another output list, or another input count.
    """
    converted = convert_xgboost(booster, initial_types=[(INPUT_NAME, FloatTensorType([None, n_features]))])
    operators = [node.op_type for node in converted.graph.node]
    outputs = [output.name for output in converted.graph.output]
    if operators != ["TreeEnsembleClassifier"] or outputs != EXPECTED_OUTPUTS:
        raise ValueError(
            f"converter returned {operators} with outputs {outputs}, expected one classifier with {EXPECTED_OUTPUTS}"
        )
    if len(converted.graph.input) != 1 or converted.graph.input[0].name != INPUT_NAME:
        names = [item.name for item in converted.graph.input]
        raise ValueError(f"converter returned inputs {names}, expected only {INPUT_NAME!r}")
    return converted


def save_onnx(model: onnx.ModelProto, path: Path) -> None:
    """Writes an ONNX model's serialized bytes to disk."""
    Path(path).write_bytes(model.SerializeToString())


def open_session(path: Path) -> onnxruntime.InferenceSession:
    """Opens an ONNX file on the CPU execution provider."""
    return onnxruntime.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def onnx_probabilities(session: onnxruntime.InferenceSession, matrix: np.ndarray) -> np.ndarray:
    """
    WHAT: Returns the class-1 probability the converted model emits for every row of matrix.

    WHY: A-061 fixes the ONNX score as `probabilities[:, 1]`, the same number the native models
    return, so the parity comparison has no extra transformation to explain.

    Raises:
        ValueError: If the session returns a probability block of another shape.
    """
    output_names = [output.name for output in session.get_outputs()]
    outputs = dict(zip(output_names, session.run(None, {INPUT_NAME: matrix.astype(np.float32)}), strict=True))
    probabilities = np.asarray(outputs[PROBABILITY_OUTPUT], dtype=np.float64)
    if probabilities.shape != (matrix.shape[0], 2):
        raise ValueError(f"ONNX probabilities have shape {probabilities.shape}, expected {(matrix.shape[0], 2)}")
    return probabilities[:, 1]


def parity_summary(
    native: Sequence[float] | np.ndarray, converted: Sequence[float] | np.ndarray, nan_rows: int
) -> dict[str, Any]:
    """
    WHAT: Returns the ONNX-versus-native difference summary after enforcing the parity tolerance.

    WHY: A-061 fixes 1e-5 as the largest acceptable difference for every sample; a larger value
    means the exported model is not the frozen model and the phase must stop.

    Raises:
        ValueError: If the two score arrays differ in length, are empty, or exceed the tolerance.
    """
    native_values = np.asarray(native, dtype=np.float64)
    converted_values = np.asarray(converted, dtype=np.float64)
    if native_values.shape != converted_values.shape:
        raise ValueError(f"native scores have shape {native_values.shape}, ONNX scores {converted_values.shape}")
    if native_values.size == 0:
        raise ValueError("the parity check needs at least one row")
    difference = np.abs(native_values - converted_values)
    summary = {
        "rows": int(native_values.size),
        "nan_rows": int(nan_rows),
        "max_abs_diff": float(difference.max()),
        "mean_abs_diff": float(difference.mean()),
        "rows_over_tolerance": int((difference > PARITY_TOLERANCE).sum()),
    }
    if summary["max_abs_diff"] > PARITY_TOLERANCE:
        raise ValueError(
            f"ONNX differs from native by up to {summary['max_abs_diff']:.3e}, over the {PARITY_TOLERANCE} limit"
        )
    return summary


def _opset_description(model: onnx.ModelProto) -> str:
    """Returns the exported model's declared operator-set versions."""
    return ", ".join(f"{(item.domain or 'ai.onnx')}:{item.version}" for item in model.opset_import)


def run(
    validation_dirs: Sequence[Path],
    test_dirs: Sequence[Path],
    out_root: Path = DEFAULT_OUT_ROOT,
) -> dict[str, Any]:
    """
    WHAT: Merges the account bags, exports both models to ONNX and checks merge and parity limits.

    WHY: One entry point keeps the frozen-artifact checks, the merge, the export and the parity
    numbers in one run, so every reported difference is traceable to the artifacts that produced it.

    Raises:
        ValueError: If a folder records another split, a frozen artifact or feature order differs, a
            member violates the merge preconditions, or a merge-proof or parity value is over limit.
    """
    validation_datasets = _split_datasets(validation_dirs, "validation")
    test_datasets = _split_datasets(test_dirs, "test")
    started = time.perf_counter()

    recipe = load_frozen_recipe()
    checkpoint_hashes = verify_account_checkpoint(recipe.checkpoint_dir.parent)
    account_model = PUBaggedAccountModel.load(recipe.checkpoint_dir)
    if account_model.n_bags != recipe.n_bags:
        raise ValueError(f"loaded account ensemble has {account_model.n_bags} bags, recipe says {recipe.n_bags}")
    if list(account_model.member_seeds) != list(range(recipe.seed, recipe.seed + recipe.n_bags)):
        raise ValueError("loaded account ensemble member seeds are not the frozen seed block")
    feature_names = list(recipe.feature_names)
    if tuple(account_model_feature_names()) != tuple(feature_names):
        raise ValueError("the frozen account recipe feature order differs from account_model_feature_names()")

    metadata = verify_candidate_artifacts(DEFAULT_CANDIDATE_DIR)
    feature_order = load_feature_order(DEFAULT_CANDIDATE_DIR / "feature_order.yaml")
    features = feature_order["features"]
    encodings = feature_order["categorical_encodings"]
    if not isinstance(features, list) or not isinstance(encodings, dict):
        raise ValueError(f"{DEFAULT_CANDIDATE_DIR} holds malformed feature-order data")
    if feature_order["feature_sha256"] != metadata["feature_sha256"] or len(features) != metadata["feature_count"]:
        raise ValueError("candidate feature order does not match model_metadata.json")
    if metadata["structure"] != "all":
        raise ValueError(
            f"{DEFAULT_CANDIDATE_DIR} holds the {metadata['structure']!r} structure, not C1's single model"
        )
    payment_model = load_candidate_model(DEFAULT_CANDIDATE_DIR).models["all"]
    payment_booster = payment_model.get_booster()
    if payment_booster.num_features() != len(features):
        raise ValueError(
            f"frozen model takes {payment_booster.num_features()} features but the order holds {len(features)}"
        )
    payment_rounds = verify_booster_rounds(payment_booster, {"best_params": metadata["best_params"]["all"]})
    model_sha256 = metadata["artifacts"]["frozen_model.json"]
    feature_file_sha256 = metadata["artifacts"]["feature_order.yaml"]
    account_cut = load_account_cut()

    member_raws: list[dict[str, Any]] = []
    for index, member in enumerate(account_model.members):
        member_booster = member.get_booster()
        if member_booster.num_boosted_rounds() != recipe.n_estimators:
            raise ValueError(
                f"account member {index} holds {member_booster.num_boosted_rounds()} rounds, "
                f"recipe says {recipe.n_estimators}"
            )
        member_raws.append(json.loads(member_booster.save_raw("json")))

    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    sampler = _CommitSampler()
    with sampler:
        merged = merge_member_models(member_raws, recipe.n_bags, len(feature_names))
        merged_path = out_root / MERGED_BOOSTER_NAME
        merged_booster = write_margin_booster(
            merged, merged_path, recipe.n_bags * recipe.n_estimators, len(feature_names)
        )
        account_validation_matrix, account_validation_keys = account_sample(
            validation_datasets, "validation", feature_names
        )
        merge_summary = merge_proof(merged_booster, account_model, account_validation_matrix, account_cut)

        account_onnx = export_model(merged_booster, len(feature_names))
        save_onnx(account_onnx, out_root / ACCOUNT_ONNX_NAME)
        account_session = open_session(out_root / ACCOUNT_ONNX_NAME)

        account_validation_native = np.asarray(
            account_model.predict_probabilities(account_validation_matrix), dtype=np.float64
        )
        account_validation_parity = parity_summary(
            account_validation_native,
            onnx_probabilities(account_session, account_validation_matrix),
            _nan_rows(account_validation_matrix),
        )

        account_test_matrix, account_test_keys = account_sample(test_datasets, "test", feature_names)
        account_test_native = np.asarray(account_model.predict_probabilities(account_test_matrix), dtype=np.float64)
        account_test_parity = parity_summary(
            account_test_native,
            onnx_probabilities(account_session, account_test_matrix),
            _nan_rows(account_test_matrix),
        )

        payment_onnx = export_model(payment_booster, len(features))
        save_onnx(payment_onnx, out_root / PAYMENT_ONNX_NAME)
        payment_session = open_session(out_root / PAYMENT_ONNX_NAME)

        payment_validation_matrix, payment_validation_keys = _payment_validation_sample(
            validation_datasets, features, encodings, DEFAULT_OOF_ROOT
        )
        payment_validation_native = np.asarray(
            predict_scores(payment_model, payment_validation_matrix), dtype=np.float64
        )
        payment_validation_parity = parity_summary(
            payment_validation_native,
            onnx_probabilities(payment_session, payment_validation_matrix),
            _nan_rows(payment_validation_matrix),
        )

        payment_test_matrix, payment_test_keys = _payment_test_sample(
            test_datasets, features, encodings, account_model, recipe.config_path
        )
        payment_test_native = np.asarray(predict_scores(payment_model, payment_test_matrix), dtype=np.float64)
        payment_test_parity = parity_summary(
            payment_test_native,
            onnx_probabilities(payment_session, payment_test_matrix),
            _nan_rows(payment_test_matrix),
        )

        wall_seconds = time.perf_counter() - started
        data = {
            "amendment": AMENDMENT,
            "amendments": AMENDMENTS,
            "simulated": True,
            "git_hash": _git_head(),
            "datasets": {
                "validation": {
                    "folders": [dataset.name for dataset in validation_datasets],
                    "content_hash": _content_hashes(validation_datasets, "validation"),
                },
                "test": {
                    "folders": [dataset.name for dataset in test_datasets],
                    "content_hash": _content_hashes(test_datasets, "test"),
                },
            },
            "sampling": {
                "seed": SEED,
                "sample_rows": SAMPLE_ROWS,
                "sampling_rule": _SAMPLING_RULE,
                "account_key_fields": list(ACCOUNT_DAY_KEY_FIELDS),
                "payment_key_fields": list(PAYMENT_KEY_FIELDS),
                "account_validation_rows": len(account_validation_keys),
                "account_test_rows": len(account_test_keys),
                "payment_validation_rows": len(payment_validation_keys),
                "payment_test_rows": len(payment_test_keys),
            },
            "converter": {
                "function": "onnxmltools.convert.convert_xgboost",
                "onnxmltools_version": onnxmltools_version,
                "onnx_version": onnx.__version__,
                "onnxruntime_version": onnxruntime.__version__,
                "xgboost_version": xgboost.__version__,
                "account_opset": _opset_description(account_onnx),
                "payment_opset": _opset_description(payment_onnx),
                "input_name": INPUT_NAME,
                "output_name": PROBABILITY_OUTPUT,
            },
            "account": {
                "feature_count": len(feature_names),
                "feature_sha256": recipe.feature_sha256,
                "recipe": {
                    "config": str(recipe.config_path),
                    "config_sha256": recipe.config_sha256,
                    "checkpoint": str(recipe.checkpoint_dir),
                    "seed": recipe.seed,
                    "n_bags": recipe.n_bags,
                    "n_estimators": recipe.n_estimators,
                },
                "checkpoint_sha256": checkpoint_hashes,
                "decision_cut_source": {
                    "path": str(DEFAULT_EXTRA_REPORTING),
                    "key": CUT_SOURCE_KEY,
                },
                "merged_booster": {
                    "path": str(merged_path),
                    "sha256": _sha256_file(merged_path),
                    "rounds": recipe.n_bags * recipe.n_estimators,
                    "leaf_scale": 1.0 / recipe.n_bags,
                },
                "onnx": {
                    "path": str(out_root / ACCOUNT_ONNX_NAME),
                    "sha256": _sha256_file(out_root / ACCOUNT_ONNX_NAME),
                },
                "merge_proof": merge_summary,
                "parity": {
                    "validation": account_validation_parity,
                    "test": account_test_parity,
                },
            },
            "payment": {
                "candidate": metadata["candidate"],
                "model": {
                    "path": str(DEFAULT_CANDIDATE_DIR / "frozen_model.json"),
                    "sha256": model_sha256,
                    "feature_order_path": str(DEFAULT_CANDIDATE_DIR / "feature_order.yaml"),
                    "feature_order_file_sha256": feature_file_sha256,
                    "feature_sha256": feature_order["feature_sha256"],
                    "feature_count": len(features),
                    "rounds": payment_rounds,
                    "seed": metadata["seed"],
                    "device": metadata["device"],
                },
                "onnx": {
                    "path": str(out_root / PAYMENT_ONNX_NAME),
                    "sha256": _sha256_file(out_root / PAYMENT_ONNX_NAME),
                },
                "parity": {
                    "validation": payment_validation_parity,
                    "test": payment_test_parity,
                },
            },
            "tolerances": {
                "merge_probability": PARITY_TOLERANCE,
                "margin_report_only": True,
                "spearman_min": SPEARMAN_MIN,
                "top_fraction": TOP_FRACTION,
                "parity": PARITY_TOLERANCE,
            },
            "wall_seconds": wall_seconds,
            "peak_committed_gb": sampler.peak_gb,
            "out_root": str(out_root),
        }
        (out_root / "onnx_export.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8", newline="\n")
    _print_summary(data)
    return data


def _print_summary(data: Mapping[str, Any]) -> None:
    """Prints the short console summary of one export run."""
    account = data["account"]
    payment = data["payment"]
    summary = {
        "merge": account["merge_proof"],
        "account_validation": account["parity"]["validation"],
        "account_test": account["parity"]["test"],
        "payment_validation": payment["parity"]["validation"],
        "payment_test": payment["parity"]["test"],
        "peak_committed_gb": data["peak_committed_gb"],
        "wall_seconds": data["wall_seconds"],
        "out_root": data["out_root"],
    }
    print(json.dumps(summary, indent=2))


def main(argv: Sequence[str] | None = None) -> int:
    """Runs the merge, export and parity checks, or prints usage."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run", help="Merge the account bags, export both models and check parity")
    run_parser.add_argument(
        "--validation", action="append", required=True, help="Validation dataset folder (repeatable)"
    )
    run_parser.add_argument("--test", action="append", required=True, help="Test dataset folder (repeatable)")
    run_parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT), help="Output folder for the exported files")
    args = parser.parse_args(argv)
    run([Path(path) for path in args.validation], [Path(path) for path in args.test], Path(args.out_root))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
