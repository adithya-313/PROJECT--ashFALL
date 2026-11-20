"""
WHAT: Runs the Level 18 full-scale account model in four resumable stages: a learning curve over
training-world counts, a hyperparameter tune of each PU method, a PU-method choice on validation
truth, and the existing pilot pipeline on the chosen method.

WHY: The full-scale run must decide how many training worlds are enough before it decides which
model to ship, and every decision must be visible as a file with the fingerprint of its inputs so a
crash or a config change can never silently reuse a stale stage.

TERM: stage: one of learning_curve, tune_each_pu_method, choose_pu_method and
final_validation_and_test; each writes <out-root>/stages/<stage>.json.
TERM: fingerprint: the code version, model config sha256, dataset ids and world counts a stage ran
with; a stage file whose fingerprint differs from the current inputs is refused, not reused.

Validation is scored once per stage and test is scored only in stage 4. Every stage records its
seconds and resident memory. Missing or malformed inputs raise a plain ValueError naming the path.
"""

import argparse
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy
import orjson
import psutil
import xgboost
import yaml

from decisions.anomaly_config import load_anomaly_detector_config
from decisions.mule_account_model import (
    MuleModelConfig,
    active_mule_model_config_path,
    load_mule_model_config,
    predict_account_days,
    set_mule_model_config_path,
    train_account_model,
)
from decisions.pu_learning import PUBaggedAccountModel, train_elkan_noto_model, train_pu_bagged_model
from decisions.rules_config import (
    RulesConfig,
    load_rules_config,
    load_rules_config_with_labels,
)
from evaluation.compact_rows import CompactDayRows, CompactRows, load_compact_split
from evaluation.hyperparameter_search import run_optuna_search, validation_truth_average_precision
from evaluation.level17_scoring import ScoredDay
from evaluation.operating_points import load_harm_budgets
from evaluation.rules_tuning import (
    TuningConfig,
    load_labels_tuning_config,
    load_tuning_config,
    tune_anomaly_threshold,
)
from evaluation.run_level_17 import (
    _sha256_of_file,
    run_key,
)
from evaluation.run_level_18_pilot import (
    N_ESTIMATOR_CANDIDATES,
    _compact_row_keys,
    _optuna_requirements,
    _validation_scored_in_model_row_order,
    _xgb_hyperparameters,
    run_pilot,
)
from evaluation.trivial_world import average_precision
from showcase.framework import get_code_version

__all__ = ["main", "run_full"]

#: Boosting rounds the learning-curve probe holds fixed, so only the world count varies.
LEARNING_CURVE_N_ESTIMATORS = 100
#: The training-world count doubles until a doubling gains less than this truth AP.
LEARNING_CURVE_FLAT_IMPROVEMENT = 0.01
#: PU methods scored in stage 2, in the order stage 3 prefers them on an exact tie.
PU_METHODS: tuple[str, ...] = ("none", "elkan_noto", "bagging")
#: Stage 1 variants in the order a tie prefers them: keep every day, keep every day with trust
#: weights, then drop the committed window. The tie tolerance itself lives in the model config.
MATURITY_VARIANT_PREFERENCE: tuple[str, ...] = ("all_days", "all_days_trust_weighted", "drop_28")
#: Order stage 1 trains the variants in; the committed variant first, as the two-way check did.
MATURITY_VARIANT_TRAINING_ORDER: tuple[str, ...] = ("drop_28", "all_days", "all_days_trust_weighted")

STAGE_NAMES: tuple[str, ...] = (
    "maturity_check",
    "learning_curve",
    "tune_each_pu_method",
    "choose_pu_method",
    "final_validation_and_test",
)


def _read_dataset_manifest(dataset_dir: Path, split: str) -> dict[str, Any]:
    """
    WHAT: Reads one dataset's manifest and checks it belongs to the expected split.

    WHY: The driver only needs the manifest's worlds and identity before any parquet is read, but a
    folder passed under the wrong split would still train on or score the wrong worlds, so the
    recorded split and the superseded marker are checked here exactly as the loaders would.

    Args:
        dataset_dir: Dataset folder written by write_dataset.
        split: Expected split: training, validation or test.

    Returns:
        The parsed manifest mapping.

    Raises:
        ValueError: If the manifest is missing or malformed, records another split, or names a
            successor in superseded_by.
    """
    manifest_path = dataset_dir / "manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"Dataset manifest is missing: {manifest_path}")
    manifest = orjson.loads(manifest_path.read_bytes())
    if not isinstance(manifest, dict):
        raise ValueError(f"Dataset manifest {manifest_path} must hold a mapping")
    recorded_split = manifest.get("split")
    if recorded_split != split:
        raise ValueError(f"Dataset {dataset_dir} records split {recorded_split!r}, but it was passed as {split!r}")
    successor = manifest.get("superseded_by")
    if successor:
        raise ValueError(f"Dataset {dataset_dir} is superseded by {successor}")
    worlds = manifest.get("worlds")
    if not isinstance(worlds, list) or not worlds:
        raise ValueError(f"Dataset manifest {manifest_path} must list at least one world")
    return manifest


def _split_world_ids(split: str, dataset_dirs: Sequence[Path]) -> list[str]:
    """
    WHAT: Returns the sorted world ids of one split's datasets, refusing duplicates.

    WHY: A training subset is the first N worlds in sorted order, so every stage and every resume
    must agree on that order; a world repeated across datasets would otherwise be counted twice.

    Args:
        split: Expected split of every folder.
        dataset_dirs: Dataset folders of that split.

    Returns:
        Unique world ids in sorted order.

    Raises:
        ValueError: If a manifest is malformed or the same world id appears twice.
    """
    world_ids: list[str] = []
    for dataset_dir in dataset_dirs:
        manifest = _read_dataset_manifest(dataset_dir, split)
        for world in manifest["worlds"]:
            world_id = world.get("world_id") if isinstance(world, dict) else None
            if not isinstance(world_id, str) or not world_id:
                raise ValueError(f"Dataset {dataset_dir} holds a world without a string world_id")
            world_ids.append(world_id)
    if len(set(world_ids)) != len(world_ids):
        duplicates = sorted({world_id for world_id in world_ids if world_ids.count(world_id) > 1})
        raise ValueError(f"World ids repeat across the {split} datasets: {duplicates}")
    return sorted(world_ids)


def _dataset_ids(dataset_dirs: Sequence[Path]) -> list[str]:
    """Returns the dataset folder names, the identity run_key hashes."""
    return [dataset_dir.name for dataset_dir in dataset_dirs]


def _parse_training_world_counts(text: str) -> tuple[int, ...]:
    """
    WHAT: Parses the comma-separated --training-world-counts value into ascending distinct counts.

    WHY: The learning curve doubles the count, so the list must be non-empty, positive and strictly
    increasing; anything else is a typed-in mistake the run must refuse before training.

    Args:
        text: Comma-separated integers, for example "40,80,160".

    Returns:
        Counts in ascending order.

    Raises:
        ValueError: If a value is not a positive integer, a count repeats, or the list is empty.
    """
    counts: list[int] = []
    for piece in text.split(","):
        stripped = piece.strip()
        if not stripped:
            raise ValueError(f"--training-world-counts holds an empty value in {text!r}")
        try:
            count = int(stripped)
        except ValueError as error:
            raise ValueError(f"--training-world-counts value {stripped!r} is not an integer") from error
        if count <= 0:
            raise ValueError(f"--training-world-counts value {count} must be positive")
        counts.append(count)
    if not counts:
        raise ValueError("--training-world-counts must hold at least one count")
    if len(set(counts)) != len(counts):
        raise ValueError(f"--training-world-counts holds a repeated count in {text!r}")
    return tuple(sorted(counts))


def _inputs_fingerprint(
    config_path: Path,
    training_dataset_ids: Sequence[str],
    validation_dataset_ids: Sequence[str],
    test_dataset_ids: Sequence[str],
    stress_dataset_ids: Sequence[str],
    training_world_counts: Sequence[int],
) -> dict[str, Any]:
    """
    WHAT: Builds the record of inputs a stage result is valid for.

    WHY: Resume may only reuse a stage whose code, config, datasets and world counts are identical;
    storing them next to the result makes a mismatched resume refuse instead of mixing results.

    Args:
        config_path: Active mule model config file.
        training_dataset_ids: Training dataset folder names, in flag order.
        validation_dataset_ids: Validation dataset folder names, in flag order.
        test_dataset_ids: Test dataset folder names, in flag order.
        stress_dataset_ids: Stress dataset folder names, in flag order; empty when none is passed.
        training_world_counts: World counts the learning curve will try.

    Returns:
        The fingerprint mapping written into every stage file.
    """
    return {
        "code_version": get_code_version(),
        "mule_model_config_sha256": _sha256_of_file(config_path),
        "training_dataset_ids": list(training_dataset_ids),
        "validation_dataset_ids": list(validation_dataset_ids),
        "test_dataset_ids": list(test_dataset_ids),
        "stress_dataset_ids": list(stress_dataset_ids),
        "training_world_counts": list(training_world_counts),
    }


def _read_matching_stage(
    path: Path,
    name: str,
    fingerprint: dict[str, Any],
) -> dict[str, Any] | None:
    """
    WHAT: Returns a finished stage's result when its fingerprint matches, or None when it is absent.

    WHY: A stage file with different inputs is not merely stale but dangerous, so it is refused with
    the changed keys named rather than overwritten or silently reused.

    Args:
        path: Stage JSON file.
        name: Stage name, used in error messages.
        fingerprint: Current inputs.

    Returns:
        The recorded result, or None when the file does not exist.

    Raises:
        ValueError: If the file is malformed or its fingerprint differs from the current inputs.
    """
    if not path.is_file():
        return None
    payload = orjson.loads(path.read_bytes())
    if not isinstance(payload, dict) or "fingerprint" not in payload or "result" not in payload:
        raise ValueError(f"Stage file {path} must hold a fingerprint and a result")
    recorded = payload["fingerprint"]
    if recorded != fingerprint:
        keys = sorted(
            key
            for key in set(recorded) | set(fingerprint)
            if not isinstance(recorded, dict) or recorded.get(key) != fingerprint.get(key)
        )
        raise ValueError(
            f"Stage {name!r} at {path} was run with different inputs ({', '.join(keys)}); "
            "move that file away to rerun the stage"
        )
    return payload["result"]


def _write_stage(
    path: Path,
    name: str,
    fingerprint: dict[str, Any],
    result: dict[str, Any],
    started: float,
) -> None:
    """
    WHAT: Writes one stage's fingerprint, result, seconds and resident memory as JSON.

    WHY: Every stage must be resumable and auditable from its file alone, including how long it took
    and how much memory it held, because the full run must fit the laptop's memory budget.

    Args:
        path: Destination stage file.
        name: Stage name.
        fingerprint: Inputs the result is valid for.
        result: Stage-specific result mapping.
        started: time.perf_counter() value taken when the stage began.
    """
    payload = {
        "stage": name,
        "fingerprint": fingerprint,
        "seconds": time.perf_counter() - started,
        "peak_rss_mb": psutil.Process().memory_info().rss / (1024.0 * 1024.0),
        "result": result,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(orjson.dumps(payload, option=orjson.OPT_SORT_KEYS | orjson.OPT_INDENT_2) + b"\n")


def _run_stage(
    name: str,
    directory: Path,
    fingerprint: dict[str, Any],
    compute: Any,
) -> dict[str, Any]:
    """
    WHAT: Returns a stage's result from its file when resumable, otherwise computes and writes it.

    WHY: One place owns the skip-or-run decision, so every stage prints the same skip line and no
    stage can quietly run on top of an incompatible previous result.

    Args:
        name: Stage name.
        directory: Folder holding the stage files.
        fingerprint: Current inputs.
        compute: Zero-argument callable returning the stage result mapping.

    Returns:
        The stage result.

    Side effects:
        Prints a SKIP line for a resumable stage or a DONE line with its seconds otherwise.
    """
    path = directory / f"{name}.json"
    existing = _read_matching_stage(path, name, fingerprint)
    if existing is not None:
        print(f"SKIP stage {name}: fingerprint matches {path}")
        return existing
    started = time.perf_counter()
    result = compute()
    seconds = time.perf_counter() - started
    _write_stage(path, name, fingerprint, result, started)
    print(f"DONE stage {name}: {seconds:.1f} s -> {path}")
    return result


def _write_maturity_config(
    source: Path,
    destination: Path,
    maturity_days: int,
    label_trust_weighting: bool,
) -> None:
    """
    WHAT: Writes a copy of the model config whose label_maturity_days and label_trust_weighting
    values are the given ones.

    WHY: The compact loader reads both switches from the process-wide config, so the check must
    hand it one file per variant instead of editing the committed config, and the chosen variant's
    file must carry both values so every later stage trains under that variant.

    Args:
        source: Active config file the run started from.
        destination: Copy to write, inside the stage folder.
        maturity_days: Day count to store; 0 drops nothing.
        label_trust_weighting: Trust weighting switch to store.

    Raises:
        ValueError: If the config is not a mapping or lacks the label_maturity_days or
            label_trust_weighting block.
    """
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Model config {source} must hold a mapping at the root")
    parameters = raw.get("parameters")
    if not isinstance(parameters, dict):
        raise ValueError(f"Model config {source} must hold a parameters mapping")
    maturity_block = parameters.get("label_maturity_days")
    if not isinstance(maturity_block, dict):
        raise ValueError(f"Model config {source} must hold a label_maturity_days block")
    weighting_block = parameters.get("label_trust_weighting")
    if not isinstance(weighting_block, dict):
        raise ValueError(f"Model config {source} must hold a label_trust_weighting block")
    maturity_block["value"] = maturity_days
    maturity_block.pop("not_found", None)
    weighting_block["value"] = label_trust_weighting
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")


def _maturity_decision(variant_scores: Mapping[str, float], tolerance: float) -> tuple[str, str]:
    """
    WHAT: Applies the pre-registered three-way rule to the variants' validation truth APs.

    WHY: Dropping the final days of every world discards many rows, so the simpler variants that keep
    that data win unless the drop is clearly better: every variant whose AP is at least the best AP
    minus the tolerance is tied (a gap of exactly the tolerance is a tie, so a variant must lead by
    MORE than the tolerance to beat the others outright), and a tie goes to the first variant in
    MATURITY_VARIANT_PREFERENCE.

    Args:
        variant_scores: Validation truth AP by variant name; must hold exactly the preference names.
        tolerance: Largest AP gap treated as noise.

    Returns:
        (chosen variant name, one-sentence reason).

    Raises:
        ValueError: If the names differ from the preference list or the tolerance is not positive.
    """
    if set(variant_scores) != set(MATURITY_VARIANT_PREFERENCE):
        raise ValueError(
            f"Variant scores must cover exactly {list(MATURITY_VARIANT_PREFERENCE)}; got {sorted(variant_scores)}"
        )
    if tolerance <= 0.0:
        raise ValueError(f"The tie tolerance must be positive; got {tolerance}")
    best_name = max(variant_scores, key=lambda name: variant_scores[name])
    best_ap = variant_scores[best_name]
    tied_names = [name for name in MATURITY_VARIANT_PREFERENCE if variant_scores[name] >= best_ap - tolerance]
    chosen = tied_names[0]
    return chosen, (
        f"{chosen} chosen with validation truth AP {variant_scores[chosen]:.4f}: the best AP was "
        f"{best_ap:.4f} ({best_name}), and {len(tied_names)} variant(s) were within the tie tolerance "
        f"of {tolerance} of it, resolved by preferring {' > '.join(MATURITY_VARIANT_PREFERENCE)}."
    )


def _variant_settings(variant_name: str, committed_days: int) -> tuple[int, bool]:
    """
    WHAT: Returns (label_maturity_days, label_trust_weighting) for one named maturity variant.

    Raises:
        ValueError: If the name is not one of MATURITY_VARIANT_PREFERENCE.
    """
    if variant_name == "drop_28":
        return committed_days, False
    if variant_name == "all_days":
        return 0, False
    if variant_name == "all_days_trust_weighted":
        return 0, True
    raise ValueError(f"Unknown maturity variant: {variant_name!r}")


def _write_chosen_config(source: Path, destination: Path, maturity: Mapping[str, Any]) -> None:
    """
    WHAT: Writes the config copy every later stage trains under, carrying the chosen variant's
    label_maturity_days and label_trust_weighting.

    WHY: The two values come from the chosen variant's own record, so the copy can never hold one
    variant's window with another variant's weighting.

    Args:
        source: Active config file the run started from.
        destination: Chosen config copy to write.
        maturity: The maturity_check stage record.

    Raises:
        ValueError: If the record's chosen_variant has no entry in its variants list.
    """
    chosen_name = maturity["chosen_variant"]
    for variant in maturity["variants"]:
        if variant["name"] == chosen_name:
            _write_maturity_config(
                source,
                destination,
                int(variant["label_maturity_days"]),
                bool(variant["label_trust_weighting"]),
            )
            return
    raise ValueError(f"The maturity record has no variant named {chosen_name!r}")


def _maturity_check(
    training_dirs: Sequence[Path],
    training_world_ids: Sequence[str],
    validation_dirs: Sequence[Path],
    truth_root: Path,
    base_rules_config: RulesConfig,
    seed: int,
    mule_config: MuleModelConfig,
    out_root: Path,
) -> dict[str, Any]:
    """
    WHAT: Trains the fixed probe model under each of the three label-handling variants and picks one
    on validation truth.

    WHY: The measured p90 is a lower bound because most first-credit accounts are censored, and the
    28-day cut removes a large share of the training rows, so the cut must be shown to beat keeping
    every day, with or without trust weighting, before every later stage trains under a variant.

    Args:
        training_dirs: Training dataset folders.
        training_world_ids: Every training world id, sorted.
        validation_dirs: Validation dataset folders.
        truth_root: Root folder holding one truth folder per run_id.
        base_rules_config: Rules table the compact loader needs to build ScoredDays.
        seed: Model random seed.
        mule_config: Loaded model config holding the committed window and the tie tolerance.
        out_root: Driver output root; the per-variant config copies land under its stages folder.

    Returns:
        The variant records (name, label_maturity_days, label_trust_weighting, training_row_count,
        training_positive_count, training_weight_sum and validation_truth_average_precision), the
        chosen_variant, the reason and the tolerance. training_weight_sum is the row count when the
        variant is unweighted.

    Raises:
        ValueError: If the committed window is null, a split cannot be loaded, or a variant has no
            positive or negative training row.
    """
    committed_days = mule_config.label_maturity_days
    if committed_days is None:
        raise ValueError(
            "The maturity_check stage needs a committed label_maturity_days value; "
            f"the model config sets {committed_days!r}"
        )
    validation = load_compact_split(
        validation_dirs,
        truth_root,
        include_days=True,
        rules_config=base_rules_config,
    )
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
    source_config = active_mule_model_config_path().resolve()
    previous_config = active_mule_model_config_path()
    variants: list[dict[str, Any]] = []
    for variant_name in MATURITY_VARIANT_TRAINING_ORDER:
        maturity_days, trust_weighting = _variant_settings(variant_name, committed_days)
        candidate_config = out_root / "stages" / f"maturity_variant_{variant_name}.yaml"
        _write_maturity_config(source_config, candidate_config, maturity_days, trust_weighting)
        set_mule_model_config_path(candidate_config)
        try:
            training = load_compact_split(training_dirs, world_ids=training_world_ids)
        finally:
            set_mule_model_config_path(previous_config)
        targets = [int(target) for target in training.rows.targets]
        model, _ = train_account_model(
            training.rows.matrix, targets, seed, LEARNING_CURVE_N_ESTIMATORS, sample_weight=training.rows.sample_weights
        )
        probabilities = predict_account_days(model, validation.days.aligned_features)
        average_precision_truth = validation_truth_average_precision(probabilities, truth_scored)
        row_count = int(training.rows.matrix.shape[0])
        if training.rows.sample_weights is None:
            weight_sum = float(row_count)
        else:
            weight_sum = float(training.rows.sample_weights.sum(dtype=numpy.float64))
        variants.append(
            {
                "name": variant_name,
                "label_maturity_days": maturity_days,
                "label_trust_weighting": trust_weighting,
                "training_row_count": row_count,
                "training_positive_count": sum(targets),
                "training_weight_sum": weight_sum,
                "validation_truth_average_precision": float(average_precision_truth),
            }
        )
        del training, targets, model, probabilities
    chosen_variant, reason = _maturity_decision(
        {variant["name"]: float(variant["validation_truth_average_precision"]) for variant in variants},
        mule_config.maturity_tie_tolerance,
    )
    return {
        "committed_label_maturity_days": committed_days,
        "variants": variants,
        "chosen_variant": chosen_variant,
        "reason": reason,
        "tolerance": mule_config.maturity_tie_tolerance,
    }


def _learning_curve_choice(points: Sequence[dict[str, Any]]) -> tuple[int, bool, list[dict[str, Any]]]:
    """
    WHAT: Applies the doubling rule to the measured learning-curve points.

    WHY: Training more worlds is only worth its memory and time while the validation truth AP keeps
    rising; the smallest count after which a doubling gains less than the flat threshold is the
    chosen size, and a curve that never flattens is flagged rather than silently trusted.

    Args:
        points: One mapping per training-world count, ascending, each holding
            training_world_count and validation_truth_average_precision.

    Returns:
        (chosen world count, more_training_worlds_needed, one improvement record per doubling).

    Raises:
        ValueError: If there are fewer than two points or the counts do not ascend.
    """
    if len(points) < 2:
        raise ValueError("A learning curve needs at least two training-world counts")
    improvements: list[dict[str, Any]] = []
    flattened_at: int | None = None
    for previous, current in zip(points, points[1:], strict=False):
        previous_count = int(previous["training_world_count"])
        current_count = int(current["training_world_count"])
        if current_count != 2 * previous_count:
            continue
        improvement = float(current["validation_truth_average_precision"]) - float(
            previous["validation_truth_average_precision"]
        )
        improvements.append(
            {
                "from_training_worlds": previous_count,
                "to_training_worlds": current_count,
                "average_precision_improvement": improvement,
            }
        )
        if improvement < LEARNING_CURVE_FLAT_IMPROVEMENT and flattened_at is None:
            flattened_at = previous_count
    chosen = flattened_at if flattened_at is not None else int(points[-1]["training_world_count"])
    return chosen, flattened_at is None, improvements


def _learning_curve(
    training_dirs: Sequence[Path],
    training_world_ids: Sequence[str],
    validation_dirs: Sequence[Path],
    truth_root: Path,
    base_rules_config: RulesConfig,
    seed: int,
    training_world_counts: Sequence[int],
) -> dict[str, Any]:
    """
    WHAT: Trains the fixed probe model on each training-world count and scores it on validation
    truth.

    WHY: The world count is a memory and time decision, so it is measured with one fixed, untuned
    model (supervised, no search) and selection is made on hidden truth rather than on labels.

    Args:
        training_dirs: Training dataset folders.
        training_world_ids: Every training world id, sorted.
        validation_dirs: Validation dataset folders.
        truth_root: Root folder holding one truth folder per run_id.
        base_rules_config: Rules table the compact loader needs to build ScoredDays.
        seed: Model random seed.
        training_world_counts: Ascending counts to try.

    Returns:
        Points, doubling improvements, the chosen count and the more-worlds flag.

    Raises:
        ValueError: If a count exceeds the available worlds or a split cannot be loaded.
    """
    if training_world_counts[-1] > len(training_world_ids):
        raise ValueError(
            f"--training-world-counts asks for {training_world_counts[-1]} worlds, "
            f"but the training datasets hold {len(training_world_ids)}"
        )
    validation = load_compact_split(
        validation_dirs,
        truth_root,
        include_days=True,
        rules_config=base_rules_config,
    )
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
    points: list[dict[str, Any]] = []
    for count in training_world_counts:
        training = load_compact_split(training_dirs, world_ids=training_world_ids[:count])
        matrix = training.rows.matrix
        targets = [int(target) for target in training.rows.targets]
        model, _ = train_account_model(
            matrix, targets, seed, LEARNING_CURVE_N_ESTIMATORS, sample_weight=training.rows.sample_weights
        )
        probabilities = predict_account_days(model, validation.days.aligned_features)
        average_precision_truth = validation_truth_average_precision(probabilities, truth_scored)
        points.append(
            {
                "training_world_count": count,
                "training_row_count": int(matrix.shape[0]),
                "validation_truth_average_precision": float(average_precision_truth),
            }
        )
        del training, matrix, targets, model, probabilities
    chosen, more_needed, improvements = _learning_curve_choice(points)
    return {
        "training_world_counts": list(training_world_counts),
        "points": points,
        "doubling_improvements": improvements,
        "chosen_training_world_count": chosen,
        "more_training_worlds_needed": more_needed,
    }


def _select_by_label_average_precision(
    method: str,
    training_matrix: numpy.ndarray,
    training_targets: list[int],
    training_weights: numpy.ndarray | None,
    validation_matrix: numpy.ndarray,
    validation_targets: list[int],
    validation_rows_for_keys: numpy.ndarray,
    mule_config: MuleModelConfig,
    seed: int,
) -> tuple[xgboost.XGBClassifier | PUBaggedAccountModel, dict[str, Any]]:
    """
    WHAT: Reproduces the pilot's estimator-candidate loop for one PU method when the search is off.

    WHY: The off setting must keep today's path exactly: candidates are scored against the
    bank-visible validation labels and the winner is the one that reproduces them best.

    Args:
        method: One of PU_METHODS.
        training_matrix: Training feature rows.
        training_targets: Binary training labels.
        training_weights: Label trust weight per training row, or None for unweighted training.
        validation_matrix: Validation feature rows, needed by Elkan-Noto to estimate c.
        validation_targets: Binary validation labels.
        validation_rows_for_keys: Validation rows or matrix the model predicts on.
        mule_config: Loaded model config holding the PU method's own settings.
        seed: Model random seed.

    Returns:
        (fitted model, chosen parameter record).

    Raises:
        ValueError: If the method is unknown, its PU settings are missing, or no candidate scores.
    """
    best_model: xgboost.XGBClassifier | PUBaggedAccountModel | None = None
    best_ap = -1.0
    chosen: dict[str, Any] = {}
    # It holds a plain model or a bagged ensemble, depending on pu_method.
    model: xgboost.XGBClassifier | PUBaggedAccountModel
    if method == "none":
        for n_estimators in N_ESTIMATOR_CANDIDATES:
            model, _ = train_account_model(
                training_matrix,
                training_targets,
                seed,
                n_estimators=n_estimators,
                sample_weight=training_weights,
            )
            score = average_precision(predict_account_days(model, validation_rows_for_keys), validation_targets)
            if score > best_ap:
                best_ap = score
                best_model = model
                chosen = {"n_estimators": n_estimators}
    elif method == "bagging":
        n_bags = mule_config.pu_bagging_n_bags
        if n_bags is None:
            raise ValueError("pu_bagging_n_bags is required when pu_method is 'bagging'")
        candidates = mule_config.pu_unlabelled_per_positive_candidates
        if candidates is None:
            raise ValueError("pu_unlabelled_per_positive_candidates is required when pu_method is 'bagging'")
        for n_estimators in N_ESTIMATOR_CANDIDATES:
            for unlabelled_per_positive in candidates:
                model = train_pu_bagged_model(
                    training_matrix,
                    training_targets,
                    seed,
                    n_estimators,
                    n_bags,
                    unlabelled_per_positive,
                    sample_weight=training_weights,
                )
                score = average_precision(predict_account_days(model, validation_rows_for_keys), validation_targets)
                if score > best_ap:
                    best_ap = score
                    best_model = model
                    chosen = {
                        "n_estimators": n_estimators,
                        "unlabelled_per_positive": unlabelled_per_positive,
                    }
    elif method == "elkan_noto":
        probability_clip = mule_config.pu_probability_clip
        if probability_clip is None:
            raise ValueError("pu_probability_clip is required when pu_method is 'elkan_noto'")
        for n_estimators in N_ESTIMATOR_CANDIDATES:
            model, _, _, _ = train_elkan_noto_model(
                training_matrix,
                training_targets,
                validation_matrix,
                validation_targets,
                seed,
                n_estimators,
                probability_clip,
                sample_weight=training_weights,
            )
            score = average_precision(predict_account_days(model, validation_rows_for_keys), validation_targets)
            if score > best_ap:
                best_ap = score
                best_model = model
                chosen = {"n_estimators": n_estimators}
    else:
        raise ValueError(f"Unsupported pu_method: {method!r}")
    if best_model is None:
        raise ValueError(f"No {method} candidate produced a validation score")
    return best_model, chosen


def _train_pu_method(
    method: str,
    training_matrix: numpy.ndarray,
    training_targets: list[int],
    training_weights: numpy.ndarray | None,
    validation_rows: CompactRows,
    validation_days: CompactDayRows,
    truth_scored: Sequence[ScoredDay],
    mule_config: MuleModelConfig,
    seed: int,
    search_on: bool,
) -> tuple[
    xgboost.XGBClassifier | PUBaggedAccountModel,
    dict[str, Any],
    list[float],
    dict[str, Any],
]:
    """
    WHAT: Fits one PU method with the search control on or off and returns its model, parameters,
    validation probabilities and search record.

    WHY: Stage 2 must compare the three methods under the same rule, so each method is tuned by the
    seeded truth-AP search when it is on, and by today's label-AP candidate loop when it is off; the
    search record carries every trial and the importance numbers into the stage file.

    Args:
        method: One of PU_METHODS.
        training_matrix: Training feature rows of the chosen world count.
        training_targets: Binary training labels.
        training_weights: Label trust weight per training row, or None for unweighted training.
        validation_rows: Compact model rows carrying the bank-visible validation labels.
        validation_days: Compact account-day arrays aligned with the model rows.
        truth_scored: Validation account-days in model-row order, carrying the hidden truth roles.
        mule_config: Loaded model config.
        seed: Model random seed.
        search_on: Whether hyperparameter_search is optuna.

    Returns:
        (fitted model, chosen parameter record, validation probabilities in model-row order,
        search record). The search record is empty when the search is off; when it is on it holds
        the trials, trials_run, stopped_early, stop_reason, param_importances and best_score.

    Raises:
        ValueError: If the method is unknown or its settings are missing.
    """
    validation_matrix = validation_rows.matrix
    validation_targets = [int(target) for target in validation_rows.targets]
    rows_for_keys = validation_days.aligned_features
    if not search_on:
        model, chosen = _select_by_label_average_precision(
            method,
            training_matrix,
            training_targets,
            training_weights,
            validation_matrix,
            validation_targets,
            rows_for_keys,
            mule_config,
            seed,
        )
        probabilities = predict_account_days(model, rows_for_keys)
        chosen = {**chosen, "search": "off"}
        return model, chosen, probabilities, {}

    space, trial_count, optuna_seed, _, _, _ = _optuna_requirements(mule_config)
    categorical_choices: dict[str, Sequence[int]] | None = None
    n_bags = 0
    probability_clip = 0.0
    if method == "none":

        def train_none_trial(params: dict[str, int | float]) -> Sequence[float]:
            """Trains one supervised candidate and returns its validation probabilities."""
            model, _ = train_account_model(
                training_matrix,
                training_targets,
                seed,
                int(params["n_estimators"]),
                sample_weight=training_weights,
                **_xgb_hyperparameters(params),
            )
            return predict_account_days(model, rows_for_keys)

        train_trial = train_none_trial
    elif method == "bagging":
        configured_bags = mule_config.pu_bagging_n_bags
        if configured_bags is None:
            raise ValueError("pu_bagging_n_bags is required when pu_method is 'bagging'")
        n_bags = configured_bags
        candidates = mule_config.pu_unlabelled_per_positive_candidates
        if candidates is None:
            raise ValueError("pu_unlabelled_per_positive_candidates is required when pu_method is 'bagging'")
        categorical_choices = {"unlabelled_per_positive": candidates}

        def train_bagging_trial(params: dict[str, int | float]) -> Sequence[float]:
            """Trains one bagged candidate and returns its validation probabilities."""
            model = train_pu_bagged_model(
                training_matrix,
                training_targets,
                seed,
                int(params["n_estimators"]),
                n_bags,
                int(params["unlabelled_per_positive"]),
                sample_weight=training_weights,
                **_xgb_hyperparameters(params),
            )
            return predict_account_days(model, rows_for_keys)

        train_trial = train_bagging_trial
    elif method == "elkan_noto":
        configured_clip = mule_config.pu_probability_clip
        if configured_clip is None:
            raise ValueError("pu_probability_clip is required when pu_method is 'elkan_noto'")
        probability_clip = configured_clip

        def train_elkan_trial(params: dict[str, int | float]) -> Sequence[float]:
            """Trains one Elkan-Noto candidate and returns its validation probabilities."""
            model, _, _, _ = train_elkan_noto_model(
                training_matrix,
                training_targets,
                validation_matrix,
                validation_targets,
                seed,
                int(params["n_estimators"]),
                probability_clip,
                sample_weight=training_weights,
                **_xgb_hyperparameters(params),
            )
            return predict_account_days(model, rows_for_keys)

        train_trial = train_elkan_trial
    else:
        raise ValueError(f"Unsupported pu_method: {method!r}")

    search_result = run_optuna_search(
        train_trial,
        truth_scored,
        space,
        trial_count,
        optuna_seed,
        min_trials_before_stop=mule_config.optuna_min_trials_before_stop,
        stall_patience_trials=mule_config.optuna_stall_patience_trials,
        min_improvement=mule_config.optuna_min_improvement,
        categorical_choices=categorical_choices,
    )
    best_params = search_result.best_params
    if method == "none":
        model, _ = train_account_model(
            training_matrix,
            training_targets,
            seed,
            int(best_params["n_estimators"]),
            sample_weight=training_weights,
            **_xgb_hyperparameters(best_params),
        )
    elif method == "bagging":
        model = train_pu_bagged_model(
            training_matrix,
            training_targets,
            seed,
            int(best_params["n_estimators"]),
            n_bags,
            int(best_params["unlabelled_per_positive"]),
            sample_weight=training_weights,
            **_xgb_hyperparameters(best_params),
        )
    else:
        model, _, _, _ = train_elkan_noto_model(
            training_matrix,
            training_targets,
            validation_matrix,
            validation_targets,
            seed,
            int(best_params["n_estimators"]),
            probability_clip,
            sample_weight=training_weights,
            **_xgb_hyperparameters(best_params),
        )
    chosen = {**best_params, "search": "optuna"}
    probabilities = predict_account_days(model, rows_for_keys)
    search_record = {
        "trials": list(search_result.trials),
        "trials_run": search_result.trials_run,
        "stopped_early": search_result.stopped_early,
        "stop_reason": search_result.stop_reason,
        "param_importances": search_result.param_importances,
        "best_score": search_result.best_score,
    }
    return model, chosen, probabilities, search_record


def _tune_each_pu_method(
    training_dirs: Sequence[Path],
    training_world_ids: Sequence[str],
    chosen_training_world_count: int,
    validation_dirs: Sequence[Path],
    truth_root: Path,
    base_rules_config: RulesConfig,
    tuning: TuningConfig,
    main_budget_pct: float,
    mule_config: MuleModelConfig,
    seed: int,
) -> dict[str, Any]:
    """
    WHAT: Trains and scores every PU method on the chosen training-world count and the main budget.

    WHY: The method choice must compare truth AP and truth recall at the same genuine-restriction
    budget the pilot reports, so each method's cut is tuned on validation exactly as the pilot's
    XGBoost cut is.

    Args:
        training_dirs: Training dataset folders.
        training_world_ids: Every training world id, sorted.
        chosen_training_world_count: World count chosen by the learning curve.
        validation_dirs: Validation dataset folders.
        truth_root: Root folder holding one truth folder per run_id.
        base_rules_config: Rules table the compact loader needs.
        tuning: Loaded tuning config that allows validation.
        main_budget_pct: Main genuine-restriction budget the recall is measured at.
        mule_config: Loaded model config.
        seed: Model random seed.

    Returns:
        One method record per PU method, in PU_METHODS order.

    Raises:
        ValueError: If a split cannot be loaded or the chosen count exceeds the available worlds.
    """
    if chosen_training_world_count > len(training_world_ids):
        raise ValueError(
            f"Chosen training-world count {chosen_training_world_count} exceeds "
            f"the {len(training_world_ids)} available worlds"
        )
    training = load_compact_split(training_dirs, world_ids=training_world_ids[:chosen_training_world_count])
    training_matrix = training.rows.matrix
    training_targets = [int(target) for target in training.rows.targets]
    training_weights = training.rows.sample_weights
    validation = load_compact_split(
        validation_dirs,
        truth_root,
        include_days=True,
        rules_config=base_rules_config,
    )
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
    search_on = mule_config.hyperparameter_search == "optuna"
    records: list[dict[str, Any]] = []
    for method in PU_METHODS:
        started = time.perf_counter()
        model, chosen, probabilities, search_record = _train_pu_method(
            method,
            training_matrix,
            training_targets,
            training_weights,
            validation.rows,
            validation.days,
            truth_scored,
            mule_config,
            seed,
            search_on,
        )
        seconds = time.perf_counter() - started
        truth_ap = validation_truth_average_precision(probabilities, truth_scored)
        method_scored = [
            replace(day, anomaly_score=probability)
            for day, probability in zip(truth_scored, probabilities, strict=True)
        ]
        method_tuning = tune_anomaly_threshold(method_scored, tuning, "validation", main_budget_pct)
        records.append(
            {
                "pu_method": method,
                "chosen_params": chosen,
                "validation_truth_average_precision": float(truth_ap),
                "validation_rate_pct": float(method_tuning.validation_rate_pct),
                "validation_recall": float(method_tuning.validation_recall),
                "meets_budget": bool(method_tuning.meets_budget),
                "training_seconds": seconds,
                **search_record,
            }
        )
        del model, probabilities
    return {"training_world_count": chosen_training_world_count, "methods": records}


def _choose_pu_method(method_records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """
    WHAT: Applies the pre-registered PU-method rule to stage 2's records.

    WHY: Choosing the method after seeing the test set would invalidate the result, so the rule is
    fixed: highest truth recall among methods meeting the budget, then higher truth AP, then the
    order none > elkan_noto > bagging; with no method meeting the budget, lowest rate first.

    Args:
        method_records: One record per PU method from stage 2.

    Returns:
        The chosen method and the numbers the rule read.

    Raises:
        ValueError: If no method record is supplied.
    """
    if not method_records:
        raise ValueError("Choosing a PU method needs at least one method record")
    preference = {method: index for index, method in enumerate(PU_METHODS)}
    meeting = [record for record in method_records if record["meets_budget"]]
    if meeting:
        chosen = max(
            meeting,
            key=lambda record: (
                float(record["validation_recall"]),
                float(record["validation_truth_average_precision"]),
                -preference[str(record["pu_method"])],
            ),
        )
        meets_budget = True
    else:
        chosen = min(
            method_records,
            key=lambda record: (
                float(record["validation_rate_pct"]),
                -float(record["validation_truth_average_precision"]),
                preference[str(record["pu_method"])],
            ),
        )
        meets_budget = False
    return {
        "pu_method": chosen["pu_method"],
        "meets_budget": meets_budget,
        "validation_truth_average_precision": chosen["validation_truth_average_precision"],
        "validation_rate_pct": chosen["validation_rate_pct"],
        "validation_recall": chosen["validation_recall"],
    }


def _write_final_config(source: Path, destination: Path, pu_method: str) -> None:
    """
    WHAT: Writes a copy of the model config whose pu_method value is the chosen method.

    WHY: The pilot reads the single process-wide config, so stage 4 must hand it a file that keeps
    every other setting identical while recording the choice; the stage file then hashes that copy.

    Args:
        source: Active config file the run started from.
        destination: Copy to write, inside the stage folder.
        pu_method: Chosen PU method.

    Raises:
        ValueError: If the config is not a mapping with a parameters.pu_method block.
    """
    raw = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Model config {source} must hold a mapping at the root")
    parameters = raw.get("parameters")
    if not isinstance(parameters, dict):
        raise ValueError(f"Model config {source} must hold a parameters mapping")
    block = parameters.get("pu_method")
    if not isinstance(block, dict):
        raise ValueError(f"Model config {source} must hold a pu_method block")
    block["value"] = pu_method
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")


def _final_validation_and_test(
    chosen_method: str,
    training_dirs: Sequence[Path],
    validation_dirs: Sequence[Path],
    test_dirs: Sequence[Path],
    truth_root: Path,
    world_cards_root: Path,
    out_root: Path,
    config_path: Path,
    all_dataset_ids: Sequence[str],
    stress_dirs: Sequence[Path],
) -> dict[str, Any]:
    """
    WHAT: Runs the existing pilot pipeline on the chosen PU method and records its result location.

    WHY: Stage 5 is the only stage allowed to read test, so it must run the same frozen pipeline the
    pilot uses; reusing run_pilot keeps the score-cut tuning, operating points, 2x2 comparison and
    success criterion identical to a pilot run. Stress worlds are scored by the same pipeline after
    test and land in their own block.

    Args:
        chosen_method: PU method stage 4 chose.
        training_dirs: Training dataset folders.
        validation_dirs: Validation dataset folders.
        test_dirs: Test dataset folders.
        truth_root: Root folder holding one truth folder per run_id.
        world_cards_root: Root folder holding one world card per run_id.
        out_root: Driver output root; the pilot result lands under its final/ folder.
        config_path: Active config the driver started from, already carrying the chosen window.
        all_dataset_ids: Training, validation and test dataset folder names.
        stress_dirs: Optional stress dataset folders, scored after test.

    Returns:
        The pilot's exit code, result path, config copy hash, success criterion and stress summary.

    Raises:
        ValueError: If the pilot reports failed leakage checks or writes no result file.
    """
    final_root = out_root / "final"
    final_config = out_root / "stages" / "final_mule_model.yaml"
    _write_final_config(config_path, final_config, chosen_method)
    previous_config = active_mule_model_config_path()
    set_mule_model_config_path(final_config)
    try:
        exit_code = run_pilot(
            training_dirs,
            validation_dirs,
            test_dirs,
            truth_root,
            world_cards_root,
            final_root,
            stress_dirs,
        )
    finally:
        set_mule_model_config_path(previous_config)
    result_path = final_root / run_key(all_dataset_ids) / "result.json"
    if not result_path.is_file():
        raise ValueError(f"Stage 5 wrote no result file at {result_path}")
    payload = orjson.loads(result_path.read_bytes())
    if exit_code != 0:
        raise ValueError(f"The pilot's leakage checks failed (exit {exit_code}); see {result_path.parent}")
    manifest = payload.get("manifest")
    if not isinstance(manifest, dict):
        raise ValueError(f"Pilot result {result_path} holds no manifest mapping")
    stress_block = payload.get("stress")
    stress_world_count = 0
    if stress_block is not None:
        if not isinstance(stress_block, dict):
            raise ValueError(f"Pilot result {result_path} holds a malformed stress block")
        stress_world_count = int(stress_block["world_count"])
    return {
        "pu_method": chosen_method,
        "exit_code": exit_code,
        "result_path": str(result_path),
        "mule_model_config_path": str(final_config),
        "mule_model_config_sha256": _sha256_of_file(final_config),
        "success_criterion": manifest.get("success_criterion"),
        "stress_world_count": stress_world_count,
    }


def run_full(
    training_dirs: Sequence[Path],
    validation_dirs: Sequence[Path],
    test_dirs: Sequence[Path],
    truth_root: Path,
    world_cards_root: Path,
    out_root: Path,
    training_world_counts: Sequence[int],
    stress_dirs: Sequence[Path] = (),
) -> int:
    """
    WHAT: Runs the five stages in order, resuming every stage whose fingerprint matches.

    WHY: One entry point fixes the order maturity window -> world count -> method tuning -> method
    choice -> test and stress, and the fingerprint check makes a resume safe after a crash or a code
    change.

    Args:
        training_dirs: Training dataset folders.
        validation_dirs: Validation dataset folders.
        test_dirs: Test dataset folders.
        truth_root: Root folder holding one truth folder per run_id.
        world_cards_root: Root folder holding one world card per run_id.
        out_root: Root folder for the stage files and the final pilot result.
        training_world_counts: Ascending world counts the learning curve tries.
        stress_dirs: Optional stress dataset folders, scored after test in stage 5 only.

    Returns:
        Process exit code, 0 when every stage completes.

    Raises:
        ValueError: If the config is not the compact path, a count is invalid, a stage's fingerprint
            differs from the current inputs, or stage 5's pilot fails its leakage checks.
    """
    counts = tuple(int(count) for count in training_world_counts)
    if list(counts) != sorted(set(counts)) or not counts:
        raise ValueError("training_world_counts must be positive, distinct and ascending")
    mule_config = load_mule_model_config()
    if mule_config.data_path != "compact":
        raise ValueError(
            f"The full-run driver requires data_path 'compact'; the model config sets {mule_config.data_path!r}"
        )
    if mule_config.rules_plus_labels_and_registry is True:
        base_rules_config = load_rules_config_with_labels()
        tuning = load_labels_tuning_config()
    else:
        base_rules_config = load_rules_config()
        tuning = load_tuning_config()
    seed = load_anomaly_detector_config().random_state
    main_budget_pct = load_harm_budgets()[0]

    training_dataset_ids = _dataset_ids(training_dirs)
    validation_dataset_ids = _dataset_ids(validation_dirs)
    test_dataset_ids = _dataset_ids(test_dirs)
    stress_dataset_ids = _dataset_ids(stress_dirs)
    training_world_ids = _split_world_ids("training", training_dirs)
    _split_world_ids("validation", validation_dirs)
    _split_world_ids("test", test_dirs)
    if stress_dirs:
        _split_world_ids("stress", stress_dirs)

    config_path = active_mule_model_config_path().resolve()
    fingerprint = _inputs_fingerprint(
        config_path,
        training_dataset_ids,
        validation_dataset_ids,
        test_dataset_ids,
        stress_dataset_ids,
        counts,
    )
    stages_dir = out_root / "stages"

    maturity = _run_stage(
        "maturity_check",
        stages_dir,
        fingerprint,
        lambda: _maturity_check(
            training_dirs,
            training_world_ids,
            validation_dirs,
            truth_root,
            base_rules_config,
            seed,
            mule_config,
            out_root,
        ),
    )
    chosen_config = stages_dir / "chosen_mule_model.yaml"
    _write_chosen_config(config_path, chosen_config, maturity)
    previous_config = active_mule_model_config_path()
    set_mule_model_config_path(chosen_config)
    try:
        learning_curve = _run_stage(
            "learning_curve",
            stages_dir,
            fingerprint,
            lambda: _learning_curve(
                training_dirs,
                training_world_ids,
                validation_dirs,
                truth_root,
                base_rules_config,
                seed,
                counts,
            ),
        )
        chosen_count = learning_curve["chosen_training_world_count"]
        tuned_methods = _run_stage(
            "tune_each_pu_method",
            stages_dir,
            fingerprint,
            lambda: _tune_each_pu_method(
                training_dirs,
                training_world_ids,
                chosen_count,
                validation_dirs,
                truth_root,
                base_rules_config,
                tuning,
                main_budget_pct,
                mule_config,
                seed,
            ),
        )
        method_choice = _run_stage(
            "choose_pu_method",
            stages_dir,
            fingerprint,
            lambda: _choose_pu_method(tuned_methods["methods"]),
        )
        final = _run_stage(
            "final_validation_and_test",
            stages_dir,
            fingerprint,
            lambda: _final_validation_and_test(
                method_choice["pu_method"],
                training_dirs,
                validation_dirs,
                test_dirs,
                truth_root,
                world_cards_root,
                out_root,
                chosen_config,
                [
                    *training_dataset_ids,
                    *validation_dataset_ids,
                    *test_dataset_ids,
                ],
                stress_dirs,
            ),
        )
    finally:
        set_mule_model_config_path(previous_config)
    if final["exit_code"] != 0:
        return int(final["exit_code"])
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """
    WHAT: Command-line entry point for the staged Level 18 full run.

    Args:
        argv: Optional argument list; defaults to sys.argv.

    Returns:
        The driver's process exit code.
    """
    parser = argparse.ArgumentParser(
        description="Run the staged Level 18 full-scale account model and write one JSON per stage"
    )
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
        "--world-cards-root",
        type=Path,
        required=True,
        help="root folder holding one world card per run_id",
    )
    parser.add_argument("--out-root", type=Path, required=True, help="root folder for the stage files and pilot result")
    parser.add_argument(
        "--training-world-counts",
        required=True,
        help="comma-separated ascending world counts, for example 40,80,160",
    )
    parser.add_argument(
        "--stress",
        type=Path,
        action="append",
        default=[],
        help="stress dataset folder, scored after test in the final stage; repeatable",
    )
    parser.add_argument(
        "--mule-model-config",
        type=Path,
        default=None,
        help="model config to use instead of the committed config/mule_model.yaml",
    )
    args = parser.parse_args(argv)
    if args.mule_model_config is not None:
        set_mule_model_config_path(args.mule_model_config)
    try:
        return run_full(
            tuple(args.training),
            tuple(args.validation),
            tuple(args.test),
            args.truth_root,
            args.world_cards_root,
            args.out_root,
            _parse_training_world_counts(args.training_world_counts),
            tuple(args.stress),
        )
    finally:
        set_mule_model_config_path(None)


if __name__ == "__main__":
    raise SystemExit(main())
