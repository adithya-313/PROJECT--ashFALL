"""
WHAT: Runs the Level 17 evaluation end to end: trains the anomaly detector on the training
datasets, tunes the rules thresholds and the anomaly cut on validation, scores every test and
stress world with the rules, the anomaly detector and their union, and writes result.json.

WHY: The rules baseline and the model must be judged on identical account-days at matched
operating points, hidden truth may be read only inside evaluation, and every input (datasets,
truth, detector, tuned thresholds) must be recorded so that a result can be reproduced and audited.

TERM: run key: the first 12 hex characters of the SHA-256 of the sorted dataset folder names.
TERM: world card: runs/world_cards/<run_id>/base_rate.json, which carries the world's
      fraud_inflation_factor.
TERM: window: the dataset split that supplied a world, either "test" or "stress".

Every dataset folder, truth file and world card is required. A missing or mismatched input raises
a plain ValueError naming the offending path instead of producing a partial or guessed result.
"""

import argparse
import hashlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

import orjson

from core.config_validator import load_yaml
from core.event_clock import IST
from decisions.anomaly_config import load_anomaly_detector_config
from decisions.anomaly_detector import TrainedAnomalyDetector, train_anomaly_detector
from decisions.rules_config import RulesConfig, load_rules_config, with_thresholds
from evaluation.dataset_truth_join import join_truth
from evaluation.level17_evaluation import (
    evaluate_world,
    first_illicit_credit,
    require_tuned,
    world_start_by_world,
)
from evaluation.level17_scoring import one_row_per_account_day, score_account_days
from evaluation.rules_metrics import summarise_across_worlds
from evaluation.rules_tuning import (
    AnomalyTuningResult,
    RulesTuningResult,
    TuningConfig,
    load_tuning_config,
    tune_anomaly_threshold,
    tune_rules,
)
from evaluation.truth import load_truth
from replay.dataset_builder import Dataset, load_dataset
from showcase.framework import get_code_version

__all__ = [
    "EvaluationSettings",
    "TunedThresholds",
    "build_result_manifest",
    "evaluate_worlds",
    "json_safe_result",
    "load_evaluation_settings",
    "load_split_datasets",
    "main",
    "read_world_base_rate",
    "reported_accounts",
    "run_evaluation",
    "run_key",
    "snapshot_time_by_account_day",
    "summarise_worlds",
    "tune_on_validation",
    "write_result",
]

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_EVALUATION_CONFIG = REPO_ROOT / "config" / "evaluation.yaml"
DEFAULT_OUT_ROOT = REPO_ROOT / "reports" / "level_17_evaluation"

REGISTRY_LABEL_SOURCE = "registry"

_SCORERS: tuple[str, ...] = ("rules", "anomaly", "rules_and_anomaly")

_WINDOWS: tuple[str, ...] = ("test", "stress")

_HEADLINE_METRICS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("recall_at_capacity", ("recall_at_capacity",)),
    ("yield_measured", ("yield_rate", "measured")),
    ("yield_reweighted", ("yield_rate", "reweighted")),
    ("genuine_restriction_rate", ("genuine_harm", "genuine_restriction_rate")),
    ("median_time_to_detection_hours", ("time_to_detection", "median")),
)


@dataclass(frozen=True)
class EvaluationSettings:
    """
    WHAT: The policy numbers from config/evaluation.yaml that this runner needs.

    ATTRIBUTES:
        genuine_restriction_budget_pct: Tuning budget, the share of genuine account-days either
            system may restrict.
        capacity_pct_active_accounts_per_day: Share of a bank's active accounts an analyst can
            review per day.
        min_test_worlds: Smallest world count for which p10 and p90 are reported.
        persona_weights: Persona label to genuine-harm weight.
        default_persona_weight: Harm weight for personas absent from persona_weights.
    """

    genuine_restriction_budget_pct: float
    capacity_pct_active_accounts_per_day: float
    min_test_worlds: int
    persona_weights: dict[str, int]
    default_persona_weight: int


@dataclass(frozen=True)
class TunedThresholds:
    """
    WHAT: The rules configuration tuned on validation plus both tuning results.

    ATTRIBUTES:
        rules_config: Rules table carrying the tuned threshold values and statuses.
        rules_result: Search result for the rules thresholds.
        anomaly_result: Chosen anomaly cut and its validation numbers.
    """

    rules_config: RulesConfig
    rules_result: RulesTuningResult
    anomaly_result: AnomalyTuningResult


def _require_mapping(raw: Mapping[str, Any], key: str, path: Path) -> Mapping[str, Any]:
    """
    WHAT: Returns a required nested mapping from an evaluation config file.

    Raises:
        ValueError: If the block is missing or is not a mapping.
    """
    value = raw.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"{path} requires a '{key}' mapping")
    return value


def _require_number(raw: Mapping[str, Any], key: str, path: Path) -> float:
    """
    WHAT: Returns a required number from an evaluation config block.

    Raises:
        ValueError: If the value is missing, is a bool, or is not a number.
    """
    value = raw.get(key)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{path} requires a numeric '{key}'; got {value!r}")
    return float(value)


def load_evaluation_settings(path: Path | str | None = None) -> EvaluationSettings:
    """
    WHAT: Reads the capacity, tuning budget, world-count and persona-weight policy numbers.

    WHY: Every number comes from the pre-registered evaluation config, so this runner never
    carries a typed-in policy value.

    Args:
        path: Optional evaluation YAML path; defaults to config/evaluation.yaml.

    Returns:
        EvaluationSettings with the policy numbers this runner applies.

    Raises:
        ValueError: If the file or any required block or number is missing or malformed.
    """
    target = DEFAULT_EVALUATION_CONFIG if path is None else Path(path)
    raw = load_yaml(target)
    if not isinstance(raw, Mapping):
        raise ValueError(f"{target} must hold a mapping at the root")

    fair_comparison = _require_mapping(raw, "fair_comparison", target)
    capacity = _require_mapping(raw, "capacity", target)
    ml_beats_rules = _require_mapping(raw, "ml_beats_rules", target)
    min_test_worlds_block = _require_mapping(ml_beats_rules, "min_test_worlds", target)
    genuine_harm = _require_mapping(raw, "genuine_harm", target)
    persona_block = _require_mapping(genuine_harm, "persona_weights", target)
    persona_weights_raw = _require_mapping(persona_block, "weights", target)

    persona_weights: dict[str, int] = {}
    for persona, weight in persona_weights_raw.items():
        if not isinstance(persona, str) or not isinstance(weight, int) or isinstance(weight, bool):
            raise ValueError(f"{target} persona weight for {persona!r} must be an integer; got {weight!r}")
        persona_weights[persona] = weight
    default_weight = _require_number(persona_block, "default_weight", target)
    if default_weight != int(default_weight):
        raise ValueError(f"{target} default_weight must be a whole number; got {default_weight!r}")

    min_test_worlds = _require_number(min_test_worlds_block, "value", target)
    if min_test_worlds != int(min_test_worlds):
        raise ValueError(f"{target} min_test_worlds must be a whole number; got {min_test_worlds!r}")

    return EvaluationSettings(
        genuine_restriction_budget_pct=_require_number(fair_comparison, "main_genuine_restriction_rate_pct", target),
        capacity_pct_active_accounts_per_day=_require_number(capacity, "headline_pct_active_accounts_per_day", target),
        min_test_worlds=int(min_test_worlds),
        persona_weights=persona_weights,
        default_persona_weight=int(default_weight),
    )


def run_key(dataset_ids: Sequence[str]) -> str:
    """
    WHAT: Returns the stable result-folder key for a set of dataset folder names.

    WHY: Re-running on the same datasets must write to the same folder, in any flag order, so a
    result can be found from its inputs alone.

    Args:
        dataset_ids: Dataset folder names, for example "test-20261112T0000-abcd1234ef56".

    Returns:
        First 12 hex characters of the SHA-256 of the sorted names, joined by newlines.

    Raises:
        ValueError: If no dataset id is given, because an empty input has no identity.
    """
    if not dataset_ids:
        raise ValueError("run_key requires at least one dataset id")
    joined = "\n".join(sorted(dataset_ids))
    return hashlib.sha256(joined.encode("utf-8")).hexdigest()[:12]


def load_split_datasets(split: str, dataset_dirs: Sequence[Path]) -> list[Dataset]:
    """
    WHAT: Loads dataset folders and refuses any folder whose manifest records another split.

    WHY: A test dataset passed under --training would silently train on the final exam, so the
    recorded split must match the flag the folder was passed under.

    Args:
        split: Expected split, one of "training", "validation", "test" or "stress".
        dataset_dirs: Dataset folders written by write_dataset.

    Returns:
        One loaded Dataset per folder, in the given order.

    Raises:
        ValueError: If no folder is given, or a manifest records a different split.
    """
    if not dataset_dirs:
        raise ValueError(f"load_split_datasets requires at least one {split} dataset folder")
    datasets: list[Dataset] = []
    for dataset_dir in dataset_dirs:
        dataset = load_dataset(dataset_dir)
        recorded_split = dataset.manifest.get("split")
        if recorded_split != split:
            raise ValueError(
                f"Dataset {dataset_dir} records split {recorded_split!r}, but it was passed as {split!r}"
            )
        datasets.append(dataset)
    return datasets


def read_world_base_rate(world_cards_root: Path, run_id: str) -> dict[str, Any]:
    """
    WHAT: Reads one world's base_rate.json from its world card folder.

    WHY: The reweighted yield divides by that world's fraud_inflation_factor, and a missing card
    must stop the run rather than fall back to an assumed inflation.

    Args:
        world_cards_root: Root folder holding one folder per run_id.
        run_id: Simulation run whose base rate is needed.

    Returns:
        The parsed base rate mapping, including fraud_inflation_factor.

    Raises:
        ValueError: If the file is missing or does not hold a mapping.
    """
    path = world_cards_root / run_id / "base_rate.json"
    if not path.is_file():
        raise ValueError(f"World card base rate file is missing: {path}")
    base_rate = orjson.loads(path.read_bytes())
    if not isinstance(base_rate, dict):
        raise ValueError(f"World card base rate file {path} must hold a mapping")
    return base_rate


def reported_accounts(labels_rows: Sequence[Mapping[str, Any]], run_id: str) -> set[str]:
    """
    WHAT: Returns the accounts one world's bank labels name through a national-registry report.

    WHY: Hidden-mule recall is measured only among mules the registry has not reported yet, so
    registry-sourced labels define the excluded set.

    Args:
        labels_rows: Rows of a dataset labels table.
        run_id: Simulation run to keep.

    Returns:
        Account ids carrying at least one registry-source label in that run.

    Raises:
        ValueError: If a registry label row has no account_id.
    """
    accounts: set[str] = set()
    for row in labels_rows:
        if row.get("run_id") != run_id or row.get("source") != REGISTRY_LABEL_SOURCE:
            continue
        account_id = row.get("account_id")
        if not isinstance(account_id, str) or not account_id:
            raise ValueError(f"Registry label row for run {run_id} has no account_id")
        accounts.add(account_id)
    return accounts


def tune_on_validation(
    validation_dir: Path,
    truth_root: Path,
    base_rules_config: RulesConfig,
    tuning: TuningConfig,
    detector: TrainedAnomalyDetector,
    budget_pct: float,
) -> TunedThresholds:
    """
    WHAT: Tunes the rules thresholds and the anomaly cut on one validation dataset.

    WHY: Both systems must be set to the same genuine-restriction budget on validation, never on
    test or stress, so the later comparison is not made on the worlds it is judged on.

    Args:
        validation_dir: Validation dataset folder.
        truth_root: Root folder holding one truth folder per run_id.
        base_rules_config: Shipped rules table supplying the untuned_start values.
        tuning: Candidate grid and allowed tuning splits.
        detector: Trained anomaly detector used to score the validation account-days.
        budget_pct: Maximum share of genuine account-days either system may restrict.

    Returns:
        TunedThresholds holding the tuned rules table and both tuning results.

    Raises:
        ValueError: If the dataset cannot be joined, or the tuners refuse their inputs.
    """
    _, snapshots = join_truth(validation_dir, truth_root)
    day_rows = one_row_per_account_day(snapshots.to_pylist())
    scored = score_account_days(day_rows, base_rules_config, detector)

    rules_result = tune_rules(day_rows, base_rules_config, tuning, "validation", budget_pct)
    tuned_config = with_thresholds(
        base_rules_config,
        rules_result.thresholds,
        f"tuned on validation dataset {validation_dir.name}",
    )
    anomaly_result = tune_anomaly_threshold(scored, tuning, "validation", budget_pct)

    return TunedThresholds(
        rules_config=tuned_config,
        rules_result=rules_result,
        anomaly_result=anomaly_result,
    )


def _world_inflation_factor(base_rate: Mapping[str, Any], run_id: str) -> float:
    """
    WHAT: Returns the positive fraud_inflation_factor a world card records.

    Raises:
        ValueError: If the factor is missing, not a number, or not strictly positive.
    """
    factor = base_rate.get("fraud_inflation_factor")
    if not isinstance(factor, (int, float)) or isinstance(factor, bool) or factor <= 0:
        raise ValueError(f"World card for run {run_id} has no positive fraud_inflation_factor; got {factor!r}")
    return float(factor)


def snapshot_time_by_account_day(
    day_rows: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, date], datetime]:
    """
    WHAT: Maps each account-day row to the timezone-aware snapshot_time it was scored from.

    WHY: The detection flag time must be the moment the account was actually seen, so the metric
    layer needs the snapshot time of the very row that produced each scored account-day.

    Args:
        day_rows: Rows already reduced by one_row_per_account_day.

    Returns:
        (account id, IST date) -> snapshot_time of that row.

    Raises:
        ValueError: If a row lacks an account id, a datetime snapshot_time, or a timezone.
    """
    snapshot_times: dict[tuple[str, date], datetime] = {}
    for row in day_rows:
        account_id = row.get("account_id")
        if not isinstance(account_id, str) or not account_id:
            raise ValueError("Snapshot rows require a non-empty string 'account_id'")
        moment = row.get("snapshot_time")
        if not isinstance(moment, datetime):
            raise ValueError("Snapshot rows require a datetime 'snapshot_time'")
        if moment.tzinfo is None:
            raise ValueError("snapshot_time must be timezone-aware so its IST date is unambiguous")
        snapshot_times[(account_id, moment.astimezone(IST).date())] = moment
    return snapshot_times


def evaluate_worlds(
    window: str,
    datasets: Sequence[tuple[Path, Dataset]],
    truth_root: Path,
    world_cards_root: Path,
    tuned: TunedThresholds,
    detector: TrainedAnomalyDetector,
    settings: EvaluationSettings,
) -> dict[str, dict[str, Any]]:
    """
    WHAT: Scores every world of the given datasets with the rules, the anomaly detector and their
    union, returning one result block per world.

    WHY: All three scorers must see identical rows and identical truth, so the join and the world
    card lookup happen once per dataset and once per world.

    Args:
        window: The split that supplied these datasets, "test" or "stress".
        datasets: (dataset folder, loaded Dataset) pairs of that split.
        truth_root: Root folder holding one truth folder per run_id.
        world_cards_root: Root folder holding one world card per run_id.
        tuned: Rules configuration and anomaly cut chosen on validation.
        detector: Trained anomaly detector.
        settings: Capacity, persona weights and world-count policy.

    Returns:
        world_id -> {"run_id", "split", "scorers": {scorer: evaluate_world result}}.

    Raises:
        ValueError: If a world_id repeats across datasets, a base rate is missing or invalid, or
            the metrics refuse the world's account-days.
    """
    per_world: dict[str, dict[str, Any]] = {}
    for dataset_dir, dataset in datasets:
        payments, snapshots = join_truth(dataset_dir, truth_root)
        payment_rows = payments.to_pylist()
        snapshot_rows = snapshots.to_pylist()
        labels_rows = dataset.labels.to_pylist()
        for world in dataset.manifest["worlds"]:
            world_id = str(world["world_id"])
            run_id = str(world["run_id"])
            if world_id in per_world:
                raise ValueError(f"World {world_id!r} appears in more than one {window} dataset")

            world_payments = [row for row in payment_rows if row["world_id"] == world_id]
            world_snapshots = [row for row in snapshot_rows if row["world_id"] == world_id]
            day_rows = one_row_per_account_day(world_snapshots)
            scored = score_account_days(day_rows, tuned.rules_config, detector)
            snapshot_times = snapshot_time_by_account_day(day_rows)

            truth_store = load_truth(truth_root, run_id)
            first_credit = first_illicit_credit(world_payments, truth_store.story_of)
            reported = reported_accounts(labels_rows, run_id)
            inflation_factor = _world_inflation_factor(read_world_base_rate(world_cards_root, run_id), run_id)

            scorers: dict[str, Any] = {}
            for scorer in _SCORERS:
                scorers[scorer] = evaluate_world(
                    scored,
                    scorer,
                    tuned.anomaly_result.threshold,
                    settings.capacity_pct_active_accounts_per_day,
                    inflation_factor,
                    truth_store.persona_of,
                    settings.persona_weights,
                    settings.default_persona_weight,
                    snapshot_times,
                    first_credit,
                    reported,
                )
            per_world[world_id] = {"run_id": run_id, "split": window, "scorers": scorers}
    return per_world


def _metric_value(world_result: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    """
    WHAT: Walks a nested metric path, returning None when any step is absent or is None.

    Args:
        world_result: One evaluate_world result.
        path: Key path such as ("yield_rate", "measured").
    """
    current: Any = world_result
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


def summarise_worlds(
    per_world: Mapping[str, Mapping[str, Any]], min_worlds: int
) -> dict[str, dict[str, dict[str, Any]]]:
    """
    WHAT: Summarises each headline metric per scorer, separately for test and stress worlds.

    WHY: One world can flatter or condemn a scorer, but stress worlds deliberately push extreme
    situations; mixing them into the test median would move the number that is actually judged.

    Args:
        per_world: world_id -> result block from evaluate_worlds, each carrying its split.
        min_worlds: Smallest world count for which p10 and p90 are reported.

    Returns:
        window -> scorer -> metric name -> summarise_across_worlds result, for the keys "test"
        and "stress". A window with no numeric value for a metric (for example no stress worlds,
        or yield on a run that escalated nothing) is recorded as
        {"world_count": 0, "status": "no numeric values"} instead of a made-up number.

    Raises:
        ValueError: If a world block carries a split other than "test" or "stress".
    """
    for world in per_world.values():
        split = world.get("split")
        if split not in _WINDOWS:
            raise ValueError(f"World summaries require split 'test' or 'stress'; got {split!r}")

    summaries: dict[str, dict[str, dict[str, Any]]] = {}
    for window in _WINDOWS:
        window_worlds = [world for world in per_world.values() if world["split"] == window]
        window_summary: dict[str, dict[str, Any]] = {}
        for scorer in _SCORERS:
            scorer_summary: dict[str, Any] = {}
            for metric_name, path in _HEADLINE_METRICS:
                values: list[float] = []
                for world in window_worlds:
                    scorers = world.get("scorers", {})
                    result = scorers.get(scorer)
                    if result is None:
                        continue
                    value = _metric_value(result, path)
                    if value is None:
                        continue
                    values.append(float(value))
                if values:
                    scorer_summary[metric_name] = summarise_across_worlds(values, min_worlds)
                else:
                    scorer_summary[metric_name] = {"world_count": 0, "status": "no numeric values"}
            window_summary[scorer] = scorer_summary
        summaries[window] = window_summary
    return summaries


def _sha256_of_file(path: Path) -> str:
    """
    WHAT: Returns the SHA-256 hex digest of a file's bytes.

    Raises:
        ValueError: If the file does not exist.
    """
    if not path.is_file():
        raise ValueError(f"Expected file is missing: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_result_manifest(
    dataset_entries: Sequence[Mapping[str, Any]],
    truth_manifests: Mapping[str, str],
    detector: TrainedAnomalyDetector,
    anomaly_random_state: int,
    tuned: TunedThresholds,
    settings: EvaluationSettings,
) -> dict[str, Any]:
    """
    WHAT: Builds the audit block that records every input a result depends on.

    WHY: A number is only trustworthy when the datasets, truth files, detector and tuned values
    that produced it are recorded beside it. No timestamp is recorded, so the same inputs always
    produce the same file.

    Args:
        dataset_entries: {"dataset_id", "split", "content_hash"} per dataset folder.
        truth_manifests: run_id -> SHA-256 of that run's truth manifest.json.
        detector: Trained anomaly detector.
        anomaly_random_state: random_state used to fit the detector.
        tuned: Rules configuration and anomaly cut chosen on validation.
        settings: Policy numbers the run applied.

    Returns:
        The manifest mapping placed under "manifest" in result.json.
    """
    return {
        "code_version": get_code_version(),
        "datasets": [dict(entry) for entry in dataset_entries],
        "truth_manifests": {run_id: truth_manifests[run_id] for run_id in sorted(truth_manifests)},
        "detector": {
            "model_sha256": detector.model_sha256,
            "training_rows_sha256": detector.training_rows_sha256,
            "training_row_count": detector.training_row_count,
            "sklearn_version": detector.sklearn_version,
        },
        "anomaly_random_state": anomaly_random_state,
        "tuned_thresholds": {key: tuned.rules_result.thresholds[key] for key in sorted(tuned.rules_result.thresholds)},
        "tuning": {
            "rules": {
                "meets_budget": tuned.rules_result.meets_budget,
                "validation_rate_pct": tuned.rules_result.validation_rate_pct,
                "validation_recall": tuned.rules_result.validation_recall,
                "combinations_evaluated": tuned.rules_result.combinations_evaluated,
            },
            "anomaly": {
                "threshold": tuned.anomaly_result.threshold,
                "meets_budget": tuned.anomaly_result.meets_budget,
                "validation_rate_pct": tuned.anomaly_result.validation_rate_pct,
                "validation_recall": tuned.anomaly_result.validation_recall,
            },
        },
        "settings": {
            "genuine_restriction_budget_pct": settings.genuine_restriction_budget_pct,
            "capacity_pct_active_accounts_per_day": settings.capacity_pct_active_accounts_per_day,
            "min_test_worlds": settings.min_test_worlds,
            "persona_weights": {
                persona: settings.persona_weights[persona] for persona in sorted(settings.persona_weights)
            },
            "default_persona_weight": settings.default_persona_weight,
        },
    }


def json_safe_result(per_world_result: Mapping[str, Any]) -> dict[str, Any]:
    """
    WHAT: Returns a copy of the per-world results whose hidden_mule_recall_by_depth keys are text.

    WHY: JSON object keys must be text, and that metric is keyed by integer mule depth. The
    conversion happens here rather than in a serializer option so a reader can see it, and the
    payload is written with sorted keys, so depth order is preserved by the sorted keys.

    Args:
        per_world_result: world_id -> result block from evaluate_worlds.

    Returns:
        A new mapping in which every depth key is a decimal string such as "1"; all other values
        are shared with the input.
    """
    converted: dict[str, Any] = {}
    for world_id, world in per_world_result.items():
        world_copy = dict(world)
        scorer_copy: dict[str, Any] = {}
        for scorer, result in world.get("scorers", {}).items():
            result_copy = dict(result)
            depth_recall = result.get("hidden_mule_recall_by_depth")
            if depth_recall is not None:
                result_copy["hidden_mule_recall_by_depth"] = {
                    str(depth): value for depth, value in depth_recall.items()
                }
            scorer_copy[scorer] = result_copy
        world_copy["scorers"] = scorer_copy
        converted[world_id] = world_copy
    return converted


def _require_text_keys(value: Any, path: str) -> None:
    """
    WHAT: Walks a JSON payload and refuses any mapping key that is not text.

    WHY: A non-text key would either fail inside the serializer without a location or be coerced
    silently; naming the path makes the offending part of the result obvious.

    Raises:
        ValueError: Naming the path of the first key that is not a string.
    """
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if not isinstance(key, str):
                raise ValueError(f"JSON key at {path} is not a string: {key!r}")
            _require_text_keys(nested, f"{path}.{key}")
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _require_text_keys(item, f"{path}[{index}]")


def write_result(
    out_root: Path,
    key: str,
    manifest: Mapping[str, Any],
    per_world: Mapping[str, Any],
    across_worlds: Mapping[str, Any],
) -> Path:
    """
    WHAT: Writes <out_root>/<key>/result.json with sorted keys and two-space indentation.

    WHY: The file is the deliverable of an evaluation run, and marking it "simulated": true stops
    a simulated measurement from ever being read as a real-world result.

    Args:
        out_root: Root folder for evaluation results.
        key: Result-folder key from run_key.
        manifest: Audit block from build_result_manifest.
        per_world: world_id -> result block.
        across_worlds: scorer -> metric -> summary.

    Returns:
        Path of the written result.json.
    """
    result_dir = out_root / key
    result_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "simulated": True,
        "manifest": dict(manifest),
        "per_world": json_safe_result(per_world),
        "across_worlds": dict(across_worlds),
    }
    _require_text_keys(payload, "result")
    path = result_dir / "result.json"
    path.write_bytes(orjson.dumps(payload, option=orjson.OPT_SORT_KEYS | orjson.OPT_INDENT_2) + b"\n")
    return path


def _dataset_entry(dataset_dir: Path, dataset: Dataset) -> dict[str, Any]:
    """
    WHAT: Returns the manifest audit entry for one dataset folder.

    Raises:
        ValueError: If the manifest lacks a content hash.
    """
    content_hash = dataset.manifest.get("content_hash")
    if not isinstance(content_hash, str) or not content_hash:
        raise ValueError(f"Dataset {dataset_dir} has no content_hash in its manifest")
    return {
        "dataset_id": dataset_dir.name,
        "split": dataset.manifest["split"],
        "content_hash": content_hash,
    }


def run_evaluation(
    training_dir: Path,
    validation_dir: Path,
    test_dirs: Sequence[Path],
    stress_dirs: Sequence[Path],
    truth_root: Path,
    world_cards_root: Path,
    out_root: Path = DEFAULT_OUT_ROOT,
    evaluation_config_path: Path | None = None,
) -> Path:
    """
    WHAT: Runs the full Level 17 evaluation and writes result.json.

    WHY: One entry point keeps the fixed order (training, then validation tuning, then the untuned
    guard, then test and stress scoring) so no shortcut can score a test world with untuned values.

    Args:
        training_dir: Training dataset folder.
        validation_dir: Validation dataset folder.
        test_dirs: Test dataset folders.
        stress_dirs: Stress dataset folders; may be empty.
        truth_root: Root folder holding one truth folder per run_id.
        world_cards_root: Root folder holding one world card per run_id.
        out_root: Root folder for evaluation results.
        evaluation_config_path: Optional evaluation YAML path; defaults to config/evaluation.yaml.

    Returns:
        Path of the written result.json.

    Raises:
        ValueError: If any dataset split is mismatched, a world card or truth manifest is missing,
            a metric refuses the data, or the tuned thresholds fail the untuned guard.
    """
    settings = load_evaluation_settings(evaluation_config_path)
    tuning = load_tuning_config()
    anomaly_config = load_anomaly_detector_config()
    base_rules_config = load_rules_config()

    training_datasets = load_split_datasets("training", [training_dir])
    validation_datasets = load_split_datasets("validation", [validation_dir])
    test_datasets = load_split_datasets("test", test_dirs)
    stress_datasets = load_split_datasets("stress", stress_dirs) if stress_dirs else []

    training_rows = [row for dataset in training_datasets for row in dataset.snapshots.to_pylist()]
    world_starts = world_start_by_world(training_rows)
    detector = train_anomaly_detector(training_rows, "training", world_starts, anomaly_config)

    tuned = tune_on_validation(
        validation_dir,
        truth_root,
        base_rules_config,
        tuning,
        detector,
        settings.genuine_restriction_budget_pct,
    )
    require_tuned(tuned.rules_config)

    per_world = evaluate_worlds(
        "test",
        list(zip(test_dirs, test_datasets, strict=True)),
        truth_root,
        world_cards_root,
        tuned,
        detector,
        settings,
    )
    if stress_datasets:
        for world_id, world_result in evaluate_worlds(
            "stress",
            list(zip(stress_dirs, stress_datasets, strict=True)),
            truth_root,
            world_cards_root,
            tuned,
            detector,
            settings,
        ).items():
            if world_id in per_world:
                raise ValueError(f"World {world_id!r} appears in more than one test or stress dataset")
            per_world[world_id] = world_result

    across_worlds = summarise_worlds(per_world, settings.min_test_worlds)

    dataset_entries = [
        _dataset_entry(dataset_dir, dataset)
        for dataset_dir, dataset in [
            (training_dir, training_datasets[0]),
            (validation_dir, validation_datasets[0]),
            *zip(test_dirs, test_datasets, strict=True),
            *zip(stress_dirs, stress_datasets, strict=True),
        ]
    ]
    run_ids = sorted({str(world["run_id"]) for world in per_world.values()})
    truth_manifests = {run_id: _sha256_of_file(truth_root / run_id / "manifest.json") for run_id in run_ids}
    manifest = build_result_manifest(
        dataset_entries,
        truth_manifests,
        detector,
        anomaly_config.random_state,
        tuned,
        settings,
    )

    key = run_key([str(entry["dataset_id"]) for entry in dataset_entries])
    return write_result(out_root, key, manifest, per_world, across_worlds)


def main(argv: Sequence[str] | None = None) -> int:
    """
    WHAT: Command-line entry point that runs the evaluation and prints the result path.

    Args:
        argv: Optional argument list; defaults to sys.argv.

    Returns:
        Process exit code 0 after a successful write.
    """
    parser = argparse.ArgumentParser(description="Run the Level 17 evaluation and write result.json")
    parser.add_argument("--training", type=Path, required=True, help="training dataset folder")
    parser.add_argument("--validation", type=Path, required=True, help="validation dataset folder")
    parser.add_argument("--test", type=Path, action="append", required=True, help="test dataset folder; repeatable")
    parser.add_argument("--stress", type=Path, action="append", default=[], help="stress dataset folder; repeatable")
    parser.add_argument(
        "--truth-root", type=Path, required=True, help="root folder holding one truth folder per run_id"
    )
    parser.add_argument(
        "--world-cards-root", type=Path, required=True, help="root folder holding one world card per run_id"
    )
    parser.add_argument("--out-root", type=Path, default=DEFAULT_OUT_ROOT, help="root folder for evaluation results")
    args = parser.parse_args(argv)

    result_path = run_evaluation(
        args.training,
        args.validation,
        tuple(args.test),
        tuple(args.stress),
        args.truth_root,
        args.world_cards_root,
        args.out_root,
    )
    print(str(result_path))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
