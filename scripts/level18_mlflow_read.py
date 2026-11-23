"""
File: scripts/level18_mlflow_read.py
Purpose: Pure, read-only readers for the Level 18 result files so the MLflow back-fill can log them after the fact.
         Nothing here trains, scores or writes a result file, and this module deliberately imports no MLflow and no
         evaluation code, so core.truth_import_check stays green.

The readers take a Path and return plain frozen dataclasses. Missing or malformed data raises a ValueError that names
the file and key; there are no silent defaults.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import orjson
import yaml

# The exact key set a stage file must carry, in the order read_stage_file checks it.
STAGE_KEYS = ("stage", "fingerprint", "seconds", "peak_rss_mb", "result")

# Literal written when a fingerprint value is not on disk. The base-tag contract forbids an empty string or None here.
NOT_RECORDED = "not recorded"

# Fingerprint fields that every stage file in a run must agree on, checked by fingerprint_tags.
_FINGERPRINT_FIELDS = (
    "code_version",
    "mule_model_config_sha256",
    "training_dataset_ids",
    "validation_dataset_ids",
    "test_dataset_ids",
    "stress_dataset_ids",
)


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_json(path: Path) -> Any:
    return orjson.loads(path.read_bytes())


def _need(mapping: dict[str, Any], key: str, where: Path) -> Any:
    """Return mapping[key], or raise a ValueError that names the file and the missing key."""
    if key not in mapping:
        raise ValueError(f"{where}: missing key '{key}'")
    return mapping[key]


def files_under(root: Path) -> tuple[Path, ...]:
    """Every regular file below root, sorted, for the artifact listings."""
    return tuple(sorted(p for p in root.rglob("*") if p.is_file()))


def file_records(paths: Sequence[Path], base: Path) -> tuple[dict[str, Any], ...]:
    """Describe each path as relative path, size in bytes and sha256, relative to base."""
    return tuple(
        {
            "path": path.relative_to(base).as_posix(),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_of(path),
        }
        for path in paths
    )


@dataclass(frozen=True)
class StageFile:
    path: Path
    stage: str
    fingerprint: dict[str, Any]
    seconds: float
    peak_rss_mb: float
    result: dict[str, Any]


def read_stage_file(path: Path) -> StageFile:
    """Load one stage JSON file, refusing to continue if any of the five stage keys is missing."""
    data = _load_json(path)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a JSON object, found {type(data).__name__}")
    for key in STAGE_KEYS:
        _need(data, key, path)
    return StageFile(
        path=path,
        stage=data["stage"],
        fingerprint=data["fingerprint"],
        seconds=data["seconds"],
        peak_rss_mb=data["peak_rss_mb"],
        result=data["result"],
    )


def stage_files_with_fingerprint(folder: Path) -> tuple[Path, ...]:
    """The *.json files under folder/stages that carry the full five-key stage shape, in filename order."""
    stages = folder / "stages"
    if not stages.is_dir():
        return ()
    found: list[Path] = []
    for path in sorted(stages.glob("*.json")):
        data = _load_json(path)
        if isinstance(data, dict) and all(key in data for key in STAGE_KEYS):
            found.append(path)
    return tuple(found)


@dataclass(frozen=True)
class FingerprintTags:
    code_version: str
    config_sha256: str
    dataset_ids: str
    fingerprint_source: str


def fingerprint_tags(stage_files: Sequence[Path]) -> FingerprintTags:
    """Build the base tags from one or more stage fingerprints.

    With no stage files every value is NOT_RECORDED and the source is "none". With stage files the fingerprints must
    agree on every checked field, otherwise a ValueError names the field and both values. The dataset ids are the
    training, validation, test then stress ids joined by commas.
    """
    if not stage_files:
        return FingerprintTags(NOT_RECORDED, NOT_RECORDED, NOT_RECORDED, "none")

    stages = [read_stage_file(path) for path in stage_files]
    base = stages[0]
    for field in _FINGERPRINT_FIELDS:
        _need(base.fingerprint, field, base.path)
    for other in stages[1:]:
        for field in _FINGERPRINT_FIELDS:
            left = _need(base.fingerprint, field, base.path)
            right = _need(other.fingerprint, field, other.path)
            if left != right:
                raise ValueError(
                    f"fingerprint disagreement on '{field}': {other.path} has {right!r} but {base.path} has {left!r}"
                )

    fingerprint = base.fingerprint
    dataset_ids = [
        *fingerprint["training_dataset_ids"],
        *fingerprint["validation_dataset_ids"],
        *fingerprint["test_dataset_ids"],
        *fingerprint["stress_dataset_ids"],
    ]
    return FingerprintTags(
        code_version=fingerprint["code_version"],
        config_sha256=fingerprint["mule_model_config_sha256"],
        dataset_ids=",".join(dataset_ids),
        fingerprint_source="stage_file",
    )


@dataclass(frozen=True)
class MaturityVariant:
    name: str
    label_maturity_days: int
    label_trust_weighting: bool
    training_row_count: int
    training_positive_count: int
    validation_truth_average_precision: float


@dataclass(frozen=True)
class MaturityReport:
    variants: tuple[MaturityVariant, ...]
    chosen_variant: str
    reason: str
    tolerance: float


def maturity_variants(stage: StageFile) -> MaturityReport:
    """One record per label-maturity variant, plus which variant was chosen and the tolerance behind it."""
    result = stage.result
    variants = tuple(
        MaturityVariant(
            name=_need(variant, "name", stage.path),
            label_maturity_days=_need(variant, "label_maturity_days", stage.path),
            label_trust_weighting=_need(variant, "label_trust_weighting", stage.path),
            training_row_count=_need(variant, "training_row_count", stage.path),
            training_positive_count=_need(variant, "training_positive_count", stage.path),
            validation_truth_average_precision=_need(variant, "validation_truth_average_precision", stage.path),
        )
        for variant in _need(result, "variants", stage.path)
    )
    return MaturityReport(
        variants=variants,
        chosen_variant=_need(result, "chosen_variant", stage.path),
        reason=_need(result, "reason", stage.path),
        tolerance=_need(result, "tolerance", stage.path),
    )


@dataclass(frozen=True)
class LearningCurvePoint:
    training_world_count: int
    validation_truth_average_precision: float
    training_row_count: int


@dataclass(frozen=True)
class LearningCurve:
    points: tuple[LearningCurvePoint, ...]
    doubling_improvements: tuple[dict[str, Any], ...]
    chosen_training_world_count: int
    more_training_worlds_needed: bool


def learning_curve_points(stage: StageFile) -> LearningCurve:
    """The learning-curve points, kept in file order, plus the doubling improvements and the chosen count."""
    result = stage.result
    points = tuple(
        LearningCurvePoint(
            training_world_count=_need(point, "training_world_count", stage.path),
            validation_truth_average_precision=_need(point, "validation_truth_average_precision", stage.path),
            training_row_count=_need(point, "training_row_count", stage.path),
        )
        for point in _need(result, "points", stage.path)
    )
    return LearningCurve(
        points=points,
        doubling_improvements=tuple(_need(result, "doubling_improvements", stage.path)),
        chosen_training_world_count=_need(result, "chosen_training_world_count", stage.path),
        more_training_worlds_needed=_need(result, "more_training_worlds_needed", stage.path),
    )


@dataclass(frozen=True)
class Trial:
    number: int
    params: dict[str, Any]
    score: float


@dataclass(frozen=True)
class PuMethodSearch:
    method: str
    chosen_params: dict[str, Any]
    validation_truth_average_precision: float
    validation_rate_pct: float
    validation_recall: float
    meets_budget: bool
    trials: tuple[Trial, ...]
    trials_run: int
    stopped_early: bool
    stop_reason: str
    param_importances: dict[str, Any] | str


def pu_method_search(stage: StageFile) -> PuMethodSearch:
    """The chosen method, its result, every trial in file order, and the importances kept exactly as stored.

    param_importances is returned unchanged: it is a dict of importances for a method that ran, or the
    "not run: ..." string for one that did not. It is never coerced.
    """
    result = stage.result
    importances = _need(result, "param_importances", stage.path)
    if not isinstance(importances, (dict, str)):
        raise ValueError(
            f"{stage.path}: param_importances must be a dict or the 'not run' string, "
            f"found {type(importances).__name__}"
        )
    trials = tuple(
        Trial(
            number=_need(trial, "number", stage.path),
            params=_need(trial, "params", stage.path),
            score=_need(trial, "score", stage.path),
        )
        for trial in _need(result, "trials", stage.path)
    )
    return PuMethodSearch(
        method=_need(result, "pu_method", stage.path),
        chosen_params=_need(result, "chosen_params", stage.path),
        validation_truth_average_precision=_need(result, "validation_truth_average_precision", stage.path),
        validation_rate_pct=_need(result, "validation_rate_pct", stage.path),
        validation_recall=_need(result, "validation_recall", stage.path),
        meets_budget=_need(result, "meets_budget", stage.path),
        trials=trials,
        trials_run=_need(result, "trials_run", stage.path),
        stopped_early=_need(result, "stopped_early", stage.path),
        stop_reason=_need(result, "stop_reason", stage.path),
        param_importances=importances,
    )


@dataclass(frozen=True)
class Stage4Method:
    pu_method: str
    validation_rate_pct: float
    validation_recall: float
    meets_budget: bool
    retrained_validation_truth_average_precision: float
    saved_validation_rate_pct: float
    saved_validation_recall: float
    saved_meets_budget: bool
    saved_validation_truth_average_precision: float


@dataclass(frozen=True)
class Stage4Recomputed:
    chosen_method: str
    methods: tuple[Stage4Method, ...]
    head: str
    source_code_version: str
    training_world_count: int
    note: str


def stage4_recomputed(path: Path) -> Stage4Recomputed:
    """Read run_a320_stage4_fixed_c/stage4_recomputed.json.

    This file is not the five-key stage shape, so it gets its own reader over the keys it actually carries:
    chosen_method, methods, head, source_code_version, training_world_count and note.
    """
    data = _load_json(path)
    if not isinstance(data, dict):
        raise ValueError(f"{path}: expected a JSON object, found {type(data).__name__}")
    methods = tuple(
        Stage4Method(
            pu_method=_need(method, "pu_method", path),
            validation_rate_pct=_need(method, "validation_rate_pct", path),
            validation_recall=_need(method, "validation_recall", path),
            meets_budget=_need(method, "meets_budget", path),
            retrained_validation_truth_average_precision=_need(
                method, "retrained_validation_truth_average_precision", path
            ),
            saved_validation_rate_pct=_need(method, "saved_validation_rate_pct", path),
            saved_validation_recall=_need(method, "saved_validation_recall", path),
            saved_meets_budget=_need(method, "saved_meets_budget", path),
            saved_validation_truth_average_precision=_need(method, "saved_validation_truth_average_precision", path),
        )
        for method in _need(data, "methods", path)
    )
    return Stage4Recomputed(
        chosen_method=_need(data, "chosen_method", path),
        methods=methods,
        head=_need(data, "head", path),
        source_code_version=_need(data, "source_code_version", path),
        training_world_count=_need(data, "training_world_count", path),
        note=_need(data, "note", path),
    )


def single_subdir(parent: Path) -> Path:
    """The one subfolder of parent. Anything other than exactly one raises a ValueError naming what was found."""
    subdirs = sorted(p for p in parent.iterdir() if p.is_dir())
    if len(subdirs) != 1:
        raise ValueError(f"{parent}: expected exactly one subfolder, found {[p.name for p in subdirs]}")
    return subdirs[0]


@dataclass(frozen=True)
class FinalResults:
    stage5_dir: Path
    fit_dir: Path
    head: str
    code_version: str
    config_sha256: str
    dataset_ids: tuple[str, ...]
    headline_capacity_pct: float
    capacity_points_pct: tuple[float, ...]
    xgboost_cut: float
    amendment: str
    consistency_gate_status: str
    consistency_gate_worlds: int
    test_worlds_processed: int
    test_world_cards_checked: int
    stress_worlds_processed: int
    stress_world_cards_checked: int
    test_capacity: dict[str, Any]
    stress_capacity: dict[str, Any]
    test_result_medians: dict[str, dict[str, float]]
    leakage_5_6_status: str
    leakage_5_6_checks: dict[str, str]
    leakage_final_checks: dict[str, str]
    shap_status: str
    shap_rows_sampled: int


def _result_medians(block: dict[str, Any], where: Path) -> dict[str, dict[str, float]]:
    return {
        detector: {name: _need(values, "median", where) for name, values in metrics.items()}
        for detector, metrics in block.items()
    }


def final_results(stage5_dir: Path) -> FinalResults:
    """Read the stage-5 result set from final/<fit id>/ and the top-level reporting files.

    The fit id is discovered, never hard-coded: final/ must hold exactly one subfolder. The code version and config
    sha256 are read from files that record them (result.json manifest, stage5_provenance.json); the config sha in the
    provenance must equal the one in extra_reporting, otherwise a ValueError names both.
    """
    fit_dir = single_subdir(stage5_dir / "final")
    result_path = fit_dir / "result.json"
    extra_path = stage5_dir / "extra_reporting.json"
    provenance_path = stage5_dir / "stage5_provenance.json"
    leakage_5_6_path = stage5_dir / "leakage_checks_5_6.json"
    leakage_final_path = fit_dir / "leakage_checks.json"
    shap_path = fit_dir / "shap_top_features.json"

    result = _load_json(result_path)
    extra = _load_json(extra_path)
    provenance = _load_json(provenance_path)
    leakage_5_6 = _load_json(leakage_5_6_path)
    leakage_final = _load_json(leakage_final_path)
    shap = _load_json(shap_path)

    manifest = _need(result, "manifest", result_path)
    extra_manifest = _need(extra, "manifest", extra_path)
    record = _need(provenance, "record", provenance_path)

    config_sha256 = _need(record, "mule_model_config_sha256", provenance_path)
    extra_config_sha256 = _need(_need(extra_manifest, "mule_model_config", extra_path), "sha256", extra_path)
    if config_sha256 != extra_config_sha256:
        raise ValueError(
            f"config sha mismatch: provenance {config_sha256!r} vs extra_reporting {extra_config_sha256!r}"
        )

    datasets = _need(manifest, "datasets", result_path)
    dataset_ids = tuple(_need(dataset, "dataset_id", result_path) for dataset in datasets)

    consistency_gate = _need(extra_manifest, "consistency_gate", extra_path)
    extra_test = _need(extra, "test", extra_path)
    extra_stress = _need(extra, "stress", extra_path)

    leakage_5_6_checks = {
        name: _need(_need(leakage_5_6, name, leakage_5_6_path), "status", leakage_5_6_path)
        for name in ("check_5a", "check_5b", "check_6a", "check_6b")
    }
    leakage_final_checks = {
        name: _need(check, "status", leakage_final_path)
        for name, check in _need(leakage_final, "checks", leakage_final_path).items()
    }

    return FinalResults(
        stage5_dir=stage5_dir,
        fit_dir=fit_dir,
        head=_need(provenance, "head", provenance_path),
        code_version=_need(manifest, "code_version", result_path),
        config_sha256=config_sha256,
        dataset_ids=dataset_ids,
        headline_capacity_pct=_need(extra_manifest, "headline_capacity_pct", extra_path),
        capacity_points_pct=tuple(_need(extra_manifest, "capacity_points_pct", extra_path)),
        xgboost_cut=_need(extra_manifest, "xgboost_cut", extra_path),
        amendment=_need(extra_manifest, "amendment", extra_path),
        consistency_gate_status=_need(consistency_gate, "status", extra_path),
        consistency_gate_worlds=_need(consistency_gate, "worlds_checked", extra_path),
        test_worlds_processed=_need(extra_test, "worlds_processed", extra_path),
        test_world_cards_checked=_need(extra_test, "world_cards_checked", extra_path),
        stress_worlds_processed=_need(extra_stress, "worlds_processed", extra_path),
        stress_world_cards_checked=_need(extra_stress, "world_cards_checked", extra_path),
        test_capacity=_need(_need(extra_test, "aggregate", extra_path), "capacity", extra_path),
        stress_capacity=_need(_need(extra_stress, "aggregate", extra_path), "capacity", extra_path),
        test_result_medians=_result_medians(
            _need(_need(result, "across_worlds", result_path), "test", result_path), result_path
        ),
        leakage_5_6_status=_need(leakage_5_6, "status", leakage_5_6_path),
        leakage_5_6_checks=leakage_5_6_checks,
        leakage_final_checks=leakage_final_checks,
        shap_status=_need(shap, "status", shap_path),
        shap_rows_sampled=_need(shap, "rows_sampled", shap_path),
    )


def final_fingerprint_tags(results: FinalResults) -> FingerprintTags:
    """Base tags for the final run: read from result/provenance files, so the source is "partial", not a stage file."""
    return FingerprintTags(
        code_version=results.code_version,
        config_sha256=results.config_sha256,
        dataset_ids=",".join(results.dataset_ids),
        fingerprint_source="partial",
    )


@dataclass(frozen=True)
class FailedAttempt:
    name: str
    folder: Path | None
    reason: str
    fixed_by: str | None
    derived_from: str
    status: str
    reason_recorded: bool
    evidence: str | None


def failed_attempts(yaml_path: Path, runs_root: Path) -> tuple[FailedAttempt, ...]:
    """Read config/level18_failed_attempts.yaml and resolve each listed folder under runs_root.

    A listed folder that does not exist raises a ValueError naming it. An entry with folder: null (the
    account-identity bug, which no single folder holds) is allowed and yields folder None.
    """
    data = yaml.safe_load(yaml_path.read_bytes())
    if not isinstance(data, dict):
        raise ValueError(f"{yaml_path}: expected a mapping, found {type(data).__name__}")
    records: list[FailedAttempt] = []
    for entry in _need(data, "attempts", yaml_path):
        folder_name = _need(entry, "folder", yaml_path)
        folder: Path | None = None
        if folder_name is not None:
            candidate = runs_root / folder_name
            if not candidate.is_dir():
                raise ValueError(f"{yaml_path}: listed folder does not exist: {candidate}")
            folder = candidate
        records.append(
            FailedAttempt(
                name=_need(entry, "name", yaml_path),
                folder=folder,
                reason=_need(entry, "reason", yaml_path),
                fixed_by=_need(entry, "fixed_by", yaml_path),
                derived_from=_need(entry, "derived_from", yaml_path),
                status=_need(entry, "status", yaml_path),
                reason_recorded=_need(entry, "reason_recorded", yaml_path),
                evidence=_need(entry, "evidence", yaml_path),
            )
        )
    return tuple(records)
