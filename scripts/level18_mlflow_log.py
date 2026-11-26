"""
File: scripts/level18_mlflow_log.py
Purpose: Back-fill MLflow with every Level 18 experiment after the fact, from the saved result files only.

This script never trains, scores or rewrites a result: it imports the pure readers, turns their dataclasses into
MLflow runs, and is idempotent. Before creating a run it searches the experiment for a run with the same idem_key
tag (a sha256 over the source file bytes and the canonical JSON of the unit's name, tags, params and failed-attempt
record, so editing a YAML reason re-logs that attempt) and the same run name; if one exists it prints SKIP and logs
nothing. The idem_key and source_sha256 tags are set only after params, metrics, children and artifacts all
succeeded. A run that fails part-way is ended FAILED with a backfill_error tag and no idem_key, so the next pass
logs a fresh complete run and the partial run remains as evidence. That lookup is the only read from MLflow, and
nothing reads MLflow for any model decision.

Every run carries the base tags code_version, config_sha256, dataset_ids (from a stage fingerprint, or "not
recorded"), fingerprint_source, source_file, source_sha256 and simulated="true". Failed and killed attempts carry
no metrics, only tags and a file-list artifact.
"""

from __future__ import annotations

import argparse
import hashlib
import tempfile
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Sequence

import mlflow
import orjson

from core.mlflow_setup import setup_mlflow
from scripts.level18_mlflow_read import (
    NOT_RECORDED,
    FailedAttempt,
    FingerprintTags,
    driver_run_results,
    failed_attempts,
    file_records,
    files_under,
    final_fingerprint_tags,
    final_results,
    fingerprint_tags,
    learning_curve_points,
    maturity_variants,
    pu_method_search,
    read_stage_file,
    sha256_of,
    single_subdir,
    stage4_recomputed,
    stage_files_with_fingerprint,
)

EXPERIMENT = "ashFALL Level 18 — mule models"
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RUNS_ROOT = "runs/full"
FAILED_YAML = REPO_ROOT / "config" / "level18_failed_attempts.yaml"

# Which detector groups the final-run headline medians are read from (all four exist in result.json).
_DETECTORS = ("anomaly", "rules", "rules_and_anomaly", "xgboost")
_CAPACITY_KINDS = ("model", "normalized", "oracle")


@dataclass(frozen=True)
class Metric:
    name: str
    value: float
    step: int | None = None


@dataclass(frozen=True)
class Child:
    name: str
    params: dict[str, str]
    metrics: tuple[Metric, ...]
    tags: dict[str, str]


@dataclass(frozen=True)
class Listing:
    files: tuple[Path, ...]
    base: Path
    filename: str


@dataclass(frozen=True)
class Unit:
    kind: str
    name: str
    source_file: Path
    tags: dict[str, str]
    params: dict[str, str]
    metrics: tuple[Metric, ...]
    children: tuple[Child, ...]
    artifacts_copy: tuple[Path, ...]
    listings: tuple[Listing, ...]
    status: str
    failed_record: dict[str, Any] | None = None


def _as_str(value: Any) -> str:
    if isinstance(value, (dict, list)):
        return orjson.dumps(value).decode()
    return str(value)


def _bool(value: bool) -> str:
    return "true" if value else "false"


def _param_dict(values: dict[str, Any], prefix: str = "") -> dict[str, str]:
    return {f"{prefix}{key}": _as_str(value) for key, value in values.items()}


def _base_tags(tags: FingerprintTags) -> dict[str, str]:
    return {
        "code_version": tags.code_version,
        "config_sha256": tags.config_sha256,
        "dataset_ids": tags.dataset_ids,
        "fingerprint_source": tags.fingerprint_source,
    }


def _label(path: Path, roots: Sequence[Path]) -> str:
    """A readable tag value: the first root the path sits under, else the absolute path."""
    resolved = path.resolve()
    for root in roots:
        try:
            return resolved.relative_to(root.resolve()).as_posix()
        except ValueError:
            continue
    return resolved.as_posix()


def _headline_capacity_key(fr) -> str:
    key = str(fr.headline_capacity_pct)
    if key not in fr.test_capacity or key not in fr.stress_capacity:
        raise ValueError(
            f"headline capacity {fr.headline_capacity_pct!r} (key {key!r}) is not in the stress/test capacity points"
        )
    return key


def _stage_unit(
    tags: FingerprintTags,
    kind: str,
    name: str,
    source: Path,
    params: dict[str, str],
    metrics: tuple[Metric, ...],
    children: tuple[Child, ...],
    extra_tags: dict[str, str],
    artifacts_copy: tuple[Path, ...],
    listings: tuple[Listing, ...],
) -> Unit:
    return Unit(
        kind=kind,
        name=name,
        source_file=source,
        tags={**_base_tags(tags), **extra_tags},
        params=params,
        metrics=metrics,
        children=children,
        artifacts_copy=artifacts_copy,
        listings=listings,
        status="FINISHED",
    )


def _stage1_unit(tags: FingerprintTags, path: Path) -> Unit:
    report = maturity_variants(read_stage_file(path))
    children = tuple(
        Child(
            name=variant.name,
            params=_param_dict(
                {
                    "label_maturity_days": variant.label_maturity_days,
                    "label_trust_weighting": variant.label_trust_weighting,
                    "training_row_count": variant.training_row_count,
                    "training_positive_count": variant.training_positive_count,
                }
            ),
            metrics=(Metric("validation_truth_average_precision", variant.validation_truth_average_precision),),
            tags={"variant": variant.name},
        )
        for variant in report.variants
    )
    return _stage_unit(
        tags=tags,
        kind="stage1_label_maturity",
        name="stage1_label_maturity",
        source=path,
        params={"tolerance": _as_str(report.tolerance), "chosen_variant": report.chosen_variant},
        metrics=(),
        children=children,
        extra_tags={"chosen_variant": report.chosen_variant},
        artifacts_copy=(),
        listings=(),
    )


def _learning_curve_unit(tags: FingerprintTags, path: Path) -> Unit:
    curve = learning_curve_points(read_stage_file(path))
    metrics = tuple(
        Metric(
            "validation_truth_average_precision",
            point.validation_truth_average_precision,
            step=point.training_world_count,
        )
        for point in curve.points
    ) + tuple(
        Metric(
            "doubling_improvement_average_precision",
            improvement["average_precision_improvement"],
            step=improvement["to_training_worlds"],
        )
        for improvement in curve.doubling_improvements
    )
    return _stage_unit(
        tags=tags,
        kind="learning_curve",
        name="learning_curve",
        source=path,
        params={},
        metrics=metrics,
        children=(),
        extra_tags={
            "chosen_training_world_count": str(curve.chosen_training_world_count),
            "more_training_worlds_needed": _bool(curve.more_training_worlds_needed),
        },
        artifacts_copy=(),
        listings=(),
    )


def _pu_search_unit(tags: FingerprintTags, path: Path) -> Unit:
    search = pu_method_search(read_stage_file(path))
    tags_extra = {
        "pu_method": search.method,
        "meets_budget": _bool(search.meets_budget),
        "stopped_early": _bool(search.stopped_early),
        "stop_reason": search.stop_reason,
        "trials_run": str(search.trials_run),
    }
    metrics = [
        Metric("validation_truth_average_precision", search.validation_truth_average_precision),
        Metric("validation_rate_pct", search.validation_rate_pct),
        Metric("validation_recall", search.validation_recall),
    ]
    if isinstance(search.param_importances, dict):
        metrics.extend(Metric(f"importance_{key}", float(value)) for key, value in search.param_importances.items())
    else:
        tags_extra["param_importances"] = search.param_importances
    children = tuple(
        Child(
            name=f"trial_{trial.number}",
            params=_param_dict(trial.params, prefix="param_"),
            metrics=(Metric("score", trial.score, step=trial.number),),
            tags={"trial_number": str(trial.number)},
        )
        for trial in search.trials
    )
    return _stage_unit(
        tags=tags,
        kind=f"pu_search_{search.method}",
        name=f"pu_search_{search.method}",
        source=path,
        params=_param_dict(search.chosen_params, prefix="chosen_"),
        metrics=tuple(metrics),
        children=children,
        extra_tags=tags_extra,
        artifacts_copy=(),
        listings=(),
    )


def _stage4_unit(tags: FingerprintTags, path: Path) -> Unit:
    stage4 = stage4_recomputed(path)
    children = tuple(
        Child(
            name=f"stage4_{method.pu_method}",
            params={"pu_method": method.pu_method},
            metrics=(
                Metric("validation_rate_pct", method.validation_rate_pct),
                Metric("validation_recall", method.validation_recall),
                Metric(
                    "retrained_validation_truth_average_precision",
                    method.retrained_validation_truth_average_precision,
                ),
                Metric("saved_validation_rate_pct", method.saved_validation_rate_pct),
                Metric("saved_validation_recall", method.saved_validation_recall),
            ),
            tags={
                "met_budget": _bool(method.meets_budget),
                "saved_met_budget": _bool(method.saved_meets_budget),
            },
        )
        for method in stage4.methods
    )
    return _stage_unit(
        tags=tags,
        kind="stage4_recomputed",
        name="stage4_recomputed",
        source=path,
        params={"training_world_count": str(stage4.training_world_count)},
        metrics=(),
        children=children,
        extra_tags={"chosen_method": stage4.chosen_method, "source_code_version": stage4.source_code_version},
        artifacts_copy=(),
        listings=(),
    )


def _final_unit(runs_root: Path, stage5_dir: Path) -> Unit:
    fr = final_results(stage5_dir)
    tags = final_fingerprint_tags(fr)
    metrics: list[Metric] = [
        Metric("headline_capacity_pct", fr.headline_capacity_pct),
        Metric("xgboost_cut", fr.xgboost_cut),
        Metric("consistency_gate_worlds", fr.consistency_gate_worlds),
        Metric("test_worlds_processed", fr.test_worlds_processed),
        Metric("test_world_cards_checked", fr.test_world_cards_checked),
        Metric("stress_worlds_processed", fr.stress_worlds_processed),
        Metric("stress_world_cards_checked", fr.stress_world_cards_checked),
    ]
    for detector in _DETECTORS:
        for metric_name, median in fr.test_result_medians[detector].items():
            metrics.append(Metric(f"test_{detector}_{metric_name}_median", median))

    capacity_key = _headline_capacity_key(fr)
    for label, capacity in (("test", fr.test_capacity), ("stress", fr.stress_capacity)):
        for kind in _CAPACITY_KINDS:
            block = capacity[capacity_key][kind]
            metrics.append(Metric(f"{label}_capacity_{kind}_median", block["median"]))
            metrics.append(Metric(f"{label}_capacity_{kind}_world_count", block["world_count"]))

    extra_tags = {
        "head": fr.head,
        "amendment": fr.amendment,
        "consistency_gate_status": fr.consistency_gate_status,
        "leakage_5_6_status": fr.leakage_5_6_status,
        "shap_status": fr.shap_status,
    }
    for check_name, status in fr.leakage_5_6_checks.items():
        extra_tags[f"leakage_5_6_{check_name}_status"] = status
    for check_name, status in fr.leakage_final_checks.items():
        extra_tags[f"leakage_final_{check_name}_status"] = status

    artifacts_copy = (
        fr.fit_dir / "result.json",
        fr.fit_dir / "leakage_checks.json",
        fr.fit_dir / "shap_top_features.json",
        fr.fit_dir / "optuna_trials.json",
        stage5_dir / "extra_reporting.json",
        stage5_dir / "leakage_checks_5_6.json",
        stage5_dir / "stage5_provenance.json",
        stage5_dir / "extra_run_a053.log",
        stage5_dir / "stages" / "final_mule_model.yaml",
    )
    not_copied = (
        fr.fit_dir / "validation_curves.json",
        *files_under(fr.fit_dir / "checkpoint"),
        *files_under(fr.fit_dir / "account_model_bags"),
        stage5_dir / "extra_reporting_before_a053_20261123.json",
    )
    listings = (Listing(files=not_copied, base=runs_root, filename="not_copied_files.json"),)
    return _stage_unit(
        tags=tags,
        kind="final_test_and_stress",
        name="final_test_and_stress",
        source=fr.fit_dir / "result.json",
        params={},
        metrics=tuple(metrics),
        children=(),
        extra_tags=extra_tags,
        artifacts_copy=artifacts_copy,
        listings=listings,
    )


def _failed_unit(runs_root: Path, failed_yaml: Path, record: FailedAttempt) -> Unit:
    stage_files = stage_files_with_fingerprint(record.folder) if record.folder is not None else ()
    extra_tags: dict[str, str] = {}
    if stage_files:
        tags = fingerprint_tags(stage_files)
        source = stage_files[0]
    elif record.folder is not None and (record.folder / "stages" / "final_mule_model.yaml").is_file():
        config = record.folder / "stages" / "final_mule_model.yaml"
        fit_id = single_subdir(record.folder / "final").name
        tags = FingerprintTags(NOT_RECORDED, NOT_RECORDED, NOT_RECORDED, "partial")
        extra_tags["final_mule_model_yaml_sha256"] = sha256_of(config)
        extra_tags["fit_id"] = fit_id
        source = config
    else:
        tags = FingerprintTags(NOT_RECORDED, NOT_RECORDED, NOT_RECORDED, "none")
        source = failed_yaml

    extra_tags.update(
        {
            "attempt_name": record.name,
            "reason": record.reason,
            "fixed_by": record.fixed_by if record.fixed_by is not None else NOT_RECORDED,
            "derived_from": record.derived_from,
            "reason_recorded": _bool(record.reason_recorded),
        }
    )
    if record.evidence is not None:
        extra_tags["evidence"] = record.evidence

    listings = (
        (Listing(files=files_under(record.folder), base=record.folder, filename="file_list.json"),)
        if record.folder is not None
        else ()
    )
    return Unit(
        kind="failed_attempt",
        name=record.name,
        source_file=source,
        tags={**_base_tags(tags), **extra_tags},
        params={},
        metrics=(),
        children=(),
        artifacts_copy=(),
        listings=listings,
        status=record.status,
        failed_record={
            "name": record.name,
            "folder": _label(record.folder, (runs_root, REPO_ROOT)) if record.folder is not None else None,
            "reason": record.reason,
            "fixed_by": record.fixed_by,
            "derived_from": record.derived_from,
            "status": record.status,
            "reason_recorded": record.reason_recorded,
            "evidence": record.evidence,
        },
    )


def build_units(runs_root: Path, failed_yaml: Path) -> list[Unit]:
    """Plan every run from the saved files. Pure: reads files, imports no MLflow state, creates nothing."""
    stages = runs_root / "run_a320" / "stages"
    maturity_path = stages / "maturity_check.json"
    learning_path = stages / "learning_curve.json"
    tune_paths = (
        stages / "tune_method_none.json",
        stages / "tune_method_bagging.json",
        stages / "tune_method_elkan_noto.json",
    )
    stage_tags = fingerprint_tags((maturity_path, learning_path, *tune_paths))

    units: list[Unit] = [
        _stage1_unit(stage_tags, maturity_path),
        _learning_curve_unit(stage_tags, learning_path),
    ]
    units.extend(_pu_search_unit(stage_tags, path) for path in tune_paths)
    units.append(_stage4_unit(stage_tags, runs_root / "run_a320_stage4_fixed_c" / "stage4_recomputed.json"))
    units.append(_final_unit(runs_root, runs_root / "run_a320_stage5c"))
    units.extend(_failed_unit(runs_root, failed_yaml, record) for record in failed_attempts(failed_yaml, runs_root))
    return units


def _driver_final_unit(tags: FingerprintTags, run_root: Path) -> Unit:
    run = driver_run_results(run_root)
    metrics = [Metric("xgboost_cut", run.xgboost_cut), Metric("test_worlds_processed", run.test_world_count)]
    for detector in _DETECTORS:
        for metric_name, median in run.test_result_medians[detector].items():
            metrics.append(Metric(f"test_{detector}_{metric_name}_median", median))
    extra_tags = {
        "pu_method": run.pu_method,
        "final_mule_model_yaml_sha256": run.final_config_sha256,
        "success_criterion_status": run.success_status,
        "shap_status": run.shap_status,
    }
    for check_name, status in run.leakage_final_checks.items():
        extra_tags[f"leakage_final_{check_name}_status"] = status
    not_copied = (
        run.fit_dir / "validation_curves.json",
        *files_under(run.fit_dir / "checkpoint"),
        *files_under(run.fit_dir / "account_model_bags"),
    )
    return _stage_unit(
        tags=tags,
        kind="final_test",
        name="final_test",
        source=run.fit_dir / "result.json",
        params={},
        metrics=tuple(metrics),
        children=(),
        extra_tags=extra_tags,
        artifacts_copy=(
            run.fit_dir / "result.json",
            run.fit_dir / "leakage_checks.json",
            run.fit_dir / "shap_top_features.json",
            run.fit_dir / "optuna_trials.json",
            run_root / "stages" / "final_validation_and_test.json",
            run_root / "stages" / "final_mule_model.yaml",
        ),
        listings=(Listing(files=not_copied, base=run_root.parent, filename="not_copied_files.json"),),
    )


def build_run_units(run_root: Path) -> list[Unit]:
    """Plan the runs of one out-root the full driver finished, named after its folder. Pure, like build_units."""
    stages = run_root / "stages"
    maturity_path = stages / "maturity_check.json"
    learning_path = stages / "learning_curve.json"
    tune_paths = tuple(stages / f"tune_method_{method}.json" for method in ("none", "bagging", "elkan_noto"))
    tags = fingerprint_tags((maturity_path, learning_path, *tune_paths, stages / "final_validation_and_test.json"))
    units = [_stage1_unit(tags, maturity_path), _learning_curve_unit(tags, learning_path)]
    units.extend(_pu_search_unit(tags, path) for path in tune_paths)
    units.append(_driver_final_unit(tags, run_root))
    return [
        replace(unit, name=f"{run_root.name}/{unit.name}", tags={**unit.tags, "run_folder": run_root.name})
        for unit in units
    ]


def plan_counts(units: Sequence[Unit]) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for unit in units:
        counts[unit.kind] += 1 + len(unit.children)
    return dict(counts)


def _experiment_id() -> str:
    setup_mlflow(EXPERIMENT)
    experiment = mlflow.get_experiment_by_name(EXPERIMENT)
    if experiment is None:
        raise ValueError(f"experiment {EXPERIMENT!r} was not found after setup")
    return experiment.experiment_id


def _idem_key(unit: Unit) -> str:
    """sha256 over the source file bytes and the canonical JSON (sorted keys) of the unit's name, tags, params and
    failed-attempt record.

    The key therefore covers everything the run would log: editing a YAML reason, a tag or a param changes the key
    and the unit is logged again, instead of being skipped as already present.
    """
    canonical = orjson.dumps(
        {"name": unit.name, "tags": unit.tags, "params": unit.params, "failed_record": unit.failed_record},
        option=orjson.OPT_SORT_KEYS,
    )
    digest = hashlib.sha256()
    digest.update(unit.source_file.read_bytes())
    digest.update(canonical)
    return digest.hexdigest()


def _already_logged(experiment_id: str, name: str, idem_key: str) -> bool:
    found = mlflow.search_runs(
        experiment_ids=[experiment_id],
        filter_string=f"tags.idem_key = '{idem_key}'",
        output_format="list",
        max_results=100_000,
    )
    if not isinstance(found, list):
        raise ValueError(f"search_runs with output_format='list' returned {type(found).__name__}")
    return any(run.data.tags.get("mlflow.runName") == name for run in found)


def _log_metric(metric: Metric) -> None:
    if metric.step is None:
        mlflow.log_metric(metric.name, metric.value)
    else:
        mlflow.log_metric(metric.name, metric.value, step=metric.step)


def _log_unit(unit: Unit, source_sha256: str, idem_key: str, roots: Sequence[Path]) -> None:
    tags = dict(unit.tags)
    tags["source_file"] = _label(unit.source_file, roots)
    tags["simulated"] = "true"

    mlflow.start_run(run_name=unit.name, tags=tags)
    try:
        if unit.params:
            mlflow.log_params(unit.params)
        for metric in unit.metrics:
            _log_metric(metric)
        for child in unit.children:
            child_tags = {**tags, **child.tags}
            mlflow.start_run(run_name=child.name, nested=True, tags=child_tags)
            try:
                if child.params:
                    mlflow.log_params(child.params)
                for metric in child.metrics:
                    _log_metric(metric)
                mlflow.set_tag("source_sha256", source_sha256)
            finally:
                mlflow.end_run()
        for path in unit.artifacts_copy:
            if not path.is_file():
                raise ValueError(f"artifact source file is missing: {path}")
            mlflow.log_artifact(str(path))
        for listing in unit.listings:
            missing = [str(path) for path in listing.files if not path.is_file()]
            if missing:
                raise ValueError(f"listing files are missing: {missing}")
            records = file_records(listing.files, listing.base)
            mlflow.log_dict(
                {"base": _label(listing.base, roots), "files": list(records)},
                listing.filename,
            )
    except Exception as exc:
        # The tag must go before end_run because the fluent API needs an active run; the observable state is the
        # ruling's: a FAILED run carrying backfill_error and no idem_key.
        mlflow.set_tag("backfill_error", f"{type(exc).__name__}: {exc}")
        mlflow.end_run(status="FAILED")
        raise
    mlflow.set_tag("source_sha256", source_sha256)
    mlflow.set_tag("idem_key", idem_key)
    mlflow.end_run(status=unit.status)


def log_units(units: Sequence[Unit], runs_root: Path) -> dict[str, dict[str, int]]:
    """Create (or skip) every planned run. Returns created/skipped counts per kind."""
    experiment_id = _experiment_id()
    roots = (runs_root, REPO_ROOT)
    created: dict[str, int] = defaultdict(int)
    skipped: dict[str, int] = defaultdict(int)
    for unit in units:
        source_sha256 = sha256_of(unit.source_file)
        key = _idem_key(unit)
        if _already_logged(experiment_id, unit.name, key):
            print(f"SKIP {unit.kind}: {unit.name}")
            skipped[unit.kind] += 1 + len(unit.children)
            continue
        _log_unit(unit, source_sha256, key, roots)
        created[unit.kind] += 1 + len(unit.children)
        print(f"LOG {unit.kind}: {unit.name} (+{len(unit.children)} children)")
    return {"created": dict(created), "skipped": dict(skipped)}


def attach_artifact(run_name: str, path: Path) -> bool:
    """Add one file to the artifacts of an already logged, complete run. Returns False when it is already there.

    The run must be the only one of that name carrying an idem_key. A different file already stored under the same
    name is refused, never overwritten.
    """
    if not path.is_file():
        raise ValueError(f"artifact source file is missing: {path}")
    experiment_id = _experiment_id()
    client = mlflow.MlflowClient()
    found = [
        run
        for run in client.search_runs([experiment_id], filter_string=f"tags.`mlflow.runName` = '{run_name}'")
        if "idem_key" in run.data.tags
    ]
    if len(found) != 1:
        raise ValueError(f"expected exactly one complete run named {run_name!r}, found {len(found)}")
    run_id = found[0].info.run_id
    if path.name in {item.path for item in client.list_artifacts(run_id)}:
        with tempfile.TemporaryDirectory() as folder:
            stored = Path(client.download_artifacts(run_id, path.name, folder))
            if sha256_of(stored) != sha256_of(path):
                raise ValueError(f"run {run_name!r} already holds a different {path.name}")
        return False
    client.log_artifact(run_id, str(path))
    return True


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Log the saved Level 18 results to MLflow, idempotently.")
    parser.add_argument("--runs-root", default=DEFAULT_RUNS_ROOT)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--run",
        type=Path,
        action="append",
        help="out-root of a finished full-driver run; repeatable. Logs only these, not the --runs-root plan",
    )
    parser.add_argument(
        "--attach",
        nargs=2,
        metavar=("RUN_NAME", "FILE"),
        help="only add FILE to the artifacts of the logged run RUN_NAME",
    )
    args = parser.parse_args(argv)

    if args.attach:
        run_name, file_text = args.attach
        added = attach_artifact(run_name, Path(file_text).resolve())
        print(f"[attach] {run_name}: {'added' if added else 'already present'} {Path(file_text).name}")
        return 0

    if args.run:
        resolved = [path.resolve() for path in args.run]
        if len({path.parent for path in resolved}) != 1:
            raise ValueError(f"every --run folder must share one parent folder; got {[str(p) for p in resolved]}")
        runs_root = resolved[0].parent
        units = [unit for path in resolved for unit in build_run_units(path)]
    else:
        runs_root = Path(args.runs_root).resolve()
        units = build_units(runs_root, FAILED_YAML)

    if args.dry_run:
        counts = plan_counts(units)
        for kind in sorted(counts):
            print(f"[dry-run] {kind}: {counts[kind]} run(s)")
        print(f"[dry-run] total: {sum(counts.values())} run(s)")
        return 0

    result = log_units(units, runs_root)
    for kind in sorted(set(result["created"]) | set(result["skipped"])):
        print(f"[real] {kind}: created {result['created'].get(kind, 0)}, skipped {result['skipped'].get(kind, 0)}")
    print(f"[real] total created {sum(result['created'].values())}, total skipped {sum(result['skipped'].values())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
