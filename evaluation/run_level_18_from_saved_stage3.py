"""
WHAT: Recomputes stage 4 of a Level 18 full run from its saved stage 3, then runs stage 5 on the result.

WHY: Run A's stage 3 saved each method's trained parameters and validation truth AP, which do not depend
on how accounts are counted. Its recorded genuine-restriction rate, recall and budget flag did, and were
computed with the account id alone as the account's identity, so the method choice built on them is wrong.
This module keeps the saved stage 3, retrains one model per method from the saved parameters, proves it
reproduces the saved truth AP, recomputes the operating point with the fixed code and applies the
unchanged choice rule. Stage 5, the one test scoring, then runs through the driver's own function.

Two subcommands, both refused unless the source stages were produced by code that differs from the
current code only in the files that cannot change stage 3 (see ALLOWED_CHANGED_PATHS):

    recompute-stage4  --source-stages <dir> --out <dir> --training <dir> ... --validation <dir> ...
                      --truth-root <dir> --world-cards-root <dir> --mule-model-config <file.yaml>
    run-stage5        --source-stages <dir> --stage4 <stage4_recomputed.json> --out-root <dir>
                      --training <dir> ... --validation <dir> ... --test <dir> ... [--stress <dir> ...]
                      --truth-root <dir> --world-cards-root <dir> --mule-model-config <file.yaml>

--mule-model-config is the file the original run was started with; its sha256 must equal the one in the
source fingerprints.
"""

import argparse
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import orjson

from decisions.anomaly_config import load_anomaly_detector_config
from decisions.mule_account_model import (
    MuleModelConfig,
    active_mule_model_config_path,
    load_mule_model_config,
    predict_account_days,
    set_mule_model_config_path,
    train_account_model,
)
from decisions.pu_learning import train_elkan_noto_model, train_pu_bagged_model
from decisions.rules_config import load_rules_config, load_rules_config_with_labels
from evaluation.compact_rows import load_compact_split
from evaluation.hyperparameter_search import validation_truth_average_precision
from evaluation.level17_scoring import ScoredDay
from evaluation.operating_points import load_harm_budgets
from evaluation.rules_tuning import TuningConfig, load_labels_tuning_config, load_tuning_config, tune_anomaly_threshold
from evaluation.run_level_17 import _sha256_of_file
from evaluation.run_level_18_full import (
    PU_METHODS,
    _choose_pu_method,
    _dataset_ids,
    _final_validation_and_test,
    _inputs_fingerprint,
    _split_world_ids,
    _write_final_config,
)
from evaluation.run_level_18_pilot import (
    _compact_row_keys,
    _validation_scored_in_model_row_order,
    _xgb_hyperparameters,
)

__all__ = ["check_source_guard", "main"]

#: Files the source stages folder must hold before anything is recomputed.
SOURCE_REQUIRED_FILES: tuple[str, ...] = (
    "maturity_check.json",
    "learning_curve.json",
    "tune_each_pu_method.json",
    "tune_method_none.json",
    "tune_method_elkan_noto.json",
    "tune_method_bagging.json",
    "chosen_mule_model.yaml",
)
#: Source files that carry a fingerprint, so a stale or dirty code version is caught.
FINGERPRINT_FILES: tuple[str, ...] = tuple(name for name in SOURCE_REQUIRED_FILES if name.endswith(".json"))
#: Stage 5's result file; its presence means the test set was already scored from this folder.
STAGE5_RESULT_FILE = "final_validation_and_test.json"
#: The only paths allowed to differ between the source code version and the current code. These are the
#: files changed since Run A's stage 3 that cannot alter a stage 3 number: account identity in the
#: metrics, the pilot's detector rows and dry run, and this module. Anything else (the driver, the
#: hyperparameter search, the compact loader, decisions/, config/) could, so it refuses the run.
ALLOWED_CHANGED_PATHS: frozenset[str] = frozenset(
    {
        "evaluation/level17_evaluation.py",
        "evaluation/level17_scoring.py",
        "evaluation/level18_sanity.py",
        "evaluation/rules_metrics.py",
        "evaluation/rules_tuning.py",
        "evaluation/run_level_17.py",
        "evaluation/run_level_18_pilot.py",
        "evaluation/stage5_dry_run.py",
        "evaluation/run_level_18_from_saved_stage3.py",
        # Read only by stage 5's SHAP step, which runs after every stage 1-4 decision.
        "config/shap_sample.yaml",
    }
)
#: Folders whose files may change freely: tests, and docs, because markdown (amendments, notes) cannot
#: change any stage 1-3 number.
ALLOWED_CHANGED_PREFIXES: tuple[str, ...] = ("tests/", "docs/")
#: Largest accepted difference between a retrained model's validation truth AP and the saved one. The
#: same code, data, seed and parameters on the CPU give bit-identical models, so any larger difference
#: means the retraining is not the saved model. Architect ruling 2026-11-21; it is a code constant, not a
#: config value, because a config change would itself be refused by the guard above.
REPRODUCIBILITY_TOLERANCE = 1e-12


def _run_git(arguments: Sequence[str]) -> str:
    """Runs one git command and returns its stdout, raising ValueError when git fails."""
    completed = subprocess.run(["git", *arguments], capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise ValueError(f"git {' '.join(arguments)} failed: {completed.stderr.strip()}")
    return completed.stdout


def _git_changed_paths(code_version: str) -> list[str]:
    """Returns the paths that differ between the given code version and HEAD."""
    return [line for line in _run_git(["diff", "--name-only", code_version, "HEAD"]).splitlines() if line]


def _git_tree_is_clean() -> bool:
    """Returns True when the working tree has no modified, staged or untracked files."""
    return _run_git(["status", "--porcelain"]).strip() == ""


def _git_head() -> str:
    """Returns the full hash of HEAD."""
    return _run_git(["rev-parse", "HEAD"]).strip()


def _is_allowed_path(path: str) -> bool:
    """Returns True when a changed path may differ from the source code version."""
    return path in ALLOWED_CHANGED_PATHS or path.startswith(ALLOWED_CHANGED_PREFIXES)


def _read_stage_file(path: Path) -> dict[str, Any]:
    """Reads one stage JSON file and checks it holds a fingerprint and a result."""
    payload = orjson.loads(path.read_bytes())
    if not isinstance(payload, dict) or "fingerprint" not in payload or "result" not in payload:
        raise ValueError(f"Stage file {path} must hold a fingerprint and a result")
    return payload


def check_source_guard(
    source_stages: Path,
    changed_paths_of: Callable[[str], Sequence[str]] = _git_changed_paths,
    tree_is_clean: Callable[[], bool] = _git_tree_is_clean,
) -> str:
    """
    WHAT: Refuses to continue unless the source stages are complete, untouched by stage 5, produced by
    one clean code version, and that version differs from the current code only in allowed files.

    WHY: Recomputing from a saved stage is only sound when nothing that fed the saved numbers has
    changed, and stage 5 must never run twice from one folder. The two callables are seams so the
    refusals can be tested without a git repository.

    Args:
        source_stages: The stages folder of the original run.
        changed_paths_of: Returns the paths changed between a code version and HEAD.
        tree_is_clean: Returns True when the working tree is clean.

    Returns:
        The one clean code version the source stages were produced by.

    Raises:
        ValueError: If a required file is missing, stage 5 already ran, the fingerprints disagree or
            carry a dirty code version, a path outside the allow-list changed, or the tree is dirty.
    """
    missing = [name for name in SOURCE_REQUIRED_FILES if not (source_stages / name).is_file()]
    if missing:
        raise ValueError(f"Source stages {source_stages} lack required files: {missing}")
    if (source_stages / STAGE5_RESULT_FILE).exists():
        raise ValueError(f"Source stages {source_stages} already hold {STAGE5_RESULT_FILE}; stage 5 ran from them")

    code_versions: set[str] = set()
    for name in FINGERPRINT_FILES:
        fingerprint = _read_stage_file(source_stages / name)["fingerprint"]
        code_version = fingerprint.get("code_version") if isinstance(fingerprint, dict) else None
        if not isinstance(code_version, str) or not code_version:
            raise ValueError(f"Source file {name} records no code_version in its fingerprint")
        code_versions.add(code_version)
    if len(code_versions) != 1:
        raise ValueError(f"Source stages were produced by different code versions: {sorted(code_versions)}")
    source_code_version = next(iter(code_versions))
    if source_code_version.endswith("-dirty"):
        raise ValueError(f"Source code version {source_code_version!r} is dirty, so it cannot be compared with git")

    disallowed = sorted(path for path in changed_paths_of(source_code_version) if not _is_allowed_path(path))
    if disallowed:
        raise ValueError(
            f"Code changed since {source_code_version} outside the allowed files, so stage 3 may not be reusable: "
            f"{disallowed}"
        )
    if not tree_is_clean():
        raise ValueError("The working tree is not clean; commit or stash before recomputing")
    return source_code_version


def _check_inputs_match(
    source_stages: Path,
    config_path: Path,
    training_dirs: Sequence[Path],
    validation_dirs: Sequence[Path],
    test_dirs: Sequence[Path] | None,
    stress_dirs: Sequence[Path] | None,
) -> list[int]:
    """
    WHAT: Checks the config file and datasets given now are the ones the source stages were run with,
    and returns the world counts the source run used.

    WHY: Retraining on different data or config would silently produce a different model; the
    reproducibility gate would catch it only after the expensive load. Test and stress folders are only
    compared when they are given (stage 5), because stage 4 must not touch them.

    Raises:
        ValueError: If any compared fingerprint key differs, naming the keys.
    """
    saved = _read_stage_file(source_stages / "maturity_check.json")["fingerprint"]
    counts = [int(count) for count in saved["training_world_counts"]]
    current = _inputs_fingerprint(
        config_path,
        _dataset_ids(training_dirs),
        _dataset_ids(validation_dirs),
        _dataset_ids(test_dirs) if test_dirs is not None else saved["test_dataset_ids"],
        _dataset_ids(stress_dirs) if stress_dirs is not None else saved["stress_dataset_ids"],
        counts,
    )
    differing = sorted(key for key in current if key != "code_version" and current[key] != saved.get(key))
    if differing:
        raise ValueError(f"Inputs differ from the source stages in: {differing}")
    return counts


def _fit_saved_method(
    method: str,
    params: Mapping[str, Any],
    training_matrix: Any,
    training_targets: list[int],
    training_weights: Any,
    validation_matrix: Any,
    validation_targets: list[int],
    mule_config: MuleModelConfig,
    seed: int,
) -> Any:
    """
    WHAT: Fits one model for a method from its saved parameters.

    WHY: Stage 3 fitted the winning parameters with exactly these trainer calls and arguments (the
    final fit at the end of run_level_18_full._train_pu_method); repeating them with the saved
    parameters must give the same model, which the reproducibility gate then checks.

    Raises:
        ValueError: If the method is unknown or a setting its trainer needs is missing.
    """
    device = mule_config.xgboost_device
    if method == "none":
        model, _ = train_account_model(
            training_matrix,
            training_targets,
            seed,
            int(params["n_estimators"]),
            sample_weight=training_weights,
            device=device,
            **_xgb_hyperparameters(dict(params)),
        )
        return model
    if method == "bagging":
        n_bags = mule_config.pu_bagging_n_bags
        if n_bags is None:
            raise ValueError("pu_bagging_n_bags is required for the bagging method")
        return train_pu_bagged_model(
            training_matrix,
            training_targets,
            seed,
            int(params["n_estimators"]),
            n_bags,
            int(params["unlabelled_per_positive"]),
            sample_weight=training_weights,
            device=device,
            **_xgb_hyperparameters(dict(params)),
        )
    if method == "elkan_noto":
        probability_clip = mule_config.pu_probability_clip
        if probability_clip is None:
            raise ValueError("pu_probability_clip is required for the elkan_noto method")
        model, _, _, _ = train_elkan_noto_model(
            training_matrix,
            training_targets,
            validation_matrix,
            validation_targets,
            seed,
            int(params["n_estimators"]),
            probability_clip,
            sample_weight=training_weights,
            device=device,
            **_xgb_hyperparameters(dict(params)),
        )
        return model
    raise ValueError(f"Unsupported pu_method: {method!r}")


def recompute_method_record(
    method: str,
    saved_record: Mapping[str, Any],
    probabilities: Sequence[float],
    truth_scored: Sequence[ScoredDay],
    tuning: TuningConfig,
    main_budget_pct: float,
) -> dict[str, Any]:
    """
    WHAT: Checks a retrained method reproduces its saved truth AP, then recomputes its operating point
    with the current code.

    WHY: Only the genuine-restriction rate, recall and budget flag were wrong in stage 3; everything
    else in the record is kept exactly as saved. The AP check is the gate that the model being scored
    now is the model stage 3 saved.

    Args:
        method: PU method name, for messages.
        saved_record: The method's stage 3 record.
        probabilities: The retrained model's validation probabilities, in model-row order.
        truth_scored: Validation account-days in model-row order, carrying the hidden truth roles.
        tuning: Rules tuning config, whose allowed splits tune_anomaly_threshold checks.
        main_budget_pct: The committed main genuine-restriction budget.

    Returns:
        The saved record with validation_rate_pct, validation_recall and meets_budget recomputed.

    Raises:
        ValueError: If the retrained truth AP differs from the saved one by more than the tolerance.
    """
    retrained_ap = float(validation_truth_average_precision(probabilities, truth_scored))
    saved_ap = float(saved_record["validation_truth_average_precision"])
    if abs(retrained_ap - saved_ap) > REPRODUCIBILITY_TOLERANCE:
        raise ValueError(
            f"Method {method!r} did not reproduce: saved validation truth AP {saved_ap!r}, "
            f"retrained {retrained_ap!r} (difference {abs(retrained_ap - saved_ap)!r}, "
            f"tolerance {REPRODUCIBILITY_TOLERANCE})"
        )
    method_scored = [
        replace(day, anomaly_score=probability) for day, probability in zip(truth_scored, probabilities, strict=True)
    ]
    method_tuning = tune_anomaly_threshold(method_scored, tuning, "validation", main_budget_pct)
    return {
        **saved_record,
        "validation_rate_pct": float(method_tuning.validation_rate_pct),
        "validation_recall": float(method_tuning.validation_recall),
        "meets_budget": bool(method_tuning.meets_budget),
    }


def _file_hashes(directory: Path, names: Sequence[str]) -> dict[str, str]:
    """Returns the sha256 of each named file in a folder."""
    return {name: _sha256_of_file(directory / name) for name in names}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Writes one JSON file with sorted keys, creating its folder."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(orjson.dumps(payload, option=orjson.OPT_SORT_KEYS | orjson.OPT_INDENT_2) + b"\n")


def recompute_stage4(
    source_stages: Path,
    out_dir: Path,
    training_dirs: Sequence[Path],
    validation_dirs: Sequence[Path],
    truth_root: Path,
    config_path: Path,
) -> dict[str, Any]:
    """
    WHAT: Retrains each method from stage 3's saved parameters, recomputes the operating points with
    the fixed account identity, applies the unchanged choice rule and writes stage4_recomputed.json
    and final_mule_model.yaml.

    WHY: See the module docstring. Loading, config handling and trainer calls mirror stage 3 exactly:
    the config the run started from supplies the settings, and the source chosen_mule_model.yaml is the
    active config while data is loaded and models are fitted.

    Args:
        source_stages: The stages folder of the original run.
        out_dir: Folder for stage4_recomputed.json and final_mule_model.yaml.
        training_dirs: Training dataset folders, in the original flag order.
        validation_dirs: Validation dataset folders, in the original flag order.
        truth_root: Root folder holding one truth folder per run_id.
        config_path: The model config file the original run was started with.

    Returns:
        The stage4_recomputed.json content.

    Raises:
        ValueError: If the guard refuses, inputs differ from the source run, a method does not
            reproduce its saved truth AP, or the config is not on the compact path.
    """
    source_code_version = check_source_guard(source_stages)
    counts = _check_inputs_match(source_stages, config_path, training_dirs, validation_dirs, None, None)

    mule_config = load_mule_model_config()
    if mule_config.data_path != "compact":
        raise ValueError(f"The full-run driver requires data_path 'compact'; got {mule_config.data_path!r}")
    if mule_config.rules_plus_labels_and_registry is True:
        base_rules_config = load_rules_config_with_labels()
        tuning = load_labels_tuning_config()
    else:
        base_rules_config = load_rules_config()
        tuning = load_tuning_config()
    seed = load_anomaly_detector_config().random_state
    main_budget_pct = load_harm_budgets()[0]

    learning_curve = _read_stage_file(source_stages / "learning_curve.json")["result"]
    chosen_count = int(learning_curve["chosen_training_world_count"])
    saved_records = {
        method: _read_stage_file(source_stages / f"tune_method_{method}.json")["result"] for method in PU_METHODS
    }

    chosen_config = source_stages / "chosen_mule_model.yaml"
    previous_config = active_mule_model_config_path()
    set_mule_model_config_path(chosen_config)
    try:
        training_world_ids = _split_world_ids("training", training_dirs)
        training = load_compact_split(training_dirs, world_ids=training_world_ids[:chosen_count])
        training_targets = [int(target) for target in training.rows.targets]
        validation = load_compact_split(validation_dirs, truth_root, include_days=True, rules_config=base_rules_config)
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
        validation_targets = [int(target) for target in validation.rows.targets]
        records: list[dict[str, Any]] = []
        for method in PU_METHODS:
            saved_record = saved_records[method]
            model = _fit_saved_method(
                method,
                saved_record["chosen_params"],
                training.rows.matrix,
                training_targets,
                training.rows.sample_weights,
                validation.rows.matrix,
                validation_targets,
                mule_config,
                seed,
            )
            probabilities = predict_account_days(model, validation.days.aligned_features)
            records.append(
                recompute_method_record(method, saved_record, probabilities, truth_scored, tuning, main_budget_pct)
            )
            del model, probabilities
    finally:
        set_mule_model_config_path(previous_config)

    choice = _choose_pu_method(records)
    final_config = out_dir / "final_mule_model.yaml"
    _write_final_config(chosen_config, final_config, str(choice["pu_method"]))
    result = {
        "note": "Stage 4 recomputed from the saved stage 3 with accounts identified by world, bank and account id",
        "source_stages": str(source_stages),
        "source_stage_sha256": _file_hashes(source_stages, SOURCE_REQUIRED_FILES),
        "source_code_version": source_code_version,
        "head": _git_head(),
        "changed_paths_since_source": sorted(_git_changed_paths(source_code_version)),
        "training_world_counts": counts,
        "training_world_count": chosen_count,
        "methods": [
            {
                "pu_method": record["pu_method"],
                "saved_validation_truth_average_precision": saved_records[str(record["pu_method"])][
                    "validation_truth_average_precision"
                ],
                "retrained_validation_truth_average_precision": record["validation_truth_average_precision"],
                "saved_validation_rate_pct": saved_records[str(record["pu_method"])]["validation_rate_pct"],
                "saved_validation_recall": saved_records[str(record["pu_method"])]["validation_recall"],
                "saved_meets_budget": saved_records[str(record["pu_method"])]["meets_budget"],
                "validation_rate_pct": record["validation_rate_pct"],
                "validation_recall": record["validation_recall"],
                "meets_budget": record["meets_budget"],
            }
            for record in records
        ],
        "choice": choice,
        "chosen_method": choice["pu_method"],
        "final_mule_model_path": str(final_config),
        "final_mule_model_sha256": _sha256_of_file(final_config),
    }
    _write_json(out_dir / "stage4_recomputed.json", result)
    return result


def run_stage5(
    source_stages: Path,
    stage4_path: Path,
    out_root: Path,
    training_dirs: Sequence[Path],
    validation_dirs: Sequence[Path],
    test_dirs: Sequence[Path],
    stress_dirs: Sequence[Path],
    truth_root: Path,
    world_cards_root: Path,
    config_path: Path,
) -> int:
    """
    WHAT: Runs the driver's own stage 5 for the method stage 4 recomputed and records its provenance.

    WHY: Stage 5 is the only stage allowed to read test, so it must be exactly the driver's function,
    never a reimplementation, and it must run on the very code and stage 4 file that were checked.

    Args:
        source_stages: The stages folder of the original run.
        stage4_path: stage4_recomputed.json written by recompute_stage4.
        out_root: Folder for the pilot result (final/), the final config and the provenance file.
        training_dirs: Training dataset folders, in the original flag order.
        validation_dirs: Validation dataset folders, in the original flag order.
        test_dirs: Test dataset folders, in the original flag order.
        stress_dirs: Stress dataset folders, in the original flag order.
        truth_root: Root folder holding one truth folder per run_id.
        world_cards_root: Root folder holding one world card per run_id.
        config_path: The model config file the original run was started with.

    Returns:
        The pilot's exit code.

    Raises:
        ValueError: If the guard refuses, the stage 4 file was written at another HEAD, inputs differ
            from the source run, or the driver's stage 5 raises.
    """
    source_code_version = check_source_guard(source_stages)
    stage4 = orjson.loads(stage4_path.read_bytes())
    head = _git_head()
    if stage4.get("head") != head:
        raise ValueError(f"Stage 4 file was written at HEAD {stage4.get('head')!r}, but HEAD is {head!r}")
    chosen_method = stage4.get("chosen_method")
    if chosen_method not in PU_METHODS:
        raise ValueError(f"Stage 4 file names an unknown chosen_method {chosen_method!r}")
    _check_inputs_match(source_stages, config_path, training_dirs, validation_dirs, test_dirs, stress_dirs)

    record = _final_validation_and_test(
        str(chosen_method),
        training_dirs,
        validation_dirs,
        test_dirs,
        truth_root,
        world_cards_root,
        out_root,
        source_stages / "chosen_mule_model.yaml",
        [*_dataset_ids(training_dirs), *_dataset_ids(validation_dirs), *_dataset_ids(test_dirs)],
        stress_dirs,
    )
    _write_json(
        out_root / "stage5_provenance.json",
        {
            "source_stages": str(source_stages),
            "source_stage_sha256": _file_hashes(source_stages, SOURCE_REQUIRED_FILES),
            "source_code_version": source_code_version,
            "stage4_file": str(stage4_path),
            "stage4_sha256": _sha256_of_file(stage4_path),
            "head": head,
            "record": record,
        },
    )
    return int(record["exit_code"])


def main(argv: Sequence[str] | None = None) -> int:
    """
    WHAT: Command-line entry point for the two subcommands.

    Args:
        argv: Optional argument list; defaults to sys.argv.

    Returns:
        The process exit code.
    """
    parser = argparse.ArgumentParser(description="Recompute stage 4 from a saved stage 3 and run stage 5")
    subcommands = parser.add_subparsers(dest="command", required=True)
    for name in ("recompute-stage4", "run-stage5"):
        sub = subcommands.add_parser(name)
        sub.add_argument("--source-stages", type=Path, required=True, help="stages folder of the original run")
        sub.add_argument("--training", type=Path, action="append", required=True, help="training folder; repeatable")
        sub.add_argument(
            "--validation", type=Path, action="append", required=True, help="validation folder; repeatable"
        )
        sub.add_argument("--truth-root", type=Path, required=True, help="root folder of the truth folders")
        sub.add_argument("--world-cards-root", type=Path, required=True, help="root folder of the world cards")
        sub.add_argument(
            "--mule-model-config", type=Path, required=True, help="config file the original run started with"
        )
        if name == "recompute-stage4":
            sub.add_argument("--out", type=Path, required=True, help="folder for the recomputed stage 4")
        else:
            sub.add_argument("--stage4", type=Path, required=True, help="stage4_recomputed.json")
            sub.add_argument("--out-root", type=Path, required=True, help="folder for the stage 5 output")
            sub.add_argument("--test", type=Path, action="append", required=True, help="test folder; repeatable")
            sub.add_argument("--stress", type=Path, action="append", default=[], help="stress folder; repeatable")
    args = parser.parse_args(argv)

    set_mule_model_config_path(args.mule_model_config)
    try:
        if args.command == "recompute-stage4":
            result = recompute_stage4(
                args.source_stages,
                args.out,
                tuple(args.training),
                tuple(args.validation),
                args.truth_root,
                args.mule_model_config,
            )
            print(str(args.out / "stage4_recomputed.json"))
            print(f"chosen method: {result['chosen_method']}")
            return 0
        return run_stage5(
            args.source_stages,
            args.stage4,
            args.out_root,
            tuple(args.training),
            tuple(args.validation),
            tuple(args.test),
            tuple(args.stress),
            args.truth_root,
            args.world_cards_root,
            args.mule_model_config,
        )
    finally:
        set_mule_model_config_path(None)


if __name__ == "__main__":
    raise SystemExit(main())
