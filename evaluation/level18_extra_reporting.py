"""
WHAT: Extra reporting on the frozen Run A account model: pure model quality, the capacity curve
against a truth oracle, hidden-mule recall by depth with account counts, and held-out family recall,
per world and aggregated across worlds.

WHY: The single test scoring froze every choice before these metrics existed. They are reporting
only: they change no choice and no verdict, and they reuse the pilot's own scoring, escalation and
metric functions so their numbers mean the same thing. Amendment A-052 records why they were added
after the fact.

TERM: frozen model: the stage 5 fit checkpoint, loaded without retraining, re-tuning or any
threshold change.
TERM: oracle: a score of 1.0 for every account-day whose hidden truth role is a mule role and 0.0
otherwise, escalated through the pilot's own capacity rule; it is a ceiling, never a model.
TERM: normalized recall: model recall at capacity divided by oracle recall at capacity at the same
capacity point and repeat policy.

The command line writes one JSON report. Every guard fails with a plain ValueError naming the
offending value; nothing is guessed and no failure is swallowed.
"""

import argparse
import ast
import hashlib
import statistics
import subprocess
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date
from pathlib import Path
from typing import Any

import orjson
import pyarrow
import pyarrow.compute as pyarrow_compute

from decisions.mule_account_model import (
    MuleModelConfig,
    load_mule_model_config,
    predict_account_days,
    set_mule_model_config_path,
)
from decisions.pu_learning import PUBaggedAccountModel
from decisions.rules_config import (
    RulesConfig,
    load_rules_config,
    load_rules_config_with_labels,
    with_thresholds,
)
from evaluation.dataset_truth_join import join_truth
from evaluation.hyperparameter_search import validation_truth_average_precision
from evaluation.level17_scoring import (
    MULE_ROLES,
    AccountKey,
    ScoredDay,
    anomaly_restricted_days,
    daily_escalations,
    ground_sets,
    one_row_per_account_day,
    score_account_days,
)
from evaluation.level18_sanity import (
    ORACLE_GENUINE_SCORE,
    ORACLE_MULE_SCORE,
    ORACLE_THRESHOLD,
    load_escalation_repeat_policy,
)
from evaluation.rules_metrics import hidden_mule_recall_by_depth, recall_at_capacity, summarise_across_worlds
from evaluation.run_level_17 import (
    load_evaluation_settings,
    load_split_datasets,
    read_world_base_rate,
    reported_accounts,
)
from evaluation.trivial_world import load_evaluation_config
from evaluation.truth import TruthStore, load_truth

__all__ = [
    "build_split_report",
    "family_recall",
    "load_frozen_model",
    "main",
    "normalized_recall",
    "require_allowed_drift",
    "require_consistency",
    "require_mule_model_config_sha",
    "run_extra_reporting",
    "verify_code_drift",
]

REPO_ROOT = Path(__file__).resolve().parent.parent

ACCOUNT_LABELS_PATH = "features/account_labels.py"

# Ruling 3: features/account_labels.py may change since the freeze only by the exact log-level swap
# and its comment from commit 9a6690d, so the guard compares the changed lines against these.
PINNED_ACCOUNT_LABELS_CHANGED_LINES: tuple[str, ...] = (
    "-        logger.info(",
    "+        # A complaint naming another bank is expected and normal, so INFO level would flood the log with it.",
    "+        logger.debug(",
)

# Ruling 1: every path that may differ between the checkpoint's code version and the current HEAD,
# because each one landed after the freeze and none touches offline scoring.
ALLOWED_DRIFT_EXACT: frozenset[str] = frozenset(
    {
        ".github/workflows/ci.yml",
        "config/leakage_checks.yaml",
        "evaluation/leakage_checks_5_6.py",
        "evaluation/level18_extra_reporting.py",
        "evaluation/truth.py",
        ACCOUNT_LABELS_PATH,
    }
)
ALLOWED_DRIFT_PREFIXES: tuple[str, ...] = ("docs/", "tests/")

# Ruling 3: the verified import chain, recorded in the report so a reader can retrace it.
ACCOUNT_LABELS_IMPORT_CHAIN = (
    "evaluation.level17_scoring -> features.account_snapshots -> features.m1_features -> "
    "features.decision_reads -> features.account_labels"
)

# Ruling 3: functions that write through the live complaint path; no offline module may call them.
COMPLAINT_APPLY_FUNCTIONS: frozenset[str] = frozenset({"apply_complaint_row", "apply_complaints_since"})
OFFLINE_SCAN_DIRECTORIES: tuple[str, ...] = ("evaluation", "decisions")

NORMALIZED_ORACLE_ZERO_REASON = "oracle recall at capacity is 0, so the model/oracle ratio is undefined"

PURE_METRICS: tuple[str, ...] = (
    "account_day_average_precision",
    "recall_at_cut",
    "precision_at_cut",
    "genuine_flag_rate",
)
CAPACITY_PARTS: tuple[str, ...] = ("model", "oracle", "normalized")


@dataclass(frozen=True)
class _FrozenModel:
    """
    WHAT: The frozen stage 5 model with the cuts and metadata it was fitted with.

    WHY: Every number in the report must come from this one checkpoint, so the model, the xgboost
    cut, the tuned rules thresholds and the audit record travel together.
    """

    model: PUBaggedAccountModel
    xgboost_cut: float
    rules_thresholds: dict[str, float | int]
    checkpoint_code_version: str
    checkpoint_directory: Path
    result_json: Path
    drift: dict[str, Any]


@dataclass(frozen=True)
class _WorldEvidence:
    """
    WHAT: One world's scored account-days and the truth sets every per-world metric reads.

    WHY: Building the sets once per world keeps every metric on exactly the same rows and the same
    cut, instead of rescoring the world for each metric.
    """

    world_id: str
    run_id: str
    xgboost_cut: float
    scored: list[ScoredDay]
    probabilities: list[float]
    day_rows: list[dict[str, Any]]
    mules: set[AccountKey]
    genuine_active: set[AccountKey]
    depth_of: dict[AccountKey, int]
    flagged_at_cut: set[AccountKey]
    reported: set[AccountKey]


def _git(*args: str) -> str:
    """
    WHAT: Runs one git command in the repository and returns its stdout.

    WHY: The provenance guards must ask the real repository, not a cached value, so every answer
    comes from the git binary itself.
    """
    return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=True).stdout


def _changed_lines(diff_text: str) -> list[str]:
    """
    WHAT: Returns the added and removed lines of a unified diff, excluding the file headers.

    WHY: The account_labels guard compares the diff's own +/- lines, so the comparison must see
    exactly what git reports and nothing else.
    """
    return [
        line for line in diff_text.splitlines() if line.startswith(("+", "-")) and not line.startswith(("+++", "---"))
    ]


def _is_allowed_path(path: str) -> bool:
    """True when a changed path is either allow-listed exactly or sits under an allowed prefix."""
    return path in ALLOWED_DRIFT_EXACT or path.startswith(ALLOWED_DRIFT_PREFIXES)


def require_allowed_drift(
    changed_paths: Sequence[str],
    account_labels_changed_lines: Sequence[str] | None,
) -> None:
    """
    WHAT: Refuses any path changed since the checkpoint's code version that the freeze ruling did
    not allow, and requires features/account_labels.py to carry exactly the pinned change.

    WHY: The frozen model's numbers are only comparable while no offline scoring code changed
    after the fit; any other changed path means this report would describe different code.

    Args:
        changed_paths: Paths from `git diff --name-only <checkpoint code version> HEAD`.
        account_labels_changed_lines: The +/- lines of that file's diff, or None when the file did
            not change.

    Raises:
        ValueError: If any path is not allowed, or the account_labels change is anything but the
            pinned log-level swap.
    """
    violations = sorted(path for path in changed_paths if not _is_allowed_path(path))
    if violations:
        raise ValueError(
            "These paths changed since the checkpoint's code version and are not allowed: " + ", ".join(violations)
        )
    if ACCOUNT_LABELS_PATH not in changed_paths:
        return
    if account_labels_changed_lines is None:
        raise ValueError(f"{ACCOUNT_LABELS_PATH} changed, but its diff lines were not supplied")
    lines = list(account_labels_changed_lines)
    expected = list(PINNED_ACCOUNT_LABELS_CHANGED_LINES)
    if lines != expected:
        raise ValueError(f"{ACCOUNT_LABELS_PATH} changed beyond the pinned log-level swap. Found lines: " + repr(lines))


def require_clean_worktree() -> None:
    """
    WHAT: Refuses to run when the working tree holds any change.

    WHY: The drift guard compares commits; an uncommitted edit would change the scoring code
    without appearing in that comparison.

    Raises:
        ValueError: If `git status --porcelain` reports anything.
    """
    status = _git("status", "--porcelain")
    if status.strip():
        raise ValueError("The working tree is not clean:\n" + status.rstrip())


def verify_code_drift(checkpoint_code_version: str) -> dict[str, Any]:
    """
    WHAT: Compares the repository against the checkpoint's code version, applies the allow-list and
    returns the audit record.

    WHY: The fit record names the commit the model was fitted at, so this diff is the exact list of
    code changes a reader must weigh when reading the report.

    Args:
        checkpoint_code_version: The `code_version` recorded in the checkpoint's fingerprint.

    Returns:
        A record with the baseline commit, the changed paths and the pinned account_labels lines.

    Raises:
        ValueError: If any changed path is not allowed or the account_labels change differs.
    """
    if not checkpoint_code_version:
        raise ValueError("The checkpoint fingerprint must record a non-empty code_version")
    changed_paths = [line for line in _git("diff", "--name-only", checkpoint_code_version, "HEAD").splitlines() if line]
    account_labels_lines: list[str] | None = None
    if ACCOUNT_LABELS_PATH in changed_paths:
        account_labels_lines = _changed_lines(_git("diff", checkpoint_code_version, "HEAD", "--", ACCOUNT_LABELS_PATH))
    require_allowed_drift(changed_paths, account_labels_lines)
    return {
        "baseline": checkpoint_code_version,
        "changed_paths": changed_paths,
        "account_labels_changed_lines": account_labels_lines,
    }


def _scan_offline_complaint_calls() -> tuple[int, list[str]]:
    """
    WHAT: Scans the offline packages for calls to the live complaint path.

    WHY: features/account_labels.py is imported transitively by the scoring path, so the guard must
    prove that no offline module reaches the two live-writing functions; the log-level swap is safe
    only while that holds.

    Returns:
        Number of scanned files and the offending "path:line" sites, if any.
    """
    sites: list[str] = []
    scanned = 0
    for directory in OFFLINE_SCAN_DIRECTORIES:
        for path in sorted((REPO_ROOT / directory).rglob("*.py")):
            scanned += 1
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                function = node.func
                if isinstance(function, ast.Name):
                    name = function.id
                elif isinstance(function, ast.Attribute):
                    name = function.attr
                else:
                    continue
                if name in COMPLAINT_APPLY_FUNCTIONS:
                    sites.append(f"{path.relative_to(REPO_ROOT)}:{node.lineno}")
    return scanned, sites


def _sha256_file(path: Path) -> str:
    """
    WHAT: Returns the SHA-256 hex digest of a file's bytes.

    Raises:
        ValueError: If the file is missing.
    """
    if not path.is_file():
        raise ValueError(f"Expected file is missing: {path}")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_mule_model_config_sha(config_path: Path, expected_sha256: str) -> str:
    """
    WHAT: Refuses a mule model config whose content differs from the one the checkpoint was fitted
    with.

    WHY: The feature contract and the rules-table switch both come from this file, so a different
    file would score the frozen model on a different input.

    Args:
        config_path: The config the command line was given.
        expected_sha256: The sha256 recorded in the checkpoint's fingerprint.

    Returns:
        The actual sha256, for the report manifest.

    Raises:
        ValueError: If the file is missing or its digest differs.
    """
    actual = _sha256_file(config_path)
    if actual != expected_sha256:
        raise ValueError(
            f"Mule model config {config_path} has sha256 {actual}, but the checkpoint was fitted with {expected_sha256}"
        )
    return actual


def _require_dataset_ids(recorded: Any, dataset_dirs: Sequence[Path], split: str) -> list[str]:
    """
    WHAT: Refuses a checkpoint whose recorded dataset folder names differ from the ones passed.

    WHY: The frozen model was fitted on named datasets; scoring it on any other data would make
    the report describe a different experiment.

    Args:
        recorded: The fingerprint's list of dataset folder names.
        dataset_dirs: Folders passed on the command line for this split.
        split: Split name used in the error message.

    Returns:
        The sorted folder names that were passed.
    """
    passed = sorted(directory.name for directory in dataset_dirs)
    if not isinstance(recorded, list) or not all(isinstance(name, str) for name in recorded):
        raise ValueError(
            f"The checkpoint fingerprint's {split} dataset ids must be a list of strings; got {recorded!r}"
        )
    if list(recorded) != passed:
        raise ValueError(
            f"The checkpoint's {split} dataset ids {list(recorded)} differ from the ones passed on the CLI {passed}"
        )
    return passed


def _checkpoint_directory(stage5_out: Path) -> Path:
    """
    WHAT: Finds the single fit checkpoint folder under a stage 5 output folder.

    WHY: One run writes one final fit; two checkpoints would make "the frozen model" ambiguous, so
    the run stops instead of picking one.

    Raises:
        ValueError: If zero or more than one checkpoint's fit_record.json exists.
    """
    candidates = sorted(stage5_out.glob("final/*/checkpoint/fit_record.json"))
    if len(candidates) != 1:
        raise ValueError(f"Expected exactly one final fit checkpoint under {stage5_out}, found {len(candidates)}")
    return candidates[0].parent


def _read_fit_record(path: Path) -> dict[str, Any]:
    """
    WHAT: Reads a checkpoint's fit_record.json and requires a fingerprint mapping.

    Raises:
        ValueError: If the file is missing or malformed.
    """
    if not path.is_file():
        raise ValueError(f"Checkpoint fit record is missing: {path}")
    record = orjson.loads(path.read_bytes())
    if not isinstance(record, dict):
        raise ValueError(f"Checkpoint fit record {path} must hold a mapping")
    fingerprint = record.get("fingerprint")
    if not isinstance(fingerprint, dict):
        raise ValueError(f"Checkpoint fit record {path} must hold a fingerprint mapping")
    return record


def _read_thresholds(path: Path) -> tuple[dict[str, float | int], float]:
    """
    WHAT: Reads the checkpoint's rules thresholds and xgboost cut.

    WHY: The frozen cut and the tuned rules values drive escalation ranking, so both are read from
    the checkpoint's own thresholds.json rather than from any current config.

    Raises:
        ValueError: If the file is missing, malformed, or a value is not a number.
    """
    if not path.is_file():
        raise ValueError(f"Checkpoint thresholds file is missing: {path}")
    raw = orjson.loads(path.read_bytes())
    if not isinstance(raw, dict):
        raise ValueError(f"Checkpoint thresholds file {path} must hold a mapping")
    rules_raw = raw.get("rules_thresholds")
    if not isinstance(rules_raw, dict):
        raise ValueError(f"Checkpoint thresholds file {path} must hold a rules_thresholds mapping")
    rules_thresholds: dict[str, float | int] = {}
    for key, value in rules_raw.items():
        if not isinstance(key, str) or not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"Checkpoint threshold {key!r} must be a number; got {value!r}")
        rules_thresholds[key] = value
    cut = raw.get("xgboost_cut")
    if not isinstance(cut, (int, float)) or isinstance(cut, bool):
        raise ValueError(f"Checkpoint thresholds file {path} must hold a numeric xgboost_cut; got {cut!r}")
    return rules_thresholds, float(cut)


def _checkpoint_file_sha256s(checkpoint_directory: Path) -> dict[str, str]:
    """
    WHAT: Returns the sha256 of thresholds.json and of every file under model_bags/.

    WHY: The report must name the exact bytes of the model and cut it used, so a reader can prove
    they have the same checkpoint.
    """
    digests = {"thresholds.json": _sha256_file(checkpoint_directory / "thresholds.json")}
    model_bags = checkpoint_directory / "model_bags"
    if not model_bags.is_dir():
        raise ValueError(f"Checkpoint has no model_bags folder: {model_bags}")
    for path in sorted(model_bags.rglob("*")):
        if path.is_file():
            digests[path.relative_to(checkpoint_directory).as_posix()] = _sha256_file(path)
    return digests


def load_frozen_model(
    stage5_out: Path,
    mule_model_config_path: Path,
    training_dirs: Sequence[Path],
    validation_dirs: Sequence[Path],
) -> _FrozenModel:
    """
    WHAT: Loads the frozen stage 5 model after the provenance checks the freeze ruling requires.

    WHY: The pilot's own checkpoint loader refuses a fit made by another commit, which every
    post-freeze merge triggers. This loader replaces only that refusal with explicit checks, then
    loads through the same PUBaggedAccountModel.load call, so the model and cut are still the ones
    the pilot froze.

    Args:
        stage5_out: The stage 5 output folder, for example runs/full/run_a320_stage5c.
        mule_model_config_path: The chosen mule model config the run used.
        training_dirs: Training dataset folders passed on the command line.
        validation_dirs: Validation dataset folders passed on the command line.

    Returns:
        The frozen model with its cut, rules thresholds and audit record.

    Raises:
        ValueError: If the checkpoint is missing or ambiguous, the config sha or dataset ids
            differ, an unallowed path changed since the checkpoint, or the tree is dirty.
    """
    checkpoint_directory = _checkpoint_directory(stage5_out)
    record = _read_fit_record(checkpoint_directory / "fit_record.json")
    fingerprint = record["fingerprint"]
    config_sha256 = require_mule_model_config_sha(mule_model_config_path, fingerprint["mule_model_config_sha256"])
    _require_dataset_ids(fingerprint["training_dataset_ids"], training_dirs, "training")
    _require_dataset_ids(fingerprint["validation_dataset_ids"], validation_dirs, "validation")
    checkpoint_code_version = fingerprint["code_version"]
    if not isinstance(checkpoint_code_version, str):
        raise ValueError(f"The checkpoint fingerprint's code_version must be a string; got {checkpoint_code_version!r}")
    drift = verify_code_drift(checkpoint_code_version)
    require_clean_worktree()
    pu_method = record.get("pu_method")
    if pu_method == "none" or (checkpoint_directory / "model.ubj").is_file():
        raise ValueError(
            f"Checkpoint {checkpoint_directory} does not hold a PU bagging ensemble; its pu_method is {pu_method!r}"
        )
    rules_thresholds, xgboost_cut = _read_thresholds(checkpoint_directory / "thresholds.json")
    drift["mule_model_config_sha256"] = config_sha256
    return _FrozenModel(
        model=PUBaggedAccountModel.load(checkpoint_directory / "model_bags"),
        xgboost_cut=xgboost_cut,
        rules_thresholds=rules_thresholds,
        checkpoint_code_version=checkpoint_code_version,
        checkpoint_directory=checkpoint_directory,
        result_json=checkpoint_directory.parent / "result.json",
        drift=drift,
    )


def _frozen_rules_config(mule_config: MuleModelConfig, rules_thresholds: dict[str, float | int]) -> RulesConfig:
    """
    WHAT: Rebuilds the rules table the run scored with: the configured variant plus the checkpoint's
    tuned thresholds.

    WHY: Escalation ranking sorts on rules_score first, so the report must use the same rules table
    the pilot used or the capacity numbers would move.

    Raises:
        ValueError: If the checkpoint names a threshold the rules table does not hold.
    """
    base_config = (
        load_rules_config_with_labels() if mule_config.rules_plus_labels_and_registry is True else load_rules_config()
    )
    return with_thresholds(base_config, rules_thresholds, "frozen stage 5 checkpoint thresholds")


def _load_capacity_points() -> tuple[float, float, float]:
    """
    WHAT: Returns the three capacity points the config defines: sweep minimum, headline, sweep
    maximum, in that order.

    WHY: The brief fixes the points to exactly the config's, so this loader validates them instead
    of accepting a caller-supplied list.

    Raises:
        ValueError: If the capacity block or a value is missing or malformed, or the points are not
            strictly increasing.
    """
    raw = load_evaluation_config()
    if not isinstance(raw, Mapping):
        raise ValueError("config/evaluation.yaml must hold a mapping at the root")
    capacity = raw.get("capacity")
    if not isinstance(capacity, Mapping):
        raise ValueError("config/evaluation.yaml requires a 'capacity' mapping")
    points: list[float] = []
    for key in (
        "sweep_min_pct_active_accounts_per_day",
        "headline_pct_active_accounts_per_day",
        "sweep_max_pct_active_accounts_per_day",
    ):
        value = capacity.get(key)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(f"config/evaluation.yaml capacity requires a numeric '{key}'; got {value!r}")
        points.append(float(value))
    sweep_min, headline, sweep_max = points
    if not sweep_min < headline < sweep_max:
        raise ValueError(
            f"Capacity points must be strictly increasing; got min {sweep_min}, headline {headline}, max {sweep_max}"
        )
    return sweep_min, headline, sweep_max


def _load_held_out_families() -> list[str]:
    """
    WHAT: Returns the active held-out tactic families from config/evaluation.yaml.

    WHY: The report covers exactly the families the config declares held out, so the list comes
    from the pre-registered config rather than from the data.

    Raises:
        ValueError: If the block, its tactics list, or a name is malformed, or a name repeats.
    """
    raw = load_evaluation_config()
    block = raw.get("held_out_tactics") if isinstance(raw, Mapping) else None
    if not isinstance(block, Mapping):
        raise ValueError("config/evaluation.yaml requires a 'held_out_tactics' mapping")
    tactics = block.get("tactics")
    if not isinstance(tactics, list) or not tactics:
        raise ValueError(f"config/evaluation.yaml held_out_tactics.tactics must be a non-empty list; got {tactics!r}")
    families: list[str] = []
    for name in tactics:
        if not isinstance(name, str) or not name:
            raise ValueError(f"Held-out family names must be non-empty strings; got {name!r}")
        families.append(name)
    if len(set(families)) != len(families):
        raise ValueError(f"Held-out family names must not repeat; got {families}")
    return families


def _rows_for_world(table: pyarrow.Table, column: str, value: str) -> list[dict[str, Any]]:
    """
    WHAT: Returns one world's rows of an Arrow table as Python dicts.

    WHY: Filtering on the Arrow table first keeps only one world as Python rows, which is how the
    pilot keeps its memory flat at full scale.
    """
    return table.filter(pyarrow_compute.equal(table.column(column), value)).to_pylist()


def _world_evidence(
    world_id: str,
    run_id: str,
    snapshots: pyarrow.Table,
    labels: pyarrow.Table,
    model: PUBaggedAccountModel,
    rules_config: RulesConfig,
    xgboost_cut: float,
) -> _WorldEvidence:
    """
    WHAT: Scores one world exactly as the pilot's xgboost scorer does and builds its truth sets.

    WHY: The consistency gate is only meaningful when the recomputed score comes from the same
    pipeline as the stored one: one row per account-day, rules for ranking, then the frozen model's
    probability in the anomaly slot.

    Args:
        world_id: World to keep from the joined tables.
        run_id: Run the world belongs to, used to read that run's registry labels.
        snapshots: Joined snapshot table of the dataset.
        labels: The dataset's labels table.
        model: The frozen account model.
        rules_config: Rules table with the checkpoint's tuned thresholds.
        xgboost_cut: The frozen inclusive account-day cut.

    Returns:
        The world's scored rows, probabilities and truth sets.
    """
    day_rows = one_row_per_account_day(_rows_for_world(snapshots, "world_id", world_id))
    base_scored = score_account_days(day_rows, rules_config, None)
    probabilities = predict_account_days(model, day_rows)
    scored = [
        replace(scored_day, anomaly_score=probability)
        for scored_day, probability in zip(base_scored, probabilities, strict=True)
    ]
    mules, genuine_active, depth_of, _ = ground_sets(scored)
    flagged_at_cut = set(anomaly_restricted_days(scored, xgboost_cut))
    reported_ids = reported_accounts(_rows_for_world(labels, "run_id", run_id), run_id)
    reported = {account for account in depth_of if account[2] in reported_ids}
    return _WorldEvidence(
        world_id=world_id,
        run_id=run_id,
        xgboost_cut=xgboost_cut,
        scored=scored,
        probabilities=probabilities,
        day_rows=day_rows,
        mules=mules,
        genuine_active=genuine_active,
        depth_of=depth_of,
        flagged_at_cut=flagged_at_cut,
        reported=reported,
    )


def normalized_recall(model_recall: float, oracle_recall: float) -> float | None:
    """
    WHAT: Returns model recall divided by oracle recall, or None when the oracle recall is 0.

    WHY: The ratio says how much of the achievable ceiling the model reached; it is undefined when
    the ceiling is 0 and is deliberately kept above 1.0 when the model over-reaches the capped
    oracle on that world.

    Args:
        model_recall: Model recall at capacity at one capacity point.
        oracle_recall: Oracle recall at capacity at the same point and repeat policy.

    Returns:
        The ratio, or None when oracle_recall is 0.
    """
    if oracle_recall == 0:
        return None
    return model_recall / oracle_recall


def require_consistency(comparisons: Sequence[tuple[str, float, float]]) -> None:
    """
    WHAT: Refuses the run when any world's recomputed recall at capacity differs from the value the
    frozen run stored for it.

    WHY: The stored value was produced by the exact same pipeline before the model froze; equality
    is the proof that this report is scoring the same model on the same rows with the same cut.

    Args:
        comparisons: (world id, stored recall, recomputed recall) triples.

    Raises:
        ValueError: If no world was compared, or any triple's values differ; at most five worlds
            are named.
    """
    if not comparisons:
        raise ValueError("The consistency gate requires at least one world to compare")
    mismatched = [
        (world_id, stored, recomputed) for world_id, stored, recomputed in comparisons if stored != recomputed
    ]
    if mismatched:
        lines = [
            f"{world_id}: stored {stored!r}, recomputed {recomputed!r}"
            for world_id, stored, recomputed in mismatched[:5]
        ]
        raise ValueError(
            "Consistency gate failed: recomputed recall at capacity does not match the stored value:\n"
            + "\n".join(lines)
        )


def _stored_test_recalls(result_json: Path) -> dict[str, float]:
    """
    WHAT: Reads every test world's stored xgboost recall at capacity from the frozen run's
    result.json.

    WHY: Those stored numbers are the reference the consistency gate compares against, so they are
    read once and required for every world.

    Raises:
        ValueError: If the file is missing, misstructured, or a stored value is not a number.
    """
    if not result_json.is_file():
        raise ValueError(f"Frozen run result file is missing: {result_json}")
    result = orjson.loads(result_json.read_bytes())
    per_world = result.get("per_world") if isinstance(result, Mapping) else None
    if not isinstance(per_world, Mapping) or not per_world:
        raise ValueError(f"{result_json} must hold a non-empty per_world mapping")
    recalls: dict[str, float] = {}
    for world_id, block in per_world.items():
        scorers = block.get("scorers") if isinstance(block, Mapping) else None
        xgboost = scorers.get("xgboost") if isinstance(scorers, Mapping) else None
        value = xgboost.get("recall_at_capacity") if isinstance(xgboost, Mapping) else None
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise ValueError(
                f"{result_json} world {world_id!r} has no numeric xgboost recall_at_capacity; got {value!r}"
            )
        recalls[str(world_id)] = float(value)
    return recalls


def _consistency_comparisons(
    test_dirs: Sequence[Path],
    truth_root: Path,
    frozen: _FrozenModel,
    rules_config: RulesConfig,
    headline_pct: float,
    repeat_policy: str,
    stored_recalls: Mapping[str, float],
) -> list[tuple[str, float, float]]:
    """
    WHAT: Recomputes every test world's xgboost recall at capacity at the headline point and pairs
    it with the stored value.

    WHY: This is the mandatory gate. It stops after five mismatches, because the brief asks for the
    first five mismatching worlds rather than a full scan of an already-invalid run.

    Raises:
        ValueError: If a world has no stored value, or a world repeats across datasets.
    """
    comparisons: list[tuple[str, float, float]] = []
    seen: set[str] = set()
    mismatches = 0
    for dataset_dir, dataset in zip(test_dirs, load_split_datasets("test", test_dirs), strict=True):
        _, snapshots = join_truth(dataset_dir, truth_root)
        labels = dataset.labels
        for world in dataset.manifest["worlds"]:
            world_id = str(world["world_id"])
            run_id = str(world["run_id"])
            if world_id in seen:
                raise ValueError(f"World {world_id!r} repeats across the test datasets")
            seen.add(world_id)
            if world_id not in stored_recalls:
                raise ValueError(f"The frozen run stored no recall at capacity for test world {world_id!r}")
            evidence = _world_evidence(
                world_id, run_id, snapshots, labels, frozen.model, rules_config, frozen.xgboost_cut
            )
            escalations = daily_escalations(evidence.scored, "anomaly", frozen.xgboost_cut, headline_pct, repeat_policy)
            recomputed = recall_at_capacity(escalations, evidence.mules)
            stored = stored_recalls[world_id]
            comparisons.append((world_id, stored, recomputed))
            if stored != recomputed:
                mismatches += 1
                if mismatches >= 5:
                    break
        if mismatches >= 5:
            break
    return comparisons


def _never_mule_genuine(evidence: _WorldEvidence) -> set[AccountKey]:
    """
    WHAT: Returns the world's genuine-role accounts that are never mule accounts.

    WHY: A recruited mule carries genuine days before its mule days, so ground_sets records it in
    both populations; counting a flagged mule as an innocent flag would overstate genuine harm.

    Args:
        evidence: The world's scored account-days and truth sets.

    Returns:
        The accounts with at least one genuine day and no mule day.
    """
    return evidence.genuine_active - evidence.mules


def _pure_metrics(evidence: _WorldEvidence) -> dict[str, Any]:
    """
    WHAT: Computes the cut-independent and cut-based account metrics of one world.

    WHY: Average precision judges the ranking over every account-day, while recall, precision and
    the genuine flag rate judge the single frozen account-level decision; the genuine flag rate
    divides by genuine accounts that are never mules, because a recruited mule's genuine days
    would otherwise count a correctly flagged mule as an innocent flag.

    Returns:
        A mapping from metric name to its number, or to {"skipped": reason} where the world holds
        no population for it.
    """
    pure: dict[str, Any] = {}
    has_mule_days = any(scored_day.role in MULE_ROLES for scored_day in evidence.scored)
    has_non_mule_days = any(scored_day.role not in MULE_ROLES for scored_day in evidence.scored)
    if has_mule_days and has_non_mule_days:
        pure["account_day_average_precision"] = validation_truth_average_precision(
            evidence.probabilities, evidence.scored
        )
    else:
        pure["account_day_average_precision"] = {
            "skipped": "world holds only mule-role account-days or only non-mule account-days"
        }

    if evidence.mules:
        recalled = evidence.flagged_at_cut & evidence.mules
        pure["recall_at_cut"] = len(recalled) / len(evidence.mules)
    else:
        pure["recall_at_cut"] = {"skipped": "world has no mule account"}

    if evidence.flagged_at_cut:
        pure["precision_at_cut"] = len(evidence.flagged_at_cut & evidence.mules) / len(evidence.flagged_at_cut)
    else:
        pure["precision_at_cut"] = {"skipped": "no account reached the frozen cut"}

    never_mule_genuine = _never_mule_genuine(evidence)
    if never_mule_genuine:
        pure["genuine_flag_rate"] = len(evidence.flagged_at_cut & never_mule_genuine) / len(never_mule_genuine)
    else:
        pure["genuine_flag_rate"] = {"skipped": "world has no genuine account"}
    return pure


def _capacity_metrics(
    evidence: _WorldEvidence,
    capacity_points: Sequence[float],
    headline_pct: float,
    repeat_policy: str,
) -> tuple[dict[str, Any], dict[date, list[AccountKey]]]:
    """
    WHAT: Computes model, oracle and normalized recall at capacity at every configured point.

    WHY: The oracle applies the pilot's own escalation and capacity rule to a perfect truth score,
    so the normalized value says how much of the achievable ceiling the frozen model reached.

    Returns:
        The per-point block and the model's escalations at the headline point, which the family
        metric reuses.
    """
    oracle_scored = [
        replace(
            scored_day,
            anomaly_score=ORACLE_MULE_SCORE if scored_day.role in MULE_ROLES else ORACLE_GENUINE_SCORE,
        )
        for scored_day in evidence.scored
    ]
    block: dict[str, Any] = {}
    headline_escalations: dict[date, list[AccountKey]] | None = None
    for point in capacity_points:
        model_escalations = daily_escalations(evidence.scored, "anomaly", evidence.xgboost_cut, point, repeat_policy)
        if point == headline_pct:
            headline_escalations = model_escalations
        entry: dict[str, Any] = {}
        if not evidence.mules:
            entry["model"] = {"skipped": "world has no mule account"}
            entry["oracle"] = {"skipped": "world has no mule account"}
            entry["normalized"] = {"skipped": "world has no mule account"}
        else:
            oracle_escalations = daily_escalations(oracle_scored, "anomaly", ORACLE_THRESHOLD, point, repeat_policy)
            model_recall = recall_at_capacity(model_escalations, evidence.mules)
            oracle_recall = recall_at_capacity(oracle_escalations, evidence.mules)
            entry["model"] = model_recall
            entry["oracle"] = oracle_recall
            ratio = normalized_recall(model_recall, oracle_recall)
            entry["normalized"] = {"skipped": NORMALIZED_ORACLE_ZERO_REASON} if ratio is None else ratio
        block[str(point)] = entry
    if headline_escalations is None:
        raise ValueError("Capacity points must include the headline point")
    return block, headline_escalations


def _hidden_counts_by_depth(depth_of: Mapping[AccountKey, int], reported: set[AccountKey]) -> dict[int, int]:
    """
    WHAT: Counts the unreported mule accounts at each depth.

    WHY: hidden_mule_recall_by_depth reports a share per depth but not the population it divides
    by, and a share over five accounts must not be read like a share over five hundred.
    """
    counts: dict[int, int] = {}
    for account, depth in depth_of.items():
        if account not in reported:
            counts[depth] = counts.get(depth, 0) + 1
    return counts


def family_recall(
    day_rows: Sequence[Mapping[str, Any]],
    flagged: set[AccountKey],
    escalations: dict[date, list[AccountKey]],
    store: TruthStore,
    families: Sequence[str],
) -> dict[str, dict[str, float | int]]:
    """
    WHAT: For each held-out family present in the world, counts its mule accounts and measures
    their recall at capacity and at the frozen cut.

    WHY: A family the model was never trained on must be judged on its own population; pooling it
    with every other mule would hide a weakness in one tactic behind strength in another.

    Args:
        day_rows: One row per account-day of the world, carrying world_id, bank_id, account_id,
            truth_role and truth_owner.
        flagged: Account keys whose peak score reached the frozen cut.
        escalations: The model's escalations at the headline capacity point.
        store: The world's run truth store, which resolves each owner gang's family.
        families: The configured held-out families.

    Returns:
        family -> {accounts, recall_at_capacity, recall_at_cut}; a family absent from the world is
        omitted.

    Raises:
        ValueError: If a mule-role row has no owner gang, or an account carries rows from two
            different families.
    """
    family_of_account: dict[AccountKey, str] = {}
    for row in day_rows:
        if row.get("truth_role") not in MULE_ROLES:
            continue
        identifiers: list[str] = []
        for field in ("world_id", "bank_id", "account_id"):
            value = row.get(field)
            if not isinstance(value, str) or not value:
                raise ValueError(f"family_recall requires a non-empty string '{field}' on every row")
            identifiers.append(value)
        owner = row.get("truth_owner")
        if not isinstance(owner, str) or not owner:
            raise ValueError(f"Mule-role row {identifiers} has no owner gang")
        family = store.gang_family(owner)
        account = (identifiers[0], identifiers[1], identifiers[2])
        known = family_of_account.get(account)
        if known is not None and known != family:
            raise ValueError(f"Mule account {account} carries rows from families {known} and {family}")
        family_of_account[account] = family

    results: dict[str, dict[str, float | int]] = {}
    for family in families:
        members = {account for account, account_family in family_of_account.items() if account_family == family}
        if not members:
            continue
        results[family] = {
            "accounts": len(members),
            "recall_at_capacity": recall_at_capacity(escalations, members),
            "recall_at_cut": len(flagged & members) / len(members),
        }
    return results


def _world_metrics(
    evidence: _WorldEvidence,
    store: TruthStore,
    families: Sequence[str],
    capacity_points: Sequence[float],
    headline_pct: float,
    repeat_policy: str,
    stored_recall: float | None,
) -> dict[str, Any]:
    """
    WHAT: Assembles every per-world metric block of the report.

    WHY: One function per world keeps the split scan flat in memory and makes each world's record
    a single auditable object, including its consistency value for test worlds.
    """
    record: dict[str, Any] = {
        "run_id": evidence.run_id,
        "account_days": len(evidence.scored),
        "mule_accounts": len(evidence.mules),
        "genuine_accounts": len(_never_mule_genuine(evidence)),
    }
    record["pure"] = _pure_metrics(evidence)
    capacity, headline_escalations = _capacity_metrics(evidence, capacity_points, headline_pct, repeat_policy)
    if stored_recall is not None:
        record["consistency"] = {
            "stored_recall_at_capacity": stored_recall,
            "recomputed_recall_at_capacity": capacity[str(headline_pct)]["model"],
        }
    record["capacity"] = capacity

    depth_recall = hidden_mule_recall_by_depth(evidence.flagged_at_cut, evidence.depth_of, evidence.reported)
    depth_counts = _hidden_counts_by_depth(evidence.depth_of, evidence.reported)
    record["hidden_by_depth"] = {
        str(depth): {"recall": value, "accounts": depth_counts[depth]} for depth, value in sorted(depth_recall.items())
    }
    record["families"] = family_recall(
        evidence.day_rows, evidence.flagged_at_cut, headline_escalations, store, families
    )
    return record


def _summarise_metric(
    values: list[float],
    skipped: Counter[str],
    min_worlds: int,
) -> dict[str, Any]:
    """
    WHAT: Summarises one metric across the worlds that produced it.

    WHY: A median and spread matter only beside how many worlds were used and why the others were
    left out, so the skipped counts travel with the summary.
    """
    summary: dict[str, Any] = {
        "worlds_used": len(values),
        "worlds_skipped": sum(skipped.values()),
    }
    if values:
        summary.update(summarise_across_worlds(values, min_worlds))
    else:
        summary.update({"world_count": 0, "median": None, "spread": "no world supplied this metric"})
    if skipped:
        summary["skip_reasons"] = dict(sorted(skipped.items()))
    return summary


def _collect_metric(
    records: Sequence[Mapping[str, Any]],
    lookup: Any,
) -> tuple[list[float], Counter[str]]:
    """
    WHAT: Splits one metric's per-world entries into numbers and skip reasons.

    WHY: Every aggregate in this module reads that split identically, so the unwrapping lives once.
    """
    values: list[float] = []
    skipped: Counter[str] = Counter()
    for record in records:
        entry = lookup(record)
        if isinstance(entry, Mapping):
            skipped[str(entry["skipped"])] += 1
        else:
            values.append(float(entry))
    return values, skipped


def _aggregate(
    per_world: Mapping[str, Mapping[str, Any]],
    capacity_points: Sequence[float],
    families: Sequence[str],
    min_worlds: int,
) -> dict[str, Any]:
    """
    WHAT: Aggregates every per-world metric across the split.

    WHY: A single world can flatter or condemn the model, and the brief reports a median with a
    10th-to-90th-percentile band plus the worlds used and skipped.
    """
    records = list(per_world.values())
    pure = {
        metric: _summarise_metric(
            *_collect_metric(records, lambda record, name=metric: record["pure"][name]), min_worlds
        )
        for metric in PURE_METRICS
    }

    capacity: dict[str, Any] = {}
    for point in capacity_points:
        key = str(point)
        capacity[key] = {
            part: _summarise_metric(
                *_collect_metric(records, lambda record, name=key, which=part: record["capacity"][name][which]),
                min_worlds,
            )
            for part in CAPACITY_PARTS
        }

    depths = sorted({int(depth) for record in records for depth in record["hidden_by_depth"]})
    hidden_by_depth: dict[str, Any] = {}
    for depth in depths:
        key = str(depth)
        values: list[float] = []
        accounts: list[int] = []
        for record in records:
            entry = record["hidden_by_depth"].get(key)
            if entry is None:
                continue
            values.append(entry["recall"])
            accounts.append(entry["accounts"])
        hidden_by_depth[key] = {
            "recall": _summarise_metric(values, Counter(), min_worlds),
            "accounts_total": sum(accounts),
            "accounts_median": statistics.median(accounts),
        }

    family_blocks: dict[str, Any] = {}
    for family in families:
        accounts_total = 0
        capacity_values: list[float] = []
        cut_values: list[float] = []
        for record in records:
            entry = record["families"].get(family)
            if entry is None:
                continue
            accounts_total += entry["accounts"]
            capacity_values.append(entry["recall_at_capacity"])
            cut_values.append(entry["recall_at_cut"])
        family_blocks[family] = {
            "accounts_total": accounts_total,
            "worlds_used": len(capacity_values),
            "worlds_absent": len(records) - len(capacity_values),
            "recall_at_capacity": _summarise_metric(capacity_values, Counter(), min_worlds),
            "recall_at_cut": _summarise_metric(cut_values, Counter(), min_worlds),
        }

    return {
        "worlds_processed": len(records),
        "pure": pure,
        "capacity": capacity,
        "hidden_by_depth": hidden_by_depth,
        "families": family_blocks,
    }


def build_split_report(
    split: str,
    dataset_dirs: Sequence[Path],
    truth_root: Path,
    world_cards_root: Path,
    frozen: _FrozenModel,
    rules_config: RulesConfig,
    capacity_points: Sequence[float],
    headline_pct: float,
    families: Sequence[str],
    repeat_policy: str,
    min_worlds: int,
    stored_recalls: Mapping[str, float] | None,
) -> dict[str, Any]:
    """
    WHAT: Processes one split world by world and returns its per-world records and aggregate.

    WHY: Test and stress must be measured by identical code on identical definitions; only the
    stored-recall reference and the split's dataset folders differ.

    Args:
        split: "test" or "stress"; also the split each dataset manifest must record.
        dataset_dirs: The split's dataset folders.
        truth_root: Root folder holding one truth folder per run_id.
        world_cards_root: Root folder holding one world card per run_id.
        frozen: The frozen model and its cuts.
        rules_config: Rules table with the checkpoint's tuned thresholds.
        capacity_points: The three configured capacity percentages.
        headline_pct: The configured headline capacity percentage.
        families: The configured held-out families.
        repeat_policy: Escalation repeats policy from config/evaluation.yaml.
        min_worlds: Smallest world count for which p10 and p90 are reported.
        stored_recalls: Stored test recalls keyed by world id, or None for the stress split.

    Returns:
        A block with worlds processed, world cards checked, per-world records and the aggregate.

    Raises:
        ValueError: If a dataset records another split, a world repeats, or a world card is missing.
    """
    per_world: dict[str, dict[str, Any]] = {}
    stores: dict[str, TruthStore] = {}
    world_cards_checked = 0
    for dataset_dir, dataset in zip(dataset_dirs, load_split_datasets(split, dataset_dirs), strict=True):
        _, snapshots = join_truth(dataset_dir, truth_root)
        labels = dataset.labels
        for world in dataset.manifest["worlds"]:
            world_id = str(world["world_id"])
            run_id = str(world["run_id"])
            if world_id in per_world:
                raise ValueError(f"World {world_id!r} repeats across the {split} datasets")
            read_world_base_rate(world_cards_root, run_id)
            world_cards_checked += 1
            store = stores.get(run_id)
            if store is None:
                store = load_truth(truth_root, run_id)
                stores[run_id] = store
            stored_recall = None
            if stored_recalls is not None:
                if world_id not in stored_recalls:
                    raise ValueError(f"The frozen run stored no recall at capacity for {split} world {world_id!r}")
                stored_recall = stored_recalls[world_id]
            evidence = _world_evidence(
                world_id, run_id, snapshots, labels, frozen.model, rules_config, frozen.xgboost_cut
            )
            per_world[world_id] = _world_metrics(
                evidence, store, families, capacity_points, headline_pct, repeat_policy, stored_recall
            )
    return {
        "worlds_processed": len(per_world),
        "world_cards_checked": world_cards_checked,
        "per_world": per_world,
        "aggregate": _aggregate(per_world, capacity_points, families, min_worlds),
    }


def run_extra_reporting(
    stage5_out: Path,
    training_dirs: Sequence[Path],
    validation_dirs: Sequence[Path],
    test_dirs: Sequence[Path],
    stress_dirs: Sequence[Path],
    truth_root: Path,
    world_cards_root: Path,
    mule_model_config_path: Path,
) -> dict[str, Any]:
    """
    WHAT: Runs every guard, the consistency gate and both split scans, and returns the report.

    WHY: One entry point keeps the order fixed: provenance checks, then the gate, then the metrics,
    so a broken provenance never produces metric numbers.

    Raises:
        ValueError: On any failed guard or gate, or malformed input.
    """
    frozen = load_frozen_model(stage5_out, mule_model_config_path, training_dirs, validation_dirs)
    mule_config = load_mule_model_config()
    rules_config = _frozen_rules_config(mule_config, frozen.rules_thresholds)
    settings = load_evaluation_settings()
    sweep_min, headline, sweep_max = _load_capacity_points()
    capacity_points = (sweep_min, headline, sweep_max)
    families = _load_held_out_families()
    repeat_policy = load_escalation_repeat_policy()
    scanned_files, complaint_sites = _scan_offline_complaint_calls()
    if complaint_sites:
        raise ValueError("Offline modules call the live complaint path: " + ", ".join(complaint_sites))

    stored_recalls = _stored_test_recalls(frozen.result_json)
    comparisons = _consistency_comparisons(
        test_dirs, truth_root, frozen, rules_config, headline, repeat_policy, stored_recalls
    )
    require_consistency(comparisons)

    test_report = build_split_report(
        "test",
        test_dirs,
        truth_root,
        world_cards_root,
        frozen,
        rules_config,
        capacity_points,
        headline,
        families,
        repeat_policy,
        settings.min_test_worlds,
        stored_recalls,
    )
    stress_report = build_split_report(
        "stress",
        stress_dirs,
        truth_root,
        world_cards_root,
        frozen,
        rules_config,
        capacity_points,
        headline,
        families,
        repeat_policy,
        settings.min_test_worlds,
        None,
    )

    head = _git("rev-parse", "HEAD").strip()
    manifest = {
        "stage5_out": stage5_out.as_posix(),
        "result_json": frozen.result_json.as_posix(),
        "checkpoint": frozen.checkpoint_directory.as_posix(),
        "checkpoint_code_version": frozen.checkpoint_code_version,
        "head": head,
        "drift": frozen.drift,
        "checkpoint_sha256": _checkpoint_file_sha256s(frozen.checkpoint_directory),
        "xgboost_cut": frozen.xgboost_cut,
        "rules_thresholds": frozen.rules_thresholds,
        "capacity_points_pct": list(capacity_points),
        "headline_capacity_pct": headline,
        "min_test_worlds": settings.min_test_worlds,
        "repeat_policy": repeat_policy,
        "scorer": "xgboost",
        "mule_model_config": {
            "path": mule_model_config_path.as_posix(),
            "sha256": frozen.drift["mule_model_config_sha256"],
        },
        "datasets": {
            "training": sorted(directory.name for directory in training_dirs),
            "validation": sorted(directory.name for directory in validation_dirs),
            "test": sorted(directory.name for directory in test_dirs),
            "stress": sorted(directory.name for directory in stress_dirs),
        },
        "consistency_gate": {
            "status": "pass",
            "worlds_checked": len(comparisons),
        },
        "account_labels_guard": {
            "import_chain": ACCOUNT_LABELS_IMPORT_CHAIN,
            "offline_call_sites": [],
            "scanned_files": scanned_files,
            "pinned_changed_lines": list(PINNED_ACCOUNT_LABELS_CHANGED_LINES),
        },
        "amendment": "A-052",
    }
    return {"manifest": manifest, "test": test_report, "stress": stress_report}


def main(argv: Sequence[str] | None = None) -> int:
    """
    WHAT: Command line entry point: runs the reporting pass and writes extra_reporting.json.

    WHY: The stage 5 output folder is the one place a run's numbers are looked up, so the report is
    written beside result.json's folder.

    Returns:
        Process exit code 0 after the report is written.
    """
    parser = argparse.ArgumentParser(
        description="Extra reporting on the frozen Level 18 Run A model; reporting only, no retraining."
    )
    parser.add_argument("--stage5-out", type=Path, required=True)
    parser.add_argument("--training", type=Path, action="append", required=True)
    parser.add_argument("--validation", type=Path, action="append", required=True)
    parser.add_argument("--test", type=Path, action="append", required=True)
    parser.add_argument("--stress", type=Path, action="append", required=True)
    parser.add_argument("--truth-root", type=Path, required=True)
    parser.add_argument("--world-cards-root", type=Path, required=True)
    parser.add_argument("--mule-model-config", type=Path, required=True)
    args = parser.parse_args(argv)

    set_mule_model_config_path(args.mule_model_config)
    try:
        report = run_extra_reporting(
            stage5_out=args.stage5_out,
            training_dirs=args.training,
            validation_dirs=args.validation,
            test_dirs=args.test,
            stress_dirs=args.stress,
            truth_root=args.truth_root,
            world_cards_root=args.world_cards_root,
            mule_model_config_path=args.mule_model_config,
        )
    finally:
        set_mule_model_config_path(None)

    output_path = args.stage5_out / "extra_reporting.json"
    output_path.write_bytes(orjson.dumps(report, option=orjson.OPT_SORT_KEYS | orjson.OPT_INDENT_2) + b"\n")
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
