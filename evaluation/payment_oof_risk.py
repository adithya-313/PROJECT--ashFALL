# File: evaluation/payment_oof_risk.py
# Purpose: Level 18 out-of-fold payee risk for the payment model: score each payment's receiver with
#          an account model that never saw that payment's own world, under the pre-registered rules.
"""
WHAT:
    Builds the `payee_risk` input of the Level 18 payment model. For every payment it attaches the
    account model's score for the receiver's latest account-day whose snapshot_time is strictly
    earlier than the payment's decision_time, but only when the receiver sits at the payment row's
    own bank; otherwise it records a NaN and the reason.

WHY:
    The payment model may not learn from a receiver score produced by an account model that was
    trained on the same world, or the payment model is scored on data it effectively memorised. So
    the training worlds are split into five world folds; each fold is scored by a model trained on
    the other four folds only, and the held-out model is then discarded. Validation worlds are
    scored by the frozen Run A model, exactly as pre-registered in A-055.

TERM: fold: one of five groups of whole training worlds; position `i` of the world ids sorted as
      strings belongs to fold `i mod 5`.
TERM: account-day: one row per (world, bank, account, IST day), the latest snapshot of that day,
      the same unit the account model scores.

Test and stress datasets are never read: both entry points accept only training and validation
folders and raise on any other manifest split. Every missing input raises a plain ValueError.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import subprocess
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

from decisions.mule_account_model import (
    account_model_feature_names,
    mule_model_config_override,
    predict_account_days,
    set_mule_model_config_path,
)
from decisions.pu_learning import PUBaggedAccountModel, train_pu_bagged_model
from evaluation.compact_rows import CompactRows, load_compact_split
from evaluation.level17_scoring import one_row_per_account_day
from evaluation.run_level_18_pilot import _xgb_hyperparameters

REPO_ROOT = Path(__file__).resolve().parents[1]

#: Number of world folds fixed by A-055.
FOLD_COUNT = 5

#: NaN reasons recorded per payment when no receiver score can be attached.
REASON_BANK_UNKNOWN = "receiver_bank_unknown"
REASON_OTHER_BANK = "receiver_at_other_bank"
REASON_NO_PRIOR_DAY = "no_prior_account_day"

#: The frozen Run A artifacts A-055 names as the recipe source.
DEFAULT_FROZEN_CONFIG = REPO_ROOT / "runs" / "full" / "run_a320_stage5c" / "stages" / "final_mule_model.yaml"
DEFAULT_FIT_RECORD = (
    REPO_ROOT / "runs" / "full" / "run_a320_stage5c" / "final" / "45c496799170" / "checkpoint" / "fit_record.json"
)
DEFAULT_ENSEMBLE = (
    REPO_ROOT
    / "runs"
    / "full"
    / "run_a320_stage5c"
    / "final"
    / "45c496799170"
    / "checkpoint"
    / "model_bags"
    / "ensemble.json"
)
DEFAULT_CHECKPOINT = (
    REPO_ROOT / "runs" / "full" / "run_a320_stage5c" / "final" / "45c496799170" / "checkpoint" / "model_bags"
)

#: The model feature order is the one the fitted Run A members were built for; A-055 fixes it.
EXPECTED_FEATURE_SHA256 = "075183a2b9192ab889029e16c06a01f4f837db1c331218cfb93c2acd6d6efa20"
EXPECTED_FEATURE_COUNT = 42


@dataclass(frozen=True)
class AccountDayScore:
    """One scored account-day: the kept snapshot time and the model's probability for it."""

    snapshot_time: datetime
    score: float


@dataclass(frozen=True)
class FrozenRecipe:
    """The exact Run A recipe A-055 reproduces for every fold model."""

    params: dict[str, float | int]
    n_estimators: int
    n_bags: int
    unlabelled_per_positive: int
    seed: int
    feature_names: tuple[str, ...]
    feature_sha256: str
    config_path: Path
    config_sha256: str
    checkpoint_dir: Path
    training_dataset_ids: tuple[str, ...]
    validation_dataset_ids: tuple[str, ...]


def _require_manifest(dataset_dir: Path) -> dict[str, Any]:
    """
    WHAT: Reads a dataset folder's manifest.

    WHY: The split and world list are both read from the manifest, so a missing or malformed one
    names the offending folder instead of being guessed.

    Raises:
        ValueError: If the manifest is absent or does not hold a mapping.
    """
    path = Path(dataset_dir) / "manifest.json"
    if not path.is_file():
        raise ValueError(f"Dataset manifest is missing: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict):
        raise ValueError(f"Dataset manifest {path} must hold a mapping")
    return manifest


def validate_dataset_split(dataset_dir: Path, expected_split: str) -> dict[str, Any]:
    """
    WHAT: Returns a folder's manifest after proving its split is training or validation and equals
    the split the caller passes.

    WHY: Phases 0-3 must never read test or stress data. This is the single guard every entry point
    uses, so a folder placed on the wrong side raises instead of being trained or scored.

    Args:
        dataset_dir: Dataset folder holding manifest.json.
        expected_split: Either "training" or "validation".

    Returns:
        The parsed manifest.

    Raises:
        ValueError: If the split is test or stress, differs from expected_split, or expected_split
            is not training/validation.
    """
    if expected_split not in ("training", "validation"):
        raise ValueError(f"expected_split must be 'training' or 'validation'; got {expected_split!r}")
    manifest = _require_manifest(dataset_dir)
    split = manifest.get("split")
    if split not in ("training", "validation"):
        raise ValueError(
            f"{Path(dataset_dir).name} has manifest split {split!r}; phases 0-3 accept only "
            "training and validation datasets and never read test or stress"
        )
    if split != expected_split:
        raise ValueError(f"{Path(dataset_dir).name} has manifest split {split!r} but was passed as {expected_split!r}")
    return manifest


def _world_sources(dataset_dirs: Sequence[Path]) -> dict[str, Path]:
    """
    WHAT: Maps every world id in the folders to the folder that holds it.

    WHY: Scoring reads each world's own payments and snapshots, and a world id must resolve to one
    folder so a duplicate or missing world raises instead of silently merging rows.

    Raises:
        ValueError: If a folder holds no worlds or a world id repeats across folders.
    """
    sources: dict[str, Path] = {}
    for dataset_dir in dataset_dirs:
        manifest = _require_manifest(dataset_dir)
        worlds = manifest.get("worlds")
        if not isinstance(worlds, list) or not worlds:
            raise ValueError(f"{Path(dataset_dir).name} lists no worlds")
        for world in worlds:
            world_id = world.get("world_id") if isinstance(world, dict) else None
            if not isinstance(world_id, str) or not world_id:
                raise ValueError(f"{Path(dataset_dir).name} has a world entry without world_id")
            if world_id in sources:
                raise ValueError(f"world {world_id!r} appears in more than one dataset folder")
            sources[world_id] = Path(dataset_dir)
    return sources


def assign_folds(world_ids: Sequence[str]) -> dict[str, int]:
    """
    WHAT: Assigns every world to one of five folds by its position in the string-sorted id list.

    WHY: A-055 fixes folds by whole world and by sorted position so the split is deterministic and
    reproducible before any result exists. Sorting as strings (not by seed or size) means no model
    could influence the assignment.

    Args:
        world_ids: Training world ids.

    Returns:
        world_id -> fold index in 0..FOLD_COUNT-1.

    Raises:
        ValueError: If the ids hold a duplicate or are not strings.
    """
    ordered = sorted(world_ids)
    for world_id in ordered:
        if not isinstance(world_id, str) or not world_id:
            raise ValueError(f"world ids must be non-empty strings; got {world_id!r}")
    if len(set(ordered)) != len(ordered):
        raise ValueError("world ids must be unique; a duplicate cannot be assigned to one fold")
    return {world_id: position % FOLD_COUNT for position, world_id in enumerate(ordered)}


def latest_prior_score(
    scores: Sequence[AccountDayScore],
    decision_time: datetime,
) -> AccountDayScore | None:
    """
    WHAT: Returns the account-day with the latest snapshot_time strictly earlier than decision_time.

    WHY: The bank may only use what it already saw at the decision instant, so a snapshot taken at
    exactly the decision_time (or later) is not available. The comparison is strict for that reason.

    Args:
        scores: One account's account-days, sorted ascending by snapshot_time.
        decision_time: The payment's decision instant.

    Returns:
        The latest strictly-earlier account-day, or None when none is earlier.
    """
    prior: AccountDayScore | None = None
    for day in scores:
        if day.snapshot_time < decision_time:
            prior = day
        else:
            break
    return prior


def payee_risk(
    scores: Sequence[AccountDayScore],
    decision_time: datetime,
    receiver_bank: str | None,
    payment_bank: str,
) -> tuple[float, str]:
    """
    WHAT: Returns (score, "") for a same-bank receiver with an earlier account-day, else
    (NaN, reason).

    WHY: The pre-registered rule combines three facts: the receiver must be known to a bank, that
    bank must be the payment row's bank, and the receiver must have an account-day before the
    decision. Each failing fact gets its own recorded reason so the NaN share is explainable.

    Args:
        scores: The receiver's account-days, sorted ascending by snapshot_time.
        decision_time: The payment's decision instant.
        receiver_bank: The receiver account's bank from snapshots, or None when absent.
        payment_bank: The payment row's bank_id.

    Returns:
        (probability, "") on success, otherwise (NaN, one of the REASON_* constants).

    Raises:
        ValueError: If the chosen score is not finite, which would be a model fault, not missing data.
    """
    if receiver_bank is None:
        return math.nan, REASON_BANK_UNKNOWN
    if receiver_bank != payment_bank:
        return math.nan, REASON_OTHER_BANK
    prior = latest_prior_score(scores, decision_time)
    if prior is None:
        return math.nan, REASON_NO_PRIOR_DAY
    if not math.isfinite(prior.score):
        raise ValueError(f"account-model score is not finite: {prior.score!r}")
    return prior.score, ""


def _sha256_of_file(path: Path) -> str:
    """Returns the SHA-256 hex digest of a file's bytes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _git_head() -> str:
    """Returns the current git HEAD commit hash, refusing any failure."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    head = result.stdout.strip()
    if not head:
        raise ValueError("git rev-parse HEAD returned no commit hash")
    return head


def load_frozen_recipe(
    config_path: Path = DEFAULT_FROZEN_CONFIG,
    fit_record_path: Path = DEFAULT_FIT_RECORD,
    ensemble_path: Path = DEFAULT_ENSEMBLE,
    checkpoint_dir: Path = DEFAULT_CHECKPOINT,
) -> FrozenRecipe:
    """
    WHAT: Loads the exact Run A recipe from the frozen fit record, ensemble sidecar and config copy.

    WHY: The recipe must be the one Phase 0c recorded, not a retyped copy. Reading the three files
    the fit wrote and checking they agree means a changed artifact raises instead of silently
    training a different model.

    Args:
        config_path: Frozen `final_mule_model.yaml` the fit ran under.
        fit_record_path: The fit record holding the best parameters and PU method.
        ensemble_path: The bagged ensemble's sidecar holding n_bags and member seeds.
        checkpoint_dir: The frozen Run A ensemble folder used for validation worlds.

    Returns:
        The recipe every fold model is fitted with.

    Raises:
        ValueError: If the files disagree, the PU method is not bagging, the feature order or count
            is not the frozen one, or the config hash differs from the fit fingerprint.
    """
    config_path = Path(config_path)
    fit_record = json.loads(Path(fit_record_path).read_text(encoding="utf-8"))
    ensemble = json.loads(Path(ensemble_path).read_text(encoding="utf-8"))
    if not isinstance(fit_record, dict) or not isinstance(ensemble, dict):
        raise ValueError("fit_record and ensemble sidecar must both hold mappings")

    if fit_record.get("pu_method") != "bagging":
        raise ValueError(f"frozen PU method must be 'bagging'; got {fit_record.get('pu_method')!r}")
    search = fit_record.get("search")
    if not isinstance(search, dict):
        raise ValueError("fit_record holds no 'search' mapping")
    params = search.get("best_params")
    if not isinstance(params, dict):
        raise ValueError("fit_record search holds no 'best_params' mapping")

    n_estimators = fit_record.get("best_n_estimators")
    if not isinstance(n_estimators, int) or isinstance(n_estimators, bool):
        raise ValueError(f"fit_record best_n_estimators must be an integer; got {n_estimators!r}")
    if int(params["n_estimators"]) != n_estimators:
        raise ValueError(
            f"best_params n_estimators {params['n_estimators']!r} disagrees with best_n_estimators {n_estimators}"
        )
    if ensemble.get("n_estimators") != n_estimators:
        raise ValueError(
            f"ensemble n_estimators {ensemble.get('n_estimators')!r} disagrees with fit record {n_estimators}"
        )

    n_bags = ensemble.get("n_bags")
    if not isinstance(n_bags, int) or isinstance(n_bags, bool) or n_bags <= 0:
        raise ValueError(f"ensemble n_bags must be a positive integer; got {n_bags!r}")
    unlabelled_per_positive = ensemble.get("unlabelled_per_positive")
    if not isinstance(unlabelled_per_positive, int) or isinstance(unlabelled_per_positive, bool):
        raise ValueError(f"ensemble unlabelled_per_positive must be an integer; got {unlabelled_per_positive!r}")
    if int(params["unlabelled_per_positive"]) != unlabelled_per_positive:
        raise ValueError("best_params unlabelled_per_positive disagrees with the ensemble sidecar")

    # The frozen config supplies the seed constant stage 3 used and proves the fit's config hash.
    from decisions.mule_account_model import load_mule_model_config

    config = load_mule_model_config(config_path)
    seed = config.optuna_seed
    if seed is None:
        raise ValueError("frozen config holds no optuna_seed to use as the model seed constant")
    member_seeds = ensemble.get("member_seeds")
    if member_seeds != list(range(seed, seed + n_bags)):
        raise ValueError(f"ensemble member seeds are not {seed}..{seed + n_bags - 1}; the seed constant disagrees")

    config_sha256 = _sha256_of_file(config_path)
    fingerprint = fit_record.get("fingerprint")
    if not isinstance(fingerprint, dict):
        raise ValueError("fit_record holds no 'fingerprint' mapping")
    if fingerprint.get("mule_model_config_sha256") != config_sha256:
        raise ValueError("frozen config sha256 does not match the fit record's mule_model_config_sha256")

    # Feature order is evaluated under the frozen config so the label-network switch is the fit's.
    previous = mule_model_config_override()
    set_mule_model_config_path(config_path)
    try:
        feature_names = account_model_feature_names()
    finally:
        set_mule_model_config_path(previous)
    feature_sha256 = hashlib.sha256("\n".join(feature_names).encode("utf-8")).hexdigest()
    if len(feature_names) != EXPECTED_FEATURE_COUNT or feature_sha256 != EXPECTED_FEATURE_SHA256:
        raise ValueError(
            "account model feature order is not the frozen Run A list: "
            f"{len(feature_names)} names, sha256 {feature_sha256}"
        )

    training_dataset_ids = tuple(fingerprint.get("training_dataset_ids", ()))
    validation_dataset_ids = tuple(fingerprint.get("validation_dataset_ids", ()))
    if not training_dataset_ids or not validation_dataset_ids:
        raise ValueError("fit_record fingerprint lists no training or validation dataset ids")

    return FrozenRecipe(
        params=dict(params),
        n_estimators=n_estimators,
        n_bags=n_bags,
        unlabelled_per_positive=unlabelled_per_positive,
        seed=seed,
        feature_names=feature_names,
        feature_sha256=feature_sha256,
        config_path=config_path,
        config_sha256=config_sha256,
        checkpoint_dir=Path(checkpoint_dir),
        training_dataset_ids=training_dataset_ids,
        validation_dataset_ids=validation_dataset_ids,
    )


class _CommitSampler:
    """
    WHAT: Samples the Windows commit charge on a background thread and keeps its peak.

    WHY: A longer-than-expected run can exhaust the page file; the pre-registered laptop rule stops
    at 85 percent of the commit limit, so the fold record needs the peak this process reached.
    """

    def __init__(self, interval_seconds: float = 2.0) -> None:
        self._interval = interval_seconds
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.peak_gb = 0.0

    def __enter__(self) -> "_CommitSampler":
        self._thread.start()
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        self._stop.set()
        self._thread.join()

    def _run(self) -> None:
        while not self._stop.wait(self._interval):
            self.peak_gb = max(self.peak_gb, committed_gb())
        self.peak_gb = max(self.peak_gb, committed_gb())


def committed_gb() -> float:
    """Returns the system commit charge in GiB (page-file committed bytes)."""
    import ctypes
    from ctypes import wintypes

    class _MemoryStatusEx(ctypes.Structure):
        _fields_ = [
            ("dwLength", wintypes.DWORD),
            ("dwMemoryLoad", wintypes.DWORD),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    status = _MemoryStatusEx()
    status.dwLength = ctypes.sizeof(_MemoryStatusEx)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
        raise ValueError("GlobalMemoryStatusEx failed to read the commit charge")
    return (status.ullTotalPageFile - status.ullAvailPageFile) / 1024**3


def _build_fold_rows(
    training_dirs: Sequence[Path],
    train_worlds: Sequence[str],
    frozen_config: Path,
) -> CompactRows:
    """
    WHAT: Builds the compact training rows for one fold's worlds under the frozen config.

    WHY: `load_compact_split` reproduces `build_training_rows` per world, including the label cut,
    maturity window and trust weights, so the fold model is trained on exactly the rows stage 5
    used. The frozen config must be active so the maturity and trust switches are the fit's.

    Returns:
        The fold's CompactRows (matrix, targets, keys and trust weights).
    """
    previous = mule_model_config_override()
    set_mule_model_config_path(frozen_config)
    try:
        split = load_compact_split(list(training_dirs), world_ids=list(train_worlds))
    finally:
        set_mule_model_config_path(previous)
    return split.rows


def _fit_fold_model(rows: CompactRows, recipe: FrozenRecipe) -> PUBaggedAccountModel:
    """
    WHAT: Fits one fold's bagged account model with the frozen recipe on CPU.

    WHY: A-055 fixes the fold recipe to be Phase 0c exactly, so every value is read from the frozen
    artifacts and converted through the pilot's own trainer-argument helper.
    """
    model = train_pu_bagged_model(
        rows.matrix,
        rows.targets,
        recipe.seed,
        recipe.n_estimators,
        recipe.n_bags,
        recipe.unlabelled_per_positive,
        sample_weight=rows.sample_weights,
        device="cpu",
        **_xgb_hyperparameters(recipe.params),
    )
    _assert_recipe_reproduction(model, recipe)
    return model


def _assert_recipe_reproduction(model: PUBaggedAccountModel, recipe: FrozenRecipe) -> None:
    """
    WHAT: Proves a fitted fold model carries the frozen recipe's bags, rounds, ratio, seeds and CPU
    device.

    WHY: A .ubj file does not keep the tree hyperparameters, so reproduction is proven from what a
    fitted ensemble does expose: the bag count, each member's boosted rounds, the sampled ratio, the
    exact member seeds, and the training device recorded in each booster.
    """
    if model.n_bags != recipe.n_bags:
        raise ValueError(f"fold model has {model.n_bags} bags, recipe says {recipe.n_bags}")
    if model.n_estimators != recipe.n_estimators:
        raise ValueError(f"fold model has {model.n_estimators} estimators, recipe says {recipe.n_estimators}")
    if model.unlabelled_per_positive != recipe.unlabelled_per_positive:
        raise ValueError(
            f"fold model ratio {model.unlabelled_per_positive} disagrees with recipe {recipe.unlabelled_per_positive}"
        )
    if list(model.member_seeds) != list(range(recipe.seed, recipe.seed + recipe.n_bags)):
        raise ValueError("fold model member seeds are not the frozen contiguous seed block")
    for member in model.members:
        booster = member.get_booster()
        if booster.num_boosted_rounds() != recipe.n_estimators:
            raise ValueError(
                f"a fold member holds {booster.num_boosted_rounds()} rounds, recipe says {recipe.n_estimators}"
            )
        config = json.loads(booster.save_config())
        device = config["learner"]["generic_param"].get("device")
        if device != "cpu":
            raise ValueError(f"fold member was fitted on device {device!r}, recipe says 'cpu'")


def _read_world_snapshots(dataset_dir: Path, world_id: str) -> list[dict[str, Any]]:
    """Returns one world's snapshot rows from its dataset folder."""
    table = pq.read_table(Path(dataset_dir) / "snapshots.parquet", filters=[("world_id", "=", world_id)])
    rows = table.to_pylist()
    if not rows:
        raise ValueError(f"world {world_id!r} has no snapshot rows in {Path(dataset_dir).name}")
    return rows


def _read_world_payments(dataset_dir: Path, world_id: str) -> list[dict[str, Any]]:
    """Returns one world's payment rows with the columns payee risk needs."""
    table = pq.read_table(
        Path(dataset_dir) / "payments.parquet",
        columns=["payment_id", "bank_id", "receiver_account_id", "decision_time"],
        filters=[("world_id", "=", world_id)],
    )
    rows = table.to_pylist()
    if not rows:
        raise ValueError(f"world {world_id!r} has no payment rows in {Path(dataset_dir).name}")
    return rows


def score_world_account_days(
    model: PUBaggedAccountModel,
    snapshot_rows: Sequence[dict[str, Any]],
) -> tuple[dict[str, list[AccountDayScore]], dict[str, str]]:
    """
    WHAT: Scores one world's account-days and returns per-account sorted scores and each account's
    bank.

    WHY: Reuses the production `one_row_per_account_day` selector and `predict_account_days`
    scorer, so the scored unit and feature order match stage 5. Sorting each account's days once
    makes the strict earlier-than lookup a linear scan per payment's receiver.

    Args:
        model: The fold model (training world) or frozen Run A model (validation world).
        snapshot_rows: The world's snapshot rows.

    Returns:
        (scores_by_account, bank_by_account). Scores are sorted ascending by snapshot_time.

    Raises:
        ValueError: If a scored account-day row lacks an account id or bank.
    """
    day_rows = one_row_per_account_day(list(snapshot_rows))
    probabilities = predict_account_days(model, day_rows)
    scores_by_account: dict[str, list[AccountDayScore]] = {}
    bank_by_account: dict[str, str] = {}
    for row, probability in zip(day_rows, probabilities, strict=True):
        account_id = row.get("account_id")
        bank_id = row.get("bank_id")
        moment = row.get("snapshot_time")
        if not isinstance(account_id, str) or not account_id:
            raise ValueError("an account-day row has no account_id")
        if not isinstance(bank_id, str) or not bank_id:
            raise ValueError(f"account {account_id!r} has no bank_id")
        if not isinstance(moment, datetime):
            raise ValueError(f"account {account_id!r} has no snapshot_time")
        previous_bank = bank_by_account.get(account_id)
        if previous_bank is not None and previous_bank != bank_id:
            raise ValueError(f"account {account_id!r} appears at two banks in one world")
        bank_by_account[account_id] = bank_id
        scores_by_account.setdefault(account_id, []).append(AccountDayScore(moment, float(probability)))
    for scores in scores_by_account.values():
        scores.sort(key=lambda day: day.snapshot_time)
    return scores_by_account, bank_by_account


def build_world_payee_risk(
    payment_rows: Sequence[dict[str, Any]],
    scores_by_account: dict[str, list[AccountDayScore]],
    bank_by_account: dict[str, str],
    world_id: str,
    split: str,
    fold: int,
) -> list[dict[str, Any]]:
    """
    WHAT: Turns one world's payments into payee-risk rows under the pre-registered rule.

    WHY: Every payment row must carry either a finite score or a recorded reason, so the downstream
    check can prove the NaN share is fully explained instead of silently missing.

    Raises:
        ValueError: If a payment row lacks a required field.
    """
    rows: list[dict[str, Any]] = []
    for payment in payment_rows:
        payment_id = payment.get("payment_id")
        payment_bank = payment.get("bank_id")
        receiver = payment.get("receiver_account_id")
        decision_time = payment.get("decision_time")
        if not isinstance(payment_id, str) or not payment_id:
            raise ValueError(f"world {world_id!r} has a payment without payment_id")
        if not isinstance(payment_bank, str) or not payment_bank:
            raise ValueError(f"payment {payment_id!r} has no bank_id")
        if not isinstance(receiver, str) or not receiver:
            raise ValueError(f"payment {payment_id!r} has no receiver_account_id")
        if not isinstance(decision_time, datetime):
            raise ValueError(f"payment {payment_id!r} has no decision_time")
        receiver_bank = bank_by_account.get(receiver)
        value, reason = payee_risk(scores_by_account.get(receiver, ()), decision_time, receiver_bank, payment_bank)
        rows.append(
            {
                "world_id": world_id,
                "split": split,
                "fold": fold,
                "payment_id": payment_id,
                "decision_time": decision_time,
                "receiver_account_id": receiver,
                "payment_bank_id": payment_bank,
                "receiver_bank_id": receiver_bank,
                "same_bank": None if receiver_bank is None else receiver_bank == payment_bank,
                "payee_risk": value,
                "nan_reason": reason,
            }
        )
    return rows


_PAYEE_RISK_SCHEMA = pa.schema(
    [
        ("world_id", pa.string()),
        ("split", pa.string()),
        ("fold", pa.int16()),
        ("payment_id", pa.string()),
        ("decision_time", pa.timestamp("us", tz="UTC")),
        ("receiver_account_id", pa.string()),
        ("payment_bank_id", pa.string()),
        ("receiver_bank_id", pa.string()),
        ("same_bank", pa.bool_()),
        ("payee_risk", pa.float64()),
        ("nan_reason", pa.string()),
    ]
)


def _write_world_parquet(out_dir: Path, world_id: str, rows: list[dict[str, Any]]) -> Path:
    """Writes one world's payee-risk rows to per_world/<world_id>.parquet and returns the path."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{world_id}.parquet"
    pq.write_table(pa.Table.from_pylist(rows, schema=_PAYEE_RISK_SCHEMA), path)
    return path


def _reason_counts(rows: Sequence[dict[str, Any]]) -> dict[str, int]:
    """Counts the recorded NaN reasons across rows, with scored rows counted as 'scored'."""
    counts: dict[str, int] = {}
    for row in rows:
        key = row["nan_reason"] or "scored"
        counts[key] = counts.get(key, 0) + 1
    return counts


def run_oof(
    training_dirs: Sequence[Path],
    validation_dirs: Sequence[Path],
    out_root: Path,
    *,
    frozen_config: Path = DEFAULT_FROZEN_CONFIG,
    fit_record: Path = DEFAULT_FIT_RECORD,
    ensemble: Path = DEFAULT_ENSEMBLE,
    checkpoint_dir: Path = DEFAULT_CHECKPOINT,
) -> dict[str, Any]:
    """
    WHAT: Builds payee risk for every training payment out-of-fold and every validation payment
    with the frozen Run A model, writing per-world parquet and fold_record.json.

    WHY: This is the one entry point Phase 1 needs; it enforces the split guard, the fold recipe,
    the no-self-scoring assertion and the per-row score-or-reason invariant before it returns.

    Args:
        training_dirs: Training dataset folders only.
        validation_dirs: Validation dataset folders only.
        out_root: Output folder; per-world files land under its per_world/ subfolder.
        frozen_config: Frozen Run A config copy.
        fit_record: Frozen Run A fit record.
        ensemble: Frozen Run A ensemble sidecar.
        checkpoint_dir: Frozen Run A ensemble folder.

    Returns:
        The fold_record mapping that was written.

    Raises:
        ValueError: On any split guard failure, a world overlap, a recipe mismatch, or a payment
            row with neither a score nor a reason.
    """
    for dataset_dir in training_dirs:
        validate_dataset_split(dataset_dir, "training")
    for dataset_dir in validation_dirs:
        validate_dataset_split(dataset_dir, "validation")

    training_sources = _world_sources(training_dirs)
    validation_sources = _world_sources(validation_dirs)
    overlap = set(training_sources) & set(validation_sources)
    if overlap:
        raise ValueError(f"worlds appear in both training and validation folders: {sorted(overlap)}")

    recipe = load_frozen_recipe(frozen_config, fit_record, ensemble, checkpoint_dir)
    training_worlds = sorted(training_sources)
    folds = assign_folds(training_worlds)
    validation_worlds = sorted(validation_sources)

    out_root = Path(out_root)
    per_world_dir = out_root / "per_world"
    started = time.perf_counter()
    fold_records: dict[str, Any] = {}
    training_reason_totals: dict[str, int] = {}

    previous_config = mule_model_config_override()
    set_mule_model_config_path(recipe.config_path)
    try:
        for fold in range(FOLD_COUNT):
            held_out = [world_id for world_id in training_worlds if folds[world_id] == fold]
            train_worlds = [world_id for world_id in training_worlds if folds[world_id] != fold]
            if set(held_out) & set(train_worlds):
                raise ValueError("a fold would train and score the same world")
            if any(folds[world_id] != fold for world_id in held_out):
                raise ValueError("a held-out world is not in the fold being scored")

            rows = _build_fold_rows(training_dirs, train_worlds, recipe.config_path)
            with _CommitSampler() as sampler:
                fit_started = time.perf_counter()
                model = _fit_fold_model(rows, recipe)
                fit_seconds = time.perf_counter() - fit_started
            del rows

            fold_reason: dict[str, int] = {}
            for world_id in held_out:
                snapshots = _read_world_snapshots(training_sources[world_id], world_id)
                scores_by_account, bank_by_account = score_world_account_days(model, snapshots)
                payments = _read_world_payments(training_sources[world_id], world_id)
                world_rows = build_world_payee_risk(
                    payments, scores_by_account, bank_by_account, world_id, "training", fold
                )
                _assert_score_or_reason(world_rows)
                _write_world_parquet(per_world_dir, world_id, world_rows)
                for key, value in _reason_counts(world_rows).items():
                    fold_reason[key] = fold_reason.get(key, 0) + value
                    training_reason_totals[key] = training_reason_totals.get(key, 0) + value
            del model
            fold_records[str(fold)] = {
                "world_ids": held_out,
                "training_world_count": len(train_worlds),
                "fit_seconds": fit_seconds,
                "peak_committed_gb": sampler.peak_gb,
                "reason_counts": fold_reason,
            }
            print(
                f"DONE fold {fold}: {len(train_worlds)} training worlds, {len(held_out)} scored, "
                f"{fit_seconds:.1f} s, peak committed {sampler.peak_gb:.2f} GiB"
            )

        frozen_model = PUBaggedAccountModel.load(recipe.checkpoint_dir)
        validation_started = time.perf_counter()
        validation_reason: dict[str, int] = {}
        for world_id in validation_worlds:
            snapshots = _read_world_snapshots(validation_sources[world_id], world_id)
            scores_by_account, bank_by_account = score_world_account_days(frozen_model, snapshots)
            payments = _read_world_payments(validation_sources[world_id], world_id)
            world_rows = build_world_payee_risk(
                payments, scores_by_account, bank_by_account, world_id, "validation", -1
            )
            _assert_score_or_reason(world_rows)
            _write_world_parquet(per_world_dir, world_id, world_rows)
            for key, value in _reason_counts(world_rows).items():
                validation_reason[key] = validation_reason.get(key, 0) + value
        validation_seconds = time.perf_counter() - validation_started
    finally:
        set_mule_model_config_path(previous_config)

    total_seconds = time.perf_counter() - started
    record = {
        "fold_count": FOLD_COUNT,
        "folds": fold_records,
        "training_world_count": len(training_worlds),
        "validation_world_count": len(validation_worlds),
        "validation_seconds": validation_seconds,
        "validation_reason_counts": validation_reason,
        "training_reason_counts": training_reason_totals,
        "config_sha256": recipe.config_sha256,
        "git_hash": _git_head(),
        "recipe": {
            "params": recipe.params,
            "n_estimators": recipe.n_estimators,
            "n_bags": recipe.n_bags,
            "unlabelled_per_positive": recipe.unlabelled_per_positive,
            "seed": recipe.seed,
            "feature_sha256": recipe.feature_sha256,
            "feature_count": len(recipe.feature_names),
            "training_dataset_ids": list(recipe.training_dataset_ids),
            "validation_dataset_ids": list(recipe.validation_dataset_ids),
            "validation_checkpoint": str(recipe.checkpoint_dir),
        },
        "total_seconds": total_seconds,
    }
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "fold_record.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"DONE payee risk: {total_seconds:.1f} s -> {out_root}")
    return record


def _assert_score_or_reason(rows: Sequence[dict[str, Any]]) -> None:
    """
    WHAT: Raises unless every row has a finite score with no reason, or a NaN with a reason.

    WHY: The pre-registered gate is that every payment row carries payee_risk or a recorded NaN
    reason; this proves it per world before the run reports success.
    """
    for row in rows:
        value = row["payee_risk"]
        reason = row["nan_reason"]
        if math.isfinite(value) and reason == "":
            continue
        if math.isnan(value) and reason != "":
            continue
        raise ValueError(f"payment {row['payment_id']!r} has payee_risk {value!r} with reason {reason!r}")


def main(argv: Sequence[str] | None = None) -> int:
    """Runs the out-of-fold payee-risk build from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training", action="append", required=True, help="Training dataset folder (repeatable)")
    parser.add_argument("--validation", action="append", required=True, help="Validation dataset folder (repeatable)")
    parser.add_argument(
        "--out-root",
        default=str(REPO_ROOT / "runs" / "full" / "payment_oof"),
        help="Output folder for per-world files and fold_record.json",
    )
    args = parser.parse_args(argv)
    run_oof(
        [Path(path) for path in args.training],
        [Path(path) for path in args.validation],
        Path(args.out_root),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
