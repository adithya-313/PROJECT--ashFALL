# File: evaluation/payment_candidates.py
# Purpose: Pre-registered payment-model v2 candidates (monotone, segmented, both), their selection
# and the once-only test and stress scoring of the selected winner, under A-066.
"""
WHAT: Trains and compares the three A-066 payment-model v2 candidates (C1 monotone, C2 segmented,
C3 both) on the frozen v1 rows, validation and feature order, selects one by the pre-registered rule,
and scores the selected winner once on test and stress with A-058's procedure.

WHY: A replacement for the frozen payment model may only be chosen on validation truth and only under
a rule written before the first fit. Every choice, device rule, tie break and gate lives in A-066, so
the reported test and stress numbers are a held-out measurement rather than a choice made on them.
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
import numpy.typing as npt
import pyarrow.compute as pc
import pyarrow.parquet as pq
import xgboost
import yaml

from decisions.mule_account_model import (
    load_mule_model_config,
    mule_model_config_override,
    set_mule_model_config_path,
)
from decisions.pu_learning import PUBaggedAccountModel
from evaluation.dataset_truth_join import join_truth
from evaluation.hyperparameter_search import run_optuna_search, validation_truth_average_precision
from evaluation.payment_diagnosis import dedupe_payees, feature_truth_ap
from evaluation.payment_model import (
    DEFAULT_FEATURE_ORDER_PATH,
    DEFAULT_MULE_CONFIG_PATH,
    DEFAULT_OOF_ROOT,
    DEFAULT_TRUTH_ROOT,
    KEY_COLUMNS,
    OBJECTIVE,
    PAYEE_RISK,
    TREE_METHOD,
    _fixed_device_params,
    build_matrix,
    choose_device,
    load_feature_order,
    load_split_rows,
    load_validation_roles,
    payment_scored,
    predict_scores,
    target_labels,
    truth_labels,
)
from evaluation.payment_oof_risk import (
    _read_world_snapshots,
    build_world_payee_risk,
    load_frozen_recipe,
    score_world_account_days,
    validate_dataset_split,
)
from evaluation.payment_test_score import (
    HOLD_SHARE,
    WorldScores,
    aggregate,
    attach_payee_risk,
    evaluate_splits,
    hold_threshold,
    validate_scoring_split,
    verify_booster_rounds,
    verify_frozen_artifacts,
)
from evaluation.trivial_world import average_precision

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT_ROOT = REPO_ROOT / "runs" / "full" / "payment_model_v2"
DEFAULT_V1_MODEL_DIR = REPO_ROOT / "runs" / "full" / "payment_model_v1"
DEFAULT_ACCOUNT_CONFIG = REPO_ROOT / "runs" / "full" / "run_a320_stage5c" / "stages" / "final_mule_model.yaml"

CANDIDATE_ORDER = ("C1", "C2", "C3")
CANDIDATE_TITLE = {
    "C1": "monotone: +1 on payee_risk only",
    "C2": "segmented: payee_risk present versus NaN, each searched separately",
    "C3": "monotone and segmented",
}
CANDIDATE_MONOTONE = {"C1": True, "C2": False, "C3": True}
CANDIDATE_SEGMENTED = {"C1": False, "C2": True, "C3": True}
TIE_WINDOW = 0.002
TIE_PRIORITY = ("C1", "C3", "C2")
V1_VALIDATION_AP_FLOOR = 0.4659
# Reloading the CUDA-frozen v1 booster on cpu reproduces its validation AP to ~1.6e-8, so the
# sanity guard below allows prediction noise while still catching a wrong booster or wrong rows.
V1_REPRODUCTION_TOLERANCE = 1e-6

PRESENT = "present"
ABSENT = "nan"
SINGLE = "all"
SEGMENTS = (PRESENT, ABSENT)
MODEL_FILE = {SINGLE: "frozen_model.json", PRESENT: "frozen_model_present.json", ABSENT: "frozen_model_nan.json"}

EXPECTED_TRAINING_ROWS = 4_911_546
EXPECTED_TRAINING_POSITIVES = 70_821
EXPECTED_VALIDATION_ROWS = 1_834_135
EXPECTED_VALIDATION_POSITIVES = 27_393

REGENERATION_IF_REPLACED = (
    "ONNX export and the ONNX-versus-native parity check (A-061a/b): evaluation/onnx_export.py, "
    "reports/level_18/console/onnx_parity.json",
    "Triton model repository for mule_payment_xgb: serving/model_repository/mule_payment_xgb/1/model.onnx "
    "and, if the input or output shape changes, config.pbtxt (the handoff in docs/handoff/level_18b_partial.md)",
    "Payment isotonic calibration fitted on validation and reports/level_18/console/calibration.json (A-059/Phase 7)",
    "Payment reason codes and reports/level_18/console/reason_codes.json (A-059)",
    "Payment test and stress score and diagnosis: runs/full/payment_model_v1/test_score/result.json, "
    "runs/full/payment_model_v1/diagnosis/diagnosis.json and the console payment_results.json",
    "Level 18 model cards reports/level_18/model_cards/mule_payment_xgb.{json,md} and the mule model quality gate",
    "Level 18 showcase console and report: reports/level_18/console/*.json built from payment results, "
    "and the showcase checks that read them",
    "MLflow backfill for the replaced run (scripts/level18_mlflow_log.py)",
)


def _sha256_file(path: Path) -> str:
    """Returns the sha256 hex digest of a file's bytes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def monotone_vector(features: Sequence[str]) -> list[int]:
    """
    WHAT: Returns the A-066 monotone constraint vector: +1 at `payee_risk`, 0 at every other feature.

    WHY: XGBoost takes one constraint per feature in the trained column order; a vector of the wrong
    length or with a second constrained feature would silently be a different model.

    Raises:
        ValueError: If `payee_risk` is absent from the feature order.
    """
    if PAYEE_RISK not in features:
        raise ValueError(f"monotone candidate needs {PAYEE_RISK!r} in the feature order")
    vector = [0] * len(features)
    vector[list(features).index(PAYEE_RISK)] = 1
    return vector


def segment_masks(payee_risk: npt.ArrayLike) -> dict[str, np.ndarray]:
    """
    WHAT: Returns the A-066 segment masks: present rows are those with a finite `payee_risk`, NaN rows
    are the rest.

    WHY: The two boosters must see exactly the rows the pre-registration names; an empty segment
    cannot be fitted and is a stop rather than a silently dropped study.

    Raises:
        ValueError: If the values are not one-dimensional or a segment holds no rows.
    """
    values = np.asarray(payee_risk, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"payee_risk must be one-dimensional; got shape {values.shape}")
    finite = np.isfinite(values)
    masks = {PRESENT: finite, ABSENT: ~finite}
    for name, mask in masks.items():
        if not mask.any():
            raise ValueError(f"segment {name!r} holds no rows")
    return masks


def fit_candidate_model(
    matrix: np.ndarray,
    labels: np.ndarray,
    params: Mapping[str, int | float],
    device: str,
    seed: int,
    monotone_constraints: Sequence[int] | None,
) -> xgboost.XGBClassifier:
    """
    WHAT: Fits one candidate booster with the pre-registered estimator settings and, when given, the
    monotone vector.

    WHY: C1 and C3 must carry the constraint on every fit and C2 must carry none; making the vector
    an explicit argument keeps the difference between the candidates in one place.

    Raises:
        ValueError: If the constraint vector's length does not match the matrix's column count.
    """
    kwargs: dict[str, Any] = dict(params)
    if monotone_constraints is not None:
        constraints = list(monotone_constraints)
        if len(constraints) != matrix.shape[1]:
            raise ValueError(
                f"monotone constraints hold {len(constraints)} entries but the matrix has {matrix.shape[1]} columns"
            )
        kwargs["monotone_constraints"] = tuple(constraints)
    model = xgboost.XGBClassifier(
        objective=OBJECTIVE,
        tree_method=TREE_METHOD,
        device=device,
        random_state=seed,
        **kwargs,
    )
    model.fit(matrix, labels)
    return model


@dataclass(frozen=True)
class CandidateModel:
    """
    WHAT: One candidate's fitted booster(s) and the routing rule that turns a feature matrix into one
    probability per row.

    WHY: A segmented candidate is two models with a fixed routing rule; keeping the rule beside the
    models makes the pooled score impossible to compute against the wrong rows.
    """

    candidate: str
    payee_risk_index: int
    models: Mapping[str, xgboost.XGBClassifier]

    def predict(self, matrix: np.ndarray) -> np.ndarray:
        """
        WHAT: Returns one probability per row, routing each row to its own segment's booster.

        Raises:
            ValueError: If a single-model candidate does not hold exactly the single booster.
        """
        if SINGLE in self.models:
            if len(self.models) != 1:
                raise ValueError(f"single-model candidate {self.candidate} holds {sorted(self.models)}")
            return self.models[SINGLE].predict_proba(matrix)[:, 1]
        risk = matrix[:, self.payee_risk_index]
        present = np.isfinite(risk)
        scores = np.empty(matrix.shape[0], dtype=np.float64)
        scores[present] = self.models[PRESENT].predict_proba(matrix[present])[:, 1]
        scores[~present] = self.models[ABSENT].predict_proba(matrix[~present])[:, 1]
        return scores


def fit_candidate_from_params(
    candidate: str,
    params_by_segment: Mapping[str, Mapping[str, int | float]],
    constraints: Sequence[int] | None,
    train_matrix: np.ndarray,
    train_labels: np.ndarray,
    train_masks: Mapping[str, np.ndarray],
    device: str,
    seed: int,
) -> CandidateModel:
    """
    WHAT: Fits every booster of one candidate from its best parameters.

    WHY: Freezing refits the searched parameters, exactly as the v1 freeze did, so the written model
    is the one the recorded search score belongs to.
    """
    models: dict[str, xgboost.XGBClassifier] = {}
    for segment, params in params_by_segment.items():
        if segment == SINGLE:
            rows = np.ones(train_matrix.shape[0], dtype=bool)
        else:
            rows = train_masks[segment]
        models[segment] = fit_candidate_model(train_matrix[rows], train_labels[rows], params, device, seed, constraints)
    return CandidateModel(candidate=candidate, payee_risk_index=0, models=models)


def pooled_truth_ap(scores: Sequence[float], roles: Sequence[str]) -> float:
    """Returns validation truth AP over every row, using the shared search objective."""
    return validation_truth_average_precision(list(scores), payment_scored(roles))


def _world_ap_or_none(scores: Sequence[float], labels: Sequence[int]) -> float | None:
    """Returns a world's AP, or None when it lacks a positive or a negative (A-058 ruling 4)."""
    positives = sum(labels)
    if positives == 0 or positives == len(labels):
        return None
    return average_precision(list(scores), list(labels))


def world_truth_ap(
    world_ids: Sequence[str],
    scores: Sequence[float],
    labels: Sequence[int],
) -> list[dict[str, Any]]:
    """
    WHAT: Returns per-world rows, positives and truth AP for one split's aligned arrays.

    Raises:
        ValueError: If the three sequences differ in length.
    """
    if not (len(world_ids) == len(scores) == len(labels)):
        raise ValueError(f"world ids {len(world_ids)}, scores {len(scores)} and labels {len(labels)} differ in length")
    order: dict[str, list[int]] = {}
    for index, world_id in enumerate(world_ids):
        order.setdefault(world_id, []).append(index)
    rows: list[dict[str, Any]] = []
    for world_id, indices in order.items():
        world_scores = [scores[index] for index in indices]
        world_labels = [labels[index] for index in indices]
        rows.append(
            {
                "world_id": world_id,
                "rows": len(indices),
                "positives": int(sum(world_labels)),
                "truth_ap": _world_ap_or_none(world_scores, world_labels),
            }
        )
    return rows


def world_p10(world_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Returns the A-066 cross-world aggregate of per-world truth AP (median, p10, p90, worlds used)."""
    return aggregate([row["truth_ap"] for row in world_rows])


def select_candidate(records: Sequence[Mapping[str, Any]], v1_p10: float) -> dict[str, Any]:
    """
    WHAT: Applies A-066's selection: p10 eligibility, highest validation AP, and the 0.002 tie
    window broken by C1, then C3, then C2.

    WHY: The priority is fixed before any score exists, so a tie cannot be resolved by inspecting
    which candidate looks better.

    Raises:
        ValueError: If a record names an unknown candidate or lacks a numeric AP or p10.
    """
    for record in records:
        if record.get("candidate") not in CANDIDATE_ORDER:
            raise ValueError(f"unknown candidate {record.get('candidate')!r}")
        for field in ("validation_ap", "validation_p10"):
            value = record.get(field)
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise ValueError(f"{record.get('candidate')} has no finite {field}")
    eligible = [record for record in records if float(record["validation_p10"]) >= v1_p10]
    trace: dict[str, Any] = {
        "v1_p10": v1_p10,
        "eligible": [record["candidate"] for record in eligible],
        "tie_window": TIE_WINDOW,
        "winner": None,
        "reason": "",
    }
    if not eligible:
        trace["reason"] = "no candidate reaches v1's p10, so v1 stays"
        return trace
    best_ap = max(float(record["validation_ap"]) for record in eligible)
    tied = [record for record in eligible if best_ap - float(record["validation_ap"]) <= TIE_WINDOW]
    winner = min(tied, key=lambda record: TIE_PRIORITY.index(record["candidate"]))
    trace["best_ap"] = best_ap
    trace["tied"] = [record["candidate"] for record in tied]
    trace["winner"] = winner["candidate"]
    trace["reason"] = (
        f"{winner['candidate']} has validation AP {float(winner['validation_ap']):.12f} within "
        f"{TIE_WINDOW} of the best and the registered tie order is {', '.join(TIE_PRIORITY)}"
        if len(tied) > 1
        else f"{winner['candidate']} has the highest validation AP among the eligible candidates"
    )
    return trace


def replacement_decision(winner_ap: float, winner_p10: float, v1_p10: float) -> dict[str, Any]:
    """Applies A-066's replacement gate: the literal 0.4659 floor and v1's p10, both inclusive."""
    replaces = winner_ap >= V1_VALIDATION_AP_FLOOR and winner_p10 >= v1_p10
    return {
        "validation_ap": winner_ap,
        "validation_ap_floor": V1_VALIDATION_AP_FLOOR,
        "validation_p10": winner_p10,
        "v1_p10": v1_p10,
        "replaces_v1": replaces,
    }


def write_candidate_artifacts(
    out_dir: Path,
    candidate: str,
    model: CandidateModel,
    features: Sequence[str],
    feature_order_path: Path,
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """
    WHAT: Writes the candidate's frozen booster(s), feature order and metadata with their sha256
    values and returns the completed metadata.

    WHY: The scoring step verifies these hashes, so the written artifact and the recorded provenance
    cannot drift apart.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    artifacts: dict[str, str] = {}
    for segment, booster in model.models.items():
        name = MODEL_FILE[segment]
        booster.save_model(out_dir / name)
        artifacts[name] = _sha256_file(out_dir / name)
    feature_order = load_feature_order(feature_order_path)
    (out_dir / "feature_order.yaml").write_text(
        yaml.safe_dump(feature_order, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    artifacts["feature_order.yaml"] = _sha256_file(out_dir / "feature_order.yaml")
    completed = dict(metadata)
    completed["candidate"] = candidate
    completed["title"] = CANDIDATE_TITLE[candidate]
    completed["structure"] = SINGLE if SINGLE in model.models else "segmented"
    completed["monotone_constraints"] = monotone_vector(features) if CANDIDATE_MONOTONE[candidate] else None
    completed["artifacts"] = artifacts
    (out_dir / "model_metadata.json").write_text(json.dumps(completed, indent=2), encoding="utf-8")
    return completed


def verify_candidate_artifacts(model_dir: Path) -> dict[str, Any]:
    """
    WHAT: Returns a candidate's metadata after proving every recorded artifact sha256 still matches.

    WHY: A-066 scores the frozen candidate once; a changed booster or feature order must stop the run
    instead of being scored as the frozen selection.

    Raises:
        ValueError: If the metadata lacks artifacts or any recorded file differs or is missing.
    """
    metadata_path = Path(model_dir) / "model_metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    artifacts = metadata.get("artifacts")
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError(f"{metadata_path} records no artifact hashes")
    mismatches: list[str] = []
    for name, expected in artifacts.items():
        path = Path(model_dir) / name
        if not path.is_file():
            mismatches.append(f"{name} is missing")
            continue
        digest = _sha256_file(path)
        if digest != expected:
            mismatches.append(f"{name} sha256 {digest} != recorded {expected}")
    if mismatches:
        raise ValueError(f"candidate artifacts do not match {metadata_path}: " + "; ".join(mismatches))
    return metadata


def load_candidate_model(model_dir: Path) -> CandidateModel:
    """
    WHAT: Loads a verified candidate's boosters and checks their feature count and round counts.

    Raises:
        ValueError: If the structure is unknown, the feature order disagrees with the metadata, or a
            booster's shape differs from the recorded one.
    """
    metadata = verify_candidate_artifacts(model_dir)
    payload = load_feature_order(Path(model_dir) / "feature_order.yaml")
    features = payload["features"]
    if payload["feature_sha256"] != metadata["feature_sha256"] or len(features) != metadata["feature_count"]:
        raise ValueError(f"{model_dir} feature order does not match its metadata")
    if features[0] != PAYEE_RISK:
        raise ValueError(f"{model_dir} feature order does not start with {PAYEE_RISK!r}")
    structure = metadata.get("structure")
    segments: tuple[str, ...]
    if structure == SINGLE:
        segments = (SINGLE,)
    elif structure == "segmented":
        segments = SEGMENTS
    else:
        raise ValueError(f"{model_dir} has unknown structure {structure!r}")
    models: dict[str, xgboost.XGBClassifier] = {}
    for segment in segments:
        model = xgboost.XGBClassifier()
        model.load_model(str(Path(model_dir) / MODEL_FILE[segment]))
        booster = model.get_booster()
        if booster.num_features() != len(features):
            raise ValueError(f"{MODEL_FILE[segment]} takes {booster.num_features()} features, not {len(features)}")
        verify_booster_rounds(booster, {"best_params": metadata["best_params"][segment]})
        models[segment] = model
    return CandidateModel(candidate=str(metadata["candidate"]), payee_risk_index=0, models=models)


def run_candidate_device_check(
    candidate: str,
    train_matrix: np.ndarray,
    train_labels: np.ndarray,
    train_masks: Mapping[str, np.ndarray],
    validation_matrix: np.ndarray,
    validation_roles: Sequence[str],
    fixed_params: Mapping[str, int | float],
    seed: int,
    constraints: Sequence[int] | None,
) -> dict[str, Any]:
    """
    WHAT: Fits the candidate at A-056's fixed parameters once on cpu and once on cuda, scores both on
    pooled validation truth AP, and applies `choose_device`.

    WHY: A-066 requires a fresh device decision per candidate before its search, so no candidate's
    search can run on a device picked after seeing its scores.

    Raises:
        ValueError: If the cpu fit produces a non-finite truth AP.
    """
    segments = (SINGLE,) if not CANDIDATE_SEGMENTED[candidate] else SEGMENTS
    params_by_segment = {segment: dict(fixed_params) for segment in segments}
    device_records: dict[str, dict[str, Any]] = {}
    for device in ("cpu", "cuda"):
        started = time.perf_counter()
        error: str | None = None
        truth_ap: float | None = None
        try:
            model = fit_candidate_from_params(
                candidate, params_by_segment, constraints, train_matrix, train_labels, train_masks, device, seed
            )
        except (xgboost.core.XGBoostError, RuntimeError, OSError) as fit_error:
            error = f"{type(fit_error).__name__}: {fit_error}"
        seconds = time.perf_counter() - started
        if error is None:
            truth_ap = pooled_truth_ap(model.predict(validation_matrix).tolist(), validation_roles)
        device_records[device] = {"truth_ap": truth_ap, "seconds": seconds, "error": error}
        if device == "cpu":
            if truth_ap is None or not math.isfinite(truth_ap):
                raise ValueError(f"{candidate} cpu device check has no finite truth AP: {error}")
    chosen, reason = choose_device(
        float(device_records["cpu"]["truth_ap"]),
        float(device_records["cpu"]["seconds"]),
        device_records["cuda"]["truth_ap"],
        device_records["cuda"]["seconds"],
        device_records["cuda"]["error"],
    )
    return {
        "candidate": candidate,
        "fixed_params": dict(fixed_params),
        "seed": seed,
        "objective": OBJECTIVE,
        "tree_method": TREE_METHOD,
        "monotone_constraints": list(constraints) if constraints is not None else None,
        "cpu": device_records["cpu"],
        "cuda": device_records["cuda"],
        "ap_tolerance": 0.005,
        "chosen_device": chosen,
        "reason": reason,
    }


def run_candidate_search(
    candidate: str,
    train_matrix: np.ndarray,
    train_labels: np.ndarray,
    train_masks: Mapping[str, np.ndarray],
    validation_matrix: np.ndarray,
    validation_masks: Mapping[str, np.ndarray],
    validation_roles: Sequence[str],
    device: str,
    space: Mapping[str, Any],
    trial_count: int,
    seed: int,
    constraints: Sequence[int] | None,
    min_trials_before_stop: int,
    stall_patience_trials: int,
    min_improvement: float,
) -> dict[str, Any]:
    """
    WHAT: Runs A-066's search: one study over all rows for C1, and one study per segment for C2 and
    C3, with each study's objective that study's own validation truth AP.

    WHY: The studies reuse the shared `run_optuna_search` with the config's space, seed and stall
    rule, so the candidate scores are comparable with every other searched model in the project.

    Returns:
        The best parameters per segment, every study's record, importances and the search seconds.
    """
    segments = (SINGLE,) if not CANDIDATE_SEGMENTED[candidate] else SEGMENTS
    best: dict[str, dict[str, int | float]] = {}
    studies: dict[str, Any] = {}
    importances: dict[str, Any] = {}
    started = time.perf_counter()
    for segment in segments:
        if segment == SINGLE:
            train_rows = np.ones(train_matrix.shape[0], dtype=bool)
            validation_indices = np.arange(validation_matrix.shape[0])
        else:
            train_rows = train_masks[segment]
            validation_indices = np.nonzero(validation_masks[segment])[0]
        segment_matrix = train_matrix[train_rows]
        segment_labels = train_labels[train_rows]
        segment_validation = validation_matrix[validation_indices]
        segment_roles = [validation_roles[int(index)] for index in validation_indices]

        def train_trial(
            params: dict[str, int | float],
            matrix: np.ndarray = segment_matrix,
            labels: np.ndarray = segment_labels,
            scored_matrix: np.ndarray = segment_validation,
            pair: str = segment,
        ) -> Sequence[float]:
            """Trains one trial's booster and returns its segment's validation probabilities."""
            model = fit_candidate_model(matrix, labels, params, device, seed, constraints)
            return predict_scores(model, scored_matrix)

        search = run_optuna_search(
            train_trial,
            payment_scored(segment_roles),
            space,
            trial_count,
            seed,
            min_trials_before_stop=min_trials_before_stop,
            stall_patience_trials=stall_patience_trials,
            min_improvement=min_improvement,
        )
        best[segment] = dict(search.best_params)
        studies[segment] = {
            "trials_run": search.trials_run,
            "stopped_early": search.stopped_early,
            "stop_reason": search.stop_reason,
            "best_params": dict(search.best_params),
            "best_score": search.best_score,
            "trials": list(search.trials),
        }
        importances[segment] = search.param_importances
    return {
        "best_params": best,
        "studies": studies,
        "importances": importances,
        "search_seconds": time.perf_counter() - started,
    }


def measure_v1_validation(
    model_dir: Path,
    validation_matrix: np.ndarray,
    world_ids: Sequence[str],
    validation_roles: Sequence[str],
) -> dict[str, Any]:
    """
    WHAT: Scores the frozen v1 model on the validation rows and returns its pooled AP and per-world
    p10, after A-058's artifact and shape checks.

    WHY: A-066's selection compares every candidate's p10 with v1's p10; measuring it from the frozen
    model on validation rows is a measurement, not a fit, and must happen before any candidate
    search.

    Raises:
        ValueError: If the frozen artifacts, feature order, booster shape or validation AP disagree
            with the recorded v1 metadata.
    """
    model_dir = Path(model_dir)
    verify_frozen_artifacts(model_dir)
    metadata = json.loads((model_dir / "model_metadata.json").read_text(encoding="utf-8"))
    payload = load_feature_order(model_dir / "feature_order.yaml")
    features = payload["features"]
    if payload["feature_sha256"] != metadata["feature_sha256"] or len(features) != metadata["feature_count"]:
        raise ValueError(f"{model_dir} feature order does not match its metadata")
    model = xgboost.XGBClassifier()
    model.load_model(str(model_dir / "frozen_model.json"))
    booster = model.get_booster()
    if booster.num_features() != len(features):
        raise ValueError(f"v1 frozen model takes {booster.num_features()} features, not {len(features)}")
    verify_booster_rounds(booster, metadata)
    scores = predict_scores(model, validation_matrix)
    truth_ap = pooled_truth_ap(scores, validation_roles)
    recorded = float(metadata["validation_truth_ap"])
    if not math.isclose(truth_ap, recorded, rel_tol=0.0, abs_tol=V1_REPRODUCTION_TOLERANCE):
        raise ValueError(f"v1 validation truth AP {truth_ap} does not reproduce its recorded {recorded}")
    world_rows = world_truth_ap(world_ids, scores, truth_labels(validation_roles))
    return {
        "model_dir": str(model_dir),
        "artifacts": {name: _sha256_file(model_dir / name) for name in ("frozen_model.json", "feature_order.yaml")},
        "validation_truth_ap": truth_ap,
        "recorded_validation_truth_ap": recorded,
        "validation_worlds": world_rows,
        "validation_p10": world_p10(world_rows),
    }


def freeze_candidate(
    candidate: str,
    search: Mapping[str, Any],
    device: str,
    train_matrix: np.ndarray,
    train_labels: np.ndarray,
    train_masks: Mapping[str, np.ndarray],
    validation_matrix: np.ndarray,
    validation_world_ids: Sequence[str],
    validation_roles: Sequence[str],
    features: Sequence[str],
    feature_order_path: Path,
    out_dir: Path,
    seed: int,
) -> dict[str, Any]:
    """
    WHAT: Refits the searched parameters, writes the candidate's artifacts and records its pooled
    validation AP and per-world p10.

    Raises:
        ValueError: If the written model's validation AP is not finite.
    """
    constraints = monotone_vector(features) if CANDIDATE_MONOTONE[candidate] else None
    model = fit_candidate_from_params(
        candidate, search["best_params"], constraints, train_matrix, train_labels, train_masks, device, seed
    )
    scores = model.predict(validation_matrix).tolist()
    truth_ap = pooled_truth_ap(scores, validation_roles)
    if not math.isfinite(truth_ap):
        raise ValueError(f"{candidate} validation truth AP is not finite: {truth_ap!r}")
    world_rows = world_truth_ap(validation_world_ids, scores, truth_labels(validation_roles))
    segments = (SINGLE,) if not CANDIDATE_SEGMENTED[candidate] else SEGMENTS
    segment_rows = {
        segment: train_masks[segment] if segment != SINGLE else np.ones(len(train_labels), dtype=bool)
        for segment in segments
    }
    metadata = {
        "feature_sha256": load_feature_order(feature_order_path)["feature_sha256"],
        "feature_count": len(features),
        "device": device,
        "seed": seed,
        "best_params": search["best_params"],
        "trials_run": {segment: search["studies"][segment]["trials_run"] for segment in segments},
        "stopped_early": {segment: search["studies"][segment]["stopped_early"] for segment in segments},
        "stop_reason": {segment: search["studies"][segment]["stop_reason"] for segment in segments},
        "search_best_score": {segment: search["studies"][segment]["best_score"] for segment in segments},
        "training_rows": {segment: int(segment_rows[segment].sum()) for segment in segments},
        "training_positives": {segment: int(train_labels[segment_rows[segment]].sum()) for segment in segments},
        "validation_truth_ap": truth_ap,
        "validation_rows": len(validation_roles),
        "validation_positives": int(sum(truth_labels(validation_roles))),
        "validation_worlds": world_rows,
        "validation_p10": world_p10(world_rows),
        "search_seconds": search["search_seconds"],
    }
    written = write_candidate_artifacts(out_dir, candidate, model, features, feature_order_path, metadata)
    (out_dir / "search_trials.json").write_text(json.dumps(search["studies"], indent=2), encoding="utf-8")
    (out_dir / "search_importances.json").write_text(json.dumps(search["importances"], indent=2), encoding="utf-8")
    return written


def run_candidates(
    training_dirs: Sequence[Path],
    validation_dirs: Sequence[Path],
    truth_root: Path,
    oof_root: Path,
    out_root: Path,
    feature_order_path: Path,
    v1_model_dir: Path,
    mule_config_path: Path = DEFAULT_MULE_CONFIG_PATH,
    candidates: Sequence[str] = CANDIDATE_ORDER,
) -> dict[str, Any]:
    """
    WHAT: Runs A-066 end to end on training and validation rows only: v1's p10, each candidate's
    device check, search and freeze, then the selection and the replacement gate.

    WHY: One entry point keeps the frozen rows, expected counts and the order (v1 measurement before
    any search) in one place.

    Raises:
        ValueError: If a folder is missing, an expected count or hash differs, or a candidate is
            unknown.
    """
    for name in candidates:
        if name not in CANDIDATE_ORDER:
            raise ValueError(f"unknown candidate {name!r}")
    for path in [*training_dirs, *validation_dirs, Path(truth_root), Path(oof_root), Path(feature_order_path)]:
        if not Path(path).exists():
            raise ValueError(f"required folder or file is missing: {path}")
    payload = load_feature_order(Path(feature_order_path))
    features = payload["features"]
    encodings = payload["categorical_encodings"]
    if not isinstance(features, list) or not isinstance(encodings, dict):
        raise ValueError(f"{feature_order_path} holds malformed feature order data")
    if features[0] != PAYEE_RISK:
        raise ValueError(f"{feature_order_path} does not start with {PAYEE_RISK!r}")
    config = load_mule_model_config(Path(mule_config_path))
    space = config.hyperparameter_space
    trial_count = config.optuna_trials
    seed = config.optuna_seed
    if space is None or trial_count is None or seed is None:
        raise ValueError(f"{mule_config_path} lacks hyperparameter_space, optuna_trials or optuna_seed")
    fixed_params = _fixed_device_params(space)

    print(f"loading {len(training_dirs)} training datasets", flush=True)
    training_rows = load_split_rows(training_dirs, "training", features, oof_root)
    training_labels = target_labels(training_rows)
    if training_labels.shape[0] != EXPECTED_TRAINING_ROWS or int(training_labels.sum()) != EXPECTED_TRAINING_POSITIVES:
        raise ValueError(
            f"training rows {training_labels.shape[0]} with {int(training_labels.sum())} positives, "
            f"expected {EXPECTED_TRAINING_ROWS} and {EXPECTED_TRAINING_POSITIVES}"
        )
    training_matrix = build_matrix(training_rows, features, encodings)
    training_masks = segment_masks(training_rows[PAYEE_RISK].to_numpy(zero_copy_only=False))
    del training_rows
    print(f"training rows {training_matrix.shape[0]} positives {int(training_labels.sum())}", flush=True)

    print(f"loading {len(validation_dirs)} validation datasets", flush=True)
    validation_rows = load_split_rows(validation_dirs, "validation", features, oof_root)
    validation_roles = load_validation_roles(validation_dirs, Path(truth_root), validation_rows)
    validation_labels = truth_labels(validation_roles)
    validation_positives = int(sum(validation_labels))
    if len(validation_roles) != EXPECTED_VALIDATION_ROWS or validation_positives != EXPECTED_VALIDATION_POSITIVES:
        raise ValueError(
            f"validation rows {len(validation_roles)} with {validation_positives} positives, "
            f"expected {EXPECTED_VALIDATION_ROWS} and {EXPECTED_VALIDATION_POSITIVES}"
        )
    validation_matrix = build_matrix(validation_rows, features, encodings)
    validation_masks = segment_masks(validation_rows[PAYEE_RISK].to_numpy(zero_copy_only=False))
    validation_world_ids = validation_rows["world_id"].to_pylist()
    del validation_rows
    print(f"validation rows {validation_matrix.shape[0]} worlds {len(set(validation_world_ids))}", flush=True)

    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    print("v1 validation measurement", flush=True)
    v1 = measure_v1_validation(Path(v1_model_dir), validation_matrix, validation_world_ids, validation_roles)
    (out_root / "selection.json").write_text(
        json.dumps({"v1": {key: value for key, value in v1.items() if key != "validation_worlds"}}, indent=2),
        encoding="utf-8",
    )
    print(f"v1 validation truth AP {v1['validation_truth_ap']:.12f} p10 {v1['validation_p10']['p10']:.12f}", flush=True)

    records: list[dict[str, Any]] = []
    for candidate in candidates:
        print(f"{candidate} device check", flush=True)
        constraints = monotone_vector(features) if CANDIDATE_MONOTONE[candidate] else None
        device_record = run_candidate_device_check(
            candidate,
            training_matrix,
            training_labels,
            training_masks,
            validation_matrix,
            validation_roles,
            fixed_params,
            seed,
            constraints,
        )
        (out_root / candidate).mkdir(parents=True, exist_ok=True)
        (out_root / candidate / "device_check.json").write_text(json.dumps(device_record, indent=2), encoding="utf-8")
        device = str(device_record["chosen_device"])
        print(f"{candidate} device {device}: {device_record['reason']}", flush=True)
        print(f"{candidate} search", flush=True)
        search = run_candidate_search(
            candidate,
            training_matrix,
            training_labels,
            training_masks,
            validation_matrix,
            validation_masks,
            validation_roles,
            device,
            space,
            trial_count,
            seed,
            constraints,
            config.optuna_min_trials_before_stop,
            config.optuna_stall_patience_trials,
            config.optuna_min_improvement,
        )
        metadata = freeze_candidate(
            candidate,
            search,
            device,
            training_matrix,
            training_labels,
            training_masks,
            validation_matrix,
            validation_world_ids,
            validation_roles,
            features,
            Path(feature_order_path),
            out_root / candidate,
            seed,
        )
        print(
            f"{candidate} validation AP {metadata['validation_truth_ap']:.12f} "
            f"p10 {metadata['validation_p10']['p10']:.12f}",
            flush=True,
        )
        records.append(
            {
                "candidate": candidate,
                "title": CANDIDATE_TITLE[candidate],
                "device": device,
                "trials_run": metadata["trials_run"],
                "validation_ap": metadata["validation_truth_ap"],
                "validation_p10": metadata["validation_p10"]["p10"],
                "validation_p10_worlds_used": metadata["validation_p10"]["worlds_used"],
                "model_dir": str(out_root / candidate),
            }
        )
        del search

    trace = select_candidate(records, float(v1["validation_p10"]["p10"]))
    winner = trace["winner"]
    gate: dict[str, Any] | None = None
    if winner is None:
        gate = {"replaces_v1": False, "reason": trace["reason"]}
    else:
        winner_record = next(record for record in records if record["candidate"] == winner)
        gate = replacement_decision(
            float(winner_record["validation_ap"]),
            float(winner_record["validation_p10"]),
            float(v1["validation_p10"]["p10"]),
        )
        gate["reason"] = (
            f"{winner} validation AP {gate['validation_ap']:.12f} versus the literal floor "
            f"{V1_VALIDATION_AP_FLOOR} and p10 {gate['validation_p10']:.12f} versus v1 {gate['v1_p10']:.12f}"
        )
    if not gate["replaces_v1"]:
        gate["regeneration_required"] = []
    else:
        gate["regeneration_required"] = list(REGENERATION_IF_REPLACED)
    selection = {
        "amendment": "A-066",
        "v1": {key: value for key, value in v1.items() if key != "validation_worlds"},
        "candidates": records,
        "selection": trace,
        "gate": gate,
    }
    (out_root / "selection.json").write_text(json.dumps(selection, indent=2), encoding="utf-8")
    print(json.dumps({"selection": trace, "gate": gate}, indent=2), flush=True)
    return selection


def validate_winner_scoring_folder(dataset_dir: Path, split: str) -> dict[str, Any]:
    """Returns a scored folder's manifest after A-058's test/stress guard accepts it for `split`."""
    return validate_scoring_split(Path(dataset_dir), split)


def score_candidate_split(
    dataset_dirs: Sequence[Path],
    split: str,
    features: Sequence[str],
    encodings: Mapping[str, Sequence[str]],
    model: CandidateModel,
    truth_root: Path,
    account_model: PUBaggedAccountModel,
    account_config: Path,
) -> tuple[list[WorldScores], list[list[str]], dict[str, int], list[dict[str, Any]]]:
    """
    WHAT: Scores one test or stress split's worlds with the selected candidate, attaching the frozen
    Run A `payee_risk` and the hidden receiver role, and returns per-world arrays, receiver ids,
    NaN-reason counts and dataset records.

    WHY: A-058's scoring procedure is reused function for function (split guard, payee-risk rule,
    key alignment, metric rules); only the probability source is the candidate's segment routing.

    Raises:
        ValueError: If a folder is not the expected split, truth is missing, roles or keys disagree,
            or a payment has neither a score nor a recorded reason.
    """
    worlds: list[WorldScores] = []
    receivers: list[list[str]] = []
    reason_counts: dict[str, int] = {}
    dataset_records: list[dict[str, Any]] = []
    needed = [*KEY_COLUMNS, "bank_id", "decision_time", "receiver_account_id"]
    needed.extend(name for name in features if name != PAYEE_RISK and name not in needed)

    previous = mule_model_config_override()
    set_mule_model_config_path(account_config)
    try:
        for dataset_dir in dataset_dirs:
            manifest = validate_winner_scoring_folder(dataset_dir, split)
            truth_payments, truth_snapshots = join_truth(dataset_dir, truth_root)
            roles = truth_payments.select([*KEY_COLUMNS, "truth_role_receiver"]).sort_by(
                [("world_id", "ascending"), ("payment_id", "ascending")]
            )
            del truth_snapshots, truth_payments
            dataset_rows = 0
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
                    table.to_pylist(),
                    scores_by_account,
                    bank_by_account,
                    world_id,
                    split,
                    -1,
                )
                for row in risk_rows:
                    reason = str(row["nan_reason"]) or "scored"
                    reason_counts[reason] = reason_counts.get(reason, 0) + 1
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
                labels = truth_labels(role_values)

                matrix = build_matrix(table, features, encodings)
                scores = model.predict(matrix).tolist()
                worlds.append(
                    WorldScores(
                        split=split,
                        dataset=dataset_dir.name,
                        world_id=world_id,
                        scores=scores,
                        labels=labels,
                        payee_risk=table[PAYEE_RISK].to_numpy(zero_copy_only=False),
                    )
                )
                receivers.append(table["receiver_account_id"].to_pylist())
                dataset_rows += table.num_rows
            dataset_records.append(
                {
                    "folder": dataset_dir.name,
                    "content_hash": manifest.get("content_hash"),
                    "rows": dataset_rows,
                    "worlds": len(manifest["worlds"]),
                }
            )
    finally:
        set_mule_model_config_path(previous)
    return worlds, receivers, reason_counts, dataset_records


def _held_rows(
    worlds: Sequence[WorldScores],
    receivers: Sequence[Sequence[str]],
    threshold: float,
) -> list[tuple[str, str, int]]:
    """Returns (world, receiver, label) for every payment scoring at or above the frozen threshold."""
    rows: list[tuple[str, str, int]] = []
    for world, world_receivers in zip(worlds, receivers, strict=True):
        for score, receiver, label in zip(world.scores, world_receivers, world.labels, strict=True):
            if score >= threshold:
                rows.append((world.world_id, str(receiver), int(label)))
    return rows


def score_winner(
    validation_dirs: Sequence[Path],
    test_dirs: Sequence[Path],
    stress_dirs: Sequence[Path],
    truth_root: Path,
    oof_root: Path,
    model_dir: Path,
    out_root: Path,
) -> dict[str, Any]:
    """
    WHAT: Scores the selected candidate once on validation (for the threshold), test and stress with
    A-058's procedure, and writes `result.json`.

    WHY: A-066 scores the winner once and reports it beside v1, never instead; the threshold comes
    from the winner's validation scores alone and is applied unchanged to test and stress.

    Raises:
        ValueError: If a split is empty, the candidate's artifacts disagree with their metadata, or
            any A-058 guard fails.
    """
    if not test_dirs:
        raise ValueError("winner scoring needs at least one test dataset")
    if not stress_dirs:
        raise ValueError("winner scoring needs at least one stress dataset")
    started = time.perf_counter()
    model = load_candidate_model(Path(model_dir))
    metadata = json.loads((Path(model_dir) / "model_metadata.json").read_text(encoding="utf-8"))
    payload = load_feature_order(Path(model_dir) / "feature_order.yaml")
    features = payload["features"]
    encodings = payload["categorical_encodings"]
    if not isinstance(features, list) or not isinstance(encodings, dict):
        raise ValueError(f"{model_dir} holds malformed feature order data")

    for dataset_dir in validation_dirs:
        validate_dataset_split(Path(dataset_dir), "validation")
    validation_rows = load_split_rows(validation_dirs, "validation", features, oof_root)
    validation_matrix = build_matrix(validation_rows, features, encodings)
    del validation_rows
    validation_scores = model.predict(validation_matrix).tolist()
    threshold = hold_threshold(validation_scores, HOLD_SHARE)

    recipe = load_frozen_recipe()
    account_model = PUBaggedAccountModel.load(recipe.checkpoint_dir)
    scored: dict[str, list[WorldScores]] = {}
    receivers: dict[str, list[list[str]]] = {}
    extras: dict[str, dict[str, Any]] = {}
    for split, dataset_dirs in (("test", test_dirs), ("stress", stress_dirs)):
        worlds, world_receivers, reason_counts, dataset_records = score_candidate_split(
            dataset_dirs,
            split,
            features,
            encodings,
            model,
            Path(truth_root),
            account_model,
            recipe.config_path,
        )
        scored[split] = worlds
        receivers[split] = world_receivers
        extras[split] = {"reason_counts": reason_counts, "datasets": dataset_records}

    result = evaluate_splits(validation_scores, scored)
    for split in ("test", "stress"):
        reason_counts = extras[split]["reason_counts"]
        total = sum(reason_counts.values())
        result["splits"][split]["nan_reason_counts"] = reason_counts
        result["splits"][split]["nan_payee_risk_share"] = (total - reason_counts.get("scored", 0)) / total
        result["splits"][split]["datasets"] = extras[split]["datasets"]
        held = _held_rows(scored[split], receivers[split], threshold)
        if not held:
            raise ValueError(f"split {split!r} has no payments at the frozen threshold")
        result["splits"][split]["payee_precision"] = dedupe_payees(held)
    stress = scored["stress"]
    stress_alone = []
    for world in stress:
        alone_ap, _missing = feature_truth_ap(np.asarray(world.payee_risk, dtype=np.float64), np.asarray(world.labels))
        stress_alone.append(alone_ap)
    result["splits"]["stress"]["payee_risk_alone_truth_ap"] = aggregate(stress_alone)

    result["manifest"] = {
        "amendment": "A-066",
        "candidate": metadata["candidate"],
        "model": {
            "path": str(Path(model_dir)),
            "artifacts": metadata["artifacts"],
            "feature_sha256": metadata["feature_sha256"],
            "feature_count": metadata["feature_count"],
            "seed": metadata["seed"],
            "device": metadata["device"],
        },
        "account_recipe": {
            "config": str(recipe.config_path),
            "config_sha256": recipe.config_sha256,
            "checkpoint": str(recipe.checkpoint_dir),
            "seed": recipe.seed,
            "n_bags": recipe.n_bags,
        },
        "validation_dataset_ids": [Path(path).name for path in validation_dirs],
        "scored_dataset_ids": {
            "test": [Path(path).name for path in test_dirs],
            "stress": [Path(path).name for path in stress_dirs],
        },
        "threshold": threshold,
        "hold_share": HOLD_SHARE,
        "wall_seconds": time.perf_counter() - started,
    }
    out_root = Path(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "candidate": metadata["candidate"],
                "threshold": threshold,
                "test": result["splits"]["test"]["aggregate"],
                "stress": result["splits"]["stress"]["aggregate"],
                "stress_payee_risk_alone": result["splits"]["stress"]["payee_risk_alone_truth_ap"],
            },
            indent=2,
        ),
        flush=True,
    )
    return result


def main(argv: Sequence[str] | None = None) -> int:
    """Runs the payment-candidates CLI."""
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    run_parser = commands.add_parser("run", help="Device checks, searches, freeze and selection")
    run_parser.add_argument("--training", action="append", required=True, help="Training dataset folder (repeatable)")
    run_parser.add_argument(
        "--validation", action="append", required=True, help="Validation dataset folder (repeatable)"
    )
    run_parser.add_argument("--truth-root", default=str(DEFAULT_TRUTH_ROOT), help="Hidden truth root")
    run_parser.add_argument("--oof-root", default=str(DEFAULT_OOF_ROOT), help="Phase 1 payee-risk root")
    run_parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT), help="Output root")
    run_parser.add_argument("--feature-order", default=str(DEFAULT_FEATURE_ORDER_PATH), help="Frozen feature order")
    run_parser.add_argument("--v1-model-dir", default=str(DEFAULT_V1_MODEL_DIR), help="Frozen v1 model folder")
    run_parser.add_argument("--mule-config", default=str(DEFAULT_MULE_CONFIG_PATH), help="Search config")

    score_parser = commands.add_parser("score-winner", help="Score the selected candidate once")
    score_parser.add_argument(
        "--validation", action="append", required=True, help="Validation dataset folder (repeatable)"
    )
    score_parser.add_argument("--test", action="append", required=True, help="Test dataset folder (repeatable)")
    score_parser.add_argument("--stress", action="append", required=True, help="Stress dataset folder (repeatable)")
    score_parser.add_argument("--truth-root", default=str(DEFAULT_TRUTH_ROOT), help="Hidden truth root")
    score_parser.add_argument("--oof-root", default=str(DEFAULT_OOF_ROOT), help="Phase 1 payee-risk root")
    score_parser.add_argument("--model-dir", required=True, help="Selected candidate folder")
    score_parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT / "test_score"), help="Output folder")

    args = parser.parse_args(argv)
    if args.command == "run":
        run_candidates(
            [Path(path) for path in args.training],
            [Path(path) for path in args.validation],
            Path(args.truth_root),
            Path(args.oof_root),
            Path(args.out_root),
            Path(args.feature_order),
            Path(args.v1_model_dir),
            Path(args.mule_config),
        )
        return 0
    score_winner(
        [Path(path) for path in args.validation],
        [Path(path) for path in args.test],
        [Path(path) for path in args.stress],
        Path(args.truth_root),
        Path(args.oof_root),
        Path(args.model_dir),
        Path(args.out_root),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
