"""B1-B4: validation-only improvement experiments on the frozen C1 payment recipe (A-068).

Every experiment reuses the frozen C1 parameters, feature order and cpu device, scores validation
truth only, and writes each finished item to runs/full/audit_20261125/b_results.json. B1 is the
capped residual stacking of a payee-risk-only booster and a booster on the remaining features, B2
reweights training rows by 1/count(receiver_account_id), B3 drops payee_same_bank_risk, and B4 runs
a short pinned-LightGBM Optuna search. No test or stress row is read anywhere.
"""

from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path

import lightgbm
import numpy as np
import optuna
import xgboost
from sklearn.metrics import average_precision_score

from evaluation.payment_candidates import monotone_vector
from evaluation.payment_model import (
    PAYEE_RISK,
    build_matrix,
    load_feature_order,
    load_split_rows,
    load_validation_roles,
    target_labels,
    truth_labels,
)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "runs" / "full" / "datasets"
OUT = ROOT / "runs" / "full" / "audit_20261125"
RESULTS = OUT / "b_results.json"
TRAINING = [
    DATA / "training-20261031T1830-2cdf8145a8d5",
    DATA / "training-20261031T1830-9e76fa11f9db",
    DATA / "training-20261031T1830-b16d733022fe",
    DATA / "training-20261031T1830-fb7561164f4f",
]
VALIDATION = [
    DATA / "validation-20261031T1830-5a6d507bde9c",
    DATA / "validation-20261031T1830-99fed7ecb1d1",
]
FEATURE_ORDER = ROOT / "config" / "mule_payment_feature_order.yaml"
TRUTH_ROOT = ROOT / "runs" / "full" / "truth"
OOF_ROOT = ROOT / "runs" / "full" / "payment_oof"
MODEL_DIR = ROOT / "runs" / "full" / "payment_model_v2" / "C1"
THRESHOLD_RESULT = ROOT / "runs" / "full" / "payment_model_v2" / "test_score" / "result.json"
B4_BOX_MINUTES = 25.0
B4_MAX_TRIALS = 15


def write_result(name: str, record: dict) -> None:
    """Add or replace one named result in the incremental results file."""
    document = json.loads(RESULTS.read_text(encoding="utf-8")) if RESULTS.exists() else {"simulated": True, "items": {}}
    document["items"][name] = record
    RESULTS.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(name, json.dumps(record), flush=True)


def world_ap_spread(world_ids: list[str], labels: np.ndarray, scores: np.ndarray) -> dict:
    """Returns the per-world AP p10/median/p90 over worlds that carry both classes."""
    groups: dict[str, list[int]] = {}
    for index, world in enumerate(world_ids):
        groups.setdefault(world, []).append(index)
    aps = []
    for indices in groups.values():
        idx = np.asarray(indices)
        y = labels[idx]
        if y.min() == y.max():
            continue
        aps.append(float(average_precision_score(y, scores[idx])))
    values = np.asarray(aps)
    return {
        "worlds_used": int(values.size),
        "median": float(np.percentile(values, 50)),
        "p10": float(np.percentile(values, 10)),
        "p90": float(np.percentile(values, 90)),
    }


def main() -> int:
    """Loads rows once, then runs B1, B2, B3 and B4 in order."""
    started = time.perf_counter()
    metadata = json.loads((MODEL_DIR / "model_metadata.json").read_text(encoding="utf-8"))
    params = dict(metadata["best_params"]["all"])
    feature_payload = load_feature_order(FEATURE_ORDER)
    features = feature_payload["features"]
    encodings = feature_payload["categorical_encodings"]
    threshold = float(json.loads(THRESHOLD_RESULT.read_text(encoding="utf-8"))["threshold"])

    print("loading training rows", flush=True)
    training_rows = load_split_rows(TRAINING, "training", features, OOF_ROOT)
    train_labels = target_labels(training_rows)
    train_matrix = build_matrix(training_rows, features, encodings)
    print(f"training matrix {train_matrix.shape}", flush=True)

    print("loading validation rows", flush=True)
    validation_rows = load_split_rows(VALIDATION, "validation", features, OOF_ROOT)
    validation_roles = load_validation_roles(VALIDATION, TRUTH_ROOT, validation_rows)
    validation_labels = np.asarray(truth_labels(validation_roles), dtype=np.int8)
    validation_worlds = validation_rows["world_id"].to_pylist()
    validation_matrix = build_matrix(validation_rows, features, encodings)
    del validation_rows
    base_rate = float(validation_labels.mean())
    print(f"validation matrix {validation_matrix.shape} base rate {base_rate:.8f}", flush=True)

    constants = monotone_vector(features)

    def metrics(scores: np.ndarray) -> dict:
        pooled = float(average_precision_score(validation_labels, scores))
        spread = world_ap_spread(validation_worlds, validation_labels, scores)
        return {
            "validation_ap": pooled,
            "validation_p10": spread["p10"],
            "validation_world_ap": spread,
            "validation_hold_rate_at_v2_threshold": float((scores >= threshold).mean()),
            "base_rate": base_rate,
        }

    def fit_xgboost(
        matrix: np.ndarray, labels: np.ndarray, constraints: list[int], sample_weight: np.ndarray | None = None
    ) -> xgboost.XGBClassifier:
        model = xgboost.XGBClassifier(
            objective="binary:logistic",
            tree_method="hist",
            device="cpu",
            random_state=1717,
            monotone_constraints=tuple(constraints),
            **params,
        )
        model.fit(matrix, labels, sample_weight=sample_weight)
        return model

    # B1: capped residual stacking, P on payee_risk alone and Q on the remaining 53 features.
    print("B1 residual stacking", flush=True)
    payee_index = features.index(PAYEE_RISK)
    other_columns = [index for index, name in enumerate(features) if name != PAYEE_RISK]
    p_model = fit_xgboost(train_matrix[:, [payee_index]], train_labels, [constants[payee_index]])
    q_model = fit_xgboost(train_matrix[:, other_columns], train_labels, [0] * len(other_columns))
    p_scores = p_model.predict_proba(validation_matrix[:, [payee_index]])[:, 1]
    q_scores = q_model.predict_proba(validation_matrix[:, other_columns])[:, 1]
    eps = float(np.finfo(np.float32).eps)
    p_logit = np.log(np.clip(p_scores, eps, 1.0 - eps) / (1.0 - np.clip(p_scores, eps, 1.0 - eps)))
    q_logit = np.log(np.clip(q_scores, eps, 1.0 - eps) / (1.0 - np.clip(q_scores, eps, 1.0 - eps)))
    b1_variants = {}
    for c in (0.5, 1.0, 2.0):
        stacked_logit = p_logit + np.clip(q_logit, -c, c)
        stacked = 1.0 / (1.0 + np.exp(-stacked_logit))
        b1_variants[f"c={c}"] = metrics(stacked)
    best_key = max(b1_variants, key=lambda key: (b1_variants[key]["validation_ap"], b1_variants[key]["validation_p10"]))
    b1 = dict(b1_variants[best_key])
    b1.update(
        {
            "chosen_c": float(best_key.split("=")[1]),
            "variants": b1_variants,
            "p_only_validation_ap": float(average_precision_score(validation_labels, p_scores)),
            "q_only_validation_ap": float(average_precision_score(validation_labels, q_scores)),
            "device": "cpu",
            "minutes": round((time.perf_counter() - started) / 60.0, 1),
        }
    )
    write_result("B1_capped_residual_stacking", b1)

    # B2: training-row weights 1 / count(receiver_account_id).
    print("B2 payee-deduplicated weights", flush=True)
    account_ids = training_rows["receiver_account_id"].to_pylist()
    counts = Counter(account_ids)
    weights = np.fromiter((1.0 / counts[account] for account in account_ids), dtype=np.float64, count=len(account_ids))
    del account_ids
    b2_model = fit_xgboost(train_matrix, train_labels, constants, sample_weight=weights)
    b2_scores = b2_model.predict_proba(validation_matrix)[:, 1]
    b2 = metrics(b2_scores)
    b2.update(
        {
            "distinct_payees": len(counts),
            "weight_total": float(weights.sum()),
            "device": "cpu",
            "minutes": round((time.perf_counter() - started) / 60.0, 1),
        }
    )
    del weights, counts
    write_result("B2_payee_deduplicated_weights", b2)

    # B3: drop payee_same_bank_risk with the adjusted monotone vector.
    print("B3 without payee_same_bank_risk", flush=True)
    kept = [index for index, name in enumerate(features) if name != "payee_same_bank_risk"]
    kept_constants = monotone_vector([features[index] for index in kept])
    b3_model = fit_xgboost(train_matrix[:, kept], train_labels, kept_constants)
    b3_scores = b3_model.predict_proba(validation_matrix[:, kept])[:, 1]
    b3 = metrics(b3_scores)
    b3.update(
        {
            "features_removed": ["payee_same_bank_risk"],
            "feature_count": len(kept),
            "device": "cpu",
            "minutes": round((time.perf_counter() - started) / 60.0, 1),
        }
    )
    write_result("B3_without_payee_same_bank_risk", b3)

    # B4: pinned LightGBM with a short Optuna search, validation AP as the objective.
    print("B4 lightgbm optuna", flush=True)
    b4_started = time.perf_counter()

    def objective(trial: optuna.Trial) -> float:
        trial_params = {
            "objective": "binary",
            "n_estimators": int(params["n_estimators"]),
            "num_leaves": trial.suggest_int("num_leaves", 16, 96),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.05, log=True),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "subsample_freq": 1,
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 0.9),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "min_child_weight": params["min_child_weight"],
            "monotone_constraints": constants,
            "random_state": 1717,
            "n_jobs": -1,
            "verbosity": -1,
        }
        model = lightgbm.LGBMClassifier(**trial_params)
        model.fit(train_matrix, train_labels)
        scores = np.asarray(model.predict_proba(validation_matrix))[:, 1]
        return float(average_precision_score(validation_labels, scores))

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=1717))
    trials_run = 0
    for trial_index in range(B4_MAX_TRIALS):
        elapsed_minutes = (time.perf_counter() - b4_started) / 60.0
        if trial_index > 0 and elapsed_minutes >= B4_BOX_MINUTES:
            break
        study.optimize(objective, n_trials=1)
        trials_run += 1
        print(f"B4 trial {trials_run} value {study.best_value:.6f} elapsed {elapsed_minutes:.1f}m", flush=True)
    best = study.best_params
    best_model = lightgbm.LGBMClassifier(
        objective="binary",
        n_estimators=int(params["n_estimators"]),
        min_child_weight=params["min_child_weight"],
        subsample_freq=1,
        monotone_constraints=constants,
        random_state=1717,
        n_jobs=-1,
        verbosity=-1,
        **best,
    )
    best_model.fit(train_matrix, train_labels)
    b4_scores = np.asarray(best_model.predict_proba(validation_matrix))[:, 1]
    b4 = metrics(b4_scores)
    b4.update(
        {
            "learner": "lightgbm",
            "lightgbm_version": lightgbm.__version__,
            "trials_run": trials_run,
            "max_trials": B4_MAX_TRIALS,
            "best_params": best,
            "box_minutes": B4_BOX_MINUTES,
            "partial": trials_run < B4_MAX_TRIALS,
            "device": "cpu",
            "minutes": round((time.perf_counter() - started) / 60.0, 1),
        }
    )
    write_result("B4_lightgbm_optuna", b4)

    write_result(
        "B_summary",
        {
            "wall_seconds": round(time.perf_counter() - started, 1),
            "device": "cpu",
            "c1_params": params,
            "threshold": threshold,
            "logit_epsilon": eps,
            "training_rows": int(train_matrix.shape[0]),
            "validation_rows": int(validation_matrix.shape[0]),
            "validation_base_rate": base_rate,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
