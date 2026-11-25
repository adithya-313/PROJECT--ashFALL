"""A2/A3/A4: retrain the frozen C1 payment recipe on training rows and score validation.

A2 shuffles the reported target within each training world (seed 1717) and checks the validation
AP against the base rate. A3 removes payee_risk and the eight payment-level features. A4 retrains
with seeds 1718 and 1719. Every fit uses the frozen C1 parameters and monotone vector, the chosen
device is cpu, and results are written after each fit so a stopped run keeps its finished items.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import xgboost
import yaml
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

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "runs" / "full" / "datasets"
OUT = ROOT / "runs" / "full" / "audit_20261125"
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
PAYMENT_CODE_CONFIG = ROOT / "config" / "feature_reason_codes.yaml"
RESULTS = OUT / "a_training_results.json"


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
    """Loads the frozen rows once, then runs A2, A3 and A4 in order."""
    started = time.perf_counter()
    metadata = json.loads((MODEL_DIR / "model_metadata.json").read_text(encoding="utf-8"))
    params = dict(metadata["best_params"]["all"])
    feature_payload = load_feature_order(FEATURE_ORDER)
    features = feature_payload["features"]
    encodings = feature_payload["categorical_encodings"]
    code_config = yaml.safe_load(PAYMENT_CODE_CONFIG.read_text(encoding="utf-8"))
    payment_features = sorted(name for name, code in code_config["features"].items() if code == "PAYMENT_LEVEL")
    print("payment-level features", payment_features, flush=True)

    print("loading training rows", flush=True)
    training_rows = load_split_rows(TRAINING, "training", features, OOF_ROOT)
    train_labels = target_labels(training_rows)
    train_worlds = training_rows["world_id"].to_pylist()
    train_matrix = build_matrix(training_rows, features, encodings)
    del training_rows
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

    def fit_and_score(
        matrix: np.ndarray, labels: np.ndarray, constraints: list[int], seed: int, validation: np.ndarray
    ) -> dict:
        model = xgboost.XGBClassifier(
            objective="binary:logistic",
            tree_method="hist",
            device="cpu",
            random_state=seed,
            monotone_constraints=tuple(constraints),
            **params,
        )
        fit_started = time.perf_counter()
        model.fit(matrix, labels)
        fit_seconds = time.perf_counter() - fit_started
        scores = model.predict_proba(validation)[:, 1]
        pooled = float(average_precision_score(validation_labels, scores))
        spread = world_ap_spread(validation_worlds, validation_labels, scores)
        hold_rate = float((scores >= metadata_threshold()).mean())
        return {
            "validation_ap": pooled,
            "validation_p10": spread["p10"],
            "validation_world_ap": spread,
            "validation_hold_rate_at_v2_threshold": hold_rate,
            "fit_seconds": round(fit_seconds, 1),
        }

    def metadata_threshold() -> float:
        threshold = json.loads(
            (ROOT / "runs" / "full" / "payment_model_v2" / "test_score" / "result.json").read_text(encoding="utf-8")
        )["threshold"]
        return float(threshold)

    # A2: shuffle the reported target within each training world, seed 1717.
    rng = np.random.default_rng(1717)
    shuffled = train_labels.copy()
    world_to_indices: dict[str, list[int]] = {}
    for index, world in enumerate(train_worlds):
        world_to_indices.setdefault(world, []).append(index)
    for world in sorted(world_to_indices):
        idx = np.asarray(world_to_indices[world])
        shuffled[idx] = shuffled[idx][rng.permutation(idx.size)]
    print("A2 shuffled target fit", flush=True)
    a2 = fit_and_score(train_matrix, shuffled, constants, 1717, validation_matrix)
    a2.update(
        {
            "seed": 1717,
            "shuffled_positives": int(shuffled.sum()),
            "training_positives": int(train_labels.sum()),
            "validation_base_rate": base_rate,
            "ap_over_base_rate": a2["validation_ap"] / base_rate,
            "leak_finding": a2["validation_ap"] > 2.0 * base_rate,
            "device": "cpu",
            "minutes": round((time.perf_counter() - started) / 60.0, 1),
        }
    )
    write_result("A2_label_shuffle", a2)

    # A3a: C1 features without payee_risk (constraint dropped for the removed feature).
    keep = [index for index, name in enumerate(features) if name != PAYEE_RISK]
    print("A3a no-payee-risk fit", flush=True)
    a3a = fit_and_score(train_matrix[:, keep], train_labels, [0] * len(keep), 1717, validation_matrix[:, keep])
    a3a.update(
        {
            "seed": 1717,
            "features_removed": [PAYEE_RISK],
            "feature_count": len(keep),
            "device": "cpu",
            "minutes": round((time.perf_counter() - started) / 60.0, 1),
        }
    )
    write_result("A3a_without_payee_risk", a3a)

    # A3b: C1 features without the eight payment-level features.
    keep_b = [index for index, name in enumerate(features) if name not in set(payment_features)]
    print("A3b no-payment-level fit", flush=True)
    a3b = fit_and_score(train_matrix[:, keep_b], train_labels, [0] * len(keep_b), 1717, validation_matrix[:, keep_b])
    a3b.update(
        {
            "seed": 1717,
            "features_removed": payment_features,
            "feature_count": len(keep_b),
            "device": "cpu",
            "minutes": round((time.perf_counter() - started) / 60.0, 1),
        }
    )
    write_result("A3b_without_payment_level", a3b)

    for seed in (1718, 1719):
        print(f"A4 seed {seed} fit", flush=True)
        record = fit_and_score(train_matrix, train_labels, constants, seed, validation_matrix)
        record.update(
            {
                "seed": seed,
                "device": "cpu",
                "minutes": round((time.perf_counter() - started) / 60.0, 1),
            }
        )
        write_result(f"A4_seed_{seed}", record)

    write_result(
        "A2_A3_A4_summary",
        {
            "wall_seconds": round(time.perf_counter() - started, 1),
            "device": "cpu",
            "c1_params": params,
            "training_rows": int(train_matrix.shape[0]),
            "validation_rows": int(validation_matrix.shape[0]),
            "validation_base_rate": base_rate,
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
