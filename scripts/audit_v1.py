# File: scripts/audit_v1.py
# Purpose: Level 9 - audit of model v1. The Level 5 score (PR-AUC 0.99) looks too good, so this script trains
#          COPIES of the model with parts taken away, to see where the score really comes from:
#            - ablations: without the balance columns, without the type flags, without the generated columns;
#            - an out-of-time split (train on early steps, test on late steps);
#            - every one of the 15 features ALONE.
#          Every variant is one run in the new MLflow experiment "level9_v1_audit". Model v1 itself is never touched:
#          the frozen files are hashed before and after (core/frozen_artifacts.py), nothing is written under models/,
#          and nothing is logged to the level5_* or level6_* experiments.
#
# Run from the repo root:   python scripts/audit_v1.py [--device cpu|cuda]
import argparse
import subprocess
import sys
from pathlib import Path
from typing import Any

import orjson
import pandas as pd

SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))

from core.frozen_artifacts import (  # noqa: E402
    FROZEN_ARTIFACT_PATHS,
    guarded_write_bytes,
    md5_of_file,
    verify_frozen_artifacts,
)

# --- Configuration (everything adjustable lives here) ----------------------------------------------------------------
SEED = 42
OOT_CUTOFF_STEP = 372  # train on steps <= 372, test on steps > 372
EXPERIMENT_NAME = "level9_v1_audit"
FORBIDDEN_EXPERIMENT_PREFIXES = ("level5_", "level6_")  # the audit must never write into these
TRAIN_PATH = REPO_ROOT / "data" / "processed" / "train.parquet"
TEST_PATH = REPO_ROOT / "data" / "processed" / "test.parquet"
ENRICHED_PATH = REPO_ROOT / "data" / "enriched" / "paysim_enriched.parquet"
BEST_PARAMS_PATH = REPO_ROOT / "reports" / "model" / "best_params.json"
METRICS_PATH = REPO_ROOT / "reports" / "model" / "metrics.json"
RESULTS_PATH = REPO_ROOT / "reports" / "model" / "v1_audit_results.json"
CHART_PATH = REPO_ROOT / "reports" / "model" / "v1_audit_chart.png"

# The three groups of features, defined here explicitly. "amount" stays in every variant except where it is
# tested alone.
BALANCE = ["oldbalanceOrg", "newbalanceOrig", "oldbalanceDest", "newbalanceDest"]
TYPE_FLAGS = ["type_CASH_IN", "type_CASH_OUT", "type_DEBIT", "type_PAYMENT", "type_TRANSFER"]
GENERATED = ["new_device_flag", "geo_velocity_spike", "account_avg_amount", "many_small_inbound_count",
             "large_outbound_amount"]
ALL_FEATURES = ["amount"] + BALANCE + TYPE_FLAGS + GENERATED

RAW_COLUMNS_NEEDED = ["amount", "type", "isFraud"] + BALANCE + GENERATED
GROUPS = ("original", "control", "ablation", "out_of_time", "single_feature")


# --- Small pure helpers (no training, easy to test) ----------------------------------------------------------------
def check_feature_groups() -> None:
    """The groups must cover the 15 features exactly once; stop loudly if they do not."""
    assert len(ALL_FEATURES) == 15, f"expected 15 features, found {len(ALL_FEATURES)}"
    assert len(set(ALL_FEATURES)) == 15, "a feature appears in more than one group"


def drop_group(features: pd.DataFrame, names: list[str]) -> pd.DataFrame:
    """The feature table without the named columns."""
    return features.drop(columns=names)


def keep_only(features: pd.DataFrame, name: str) -> pd.DataFrame:
    """The feature table with just one column."""
    return features[[name]]


def build_variants() -> list[dict[str, Any]]:
    """The audit plan: name, group, split and the columns to drop (or the one column to keep)."""
    variants: list[dict[str, Any]] = [
        {"name": "control_all15", "group": "control", "split": "level5_random", "drop": []},
        {"name": "ablate_no_balance", "group": "ablation", "split": "level5_random", "drop": BALANCE},
        {"name": "ablate_no_type", "group": "ablation", "split": "level5_random", "drop": TYPE_FLAGS},
        {"name": "ablate_no_generated", "group": "ablation", "split": "level5_random", "drop": GENERATED},
        {"name": "oot_all15", "group": "out_of_time", "split": "out_of_time", "drop": []},
        {"name": "oot_no_balance", "group": "out_of_time", "split": "out_of_time", "drop": BALANCE},
    ]
    for feature in ALL_FEATURES:
        variants.append({"name": f"single_{feature}", "group": "single_feature", "split": "level5_random",
                         "keep": feature})
    return variants


def validate_row(row: dict[str, Any]) -> None:
    """Raise ValueError if a result row has no MLflow run id or is missing a field."""
    for field in ("name", "group", "features", "split", "pr_auc", "mlflow_run_id", "device", "n_train", "n_test"):
        if field not in row:
            raise ValueError(f"result row is missing '{field}'")
    if not row["mlflow_run_id"]:
        raise ValueError(f"result row '{row['name']}' has no MLflow run id")
    if row["group"] not in GROUPS:
        raise ValueError(f"result row '{row['name']}' has an unknown group '{row['group']}'")


def read_json(path: Path) -> Any:
    """Read a JSON file."""
    return orjson.loads(path.read_bytes())


def frozen_hashes_on_disk() -> dict[str, str]:
    """The md5 of every frozen artifact that exists on disk (a missing file is left out)."""
    present = [path for path in FROZEN_ARTIFACT_PATHS if path.exists()]
    return {path.relative_to(REPO_ROOT).as_posix(): md5_of_file(path) for path in present}


def git_text(*args: str) -> str:
    """The output of a git command, or 'unknown' if git is not available."""
    try:
        return subprocess.run(["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, timeout=30).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return "unknown"


# --- Data and training (heavy libraries are imported only here, so the helpers above stay light) ---------------
def level5_features(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Build the 15 Level 5 features with the SAME function Level 5 uses (prepare_features)."""
    from fraud_baseline import prepare_features

    features, target = prepare_features(raw)
    assert len(features.columns) == 15, "prepare_features no longer returns 15 columns"
    assert set(features.columns) == set(ALL_FEATURES), "the audit groups no longer match prepare_features"
    return features, target


def load_level5_split() -> tuple[tuple[pd.DataFrame, pd.Series], tuple[pd.DataFrame, pd.Series]]:
    """The Level 5 train and test splits as (features, target) pairs."""
    train = level5_features(pd.read_parquet(TRAIN_PATH, columns=RAW_COLUMNS_NEEDED))
    test = level5_features(pd.read_parquet(TEST_PATH, columns=RAW_COLUMNS_NEEDED))
    return train, test


def load_oot_split() -> tuple[tuple[pd.DataFrame, pd.Series], tuple[pd.DataFrame, pd.Series]]:
    """Out-of-time split from the full dataset: train on early steps, test on later ones."""
    raw = pd.read_parquet(ENRICHED_PATH, columns=RAW_COLUMNS_NEEDED + ["step"])
    early = raw[raw["step"] <= OOT_CUTOFF_STEP].drop(columns=["step"]).reset_index(drop=True)
    late = raw[raw["step"] > OOT_CUTOFF_STEP].drop(columns=["step"]).reset_index(drop=True)
    return level5_features(early), level5_features(late)


def train_and_score(X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series,
                    device: str) -> float:
    """
    Train XGBoost the way Level 5 did (tuned parameters, scale_pos_weight, seed 42, no early stopping) and return the
    PR-AUC on the test set (the same average_precision_score Level 5 uses).
    """
    import xgboost
    from sklearn.metrics import average_precision_score

    params = read_json(BEST_PARAMS_PATH)["xgboost"]
    positives = int(y_train.sum())
    scale_pos_weight = (len(y_train) - positives) / positives  # same formula as Level 5's compute_scale_pos_weight
    model = xgboost.XGBClassifier(tree_method="hist", device=device, scale_pos_weight=scale_pos_weight,
                                  eval_metric="aucpr", random_state=SEED, **params)
    model.fit(X_train, y_train)
    return float(average_precision_score(y_test, model.predict_proba(X_test)[:, 1]))


def select_columns(variant: dict[str, Any], X: pd.DataFrame) -> pd.DataFrame:
    """Apply a variant's choice (drop some columns, or keep one) to a feature table."""
    return keep_only(X, variant["keep"]) if "keep" in variant else drop_group(X, variant["drop"])


def run_variant(variant: dict[str, Any], splits: dict[str, Any], device: str) -> dict[str, Any]:
    """Train and score one variant inside its own MLflow run, and return its result row."""
    import mlflow

    (X_train, y_train), (X_test, y_test) = splits[variant["split"]]
    X_train_v, X_test_v = select_columns(variant, X_train), select_columns(variant, X_test)
    with mlflow.start_run(run_name=variant["name"]) as run:
        score = train_and_score(X_train_v, y_train, X_test_v, y_test, device)
        mlflow.log_params({"variant": variant["name"], "features": ",".join(X_train_v.columns),
                           "split": variant["split"], "device": device, "seed": SEED, "n_train": len(X_train_v),
                           "n_test": len(X_test_v), "n_features": X_train_v.shape[1]})
        mlflow.set_tags({"git_commit": git_text("rev-parse", "--short", "HEAD"),
                         "git_branch": git_text("branch", "--show-current"), "audit_of": "model_v1"})
        mlflow.log_metric("test_pr_auc", score)
    return {"name": variant["name"], "group": variant["group"], "features": list(X_train_v.columns),
            "split": variant["split"], "pr_auc": score, "mlflow_run_id": run.info.run_id, "device": device,
            "n_train": len(X_train_v), "n_test": len(X_test_v), "test_base_rate": float(y_test.mean())}


def generated_column_diagnostics(X_train: pd.DataFrame, y_train: pd.Series) -> dict[str, dict[str, float]]:
    """For each generated column: the share of fraud rows and of non-fraud rows where the value is not zero."""
    result = {}
    for column in GENERATED:
        non_zero = X_train[column] != 0
        result[column] = {"share_nonzero_fraud": float(non_zero[y_train == 1].mean()),
                          "share_nonzero_nonfraud": float(non_zero[y_train == 0].mean())}
    return result


def log_diagnostics(diagnostics: dict[str, dict[str, float]]) -> str:
    """Log the diagnostics as one extra MLflow run and return its run id."""
    import mlflow

    with mlflow.start_run(run_name="diagnostic_generated_columns") as run:
        mlflow.set_tags({"audit_of": "model_v1", "git_commit": git_text("rev-parse", "--short", "HEAD")})
        for column, shares in diagnostics.items():
            for key, value in shares.items():
                mlflow.log_metric(f"{key}__{column}", value)
    return run.info.run_id


def original_v1_row() -> dict[str, Any]:
    """The original Level 5 score, READ from reports/model/metrics.json (never typed in)."""
    metrics = read_json(METRICS_PATH)
    return {"name": "original_v1", "group": "original", "features": list(ALL_FEATURES), "split": "level5_random",
            "pr_auc": metrics["pr_auc"], "mlflow_run_id": metrics["mlflow_evaluation_run_id"],
            "device": "cuda (Level 5)", "n_train": None, "n_test": metrics["test_rows"],
            "test_base_rate": metrics["test_fraud_rows"] / metrics["test_rows"]}


def write_results(results: dict[str, Any]) -> None:
    """Save the results JSON (through the frozen guard, so it can never land on a protected file)."""
    guarded_write_bytes(RESULTS_PATH, orjson.dumps(results, option=orjson.OPT_INDENT_2 | orjson.OPT_SORT_KEYS) + b"\n")


def main(argv: list[str] | None = None) -> int:
    """Run the audit and write reports/model/v1_audit_results.json."""
    parser = argparse.ArgumentParser(description="Audit model v1: ablations, out-of-time split, single features.")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    args = parser.parse_args(argv)

    check_feature_groups()
    assert not EXPERIMENT_NAME.startswith(FORBIDDEN_EXPERIMENT_PREFIXES), "the audit must use its own experiment"
    hashes_before = frozen_hashes_on_disk()
    problems_before = verify_frozen_artifacts(allow_missing=True)
    assert not problems_before, f"frozen artifacts already differ from their .dvc files: {problems_before}"

    from core.mlflow_setup import setup_mlflow

    print(f"MLflow experiment '{EXPERIMENT_NAME}' at {setup_mlflow(EXPERIMENT_NAME)}")
    level5, oot = load_level5_split(), load_oot_split()
    splits = {"level5_random": level5, "out_of_time": oot}
    print(f"Level 5 split: train {len(level5[0][0]):,} / test {len(level5[1][0]):,} rows; "
          f"out-of-time: train {len(oot[0][0]):,} / test {len(oot[1][0]):,} rows")

    rows = [original_v1_row()]
    for variant in build_variants():
        row = run_variant(variant, splits, args.device)
        validate_row(row)
        rows.append(row)
        print(f"  {row['name']:36} PR-AUC {row['pr_auc']:.5f}  run {row['mlflow_run_id'][:8]}")

    diagnostics = generated_column_diagnostics(*level5[0])
    results = {"base_rate": float(level5[1][1].mean()), "rows": rows, "diagnostics": diagnostics,
               "diagnostics_run_id": log_diagnostics(diagnostics), "frozen_hashes_before": hashes_before,
               "frozen_hashes_after": frozen_hashes_on_disk(), "oot_cutoff_step": OOT_CUTOFF_STEP, "seed": SEED}
    problems_after = verify_frozen_artifacts(allow_missing=True)
    if problems_after or results["frozen_hashes_after"] != hashes_before:
        print(f"ERROR: frozen artifacts changed during the audit: {problems_after}")
        return 1
    write_results(results)
    print(f"Wrote {RESULTS_PATH.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
