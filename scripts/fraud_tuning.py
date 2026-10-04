# File: scripts/fraud_tuning.py
# Purpose: Optuna hyperparameter tuning (XGBoost, CatBoost, RandomForest) scored by stratified CV PR-AUC on a
#          25% sample of the TRAIN split only; compares tuned vs baseline settings on identical folds.
#
# Important: tuning only PICKS hyperparameters. The final model is retrained on the full train data later, and
# no score produced here (from the subsample) is ever reported as a model's final result.
import argparse
import sys
import time
import warnings
from pathlib import Path

import catboost
import matplotlib
import matplotlib.pyplot as plt
import mlflow
import optuna
import optuna.visualization.matplotlib as optuna_plots
import pandas as pd
import xgboost
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold, train_test_split

# Draw plots straight to files; no window is needed.
matplotlib.use("Agg")

# Make the repo root and this scripts/ folder importable, so the file works when run directly or loaded by tests.
SCRIPTS_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(SCRIPTS_DIR))

from fraud_baseline import (  # noqa: E402
    AMOUNT_AND_BALANCE_COLUMNS,
    ENGINEERED_BOOL_COLUMNS,
    ENGINEERED_NUMBER_COLUMNS,
    TARGET_COLUMN,
    compute_scale_pos_weight,
    prepare_features,
    read_dvc_data_hash,
)

from core.mlflow_setup import setup_mlflow  # noqa: E402

# XGBoost is asked to predict with CPU data while the model sits on the GPU; it handles this itself and warns.
# The warning is harmless and would drown out our own output, so it is hidden.
warnings.filterwarnings("ignore", message=".*mismatched devices.*")

# Only the TRAIN split is ever read by this script. (The other two splits are deliberately not referenced.)
TRAIN_PATH = REPO_ROOT / "data" / "processed" / "train.parquet"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "reports" / "model"

EXPERIMENT_NAME = "level5_fraud_tuning"
RANDOM_SEED = 42
SAMPLE_FRACTION = 0.25

# Folds per model: five for the quick models, three for the slow random forest.
N_FOLDS = {"xgboost": 5, "catboost": 5, "random_forest": 3}

# Trials per model for the full study (a smoke test overrides this with a tiny number).
DEFAULT_TRIALS = {"xgboost": 40, "catboost": 30, "random_forest": 15}

# The baseline models, written out with EVERY searched hyperparameter given explicitly. These are the library
# defaults the bake-off used (checked against the fitted models: XGBoost depth 6 / learning rate 0.3 / min child
# weight 1 / subsample 1 / lambda 1; CatBoost depth 6 / l2 3; RandomForest depth None / leaf 1 / sqrt features),
# plus the tree counts the bake-off set (300, 300, 100). CatBoost picks its learning rate from the data size, so
# that one value is read from a real baseline fit (see read_catboost_default_learning_rate).
BASELINE_TRIAL_PARAMS = {
    "xgboost": {
        "max_depth": 6, "learning_rate": 0.3, "n_estimators": 300,
        "min_child_weight": 1, "subsample": 1.0, "reg_lambda": 1.0,
    },
    "catboost": {"iterations": 300, "depth": 6, "l2_leaf_reg": 3.0},
    "random_forest": {"n_estimators": 100, "max_depth": None, "min_samples_leaf": 1, "max_features": "sqrt"},
}


# --- Data ------------------------------------------------------------------------------------------------
def load_train_features() -> tuple[pd.DataFrame, pd.Series]:
    """Read ONLY the train split, and only the columns the models need, then build the 15-feature matrix."""
    # Reading just the needed columns keeps memory low (the string columns such as names are never loaded).
    needed_columns = (
        AMOUNT_AND_BALANCE_COLUMNS + ENGINEERED_BOOL_COLUMNS + ENGINEERED_NUMBER_COLUMNS + ["type", TARGET_COLUMN]
    )
    raw = pd.read_parquet(TRAIN_PATH, columns=needed_columns)
    return prepare_features(raw)


def make_tuning_sample(
    X: pd.DataFrame, y: pd.Series, fraction: float, seed: int
) -> tuple[pd.DataFrame, pd.Series]:
    """Draw a stratified sample, so the fraud ratio (0.5%) is preserved, and renumber rows 0..n-1."""
    X_sample, _, y_sample, _ = train_test_split(X, y, train_size=fraction, stratify=y, random_state=seed)
    return X_sample.reset_index(drop=True), y_sample.reset_index(drop=True)


def make_folds(y: pd.Series, n_folds: int, seed: int) -> list[tuple]:
    """Precompute the CV fold indices ONCE, so baseline and tuned settings are scored on identical folds."""
    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    return list(splitter.split(y, y))


# --- Models ----------------------------------------------------------------------------------------------
def build_xgboost_model(params: dict, scale_pos_weight: float) -> xgboost.XGBClassifier:
    """XGBoost with the fixed GPU and imbalance settings from the baseline, plus the searched hyperparameters."""
    return xgboost.XGBClassifier(
        tree_method="hist",
        device="cuda",
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=RANDOM_SEED,
        **params,
    )


def build_catboost_model(params: dict) -> catboost.CatBoostClassifier:
    """CatBoost with the fixed GPU and imbalance settings from the baseline, plus the searched hyperparameters."""
    return catboost.CatBoostClassifier(
        task_type="GPU",
        devices="0",
        auto_class_weights="Balanced",
        random_seed=RANDOM_SEED,
        verbose=0,
        allow_writing_files=False,
        **params,
    )


def build_random_forest_model(params: dict) -> RandomForestClassifier:
    """RandomForest (CPU) with the fixed imbalance setting from the baseline, plus the searched hyperparameters."""
    return RandomForestClassifier(class_weight="balanced", n_jobs=-1, random_state=RANDOM_SEED, **params)


def build_model(name: str, params: dict, scale_pos_weight: float):
    """Build the named model with the given hyperparameters."""
    if name == "xgboost":
        return build_xgboost_model(params, scale_pos_weight)
    if name == "catboost":
        return build_catboost_model(params)
    return build_random_forest_model(params)


# --- Search spaces ---------------------------------------------------------------------------------------
# Kept modest on purpose: the baselines already score 0.98-0.99, so there is little headroom to search.
def suggest_xgboost_params(trial: optuna.Trial) -> dict:
    """Search space for XGBoost: tree depth, learning rate, number of trees, and a few regularization knobs."""
    return {
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.3, log=True),
        "n_estimators": trial.suggest_int("n_estimators", 100, 500, step=50),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.01, 10.0, log=True),
    }


def suggest_catboost_params(trial: optuna.Trial) -> dict:
    """Search space for CatBoost: number of trees, depth, learning rate and L2 regularization."""
    return {
        "iterations": trial.suggest_int("iterations", 200, 600, step=100),
        "depth": trial.suggest_int("depth", 4, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.3, log=True),
        "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0, log=True),
    }


def suggest_random_forest_params(trial: optuna.Trial) -> dict:
    """Search space for RandomForest: number of trees, depth limit, leaf size and features tried per split."""
    return {
        "n_estimators": trial.suggest_int("n_estimators", 100, 200, step=50),
        "max_depth": trial.suggest_categorical("max_depth", [10, 20, 30, None]),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 5),
        "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5]),
    }


SEARCH_SPACES = {
    "xgboost": suggest_xgboost_params,
    "catboost": suggest_catboost_params,
    "random_forest": suggest_random_forest_params,
}


def read_catboost_default_learning_rate(X: pd.DataFrame, y: pd.Series, folds: list[tuple]) -> float:
    """Fit baseline CatBoost on one fold's training part and read the learning rate CatBoost chose by itself."""
    train_index, _ = folds[0]
    model = build_catboost_model(BASELINE_TRIAL_PARAMS["catboost"])
    model.fit(X.iloc[train_index], y.iloc[train_index])
    return float(model.get_all_params()["learning_rate"])


def get_baseline_trial_params(name: str, X: pd.DataFrame, y: pd.Series, folds: list[tuple]) -> dict:
    """Return the baseline hyperparameters as an Optuna trial (every searched value spelled out)."""
    params = dict(BASELINE_TRIAL_PARAMS[name])
    if name == "catboost":
        params["learning_rate"] = read_catboost_default_learning_rate(X, y, folds)
    return params


# --- Scoring ---------------------------------------------------------------------------------------------
def cv_pr_auc(
    name: str, params: dict, X: pd.DataFrame, y: pd.Series, folds: list[tuple], scale_pos_weight: float
) -> list[float]:
    """Train on each fold's training part, score PR-AUC on its validation part, and return the per-fold scores."""
    scores = []
    for train_index, valid_index in folds:
        model = build_model(name, params, scale_pos_weight)
        model.fit(X.iloc[train_index], y.iloc[train_index])
        fraud_probability = model.predict_proba(X.iloc[valid_index])[:, 1]
        # average_precision_score is the standard single-number summary of the precision-recall curve (PR-AUC).
        scores.append(average_precision_score(y.iloc[valid_index], fraud_probability))
    return scores


def make_objective(name: str, X, y, folds, scale_pos_weight: float, shared_params: dict):
    """Create the Optuna objective for one model: one trial = one CV score, logged as a nested MLflow run."""

    def objective(trial: optuna.Trial) -> float:
        params = SEARCH_SPACES[name](trial)
        start = time.perf_counter()
        with mlflow.start_run(run_name=f"{name}_trial_{trial.number}", nested=True):
            fold_scores = cv_pr_auc(name, params, X, y, folds, scale_pos_weight)
            mean_score = sum(fold_scores) / len(fold_scores)

            mlflow.log_params(shared_params)
            # Trial 0 is the baseline settings (queued before the search starts); label it so it is easy to find.
            mlflow.set_tag("baseline_trial", "yes" if trial.number == 0 else "no")
            # MLflow params are text, so values such as None are written as the word "None".
            mlflow.log_params({key: str(value) for key, value in params.items()})
            mlflow.log_metric("mean_cv_pr_auc", mean_score)
            mlflow.log_metric("trial_seconds", time.perf_counter() - start)
        seconds = time.perf_counter() - start
        print(f"    trial {trial.number}: mean CV PR-AUC {mean_score:.4f} ({seconds:.1f}s) {params}")
        return mean_score

    return objective


# --- One model's study -----------------------------------------------------------------------------------
def save_history_plot(study: optuna.Study, name: str, output_dir: Path) -> Path:
    """Save Optuna's optimization-history plot (matplotlib version, so no extra image libraries are needed)."""
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_path = output_dir / f"fraud_tuning_{name}_optimization_history.png"
    axes = optuna_plots.plot_optimization_history(study)
    axes.set_title(f"{name}: Optuna optimization history (mean CV PR-AUC)")
    axes.figure.savefig(plot_path, dpi=120, bbox_inches="tight")
    plt.close(axes.figure)
    return plot_path


def tune_one_model(name: str, X, y, n_trials: int, output_dir: Path, shared_params: dict) -> dict:
    """Score the baseline, run the Optuna study, apply the acceptance gate, and log everything to MLflow."""
    n_folds = N_FOLDS[name]
    folds = make_folds(y, n_folds, RANDOM_SEED)
    scale_pos_weight = compute_scale_pos_weight(y)
    run_params = {**shared_params, "n_folds": n_folds, "n_trials": n_trials, "model": name}
    model_start = time.perf_counter()

    print(f"\n=== {name}: {n_folds}-fold CV on {len(X):,} rows, {n_trials} trials ===")
    with mlflow.start_run(run_name=f"tune_{name}"):
        mlflow.log_params(run_params)

        # The study is repeatable (fixed seed). TPE is Optuna's default Bayesian sampler.
        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=RANDOM_SEED))

        # Queue the baseline settings as the VERY FIRST trial. This puts the baseline inside the same search, on the
        # same folds, so a tuned win is a genuine win, not a luck-of-many-tries artifact: every other trial is
        # compared with a baseline that was scored in exactly the same way.
        baseline_params = get_baseline_trial_params(name, X, y, folds)
        study.enqueue_trial(baseline_params)

        objective = make_objective(name, X, y, folds, scale_pos_weight, run_params)
        study.optimize(objective, n_trials=n_trials)

        baseline_mean = study.trials[0].value
        tuned_mean = study.best_value
        improved = study.best_trial.number != 0 and tuned_mean > baseline_mean
        # Gate: the best trial is at least as good as the baseline trial. Because the baseline is itself a trial,
        # this can never fail by construction; "improved" (strictly better than baseline) is the informative number.
        gate_passed = tuned_mean >= baseline_mean
        plot_path = save_history_plot(study, name, output_dir)

        mlflow.log_metric("baseline_mean_cv_pr_auc", baseline_mean)
        mlflow.log_metric("tuned_mean_cv_pr_auc", tuned_mean)
        mlflow.log_metric("improvement_over_baseline", tuned_mean - baseline_mean)
        mlflow.log_param("best_trial_number", study.best_trial.number)
        mlflow.log_param("improved_over_baseline", "yes" if improved else "no")
        mlflow.log_param("acceptance_gate", "PASS" if gate_passed else "FAIL")
        mlflow.log_params({f"best_{key}": str(value) for key, value in study.best_params.items()})
        mlflow.log_artifact(str(plot_path))

    print(f"  baseline (trial 0) mean CV PR-AUC {baseline_mean:.4f}")
    print(f"  best tuned mean CV PR-AUC {tuned_mean:.4f} (trial {study.best_trial.number})  params {study.best_params}")
    gate_text = "PASS" if gate_passed else "FAIL"
    print(f"  gate (best >= baseline trial, same folds): {gate_text}; strictly improved: {improved}")
    return {
        "name": name,
        "baseline": baseline_mean,
        "tuned": tuned_mean,
        "best_params": study.best_params,
        "gate_passed": gate_passed,
        "improved": improved,
        "seconds": time.perf_counter() - model_start,
        "plot": plot_path,
    }


def parse_arguments() -> argparse.Namespace:
    """Read the command-line options (all optional; the defaults run the full study)."""
    parser = argparse.ArgumentParser(description="Optuna tuning for the fraud models (train split only).")
    parser.add_argument("--models", nargs="+", default=list(N_FOLDS), choices=list(N_FOLDS))
    parser.add_argument("--trials", type=int, default=None, help="trials per model (overrides the defaults)")
    parser.add_argument("--sample-fraction", type=float, default=SAMPLE_FRACTION)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--smoke", action="store_true", help="label the MLflow runs as a smoke test")
    return parser.parse_args()


def main() -> int:
    """Load the train split, take the stratified sample, tune each requested model, and print a summary."""
    args = parse_arguments()
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    tracking_uri = setup_mlflow(EXPERIMENT_NAME)
    print(f"MLflow experiment '{EXPERIMENT_NAME}' at {tracking_uri}")

    data_hash = read_dvc_data_hash()
    X_full, y_full = load_train_features()
    X, y = make_tuning_sample(X_full, y_full, args.sample_fraction, RANDOM_SEED)
    del X_full, y_full  # free the big table; only the sample is needed from here on
    print(f"Tuning sample: {len(X):,} rows, fraud ratio {y.mean() * 100:.4f}% (full train ratio is 0.5000%)")

    shared_params = {
        "seed": RANDOM_SEED,
        "dvc_data_hash": data_hash,
        "subsample_rows": len(X),
        "subsample_fraction": args.sample_fraction,
        "smoke_test": "yes" if args.smoke else "no",
    }

    results = []
    for name in args.models:
        n_trials = args.trials if args.trials is not None else DEFAULT_TRIALS[name]
        results.append(tune_one_model(name, X, y, n_trials, args.output_dir, shared_params))

    print("\nSummary (CV on the tuning subsample; NOT final test scores):")
    for result in results:
        status = "PASS" if result["gate_passed"] else "FAIL"
        print(
            f"  {result['name']:<14} baseline {result['baseline']:.4f}  tuned {result['tuned']:.4f}  gate {status}"
            f"  improved {result['improved']}  wall-clock {result['seconds'] / 60:.1f} min"
        )
    return 0 if all(result["gate_passed"] for result in results) else 1


if __name__ == "__main__":
    sys.exit(main())
