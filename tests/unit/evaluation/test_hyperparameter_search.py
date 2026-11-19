"""
WHAT: Tests the Optuna search module, its truth average precision, the trainer defaults it relies
on, and the committed search settings.

WHY: The full-scale run chooses the account model on validation truth, so the metric, the seeded
study, the "off" path's unchanged defaults and the config ranges must all be proven on small data
before a single full-scale model is trained.
"""

from datetime import date
from pathlib import Path
from typing import Any

import numpy
import pytest
import yaml

from decisions.mule_account_model import (
    HyperparameterRange,
    load_mule_model_config,
    train_account_model,
)
from decisions.pu_learning import train_elkan_noto_model, train_pu_bagged_model
from evaluation.hyperparameter_search import (
    run_optuna_search,
    validation_truth_average_precision,
)
from evaluation.level17_scoring import ScoredDay

REPO_ROOT = Path(__file__).resolve().parents[3]
COMMITTED_CONFIG = REPO_ROOT / "config" / "mule_model.yaml"


def _scored(roles: list[str]) -> list[ScoredDay]:
    """Wraps one role per account-day into ScoredDay records with agreeing identifiers."""
    return [
        ScoredDay(
            bank_id="bank_0",
            account_id=f"acc_{index}",
            day=date(2026, 1, 1),
            rules_score=0,
            rules_fired=(),
            anomaly_score=None,
            role=role,
            depth=None,
        )
        for index, role in enumerate(roles)
    ]


def _training_set() -> tuple[list[list[float]], list[int]]:
    """Returns 40 small rows with four evenly spaced positives and a learnable signal."""
    rows: list[list[float]] = []
    targets: list[int] = []
    for index in range(40):
        value = index / 40.0
        rows.append([value, 1.0 - value, 2.0 * value - 1.0])
        targets.append(1 if index % 10 == 0 else 0)
    return rows, targets


def _truth_ap_space() -> dict[str, HyperparameterRange]:
    """Returns a tiny search space that keeps a three-trial study fast."""
    return {
        "max_depth": HyperparameterRange(minimum=2.0, maximum=4.0, log=False, integer=True),
        "n_estimators": HyperparameterRange(minimum=3.0, maximum=8.0, log=False, integer=True),
        "learning_rate": HyperparameterRange(minimum=0.05, maximum=0.5, log=True, integer=False),
        "subsample": HyperparameterRange(minimum=0.5, maximum=1.0, log=False, integer=False),
    }


def test_truth_average_precision_hand_example() -> None:
    """Truth AP averages precision at each mule day: one hit first and one hit third of two."""
    scored = _scored(["mule", "genuine", "cash_out", "victim"])

    ap = validation_truth_average_precision([0.9, 0.8, 0.7, 0.6], scored)

    assert ap == pytest.approx((1.0 + 2.0 / 3.0) / 2.0, rel=1e-12)


def test_truth_average_precision_refuses_misaligned_input() -> None:
    """A probability list that does not match the scored days raises instead of truncating."""
    scored = _scored(["mule", "genuine"])

    with pytest.raises(ValueError, match="does not match"):
        validation_truth_average_precision([0.5], scored)


def test_three_trial_study_is_deterministic_with_a_fixed_seed() -> None:
    """Two runs of the same seeded three-trial study return identical trials and best parameters."""
    rows, targets = _training_set()
    scored = _scored(["mule" if target == 1 else "genuine" for target in targets])

    def train_trial(params: dict[str, int | float]) -> list[float]:
        """Trains one tiny candidate and returns its scores on the training rows."""
        model, _ = train_account_model(
            rows,
            targets,
            17,
            int(params["n_estimators"]),
            max_depth=int(params["max_depth"]),
            learning_rate=float(params["learning_rate"]),
            subsample=float(params["subsample"]),
        )
        day_rows = numpy.asarray(rows, dtype=float)
        return [float(value) for value in model.predict_proba(day_rows)[:, 1]]

    first = run_optuna_search(train_trial, scored, _truth_ap_space(), 3, 11)
    second = run_optuna_search(train_trial, scored, _truth_ap_space(), 3, 11)

    assert len(first.trials) == 3
    assert first.trials == second.trials
    assert first.best_params == second.best_params
    assert first.best_score == second.best_score
    assert first.best_score == max(record["score"] for record in first.trials)
    assert 2 <= first.best_params["max_depth"] <= 4
    assert 3 <= first.best_params["n_estimators"] <= 8
    assert 0.05 <= first.best_params["learning_rate"] <= 0.5
    assert 0.5 <= first.best_params["subsample"] <= 1.0


def test_default_hyperparameters_reproduce_the_untuned_model() -> None:
    """train_account_model's new keyword defaults fit exactly the model the old call fitted."""
    rows, targets = _training_set()

    default_model, default_weight = train_account_model(rows, targets, 5, 30)
    explicit_model, explicit_weight = train_account_model(
        rows,
        targets,
        5,
        30,
        max_depth=6,
        learning_rate=0.3,
        min_child_weight=1.0,
        subsample=1.0,
        colsample_bytree=1.0,
    )

    default_probabilities = default_model.predict_proba(numpy.asarray(rows, dtype=float))[:, 1]
    explicit_probabilities = explicit_model.predict_proba(numpy.asarray(rows, dtype=float))[:, 1]
    assert default_weight == explicit_weight
    assert (default_probabilities == explicit_probabilities).all()


def test_pu_trainers_default_to_the_same_models() -> None:
    """Both PU trainers' new keyword defaults fit exactly the models the old calls fitted."""
    rows, targets = _training_set()

    bagged_default = train_pu_bagged_model(rows, targets, 3, 4, 2, 1)
    bagged_explicit = train_pu_bagged_model(
        rows,
        targets,
        3,
        4,
        2,
        1,
        max_depth=6,
        learning_rate=0.3,
        min_child_weight=1.0,
        subsample=1.0,
        colsample_bytree=1.0,
    )
    assert bagged_default.predict_probabilities(rows) == bagged_explicit.predict_probabilities(rows)

    elkan_default, default_c, default_weights, default_clips = train_elkan_noto_model(
        rows, targets, rows, targets, 3, 4, 0.999999
    )
    elkan_explicit, explicit_c, explicit_weights, explicit_clips = train_elkan_noto_model(
        rows,
        targets,
        rows,
        targets,
        3,
        4,
        0.999999,
        max_depth=6,
        learning_rate=0.3,
        min_child_weight=1.0,
        subsample=1.0,
        colsample_bytree=1.0,
    )
    assert default_c == explicit_c
    assert default_weights == explicit_weights
    assert default_clips == explicit_clips
    assert (elkan_default.predict_proba(rows) == elkan_explicit.predict_proba(rows)).all()


def test_committed_config_parses_the_search_settings() -> None:
    """The committed config turns the search off and carries the six architect ranges."""
    config = load_mule_model_config(COMMITTED_CONFIG)

    assert config.hyperparameter_search == "off"
    assert config.optuna_trials == 40
    assert config.optuna_seed == 1717
    assert config.hyperparameter_space is not None
    assert list(config.hyperparameter_space) == [
        "max_depth",
        "learning_rate",
        "n_estimators",
        "min_child_weight",
        "subsample",
        "colsample_bytree",
    ]
    integer_names = {name for name, bounds in config.hyperparameter_space.items() if bounds.integer}
    log_names = {name for name, bounds in config.hyperparameter_space.items() if bounds.log}
    assert integer_names == {"max_depth", "n_estimators"}
    assert log_names == {"learning_rate", "min_child_weight"}


def _write_variant(directory: Path, mutate: Any) -> Path:
    """Writes a copy of the committed config after mutate(parameters) edits it in place."""
    raw = yaml.safe_load(COMMITTED_CONFIG.read_text(encoding="utf-8"))
    mutate(raw["parameters"])
    path = directory / "mule_model_variant.yaml"
    path.write_text(yaml.safe_dump(raw, sort_keys=False), encoding="utf-8")
    return path


def test_unknown_search_setting_is_refused(tmp_path: Path) -> None:
    """A search setting other than off or optuna raises instead of silently running a grid."""

    def mutate(parameters: dict[str, Any]) -> None:
        """Sets an unsupported search method so the loader must refuse it."""
        parameters["hyperparameter_search"]["value"] = "grid"

    path = _write_variant(tmp_path, mutate)
    with pytest.raises(ValueError, match="hyperparameter_search"):
        load_mule_model_config(path)


def test_wrong_log_flag_is_refused(tmp_path: Path) -> None:
    """A space entry that disagrees with the architect's log flags raises naming the entry."""

    def mutate(parameters: dict[str, Any]) -> None:
        """Turns on the log flag of an integer parameter so the range must be refused."""
        parameters["hyperparameter_space"]["value"]["max_depth"]["log"] = True

    path = _write_variant(tmp_path, mutate)
    with pytest.raises(ValueError, match="max_depth"):
        load_mule_model_config(path)


def test_missing_search_range_is_refused(tmp_path: Path) -> None:
    """A space that omits one of the six searched parameters raises instead of defaulting it."""

    def mutate(parameters: dict[str, Any]) -> None:
        """Removes one searched parameter so the space must be refused."""
        del parameters["hyperparameter_space"]["value"]["min_child_weight"]

    path = _write_variant(tmp_path, mutate)
    with pytest.raises(ValueError, match="hyperparameter_space"):
        load_mule_model_config(path)
