"""
WHAT: Tests the Optuna search module, its truth average precision, the trainer defaults it relies
on, and the committed search settings.

WHY: The full-scale run chooses the account model on validation truth, so the metric, the seeded
study, the "off" path's unchanged defaults and the config ranges must all be proven on small data
before a single full-scale model is trained.
"""

import dataclasses
from datetime import date
from pathlib import Path
from typing import Any

import numpy
import orjson
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
            world_id="world_a",
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

    first = run_optuna_search(
        train_trial,
        scored,
        _truth_ap_space(),
        3,
        11,
        min_trials_before_stop=3,
        stall_patience_trials=1,
        min_improvement=0.001,
    )
    second = run_optuna_search(
        train_trial,
        scored,
        _truth_ap_space(),
        3,
        11,
        min_trials_before_stop=3,
        stall_patience_trials=1,
        min_improvement=0.001,
    )

    assert len(first.trials) == 3
    assert first.trials == second.trials
    assert first.best_params == second.best_params
    assert first.best_score == second.best_score
    assert first.best_score == max(record["score"] for record in first.trials)
    assert 2 <= first.best_params["max_depth"] <= 4
    assert 3 <= first.best_params["n_estimators"] <= 8
    assert 0.05 <= first.best_params["learning_rate"] <= 0.5
    assert 0.5 <= first.best_params["subsample"] <= 1.0


def test_constant_score_stops_at_the_minimum_trial_count() -> None:
    """A flat score trips the stall rule at exactly max(min_trials, patience + 1) trials."""
    scored = _scored(["mule", "genuine"])

    def train_trial(params: dict[str, int | float]) -> list[float]:
        """Returns the same probabilities every trial, so truth AP never changes."""
        return [0.9, 0.1]

    result = run_optuna_search(
        train_trial,
        scored,
        _truth_ap_space(),
        40,
        7,
        min_trials_before_stop=20,
        stall_patience_trials=10,
        min_improvement=0.001,
    )

    assert result.trials_run == max(20, 10 + 1)
    assert result.trials_run == 20
    assert result.stopped_early is True
    assert "stalled" in result.stop_reason
    assert len(result.trials) == 20


def test_strictly_improving_score_runs_every_trial() -> None:
    """A score that rises in every patience window keeps the study running to the last trial."""
    scored = _scored(["mule"] + ["genuine"] * 11)
    call_count = 0

    def train_trial(params: dict[str, int | float]) -> list[float]:
        """Lifts the positive day one rank per call, so truth AP strictly improves each time."""
        nonlocal call_count
        call_count += 1
        negatives_above = 12 - call_count
        return [0.5] + [0.9 if index < negatives_above else 0.1 for index in range(11)]

    result = run_optuna_search(
        train_trial,
        scored,
        _truth_ap_space(),
        12,
        7,
        min_trials_before_stop=3,
        stall_patience_trials=2,
        min_improvement=0.001,
    )

    assert result.trials_run == 12
    assert result.stopped_early is False
    assert result.stop_reason == "completed all 12 trials"
    assert len(result.trials) == 12


def test_minimum_trials_beyond_the_trial_count_is_refused() -> None:
    """A minimum the run can never reach raises before any trial trains."""
    scored = _scored(["mule", "genuine"])

    def train_trial(params: dict[str, int | float]) -> list[float]:
        """Fails if called, proving the refusal happens before training starts."""
        raise AssertionError("train_trial must not run when the stop settings are invalid")

    with pytest.raises(ValueError, match="must not exceed trial_count"):
        run_optuna_search(
            train_trial,
            scored,
            _truth_ap_space(),
            5,
            7,
            min_trials_before_stop=6,
            stall_patience_trials=2,
            min_improvement=0.001,
        )


def test_identical_scores_record_importance_as_not_run() -> None:
    """A stalled study whose every trial scored the same reports why importance was not run."""
    scored = _scored(["mule", "genuine"])

    def train_trial(params: dict[str, int | float]) -> list[float]:
        """Returns the same probabilities every trial, so truth AP never changes."""
        return [0.9, 0.1]

    result = run_optuna_search(
        train_trial,
        scored,
        _truth_ap_space(),
        40,
        7,
        min_trials_before_stop=20,
        stall_patience_trials=10,
        min_improvement=0.001,
    )

    assert result.stopped_early is True
    assert result.param_importances == (
        "not run: every completed trial scored identically (1.000000), "
        "so no setting can explain any difference"
    )


def test_one_parameter_drives_the_importances() -> None:
    """When only max_depth moves the truth AP, fANOVA ranks max_depth as the most important."""
    scored = _scored(["mule"] + ["genuine"] * 11)

    def train_trial(params: dict[str, int | float]) -> list[float]:
        """Returns probabilities whose truth AP rises with max_depth alone."""
        positive = 0.1 * int(params["max_depth"])
        return [positive] + [0.05 * rank for rank in range(1, 12)]

    result = run_optuna_search(
        train_trial,
        scored,
        _truth_ap_space(),
        30,
        7,
        min_trials_before_stop=30,
        stall_patience_trials=1,
        min_improvement=0.001,
    )

    importances = result.param_importances
    assert isinstance(importances, dict)
    assert max(importances, key=importances.__getitem__) == "max_depth"


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


def test_regularisation_defaults_reproduce_the_untuned_model() -> None:
    """The new regularisation defaults fit exactly the model an old call fitted on random data."""
    generator = numpy.random.default_rng(20261119)
    rows = generator.normal(size=(30, 4))
    targets = [1 if index % 6 == 0 else 0 for index in range(30)]

    default_model, default_weight = train_account_model(rows, targets, 23, n_estimators=12)
    explicit_model, explicit_weight = train_account_model(
        rows,
        targets,
        23,
        n_estimators=12,
        gamma=0.0,
        reg_alpha=0.0,
        reg_lambda=1.0,
    )

    assert default_weight == explicit_weight
    assert (default_model.predict_proba(rows) == explicit_model.predict_proba(rows)).all()


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
    """The committed config turns the search off and carries the nine architect ranges."""
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
        "gamma",
        "reg_alpha",
        "reg_lambda",
    ]
    integer_names = {name for name, bounds in config.hyperparameter_space.items() if bounds.integer}
    log_names = {name for name, bounds in config.hyperparameter_space.items() if bounds.log}
    assert integer_names == {"max_depth", "n_estimators"}
    assert log_names == {"learning_rate", "min_child_weight", "reg_alpha", "reg_lambda"}


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


def test_param_importances_are_plain_floats_that_orjson_can_write() -> None:
    """The real fANOVA importances come back as plain floats, so the stage file writer accepts them."""
    scored = _scored(["mule", "genuine", "genuine", "genuine", "genuine", "genuine", "mule", "genuine"])
    space = {
        "first_mule_score": HyperparameterRange(minimum=0.0, maximum=1.0, log=False, integer=False),
        "second_mule_score": HyperparameterRange(minimum=0.0, maximum=1.0, log=False, integer=False),
    }

    def train_trial(params: dict[str, int | float]) -> list[float]:
        """Scores the two mule days with the sampled values and every other day at 0.5."""
        probabilities = [0.5] * len(scored)
        probabilities[0] = float(params["first_mule_score"])
        probabilities[6] = float(params["second_mule_score"])
        return probabilities

    result = run_optuna_search(
        train_trial,
        scored,
        space,
        4,
        3,
        min_trials_before_stop=4,
        stall_patience_trials=1,
        min_improvement=0.5,
    )

    assert isinstance(result.param_importances, dict)
    assert set(result.param_importances) == set(space)
    for importance in result.param_importances.values():
        assert type(importance) is float
    # The same options the stage writer uses; a numpy.float64 value would raise TypeError here.
    orjson.dumps(dataclasses.asdict(result), option=orjson.OPT_SORT_KEYS | orjson.OPT_INDENT_2)
