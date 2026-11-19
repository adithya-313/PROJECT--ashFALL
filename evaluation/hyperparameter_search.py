"""
WHAT: Optuna hyperparameter search for the Level 18 account model, scored on validation truth
average precision.

WHY: The pilot's estimator loop varies only n_estimators and scores against the bank-visible
labels. The full-scale run must choose the XGBoost parameters on the hidden truth instead, so the
model that ships is the one that finds mules rather than the one that best reproduces the labels it
was trained on.
"""

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import optuna

from decisions.mule_account_model import HyperparameterRange
from evaluation.level17_scoring import MULE_ROLES, ScoredDay
from evaluation.trivial_world import average_precision

__all__ = ["OptunaSearchResult", "run_optuna_search", "validation_truth_average_precision"]


@dataclass(frozen=True)
class OptunaSearchResult:
    """
    WHAT: The outcome of one study: the best sampled parameters, their score and every trial record.

    Attributes:
        best_params: Parameter name -> sampled value of the highest-scoring trial.
        best_score: Validation truth average precision of that trial.
        trials: One record per completed trial, in trial order; each holds "number", "params" and
            "score" so the run can write every candidate it tried.
        trials_run: Completed trials, which is below the requested count when the stall rule fired.
        stopped_early: Whether the stall rule stopped the study before every requested trial.
        stop_reason: Why the study ended: the stall rule's numbers, or that all trials ran.
    """

    best_params: dict[str, int | float]
    best_score: float
    trials: tuple[dict[str, Any], ...]
    trials_run: int
    stopped_early: bool
    stop_reason: str


def validation_truth_average_precision(
    scores: Sequence[float],
    scored: Sequence[ScoredDay],
) -> float:
    """
    WHAT: Average precision of model scores against the hidden truth, where a positive day is any
    role in MULE_ROLES (mule, collector or cash-out).

    WHY: Selection on the bank-visible labels rewards a model for reproducing the label it was
    trained on. The full-scale choice must instead be made on whether the account-day is truly
    illicit, and it must use the same mule definition the pilot's recall metric uses.

    Args:
        scores: One model probability per account-day, in scored order.
        scored: Account-days the probabilities belong to, in the same order.

    Returns:
        Average precision with labels 1 for a MULE_ROLES day and 0 otherwise.

    Raises:
        ValueError: If the two sequences differ in length, or average_precision refuses the labels
            because the days hold no positive or no negative.
    """
    if len(scores) != len(scored):
        raise ValueError(f"scores length {len(scores)} does not match scored length {len(scored)}")
    labels = [1 if day.role in MULE_ROLES else 0 for day in scored]
    return average_precision(scores, labels)


def _suggest_parameters(
    trial: optuna.Trial,
    space: Mapping[str, HyperparameterRange],
    categorical_choices: Mapping[str, Sequence[int | float]] | None,
) -> dict[str, int | float]:
    """
    WHAT: Draws one value per numeric range, plus one configured category when supplied.

    WHY: Every bound comes from the validated config, so the sampler never invents a domain; the
    category exists because a PU method such as bagging also chooses how many unlabelled rows each
    bag draws, and that choice belongs to the same trial.

    Args:
        trial: Optuna trial supplying the sampler.
        space: Parameter name -> validated range.
        categorical_choices: Optional parameter name -> allowed values sampled as a category.

    Returns:
        Parameter name -> sampled value, numeric ranges first.
    """
    params: dict[str, int | float] = {}
    for name, bounds in space.items():
        if bounds.integer:
            params[name] = trial.suggest_int(
                name, int(bounds.minimum), int(bounds.maximum), log=bounds.log
            )
        else:
            params[name] = trial.suggest_float(
                name, bounds.minimum, bounds.maximum, log=bounds.log
            )
    if categorical_choices is not None:
        for name, values in categorical_choices.items():
            params[name] = int(trial.suggest_categorical(name, list(values)))
    return params


def run_optuna_search(
    train_trial: Callable[[dict[str, int | float]], Sequence[float]],
    scored: Sequence[ScoredDay],
    space: Mapping[str, HyperparameterRange],
    trial_count: int,
    seed: int,
    *,
    min_trials_before_stop: int,
    stall_patience_trials: int,
    min_improvement: float,
    categorical_choices: Mapping[str, Sequence[int | float]] | None = None,
) -> OptunaSearchResult:
    """
    WHAT: Runs one seeded TPE study whose objective is validation truth average precision.

    WHY: A tree-structured Parzen estimator concentrates later trials around the parameter regions
    that scored well, so 40 trials cover the configured ranges far better than a grid; a fixed
    sampler seed keeps the sampled sequence and the winning trial reproducible across runs. The
    stall rule stops a study that has stopped learning, so a full-scale run does not spend its
    remaining budget on trials the architect's ruling says cannot help.

    Args:
        train_trial: Callback that trains the model with the sampled parameters and returns one
            validation probability per scored account-day. It is called once per trial.
        scored: Validation account-days the returned probabilities belong to.
        space: Parameter name -> validated sampling range, read from the model config.
        trial_count: Number of trials to run before stopping normally.
        seed: Seed for the TPE sampler and, through the caller's callback, for the trained models.
        min_trials_before_stop: Completed trials required before the stall rule may stop the study.
        stall_patience_trials: Trailing completed trials whose best score is compared with the
            earlier best; reaching this many trials without the configured gain stops the study.
        min_improvement: Truth average precision the patience window must add to the earlier best
            for the study to keep running.
        categorical_choices: Optional parameter name -> allowed values sampled as a category, such
            as bagging's unlabelled rows per positive.

    Returns:
        The best trial's parameters and score, every completed trial's record, how many trials ran,
        whether the stall rule stopped the study, and the reason the study ended.

    Raises:
        ValueError: If the trial count, minimum trial count or patience is not a positive integer,
            the improvement is not a positive finite number, the minimum trial count exceeds the
            trial count, the ranges or scored days are empty, a category holds no values or reuses
            a range name, or a trial ends without a score because the callback raised.

    Side effects:
        Calls train_trial up to trial_count times and writes Optuna's own trial log lines to stderr.
    """
    if not isinstance(trial_count, int) or isinstance(trial_count, bool) or trial_count <= 0:
        raise ValueError(f"trial_count must be a positive integer; got {trial_count!r}")
    if not isinstance(seed, int) or isinstance(seed, bool):
        raise ValueError(f"seed must be an integer; got {seed!r}")
    if (
        not isinstance(min_trials_before_stop, int)
        or isinstance(min_trials_before_stop, bool)
        or min_trials_before_stop <= 0
    ):
        raise ValueError(
            f"min_trials_before_stop must be a positive integer; got {min_trials_before_stop!r}"
        )
    if min_trials_before_stop > trial_count:
        raise ValueError(
            f"min_trials_before_stop {min_trials_before_stop} must not exceed trial_count {trial_count}"
        )
    if (
        not isinstance(stall_patience_trials, int)
        or isinstance(stall_patience_trials, bool)
        or stall_patience_trials <= 0
    ):
        raise ValueError(
            f"stall_patience_trials must be a positive integer; got {stall_patience_trials!r}"
        )
    if (
        not isinstance(min_improvement, (int, float))
        or isinstance(min_improvement, bool)
        or not math.isfinite(min_improvement)
        or min_improvement <= 0.0
    ):
        raise ValueError(f"min_improvement must be a positive finite number; got {min_improvement!r}")
    if not space:
        raise ValueError("The Optuna search space must hold at least one parameter range")
    if not scored:
        raise ValueError("The Optuna search needs at least one validation account-day to score")
    if categorical_choices is not None:
        for name, values in categorical_choices.items():
            if name in space:
                raise ValueError(f"Categorical parameter {name!r} repeats a numeric search range")
            if not values:
                raise ValueError(f"Categorical parameter {name!r} holds no values")

    scores: list[float] = []
    stop_reason: str | None = None

    def objective(trial: optuna.Trial) -> float:
        """Trains one trial's candidate and returns its validation truth average precision."""
        params = _suggest_parameters(trial, space, categorical_choices)
        score = validation_truth_average_precision(train_trial(params), scored)
        scores.append(score)
        return score

    def stop_when_stalled(study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        """Stops the study once the patience window stops improving the earlier best score."""
        nonlocal stop_reason
        completed_count = len(scores)
        if completed_count < min_trials_before_stop:
            return
        if completed_count < stall_patience_trials + 1:
            return
        best_before = max(scores[:-stall_patience_trials])
        best_window = max(scores[-stall_patience_trials:])
        gain = best_window - best_before
        if gain < min_improvement:
            stop_reason = (
                f"stalled after {completed_count} trials: the last {stall_patience_trials} trials "
                f"gained {gain} truth AP over the earlier best, below {min_improvement}"
            )
            study.stop()

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=trial_count, callbacks=[stop_when_stalled])

    records: list[dict[str, Any]] = []
    for trial in study.trials:
        if trial.state is not optuna.trial.TrialState.COMPLETE:
            raise ValueError(f"Optuna trial {trial.number} ended in state {trial.state.name}")
        if trial.value is None:
            raise ValueError(f"Optuna trial {trial.number} completed without a score")
        records.append(
            {
                "number": trial.number,
                "params": dict(trial.params),
                "score": float(trial.value),
            }
        )
    best = study.best_trial
    if best.value is None:
        raise ValueError(f"Optuna best trial {best.number} completed without a score")
    return OptunaSearchResult(
        best_params=dict(best.params),
        best_score=float(best.value),
        trials=tuple(records),
        trials_run=len(records),
        stopped_early=stop_reason is not None,
        stop_reason=(
            stop_reason if stop_reason is not None else f"completed all {trial_count} trials"
        ),
    )
