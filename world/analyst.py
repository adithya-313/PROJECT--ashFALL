"""
WHAT: The simulated analyst (Level 12e, amendment A-022, issue L4): a pure function that decides one case as "mule"
or "genuine". Accuracy falls when the evidence is weak and when the queue is longer than the team can handle, and
sometimes the analyst just copies the model's score (automation bias). Level 22 will connect it to a real queue.

WHY: Human review is neither perfect nor constant, and later levels need that error in the loop. Keeping the
decision a pure function of its inputs, with the two random numbers passed in by the caller, makes it testable and
keeps every random stream under the caller's control.

TERM: queue load: open cases divided by the team's daily capacity in cases.
TERM: overload fraction: 0 when load is at most 1, otherwise (load - 1) / load.
TERM: automation bias: the analyst copying the model's score instead of judging the case.
TERM: unfloored accuracy: the accuracy formula's result before the floor is applied, kept so the floor stays visible.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from core.config_validator import load_yaml
from world.complaints import delay_from_uniform
from world.config_reader import draw_in_range, get_parameter_entry

REPO_ROOT: Path = Path(__file__).resolve().parent.parent
ANALYST_CONFIG_PATH: Path = REPO_ROOT / "config" / "analyst.yaml"


@dataclass(frozen=True)
class AnalystParams:
    """
    WHAT: The numbers that steer one world's analyst: base accuracy, the two accuracy drops, the share of cases
    decided by copying the model score, time per case, analysts on shift, shift length, the accuracy floor and the
    score threshold.

    WHY: Drawn once per world, they give each world its own analyst quality and capacity.

    Args:
        accuracy_base: Accuracy with strong evidence and a manageable queue, between 0 and 1.
        drop_weak_evidence: Accuracy lost at evidence strength 0, between 0 and 1.
        drop_overload: Accuracy lost as the overload fraction approaches 1, between 0 and 1.
        follows_score_share: Chance of copying the model score, between 0 and 1.
        minutes_per_case: Minutes one analyst spends per case, at least 1.
        analysts_on_shift: Analysts working on the day, at least 1.
        shift_minutes: Working minutes per analyst per day, at least 1.
        accuracy_floor: Lowest accuracy ever returned, between 0 and 1.
        score_threshold: A score at or above this reads as mule, between 0 and 1.

    Raises:
        ValueError: If a value is outside the limits above.

    Example:
        AnalystParams(0.9, 0.1, 0.1, 0.2, 20, 5, 480, 0.5, 0.5)
    """

    accuracy_base: float
    drop_weak_evidence: float
    drop_overload: float
    follows_score_share: float
    minutes_per_case: int
    analysts_on_shift: int
    shift_minutes: int
    accuracy_floor: float
    score_threshold: float

    def __post_init__(self) -> None:
        """Rejects shares outside [0, 1] and counts or minutes below 1 when the parameters are built."""
        shares = {
            "accuracy_base": self.accuracy_base, "drop_weak_evidence": self.drop_weak_evidence,
            "drop_overload": self.drop_overload, "follows_score_share": self.follows_score_share,
            "accuracy_floor": self.accuracy_floor, "score_threshold": self.score_threshold,
        }
        for name, value in shares.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"AnalystParams {name} {value} must be between 0 and 1")
        counts = {
            "minutes_per_case": self.minutes_per_case, "analysts_on_shift": self.analysts_on_shift,
            "shift_minutes": self.shift_minutes,
        }
        for name, count in counts.items():
            if count < 1:
                raise ValueError(f"AnalystParams {name} {count} must be at least 1")


@dataclass(frozen=True)
class AnalystOutcome:
    """
    WHAT: What one decided case gives back: the verdict, whether the analyst copied the score, and the accuracy used
    both with and without the floor.

    WHY: Returning the unfloored accuracy beside the floored one keeps the effect of the floor visible.

    Args:
        verdict_is_mule: True when the analyst says the account is a mule.
        followed_score: True when the verdict was copied from the model score.
        accuracy: The accuracy used, after the floor.
        unfloored_accuracy: The accuracy formula's value before the floor.

    Example:
        AnalystOutcome(True, False, 0.82, 0.82)
    """

    verdict_is_mule: bool
    followed_score: bool
    accuracy: float
    unfloored_accuracy: float


def load_analyst_config(path: Path | None = None) -> dict[str, Any]:
    """
    WHAT: Reads config/analyst.yaml (or the given file), validates the `analyst` block and returns it.

    WHY: A malformed file fails once, here, with a plain message instead of deep inside a decision.

    Args:
        path: Optional YAML path; defaults to config/analyst.yaml.

    Returns:
        The validated `analyst` block as a dictionary.

    Raises:
        ValueError: If the file has no `analyst` block or the block fails validate_analyst_config.

    Example:
        load_analyst_config()["shift_minutes"]["value"] == 480
    """
    config_path = path if path is not None else ANALYST_CONFIG_PATH
    data = load_yaml(config_path)
    if not isinstance(data, dict) or "analyst" not in data:
        raise ValueError(f"{config_path} must contain a top-level 'analyst' block")
    block = data["analyst"]
    validate_analyst_config(block)
    return block  # type: ignore[no-any-return]


def _require_range_pair(block: Any, name: str) -> None:
    """Checks a count range has an integer-valued pair with 1 <= low <= high for every split; raises ValueError."""
    if not isinstance(block, dict):
        raise ValueError(f"analyst config '{name}' must be a mapping")
    for split in ("training", "test", "stress"):
        if split not in block:
            raise ValueError(f"analyst config '{name}' has no '{split}' pair")
        pair = block[split]
        if not isinstance(pair, (list, tuple)) or len(pair) != 2:
            raise ValueError(f"analyst config '{name}' '{split}' must be a pair, got {pair!r}")
        if pair[0] < 1 or pair[0] > pair[1]:
            raise ValueError(f"analyst config '{name}' '{split}' must satisfy 1 <= low <= high, got {pair!r}")


def validate_analyst_config(config: dict[str, Any]) -> None:
    """
    WHAT: Checks the `analyst` block: every key is present, the four accuracy entries name a world_ranges entry, the
    two count ranges have pairs with 1 <= low <= high for every split, shift_minutes is a positive integer, and the
    floor and the threshold are numbers between 0 and 1.

    WHY: Required keys are never defaulted; a missing or out-of-range value must stop the run at load time.

    Args:
        config: The `analyst` block of config/analyst.yaml.

    Raises:
        ValueError: On the first problem found, naming the key.

    Example:
        validate_analyst_config(load_analyst_config())  # returns None when valid
    """
    required_keys = (
        "accuracy_base", "accuracy_drop_when_evidence_weak", "accuracy_drop_when_queue_overloaded",
        "follows_model_score_share", "minutes_per_case", "analysts_on_shift", "shift_minutes",
        "accuracy_floor", "score_threshold",
    )
    for key in required_keys:
        if key not in config:
            raise ValueError(f"analyst config is missing required key '{key}'")
    for key in required_keys[:4]:
        block = config[key]
        if not isinstance(block, dict) or not isinstance(block.get("world_ranges_reference"), str):
            raise ValueError(f"analyst config '{key}' must have a string 'world_ranges_reference'")
    _require_range_pair(config["minutes_per_case"], "minutes_per_case")
    _require_range_pair(config["analysts_on_shift"], "analysts_on_shift")

    shift_minutes = config["shift_minutes"].get("value") if isinstance(config["shift_minutes"], dict) else None
    if isinstance(shift_minutes, bool) or not isinstance(shift_minutes, int) or shift_minutes < 1:
        raise ValueError(f"analyst config shift_minutes value must be a positive integer, got {shift_minutes!r}")
    for key in ("accuracy_floor", "score_threshold"):
        value = config[key].get("value") if isinstance(config[key], dict) else None
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 <= value <= 1.0:
            raise ValueError(f"analyst config {key} value must be a number between 0 and 1, got {value!r}")


def draw_analyst_params(
    world_ranges: dict[str, Any], split: str, rng: np.random.Generator, config: dict[str, Any]
) -> AnalystParams:
    """
    WHAT: Draws one world's analyst parameters with exactly six draws, in this order: base accuracy, weak-evidence
    drop, overload drop and score-following share (each uniform inside its range in the GIVEN world_ranges for the
    split), then minutes per case and analysts on shift (each a uniform mapped to a whole number inside the split's
    pair in config/analyst.yaml with delay_from_uniform). Shift length, floor and threshold are fixed policy values.

    WHY: A fixed draw count and order keep the stream aligned whatever the configuration says.

    Args:
        world_ranges: The run's (possibly layered) world_ranges dictionary.
        split: Range split name.
        rng: Generator for the analyst stream; advanced by exactly six draws.
        config: The validated `analyst` block.

    Returns:
        The drawn AnalystParams.

    Raises:
        ValueError: If an entry or split is missing or a drawn value breaks AnalystParams limits.

    Example:
        draw_analyst_params(ranges, "training", np.random.default_rng(1), config)
    """
    accuracy_names = [
        config[key]["world_ranges_reference"] for key in (
            "accuracy_base", "accuracy_drop_when_evidence_weak", "accuracy_drop_when_queue_overloaded",
            "follows_model_score_share",
        )
    ]
    base, weak_drop, overload_drop, follows_share = (
        draw_in_range(get_parameter_entry(world_ranges, name), rng, split, name) for name in accuracy_names
    )
    minutes_pair = config["minutes_per_case"][split]
    analysts_pair = config["analysts_on_shift"][split]
    minutes_per_case = delay_from_uniform(float(rng.random()), int(minutes_pair[0]), int(minutes_pair[1]))
    analysts_on_shift = delay_from_uniform(float(rng.random()), int(analysts_pair[0]), int(analysts_pair[1]))
    return AnalystParams(
        accuracy_base=base,
        drop_weak_evidence=weak_drop,
        drop_overload=overload_drop,
        follows_score_share=follows_share,
        minutes_per_case=minutes_per_case,
        analysts_on_shift=analysts_on_shift,
        shift_minutes=config["shift_minutes"]["value"],
        accuracy_floor=float(config["accuracy_floor"]["value"]),
        score_threshold=float(config["score_threshold"]["value"]),
    )


def daily_capacity_cases(params: AnalystParams) -> float:
    """
    WHAT: Returns how many cases the team can review in a day: analysts on shift times shift minutes divided by
    minutes per case.

    WHY: Queue load is open cases divided by this number, so the same queue is heavier for a smaller team.

    Args:
        params: The world's AnalystParams.

    Returns:
        Cases per day, greater than 0.

    Example:
        daily_capacity_cases(AnalystParams(0.9, 0.1, 0.1, 0.2, 20, 5, 480, 0.5, 0.5)) == 120.0
    """
    return params.analysts_on_shift * params.shift_minutes / params.minutes_per_case


def overload_fraction(queue_load: float) -> float:
    """
    WHAT: Returns 0 when queue load is at most 1 and (load - 1) / load above it, so the fraction rises toward 1 as
    the queue grows without bound.

    WHY: An analyst is not hurt by a queue they can clear; beyond that, each extra case matters less and less, so
    the loss saturates instead of growing forever.

    Args:
        queue_load: Open cases divided by daily capacity; must not be negative.

    Returns:
        A number in [0, 1).

    Raises:
        ValueError: If queue_load is negative.

    Example:
        overload_fraction(4.0) == 0.75
    """
    if queue_load < 0.0:
        raise ValueError(f"queue_load {queue_load} must not be negative")
    if queue_load <= 1.0:
        return 0.0
    return (queue_load - 1.0) / queue_load


def expected_accuracy(
    params: AnalystParams, evidence_strength: float, queue_load: float
) -> tuple[float, float]:
    """
    WHAT: Returns (floored, unfloored) accuracy: base minus the weak-evidence drop times (1 - evidence strength)
    minus the overload drop times the overload fraction. The floored value is the unfloored one raised to
    accuracy_floor when it falls below it.

    WHY: Both values are returned so a reader can see when the floor, not the formula, decided the accuracy.

    Args:
        params: The world's AnalystParams.
        evidence_strength: How clear the evidence is, between 0 and 1.
        queue_load: Open cases divided by daily capacity; must not be negative.

    Returns:
        (accuracy_after_floor, accuracy_before_floor).

    Raises:
        ValueError: If evidence_strength is outside [0, 1] or queue_load is negative.

    Example:
        expected_accuracy(params, 1.0, 1.0)[0] == params.accuracy_base
    """
    if not 0.0 <= evidence_strength <= 1.0:
        raise ValueError(f"evidence_strength {evidence_strength} must be between 0 and 1")
    weak_evidence_loss = params.drop_weak_evidence * (1.0 - evidence_strength)
    overload_loss = params.drop_overload * overload_fraction(queue_load)
    unfloored = params.accuracy_base - weak_evidence_loss - overload_loss
    floored = unfloored if unfloored >= params.accuracy_floor else params.accuracy_floor
    return floored, unfloored


def decide_case(
    is_mule: bool,
    evidence_strength: float,
    queue_load: float,
    model_score: float | None,
    params: AnalystParams,
    bias_draw: float,
    accuracy_draw: float,
) -> AnalystOutcome:
    """
    WHAT: Decides one case. With a model score and bias_draw below the follow share, the verdict is "mule" exactly
    when the score is at or above the threshold (automation bias). Otherwise the verdict is the truth when
    accuracy_draw is below the accuracy and the opposite when it is not.

    WHY: A pure function of its inputs, with no generator inside, so callers keep control of every random draw and
    the same inputs always give the same outcome.

    A model_score of None means no score was given: followed_score is False and bias_draw is ignored. The caller
    still passes bias_draw, so the number of draws it makes per case stays fixed.

    Args:
        is_mule: The truth about the account.
        evidence_strength: How clear the evidence is, between 0 and 1.
        queue_load: Open cases divided by daily capacity; must not be negative.
        model_score: The model's score in [0, 1], or None for no score.
        params: The world's AnalystParams.
        bias_draw: Uniform draw in [0, 1) deciding whether the score is followed.
        accuracy_draw: Uniform draw in [0, 1) deciding whether the judgement is right.

    Returns:
        The AnalystOutcome.

    Raises:
        ValueError: If a draw is outside [0, 1), model_score is outside [0, 1], evidence_strength is outside [0, 1]
            or queue_load is negative.

    Example:
        decide_case(True, 1.0, 0.5, None, params, 0.3, 0.1).verdict_is_mule is True
    """
    for name, draw in (("bias_draw", bias_draw), ("accuracy_draw", accuracy_draw)):
        if not 0.0 <= draw < 1.0:
            raise ValueError(f"{name} {draw} must be in [0, 1)")
    if model_score is not None and not 0.0 <= model_score <= 1.0:
        raise ValueError(f"model_score {model_score} must be between 0 and 1")
    accuracy, unfloored = expected_accuracy(params, evidence_strength, queue_load)

    if model_score is not None and bias_draw < params.follows_score_share:
        return AnalystOutcome(model_score >= params.score_threshold, True, accuracy, unfloored)
    judged_correctly = accuracy_draw < accuracy
    verdict_is_mule = is_mule if judged_correctly else not is_mule
    return AnalystOutcome(verdict_is_mule, False, accuracy, unfloored)
