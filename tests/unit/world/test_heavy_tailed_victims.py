"""
Tests for heavy-tailed victim count distribution per receiving mule (Level 12c C-3b, D2).
Verifies truncated power-law mathematical properties, exact probability bounds matching
world_ranges typical/largest expectations, stochastic sampling convergence within k = 4 standard
deviations, configuration narrowing, explicit parameter range validation, and refusal when victims
exceed the eligible population.
"""

import math
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest
import yaml

import world.scam_planner as scam_planner
from world.build_world import build_world
from world.gang_config import BlockStep, GangConfig, validate_gang_config
from world.scam_config import (
    SCAM_SCHEDULES_PATH,
    resolve_victims_per_mule_largest,
    resolve_victims_per_mule_tail_exponent,
    resolve_victims_per_mule_typical,
)
from world.scam_planner import (
    _draw_victim_counts,
    compute_power_law_probabilities,
    plan_scam_step,
)


def _write_schedules(tmp_path: Path, edit: Callable[[dict[str, Any]], None]) -> Path:
    """Writes a copy of the committed schedule file after applying edit to its parsed contents."""
    data = yaml.safe_load(SCAM_SCHEDULES_PATH.read_text(encoding="utf-8"))
    edit(data)
    path = tmp_path / "scam_schedules.yaml"
    path.write_text(yaml.safe_dump(data), encoding="utf-8")
    return path


def test_probability_table_sums_to_1_and_is_decreasing() -> None:
    """Checks that compute_power_law_probabilities returns a strictly decreasing probability table summing to 1.0."""
    for cap in (1, 2, 5, 10, 50, 100):
        for exponent in (1.2, 1.8, 2.2, 2.6, 3.5):
            probs = compute_power_law_probabilities(cap, exponent)
            assert len(probs) == cap
            assert math.isclose(sum(probs), 1.0, abs_tol=1e-12)
            for i in range(len(probs) - 1):
                assert probs[i] > probs[i + 1], f"Expected decreasing at index {i} for cap={cap}, exp={exponent}"


def test_probability_table_validates_inputs() -> None:
    """Checks that invalid cap or exponent values raise ValueError with plain sentences."""
    with pytest.raises(ValueError, match="Truncation cap must be an integer of at least 1"):
        compute_power_law_probabilities(0, 2.0)
    with pytest.raises(ValueError, match="Truncation cap must be an integer of at least 1"):
        compute_power_law_probabilities(-5, 2.0)
    with pytest.raises(ValueError, match="Truncation cap must be an integer of at least 1"):
        compute_power_law_probabilities(True, 2.0)
    with pytest.raises(ValueError, match="Power law exponent must be a positive number"):
        compute_power_law_probabilities(10, 0.0)
    with pytest.raises(ValueError, match="Power law exponent must be a positive number"):
        compute_power_law_probabilities(10, -1.5)


def test_exact_check_bulk_of_mules_within_typical_range() -> None:
    """
    EXACT check (from the table, no sampling) that P(count <= high end of victims_per_mule_typical) >= 0.5
    at both ends of the training exponent range, with the cap at both ends of victims_per_mule_largest.
    """
    typical_high = resolve_victims_per_mule_typical("training")[1]
    cap_min, cap_max = resolve_victims_per_mule_largest("training")
    exp_min, exp_max = resolve_victims_per_mule_tail_exponent("training")

    for cap in (cap_min, cap_max):
        for exp in (exp_min, exp_max):
            probs = compute_power_law_probabilities(cap, exp)
            p_le_typical = sum(probs[:typical_high])
            assert p_le_typical >= 0.5, (
                f"P(count <= {typical_high}) was {p_le_typical} (< 0.5) for cap={cap}, exp={exp}."
            )


def test_sampling_converges_to_power_law_table_within_4_sd(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Checks that 20,000 draws from _draw_victim_counts with narrowed config match theoretical
    P(1) and P(count > 5) within k = 4 standard deviations, and max draw reaches at least 2 x typical high.
    """
    cap = 50
    exponent = 2.0
    monkeypatch.setattr(scam_planner, "resolve_victims_per_mule_largest", lambda split: (cap, cap))
    monkeypatch.setattr(scam_planner, "resolve_victims_per_mule_tail_exponent", lambda split: (exponent, exponent))

    probs = compute_power_law_probabilities(cap, exponent)
    typical_high = 5
    p_1 = probs[0]
    p_gt_5 = sum(probs[typical_high:])

    n_draws = 20_000
    rng = np.random.default_rng(20261021)
    step = BlockStep("scam", (("receiving_mules", 1),))

    draws: list[int] = []
    batch_size = 1000
    for _ in range(n_draws // batch_size):
        draws.extend(_draw_victim_counts(step, batch_size, "training", rng))

    draws_arr = np.array(draws)
    obs_p_1 = float(np.mean(draws_arr == 1))
    obs_p_gt_5 = float(np.mean(draws_arr > 5))

    sd_1 = math.sqrt(p_1 * (1.0 - p_1) / n_draws)
    sd_gt_5 = math.sqrt(p_gt_5 * (1.0 - p_gt_5) / n_draws)

    assert abs(obs_p_1 - p_1) <= 4.0 * sd_1, (
        f"Observed P(1) {obs_p_1:.4f} deviated from theoretical {p_1:.4f} by more than 4 SD ({4.0 * sd_1:.4f})."
    )
    assert abs(obs_p_gt_5 - p_gt_5) <= 4.0 * sd_gt_5, (
        f"Observed P(>5) {obs_p_gt_5:.4f} deviated from theoretical {p_gt_5:.4f} by > 4 SD ({4.0 * sd_gt_5:.4f})."
    )
    assert int(np.max(draws_arr)) >= 2 * typical_high, (
        f"Maximum draw {np.max(draws_arr)} was not at least 2 x typical high ({2 * typical_high})."
    )


def test_narrowed_config_exponent_read(tmp_path: Path) -> None:
    """Checks that resolve_victims_per_mule_tail_exponent reads narrowed exponent bounds from config."""
    narrowed_path = _write_schedules(
        tmp_path,
        lambda data: data["victims_per_mule_tail_exponent"].update({"training": [2.3, 2.5]}),
    )
    assert resolve_victims_per_mule_tail_exponent("training", narrowed_path) == (2.3, 2.5)


def test_explicit_count_range_validated() -> None:
    """
    Checks that an explicit victims_per_receiving_mule is range-checked against [1, largest high].
    In training split, largest high is 50.
    """
    largest_high = resolve_victims_per_mule_largest("training")[1]
    assert largest_high == 50

    # Valid boundary values: 1 and 50
    steps_1 = (
        BlockStep("acquire", (("count", 2),)),
        BlockStep("scam", (("receiving_mules", 1), ("victims_per_receiving_mule", 1))),
    )
    validate_gang_config(GangConfig("g1", 1, 100_000_00, None, steps_1, 0))

    steps_50 = (
        BlockStep("acquire", (("count", 2),)),
        BlockStep("scam", (("receiving_mules", 1), ("victims_per_receiving_mule", largest_high))),
    )
    validate_gang_config(GangConfig("g1", 1, 100_000_00, None, steps_50, 0))

    # Invalid: 0 (below 1) and 51 (above largest high)
    steps_0 = (
        BlockStep("acquire", (("count", 2),)),
        BlockStep("scam", (("receiving_mules", 1), ("victims_per_receiving_mule", 0))),
    )
    with pytest.raises(ValueError, match=r"victims_per_receiving_mule must be an integer in \[1, 50\]"):
        validate_gang_config(GangConfig("g1", 1, 100_000_00, None, steps_0, 0))

    steps_51 = (
        BlockStep("acquire", (("count", 2),)),
        BlockStep("scam", (("receiving_mules", 1), ("victims_per_receiving_mule", largest_high + 1))),
    )
    with pytest.raises(ValueError, match=r"victims_per_receiving_mule must be an integer in \[1, 50\]"):
        validate_gang_config(GangConfig("g1", 1, 100_000_00, None, steps_51, 0))


def test_refusal_when_victims_exceed_eligible_people() -> None:
    """
    Checks that plan_scam_step raises ValueError with one plain sentence naming both numbers when
    victims exceed eligible people.
    """
    # World with 10 people
    world = build_world(seed=101, n_people=10, n_shops=4)
    # 2 mules acquired, leaving 8 eligible people
    mules = ["acc_person_000000", "acc_person_000001"]
    # Request 2 receiving mules with explicit 5 victims each = 10 victims needed
    step = BlockStep("scam", (("receiving_mules", 2), ("victims_per_receiving_mule", 5)))
    config = GangConfig("g1", 1, 100_000_00, None, (step,), 0)

    with pytest.raises(ValueError, match=r"Total victims needed 10 exceeds eligible people 8\."):
        plan_scam_step(world, config, step, mules, 0, None, np.random.default_rng(1))
