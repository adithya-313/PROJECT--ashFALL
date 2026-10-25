"""
WHAT: Unit tests for Level 12h trivial-world check (A-026) and discrete AUPRC algorithm.
Tests rank evaluation hand examples, tie grouping, signal directional symmetry, trivial world failure,
configuration threshold overrides, and baseline verification for standard training worlds.

WHY: Verifies that evaluation/trivial_world.py accurately detects artificial separability, maintains pure
mathematical correctness without statistical package drift, and strictly enforces pre-registered gating rules.
"""

from pathlib import Path

import pytest
import yaml

from evaluation.trivial_world import (
    account_signals,
    average_precision,
    best_direction_ap,
    check_world,
    load_evaluation_config,
    population,
)
from world.sampled_world import run_sampled_world
from world.world_configs import read_world_configs
from world.world_sampler import load_sampler_config, sample_block


def test_average_precision_hand_examples() -> None:
    """
    WHAT: Tests average_precision against exact closed-form analytical examples.
    WHY: Proves math correctness on perfect rankings, alternating outcomes, tied thresholds, and boundary validation.
    """
    # 1. Perfect ranking where all positive predictions rank strictly above all negative predictions -> 1.0.
    assert average_precision([10.0, 5.0], [1, 0]) == 1.0

    # 2. Alternating sequence [mule, genuine, mule] with scores [3.0, 2.0, 1.0].
    # Precision at item 1 = 1/1 = 1.0; precision at item 3 = 2/3. AP = (1.0 + 2/3) / 2 = 5/6 = 0.8333...
    expected_alt = (1.0 / 1.0 + 2.0 / 3.0) / 2.0
    assert abs(average_precision([3.0, 2.0, 1.0], [1, 0, 1]) - expected_alt) < 1e-6

    # 3. Refusal when there are no positive labels present in the ground truth.
    with pytest.raises(ValueError, match="at least one positive label"):
        average_precision([1.0, 2.0], [0, 0])

    # 4. Refusal when there are no negative labels present in the ground truth.
    with pytest.raises(ValueError, match="at least one negative label"):
        average_precision([1.0, 2.0], [1, 1])


def test_tie_grouping_precision_at_top() -> None:
    """
    WHAT: Tests precision calculation when positive and negative observations share tied scores at the top.
    WHY: Verifies that tied threshold groups compute precision collectively after the whole group
         rather than individually.
    """
    # Group with tied score 3.0 containing one mule and one genuine item.
    # Group precision after the group must be 1/2 = 0.5. Followed by a positive at score 1.0.
    # AP = (1/2)*0.5 + (1/2)*(2/3) = 0.25 + 1/3 = 7/12 = 0.583333...
    scores = [3.0, 3.0, 1.0]
    labels = [1, 0, 1]
    expected_ap = 0.5 * 0.5 + 0.5 * (2.0 / 3.0)
    assert abs(average_precision(scores, labels) - expected_ap) < 1e-6

    # Direct two-item tie at the top asserting 0.5 precision after the group.
    assert average_precision([2.0, 2.0], [1, 0]) == 0.5


def test_signal_direction_symmetry() -> None:
    """
    WHAT: Tests that best_direction_ap produces identical scores for inverted signal vectors.
    WHY: Ground Rule A-026 requires scoring candidate signals in both directions so low-value indicators rank equally.
    """
    # A signal that is lower for mules ([1.0, 2.0]) than for genuine citizens ([10.0, 20.0]).
    values = [1.0, 2.0, 10.0, 20.0]
    labels = [1, 1, 0, 0]
    ap_direct = best_direction_ap(values, labels)
    ap_negated = best_direction_ap([-v for v in values], labels)
    assert ap_direct == ap_negated
    assert ap_direct == 1.0


def test_deliberately_trivial_world_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    WHAT: Asserts that injecting an artificially separable signal causes check_world to fail at AP 1.0.
    WHY: Proves the trivial-world gate rejects worlds where a single feature isolates all mules without learning.
    """
    sampler_cfg = load_sampler_config()
    ranges = read_world_configs().world_ranges
    specs = sample_block("training", 0, 1, sampler_cfg, ranges)
    output = run_sampled_world(specs[0], "small", None, sampler_cfg, "12h-test")

    # Extract true population and calculate baseline signal matrix.
    mules, _genuine = population(output)
    real_signals = account_signals(output)

    # Inject test-only 7th signal 'unique_device_flag' = 1.0 for every mule, 0.0 otherwise.
    modified_signals = {
        acc: {**metrics, "unique_device_flag": 1.0 if acc in mules else 0.0}
        for acc, metrics in real_signals.items()
    }
    monkeypatch.setattr("evaluation.trivial_world.account_signals", lambda _out, **_kw: modified_signals)

    eval_cfg = load_evaluation_config()
    result = check_world(output, eval_cfg)

    assert result["best_signal"] == "unique_device_flag"
    assert result["best_ap"] == 1.0
    assert result["passed"] is False


def test_threshold_comes_from_config(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    WHAT: Asserts that fail_threshold is dynamically loaded from config by passing a temporary evaluation.yaml.
    WHY: Confirms threshold enforcement obeys external policy files rather than hardcoded literals.
    """
    sampler_cfg = load_sampler_config()
    ranges = read_world_configs().world_ranges
    specs = sample_block("training", 0, 1, sampler_cfg, ranges)
    output = run_sampled_world(specs[0], "small", None, sampler_cfg, "12h-test")

    mules, _genuine = population(output)
    real_signals = account_signals(output)
    modified_signals = {
        acc: {**metrics, "unique_device_flag": 1.0 if acc in mules else 0.0}
        for acc, metrics in real_signals.items()
    }
    monkeypatch.setattr("evaluation.trivial_world.account_signals", lambda _out, **_kw: modified_signals)

    # Write temporary evaluation.yaml with relaxed threshold 1.01.
    tmp_config_data = load_evaluation_config()
    tmp_config_data["trivial_world_check"]["fail_threshold"] = 1.01
    tmp_config_file = tmp_path / "evaluation.yaml"
    with open(tmp_config_file, "w", encoding="utf-8") as f:
        yaml.safe_dump(tmp_config_data, f)

    with open(tmp_config_file, encoding="utf-8") as f:
        override_cfg = yaml.safe_load(f)

    # With threshold 1.01, the trivial world having AP 1.0 must pass (1.0 < 1.01).
    result = check_world(output, override_cfg)
    assert result["best_ap"] == 1.0
    assert result["passed"] is True


def test_normal_training_world_passes() -> None:
    """
    WHAT: Evaluates a standard training world (block 0, index 0, small preset) under default evaluation rules.
    WHY: Confirms standard realistic worlds generate complex behavioral distributions that pass the 0.90 gate.
    """
    sampler_cfg = load_sampler_config()
    ranges = read_world_configs().world_ranges
    specs = sample_block("training", 0, 1, sampler_cfg, ranges)
    output = run_sampled_world(specs[0], "small", None, sampler_cfg, "12h-test")

    eval_cfg = load_evaluation_config()
    result = check_world(output, eval_cfg)

    assert result["passed"] is True
    assert result["best_ap"] < 0.90
    assert len(result["top_3"]) == 3
    assert result["n_mules"] > 0
    assert result["n_genuine"] > 0
    assert result["fraud_share"] > 0.0
