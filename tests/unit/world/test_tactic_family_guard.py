"""
WHAT: Unit and integration tests for tactic family guard and pre-registered boundary validation (Level 12d, D-4).
Tests that held-out families are rejected in training worlds, permitted in test and stress worlds, unknown families
are refused, gangs without families preserve run_ids and events bit-for-bit, and broken family lists raise ValueError.

WHY: Ground rule 17 and amendment A-014 require held-out evasions to be strictly excluded from model training
and threshold tuning environments. Ensuring that any held-out family specification raises a hard ValueError
in training ranges guarantees that models cannot overfit to held-out test scenarios.

TERM: tactic family: one named way a mule syndicate moves illicit funds, e.g. "fast_pass_through".
TERM: held-out family: an evasion tactic reserved exclusively for test and stress evaluation worlds.
TERM: training family: a tactic typology permitted in training and validation simulation worlds.
"""

from pathlib import Path
from typing import Any

import pytest

from world.fast_mode import FastRunSettings
from world.gang_config import (
    BlockStep,
    GangBudget,
    GangConfig,
    gang_config_from_plain,
    gang_config_to_plain,
)
from world.interactive_records import RunnerSettings
from world.interactive_runner import run_interactive_world
from world.tactic_families import (
    check_family_lists_agree,
    load_evaluation_config,
    load_tactic_families,
)

TINY_TRAINING = FastRunSettings(101, 60, 12, 61, "training")
TINY_TEST = FastRunSettings(101, 60, 12, 61, "test")
# Stress requires n_days >= 120 because episode_length_days upper bound is 120 for stress worlds.
TINY_STRESS = FastRunSettings(101, 60, 12, 120, "stress")
GENEROUS_BUDGET = GangBudget(money_paise=50_000_000, time_minutes=500_000, max_risk=0.99)
# Computed by the architect at commit a48f81a with the same gang, settings and code_version.
FIXTURE_RUN_ID_A48F81A = "853f341529c51e29"
FIXTURE_EVENT_COUNT_A48F81A = 2936


def _build_test_gang(gang_id: str = "g_test", family: str | None = None) -> GangConfig:
    """
    WHAT: Helper creating a standard scam syndicate with optional tactic family designation.
    WHY: Provides consistent minimal gang topology across family boundary validation test cases.
    """
    steps = (
        BlockStep("acquire", (("count", 4), ("origin_mix", "all_rented"))),
        BlockStep("scam", (("receiving_mules", 2),)),
        BlockStep("route", (("layers", 1), ("accounts_per_layer", 2))),
    )
    return GangConfig(gang_id, 1, GENEROUS_BUDGET, None, steps, 1440, family=family)


def test_held_out_families_refused_in_training_worlds() -> None:
    """
    WHAT: Asserts that each pre-registered held-out tactic family is rejected with a plain ValueError
    when configured on a gang executing within a training range world.
    WHY: Amendment A-014 prohibits held-out evasions from running in training environments to guarantee
    models cannot train on held-out detection challenges.
    """
    families = load_tactic_families()
    eval_cfg = load_evaluation_config()
    held_out_names = sorted(families["held_out"].keys())
    eval_held_out = sorted(eval_cfg["held_out_tactics"]["tactics"])
    # The two lists must agree before we trust either for testing.
    assert held_out_names == eval_held_out, (
        f"tactic_families held_out {held_out_names} != evaluation.yaml {eval_held_out}"
    )

    for name in held_out_names:
        gang = _build_test_gang(gang_id=f"gang_{name}", family=name)
        expected_msg = (
            f"Gang '{gang.gang_id}' uses held-out family '{name}', "
            "which is allowed only in test and stress worlds."
        )
        with pytest.raises(ValueError, match=expected_msg):
            run_interactive_world(
                TINY_TRAINING,
                RunnerSettings(fraud_enabled=True),
                gang_configs=(gang,),
            )


def test_held_out_families_permitted_in_test_and_stress_worlds() -> None:
    """
    WHAT: Asserts that a gang configured with a held-out tactic family executes successfully without
    raising errors when run inside test and stress world partitions.
    WHY: Held-out tactics exist specifically to benchmark adversarial robustness under unseen test
    and stress conditions.
    """
    families = load_tactic_families()
    first_held_out = sorted(families["held_out"].keys())[0]
    gang = _build_test_gang(gang_id="gang_held_out_ok", family=first_held_out)

    out_test = run_interactive_world(
        TINY_TEST,
        RunnerSettings(fraud_enabled=True),
        gang_configs=(gang,),
    )
    assert len(out_test.result.run_result.events) > 0

    out_stress = run_interactive_world(
        TINY_STRESS,
        RunnerSettings(fraud_enabled=True),
        gang_configs=(gang,),
    )
    assert len(out_stress.result.run_result.events) > 0


def test_unknown_family_name_is_refused() -> None:
    """
    WHAT: Asserts that a gang specifying an unregistered tactic family name raises ValueError naming the unknown family.
    WHY: Prevents misspellings or undeclared tactical categories from silently executing without validation.
    """
    gang = _build_test_gang(gang_id="gang_unknown", family="completely_unregistered_tactic")
    with pytest.raises(ValueError, match="Gang 'gang_unknown' uses unknown family 'completely_unregistered_tactic'."):
        run_interactive_world(
            TINY_TEST,
            RunnerSettings(fraud_enabled=True),
            gang_configs=(gang,),
        )


def test_gang_with_no_family_preserves_run_id_and_events() -> None:
    """
    WHAT: Asserts that a gang built via gang_config_from_plain from a plain dict with no 'family' key
    produces the same run_id as the pinned value computed at a48f81a (byte-identical backward compatibility).
    WHY: Backward compatibility requirement: existing Level 12a-12c fixtures and simulations must remain byte-identical.
    """
    # Build a plain dict identical to what gang_config_to_plain produced at a48f81a (no 'family' key present).
    steps = (
        BlockStep("acquire", (("count", 4), ("origin_mix", "all_rented"))),
        BlockStep("scam", (("receiving_mules", 2),)),
        BlockStep("route", (("layers", 1), ("accounts_per_layer", 2))),
    )
    plain = gang_config_to_plain(GangConfig("g_plain", 1, GENEROUS_BUDGET, None, steps, 1440))
    assert "family" not in plain, "plain dict must have no 'family' key for backward-compat check"

    gang = gang_config_from_plain(plain)
    assert gang.family is None

    out = run_interactive_world(
        TINY_TRAINING,
        RunnerSettings(fraud_enabled=True),
        gang_configs=(gang,),
        code_version="family-guard-a48f81a",
    )
    assert out.result.run_id == FIXTURE_RUN_ID_A48F81A, (
        f"run_id changed: got {out.result.run_id!r}, want {FIXTURE_RUN_ID_A48F81A!r}"
    )
    assert len(out.result.run_result.events) == FIXTURE_EVENT_COUNT_A48F81A


def test_broken_copy_of_family_lists_disagreement_refused(tmp_path: Path) -> None:
    """
    WHAT: Asserts that check_family_lists_agree raises ValueError when held-out family lists differ from
    evaluation.yaml, or when any family name appears simultaneously in training and held-out categories.
    WHY: Guarantees that discrepancies between configuration specifications fail loudly before simulation planning.
    """
    families = load_tactic_families()
    eval_cfg = load_evaluation_config()

    # Base valid configuration check passes without raising.
    check_family_lists_agree(families, eval_cfg)

    # Pick the first sorted held-out name to drop/duplicate (avoids hardcoding a literal name).
    first_held_out = sorted(families["held_out"].keys())[0]

    # 1. Broken copy where a held-out tactic is missing.
    broken_missing: dict[str, Any] = {
        "training": dict(families["training"]),
        "held_out": {k: v for k, v in families["held_out"].items() if k != first_held_out},
    }
    with pytest.raises(ValueError, match="Held-out tactics lists disagree"):
        check_family_lists_agree(broken_missing, eval_cfg)

    # 2. Broken copy where an overlapping family is both training and held-out.
    broken_overlap: dict[str, Any] = {
        "training": {**families["training"], first_held_out: {}},
        "held_out": dict(families["held_out"]),
    }
    with pytest.raises(ValueError, match="cannot be both training and held-out"):
        check_family_lists_agree(broken_overlap, eval_cfg)


def test_missing_held_out_key_raises_value_error() -> None:
    """
    WHAT: Asserts that check_family_lists_agree raises ValueError when the tactic_families block
    has no 'held_out' key.
    WHY: Validates that the guard has no silent fallback for a structurally incomplete config.
    """
    families = load_tactic_families()
    eval_cfg = load_evaluation_config()
    broken: dict[str, Any] = {"training": dict(families["training"])}
    with pytest.raises(ValueError, match="missing 'held_out' key"):
        check_family_lists_agree(broken, eval_cfg)


def test_missing_held_out_tactics_in_evaluation_raises_value_error() -> None:
    """
    WHAT: Asserts that check_family_lists_agree raises ValueError when the evaluation config
    has no 'held_out_tactics' key.
    WHY: Validates that the guard has no silent fallback for a structurally incomplete evaluation config.
    """
    families = load_tactic_families()
    broken_eval: dict[str, Any] = {}
    with pytest.raises(ValueError, match="missing 'held_out_tactics' key"):
        check_family_lists_agree(families, broken_eval)
