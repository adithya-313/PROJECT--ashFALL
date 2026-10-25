"""
WHAT: Integration tests for running gang configurations through the interactive world runner (Level 12b B-2a).
Asserts reproducibility across runs, gang-sensitive run_id derivation, frozen-anchor run_id invariance,
genuine draw stability via call recording, money conservation, temporal role transitions, and fraud switch enforcement.

WHY: Connecting gang planning to the interactive runner must preserve all Level 11 and Level 12a invariants.
Testing the full execution pipeline guarantees that fraud injections and truth store generation operate without side effects.

TERM: gang run: interactive world execution parameterized with one or more gang configurations.
TERM: genuine draw: random number generation draw consumed by normal civilian payments and failure schedules.
TERM: money conservation: invariance ensuring total world balances match initial deposits plus registered injections.
TERM: temporal role: time-varying illicit role assignment evaluating genuine before valid_from and mule during tenancy.
"""

import hashlib
from pathlib import Path
from typing import Any
import pytest

from evaluation.truth import load_truth
from showcase.level_12_support_b import get_showcase_b_settings, load_showcase_b_fixture
from world.calendar_runner import run_world
from world.fast_mode import FastRunSettings
from world.gang_config import BlockStep, FixtureSource, GangBudget, GangConfig
from world.interactive_records import HookDecision, PaymentView, RunnerSettings
import world.interactive_runner as interactive_runner
from world.interactive_runner import (
    PreparedWorld,
    SWITCH_OFF_MESSAGE,
    prepare_world,
    run_interactive_world,
)
from world.truth_writer import write_truth_files

from tests.unit.world.test_interactive_runner_fraud_on import (
    CallRecorder,
    check_balances_equal_entry_sums,
    check_money_conserved,
    check_no_negative_balances,
    opening_total,
)

TINY = FastRunSettings(101, 60, 12, 61, "training")
BUDGET = GangBudget(money_paise=10_000_000, time_minutes=10_000, max_risk=0.5)
SOURCE_1 = FixtureSource("acc_gang_src_01", 0, 10_000_000, 1440)
SOURCE_2 = FixtureSource("acc_gang_src_02", 1, 10_000_000, 2880)

STEPS_1 = (
    BlockStep("acquire", (("count", 3),)),
    BlockStep("warm_up", (("days", 1),)),
    BlockStep("route", (("layers", 2), ("accounts_per_layer", 1))),
)
STEPS_2 = (
    BlockStep("acquire", (("count", 4),)),
    BlockStep("route", (("layers", 2), ("accounts_per_layer", 2))),
)

GANG_1 = GangConfig("gang_bravo", 1, BUDGET, SOURCE_1, STEPS_1)
GANG_2 = GangConfig("gang_charlie", 2, BUDGET, SOURCE_2, STEPS_2)


def _compute_truth_shas(directory: Path) -> dict[str, str]:
    """
    WHAT: Computes SHA-256 cryptographic digests for all truth files in the specified directory.
    WHY: Verifies exact byte-for-byte reproducibility across truth records, role spans, and device control files.
    Args:
        directory: Path to the directory containing written truth JSONL and metadata files.
    Returns:
        Dictionary mapping filename strings to hex SHA-256 digests.
    Example:
        # shas = _compute_truth_shas(tmp_path)
    """
    shas: dict[str, str] = {}
    for filename in ("roles.jsonl", "device_control.jsonl", "payment_stories.jsonl", "manifest.json"):
        filepath = directory / filename
        shas[filename] = hashlib.sha256(filepath.read_bytes()).hexdigest()
    return shas


def test_same_seed_twice_produces_identical_events_and_truth(tmp_path: Path) -> None:
    """
    WHAT: Asserts that running the same world seed twice with identical gang configurations produces identical
    payment event sequences, identical planned payment IDs, and identical SHA-256 hashes across all four truth files.

    WHY: Simulation determinism is mandatory; repeated runs from the same seed must replicate transaction
    timings, failure events, and hidden ground truth without divergence.

    Args:
        tmp_path: Pytest temporary directory fixture for truth file output.

    Returns:
        None. Fails if payment events, planned IDs, or truth file hashes differ between runs.

    Example:
        # test_same_seed_twice_produces_identical_events_and_truth(tmp_path)
    """
    out1 = run_interactive_world(TINY, RunnerSettings(fraud_enabled=True), gang_configs=(GANG_1,))
    out2 = run_interactive_world(TINY, RunnerSettings(fraud_enabled=True), gang_configs=(GANG_1,))

    assert out1.result.run_result.events == out2.result.run_result.events
    assert out1.result.planned_payment_ids == out2.result.planned_payment_ids

    dir1 = write_truth_files(out1, tmp_path / "run1")
    dir2 = write_truth_files(out2, tmp_path / "run2")

    shas1, shas2 = _compute_truth_shas(dir1), _compute_truth_shas(dir2)
    assert shas1 == shas2


def test_different_gang_config_produces_different_run_id() -> None:
    """
    WHAT: Asserts that varying gang tactical configurations produces distinct simulation run identifiers.
    It runs the interactive world with two distinct gang configurations and verifies run_id divergence.

    WHY: Decision D5 requires gang configurations to fold into effective_config so that any change in gang
    tactics, budget, source account, or strength produces a distinct run_id fingerprint.

    Args:
        None.

    Returns:
        None. Fails if different gang configurations yield identical run identifiers.

    Example:
        # assert run_id_a != run_id_b
    """
    out_a = run_interactive_world(TINY, RunnerSettings(fraud_enabled=True), gang_configs=(GANG_1,))
    out_b = run_interactive_world(TINY, RunnerSettings(fraud_enabled=True), gang_configs=(GANG_2,))
    assert out_a.result.run_id != out_b.result.run_id


def test_empty_gang_configs_matches_baseline_run_id() -> None:
    """
    WHAT: Asserts that running without gang configurations produces the identical run_id as today's frozen baseline.
    It verifies that effective_config omits 'gang_configs' when empty, preserving legacy run fingerprints.

    WHY: Decision D5 specifies that effective_config gains 'gang_configs' only when non-empty, ensuring that
    unmodified simulation runs never experience run_id drift.

    Args:
        None.

    Returns:
        None. Fails if omitting gang configs changes the established frozen run identifier.

    Example:
        # assert out_default.result.run_id == out_empty_gangs.result.run_id
    """
    out_default = run_interactive_world(TINY)
    out_empty_gangs = run_interactive_world(TINY, gang_configs=())
    assert out_default.result.run_id == out_empty_gangs.result.run_id


def test_genuine_draws_never_move_with_gang_present(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    WHAT: Asserts that civilian transaction and incident failure draws on run_rng are completely unaffected by gang presence.
    It records every call on run_rng during a Level 11 run and compares against an interactive run with gang configs.

    WHY: Ground Rule 13 and Decision D5 require that fraud injections never perturb genuine civilian payments;
    gang tactics must consume auxiliary child streams (child 7) without altering genuine pseudo-random state.

    Args:
        monkeypatch: Pytest monkeypatch fixture used to intercept PreparedWorld construction.

    Returns:
        None. Fails if any pseudo-random call on the run stream differs in method or arguments.

    Example:
        # assert runner_recorder.calls == level_11_recorder.calls
    """
    l11_world = prepare_world(TINY, None)
    l11_recorder = CallRecorder(l11_world.run_rng)
    run_world(
        l11_world.world, l11_world.plan, l11_recorder,  # type: ignore[arg-type]
        device_events=l11_world.device_events, episode_payments=l11_world.episode_payments
    )

    recorder: CallRecorder | None = None
    orig_prepare = interactive_runner.prepare_world

    def recording_prepare(*args: Any, **kwargs: Any) -> PreparedWorld:
        """
        WHAT: Wraps the run stream of PreparedWorld with CallRecorder to intercept and log pseudo-random draws.
        It calls the original prepare_world function, wraps its run_rng in CallRecorder, and returns a new PreparedWorld.

        WHY: Proves genuine simulation draws on run_rng remain identical regardless of gang presence per Decision D5.

        Args:
            *args: Positional arguments forwarded to prepare_world.
            **kwargs: Keyword arguments forwarded to prepare_world.

        Returns:
            PreparedWorld instance with its run_rng replaced by the CallRecorder wrapper.

        Example:
            # prep = recording_prepare(TINY, None)
            # assert prep.run_rng is recorder
        """
        nonlocal recorder
        prep = orig_prepare(*args, **kwargs)
        recorder = CallRecorder(prep.run_rng)
        return PreparedWorld(
            prep.world, prep.plan, recorder, prep.device_events, prep.episodes, prep.episode_payments, prep.episode_report  # type: ignore[arg-type]
        )

    monkeypatch.setattr(interactive_runner, "prepare_world", recording_prepare)
    run_interactive_world(TINY, RunnerSettings(fraud_enabled=True), gang_configs=(GANG_1,))
    assert recorder is not None and recorder.calls == l11_recorder.calls


def test_money_conservation_with_gang_source_opening_balance() -> None:
    """
    WHAT: Asserts that total world money remains strictly conserved when a gang fixture source is pre-funded.
    It checks that ledgers have no negative balances, balances match entry sums, and total money matches
    opening civilian deposits plus the fixture collector's opening balance.

    WHY: Ground Rule 15 and Decision D3 mandate strict double-entry ledger integrity across all bank institutions,
    guaranteeing that illicit money flows never generate unbacked currency or balance discrepancies.

    Args:
        None.

    Returns:
        None. Fails if double-entry invariants or money conservation rules are violated.

    Example:
        # assert check_money_conserved(ledgers, expected_total) == []
    """
    output = run_interactive_world(TINY, RunnerSettings(fraud_enabled=True), gang_configs=(GANG_1,))
    ledgers = output.world.ledgers
    expected_total = opening_total(TINY) + output.result.money_created_paise

    assert check_no_negative_balances(ledgers) == []
    assert check_balances_equal_entry_sums(ledgers) == []
    assert check_money_conserved(ledgers, expected_total) == []
    assert output.result.money_created_paise == GANG_1.source.opening_balance_paise


def test_rented_mule_temporal_role_is_genuine_before_valid_from(tmp_path: Path) -> None:
    """
    WHAT: Asserts that leased citizen mule accounts evaluate to role 'genuine' prior to valid_from via load_truth.
    It writes generated truth facts to disk, loads them through truth_loader, and checks temporal role resolution.

    WHY: Decision D7 establishes that rented mules are authentic citizen accounts prior to recruitment;
    truth queries before lease inception must evaluate to genuine without retroactively contaminating account history.

    Args:
        tmp_path: Pytest temporary directory fixture for writing and reading truth store files.

    Returns:
        None. Fails if account role prior to valid_from is not 'genuine' or gang is not None.

    Example:
        # assert loaded.role_at(mule_id, valid_from - 1) == "genuine"
    """
    output = run_interactive_world(TINY, RunnerSettings(fraud_enabled=True), gang_configs=(GANG_1,))
    run_dir = write_truth_files(output, tmp_path)
    loaded = load_truth(tmp_path, run_dir.name)

    mule_span = next(r for r in output.result.truth.roles if r.role == "mule")
    mule_id = mule_span.account_id
    valid_from = mule_span.valid_from

    assert loaded.role_at(mule_id, valid_from - 1) == "genuine"
    assert loaded.role_at(mule_id, valid_from) == "mule"
    assert loaded.owner_at(mule_id, valid_from - 1) is None
    assert loaded.owner_at(mule_id, valid_from) == GANG_1.gang_id


def test_gang_configs_with_fraud_switch_off_is_refused() -> None:
    """
    WHAT: Asserts that providing gang configurations when the fraud switch is off raises a ValueError with SWITCH_OFF_MESSAGE.
    WHY: Ground Rule 13 prohibits fraud inputs under fraud_enabled=False; silent omission would hide caller mistakes.

    Args:
        None.

    Returns:
        None. Fails if gang configurations are accepted with fraud disabled.

    Example:
        # with pytest.raises(ValueError, match=SWITCH_OFF_MESSAGE):
        #     run_interactive_world(TINY, RunnerSettings(fraud_enabled=False), gang_configs=(GANG_1,))
    """
    with pytest.raises(ValueError, match=SWITCH_OFF_MESSAGE):
        run_interactive_world(TINY, RunnerSettings(fraud_enabled=False), gang_configs=(GANG_1,))


def test_fact_2_regression_via_gang_configs_only() -> None:
    """
    Verifies that running the showcase-b fixture through gang_configs automatically generates
    illicit money rules and source seeds that prevent layer-1 mules from draining personal balances.
    Asserts two skipped forwards for 676,300 and 4,731,000 paise with total ledger money conserved.
    """
    gang, _ = load_showcase_b_fixture()
    settings, code_version = get_showcase_b_settings()

    def decline_collector_hook(view: PaymentView) -> HookDecision:
        """Declines payments from the gang collector account while allowing all other transfers."""
        if view.from_account == "acc_collector_01":
            return "decline"
        return "allow"

    output = run_interactive_world(
        fast_settings=settings,
        runner_settings=RunnerSettings(fraud_enabled=True),
        gang_configs=(gang,),
        hook=decline_collector_hook,
        code_version=code_version,
    )

    layer_1_mules = {"acc_person_000006", "acc_person_000010"}
    mule_successes = [
        event for event, source in zip(output.result.run_result.events, output.result.event_sources)
        if source == "fraud" and event.from_account in layer_1_mules and event.status == "success"
    ]
    assert len(mule_successes) == 0

    assert len(output.result.skipped_forwards) == 2
    assert [skipped.planned_paise for skipped in output.result.skipped_forwards] == [676300, 4731000]
    assert [skipped.from_account for skipped in output.result.skipped_forwards] == [
        "acc_person_000006",
        "acc_person_000010",
    ]
    assert all(skipped.reason == "no_illicit_money_arrived" for skipped in output.result.skipped_forwards)

    ledgers = output.world.ledgers
    expected_total = opening_total(settings) + output.result.money_created_paise
    assert check_no_negative_balances(ledgers) == []
    assert check_balances_equal_entry_sums(ledgers) == []
    assert check_money_conserved(ledgers, expected_total) == []


def test_refusal_when_gang_route_payment_lacks_rule(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verifies that the runner strictly refuses gang plans containing route payments that lack illicit money rules.
    Constructs a broken gang plan missing money rules and asserts that _apply_gang_configs raises a plain ValueError.
    """
    real_plan_gang = interactive_runner.plan_gang

    def broken_plan_gang(*args: Any, **kwargs: Any) -> Any:
        """Simulates a broken gang planner that omits illicit money rules for route transfers."""
        plan = real_plan_gang(*args, **kwargs)
        import dataclasses
        return dataclasses.replace(plan, money_rules=())

    monkeypatch.setattr(interactive_runner, "plan_gang", broken_plan_gang)
    with pytest.raises(ValueError, match="has no illicit money rule"):
        run_interactive_world(TINY, RunnerSettings(fraud_enabled=True), gang_configs=(GANG_1,))
