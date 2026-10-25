"""
Tests that a rented mule's owner keeps part of their normal life (world/rented_owner_activity.py and the pause step
in world/interactive_loop.py). Covers the kept-share draw and its config, the half-open mule span, and whole worlds
that compare a run with pausing against the same plan without it.

TERM: rented mule: a mule account whose ordinary owner still has a planned routine life.
TERM: kept share: the chance that one outgoing owner payment inside the mule span is still made.
TERM: paused payment: a genuine payment from a rented mule inside its span that is not made.
TERM: baseline run: the same gang with no origin recorded, so nothing is paused.
"""

from pathlib import Path
from typing import Any

import numpy as np
import pytest
import yaml

import world.rented_owner_activity as rented_owner_activity
from showcase.level_12_support_b import get_showcase_b_settings
from world.gang_config import BlockStep, GangBudget, GangConfig
from world.interactive_records import RunnerSettings
from world.interactive_runner import run_interactive_world
from world.rented_owner_activity import OwnerActivity, plan_owner_activity
from world.tactic_config import TACTIC_BLOCKS_PATH, WORLD_RANGES_PATH, get_rented_owner_kept_share_range
from world.truth_records import RoleSpan

BUDGET = GangBudget(money_paise=50_000_000, time_minutes=500_000, max_risk=0.99)
START_MINUTE = 1440


def make_gang(origin_mix: str | None) -> GangConfig:
    """Builds a scam gang with four mules and a two-layer route; origin_mix None leaves the parameter out."""
    acquire_params: tuple[tuple[str, Any], ...] = (("count", 4),)
    if origin_mix is not None:
        acquire_params += (("origin_mix", origin_mix),)
    steps = (
        BlockStep("acquire", acquire_params),
        BlockStep("scam", (("receiving_mules", 2), ("victims_per_receiving_mule", 2))),
        BlockStep("route", (("layers", 2), ("accounts_per_layer", 2))),
    )
    return GangConfig("g1", 1, BUDGET, None, steps, START_MINUTE)


def run_with_kept_share(
    monkeypatch: pytest.MonkeyPatch | None, kept_share: tuple[float, float] | None, origin_mix: str | None
) -> Any:
    """Runs the gang, optionally with the kept share range narrowed to the given pair."""
    if monkeypatch is not None and kept_share is not None:
        monkeypatch.setattr(rented_owner_activity, "get_rented_owner_kept_share_range", lambda split: kept_share)
    settings, code_version = get_showcase_b_settings()
    return run_interactive_world(
        fast_settings=settings, runner_settings=RunnerSettings(fraud_enabled=True),
        gang_configs=(make_gang(origin_mix),), code_version=code_version,
    )


def planned_events(output: Any) -> list[Any]:
    """Lists the events of genuine (planned) payments in execution order."""
    result = output.result
    return [event for event, source in zip(result.run_result.events, result.event_sources) if source == "planned"]


def mule_accounts(output: Any) -> set[str]:
    """Returns the accounts holding a mule role in a run's truth."""
    return {role.account_id for role in output.result.truth.roles if role.role == "mule"}


@pytest.fixture(scope="module")
def baseline() -> Any:
    """Runs the gang with no origin recorded, which pauses nothing; the other runs are compared with it."""
    return run_with_kept_share(None, None, None)


# ---- the draw and its config ----------------------------------------------------------------------------------


def span_roles() -> list[RoleSpan]:
    """Builds role spans in unsorted order: two rented mules, one business mule, one victim and one unmarked mule."""
    return [
        RoleSpan("acc_b", "mule", "g", 1, 100, 200, "rented"),
        RoleSpan("acc_a", "mule", "g", 1, 100, None, "rented"),
        RoleSpan("acc_c", "mule", "g", 2, 100, None, "business"),
        RoleSpan("acc_d", "victim", "g", None, 100, None),
        RoleSpan("acc_e", "mule", "g", 2, 100, None),
    ]


def test_only_rented_mules_get_a_kept_share_drawn_in_sorted_account_order() -> None:
    """Checks one draw per rented mule in sorted order, with the span recorded and other roles left out."""
    activity = plan_owner_activity(span_roles(), "training", np.random.default_rng(5))

    assert list(activity) == ["acc_a", "acc_b"]
    replay = np.random.default_rng(5)
    expected_first = float(replay.uniform(0.3, 0.8))
    expected_second = float(replay.uniform(0.3, 0.8))
    assert activity["acc_a"] == OwnerActivity(expected_first, 100, None)
    assert activity["acc_b"] == OwnerActivity(expected_second, 100, 200)


def test_no_rented_mule_reads_nothing_from_the_stream() -> None:
    """Checks that with no rented mule the result is empty and the generator state is untouched."""
    rng = np.random.default_rng(5)
    before = rng.bit_generator.state

    assert plan_owner_activity([RoleSpan("acc_c", "mule", "g", 1, 0, None, "business")], "training", rng) == {}
    assert rng.bit_generator.state == before


def test_several_spans_for_one_rented_account_share_one_kept_share() -> None:
    """Checks two non-overlapping spans of one rented account share one kept-share draw and covers both spans."""
    roles = [
        RoleSpan("acc_a", "mule", "g", 1, 0, 10, "rented"),
        RoleSpan("acc_a", "mule", "g", 1, 20, 30, "rented"),
    ]
    rng = np.random.default_rng(5)
    activity = plan_owner_activity(roles, "training", rng)

    single_role = [RoleSpan("acc_a", "mule", "g", 1, 0, 10, "rented")]
    single_rng = np.random.default_rng(5)
    single_activity = plan_owner_activity(single_role, "training", single_rng)

    assert activity["acc_a"].kept_share == single_activity["acc_a"].kept_share
    assert rng.bit_generator.state == single_rng.bit_generator.state

    act = activity["acc_a"]
    assert act.covers(0) is True
    assert act.covers(5) is True
    assert act.covers(9) is True
    assert act.covers(10) is False
    assert act.covers(15) is False
    assert act.covers(19) is False
    assert act.covers(20) is True
    assert act.covers(25) is True
    assert act.covers(29) is True
    assert act.covers(30) is False


def test_overlapping_spans_for_one_rented_account_are_refused() -> None:
    """Checks overlapping spans of one rented account raise a ValueError because intervals must not collide."""
    roles = [
        RoleSpan("acc_a", "mule", "g", 1, 0, 15, "rented"),
        RoleSpan("acc_a", "mule", "g", 1, 10, 25, "rented"),
    ]
    with pytest.raises(ValueError, match="overlapping mule spans"):
        plan_owner_activity(roles, "training", np.random.default_rng(5))


def test_span_is_half_open() -> None:
    """Checks the span includes its first minute and excludes the minute it ends, and an open span never ends."""
    activity = OwnerActivity(0.5, 100, 200)

    assert [activity.covers(minute) for minute in (99, 100, 199, 200)] == [False, True, True, False]
    assert OwnerActivity(0.5, 100, None).covers(10**9)


def test_kept_share_range_follows_narrowed_config(tmp_path: Path) -> None:
    """Narrows the world range and the reference and checks the resolver and the draw both follow."""
    assert get_rented_owner_kept_share_range("training") == (0.3, 0.8)
    ranges = yaml.safe_load(WORLD_RANGES_PATH.read_text(encoding="utf-8"))
    ranges["parameters"]["rented_mule_owner_activity_kept_share"]["training"] = [0.9, 0.95]
    ranges["parameters"]["scam_payments_per_victim"]["training"] = [1, 6]
    ranges_path = tmp_path / "world_ranges.yaml"
    ranges_path.write_text(yaml.safe_dump(ranges), encoding="utf-8")
    assert get_rented_owner_kept_share_range("training", None, ranges_path) == (0.9, 0.95)

    blocks = yaml.safe_load(TACTIC_BLOCKS_PATH.read_text(encoding="utf-8"))
    blocks["rented_owner_activity"]["kept_share"]["world_ranges_reference"] = "scam_payments_per_victim"
    blocks_path = tmp_path / "tactic_blocks.yaml"
    blocks_path.write_text(yaml.safe_dump(blocks), encoding="utf-8")
    assert get_rented_owner_kept_share_range("training", blocks_path) == (1.0, 6.0)


def test_kept_share_config_refusals(tmp_path: Path) -> None:
    """Checks a missing reference, an unknown parameter and a missing split are each refused in one sentence."""
    blocks = yaml.safe_load(TACTIC_BLOCKS_PATH.read_text(encoding="utf-8"))
    blocks_path = tmp_path / "tactic_blocks.yaml"
    blocks["rented_owner_activity"]["kept_share"]["world_ranges_reference"] = "nope"
    blocks_path.write_text(yaml.safe_dump(blocks), encoding="utf-8")
    with pytest.raises(ValueError, match="unknown world_ranges parameter 'nope'"):
        get_rented_owner_kept_share_range("training", blocks_path)

    del blocks["rented_owner_activity"]["kept_share"]["world_ranges_reference"]
    blocks_path.write_text(yaml.safe_dump(blocks), encoding="utf-8")
    with pytest.raises(ValueError, match="must name a world_ranges_reference"):
        get_rented_owner_kept_share_range("training", blocks_path)

    with pytest.raises(ValueError, match="no range for split 'nope'"):
        get_rented_owner_kept_share_range("nope")


# ---- whole worlds ---------------------------------------------------------------------------------------------


def test_baseline_pauses_nothing(baseline: Any) -> None:
    """Checks a gang with no recorded origin pauses nothing, so earlier gangs behave exactly as before."""
    assert baseline.result.paused_owner_payments == []


def test_full_kept_share_pauses_nothing_and_changes_no_event(
    monkeypatch: pytest.MonkeyPatch, baseline: Any
) -> None:
    """Checks a kept share of 1 pauses nothing and leaves every genuine event equal to the baseline's."""
    output = run_with_kept_share(monkeypatch, (1.0, 1.0), "all_rented")

    assert output.result.paused_owner_payments == []
    assert planned_events(output) == planned_events(baseline)


def test_zero_kept_share_pauses_every_outgoing_payment_inside_the_span_and_none_outside(
    monkeypatch: pytest.MonkeyPatch, baseline: Any
) -> None:
    """Checks a kept share of 0 pauses all outgoing mule payments inside the span and none outside."""
    output = run_with_kept_share(monkeypatch, (0.0, 0.0), "all_rented")
    base_events = planned_events(baseline)
    mule_spans = {
        role.account_id: (role.valid_from, role.burned_from)
        for role in output.result.truth.roles
        if role.role == "mule"
    }

    inside = [
        e for e in base_events
        if e.from_account in mule_spans
        and mule_spans[e.from_account][0] <= e.minute < (mule_spans[e.from_account][1] or float("inf"))
    ]
    outside = [
        e for e in base_events
        if e.from_account in mule_spans
        and (
            e.minute < mule_spans[e.from_account][0]
            or (mule_spans[e.from_account][1] is not None and e.minute >= mule_spans[e.from_account][1])
        )
    ]
    assert inside and outside

    # At least one baseline owner payment has minute >= burned_from, and none of those is paused
    after_burned = [
        e for e in base_events
        if e.from_account in mule_spans
        and mule_spans[e.from_account][1] is not None
        and e.minute >= mule_spans[e.from_account][1]
    ]
    assert after_burned
    paused_set = {
        (p.minute, p.from_account, p.to_account, p.amount_paise)
        for p in output.result.paused_owner_payments
    }
    assert not any(
        (e.minute, e.from_account, e.to_account, e.amount_paise) in paused_set
        for e in after_burned
    )

    paused_totals: dict[tuple[int, str, str], int] = {}
    for paused in output.result.paused_owner_payments:
        assert paused.from_account in mule_spans
        span_from, span_burned = mule_spans[paused.from_account]
        assert span_from <= paused.minute < (span_burned or float("inf"))
        key = (paused.minute, paused.from_account, paused.to_account)
        paused_totals[key] = paused_totals.get(key, 0) + paused.amount_paise
    base_totals: dict[tuple[int, str, str], int] = {}
    for event in inside:
        key = (event.minute, event.from_account, event.to_account)
        base_totals[key] = base_totals.get(key, 0) + event.amount_paise
    assert paused_totals == base_totals

    new_events = planned_events(output)
    inside_new = [
        e for e in new_events
        if e.from_account in mule_spans
        and mule_spans[e.from_account][0] <= e.minute < (mule_spans[e.from_account][1] or float("inf"))
    ]
    assert not inside_new
    outside_new = [
        e for e in new_events
        if e.from_account in mule_spans
        and (
            e.minute < mule_spans[e.from_account][0]
            or (mule_spans[e.from_account][1] is not None and e.minute >= mule_spans[e.from_account][1])
        )
    ]
    assert outside_new == outside


def test_incoming_payments_to_the_owner_are_never_paused(monkeypatch: pytest.MonkeyPatch, baseline: Any) -> None:
    """Checks every genuine payment into a rented mule from another account is still made, as in the baseline."""
    output = run_with_kept_share(monkeypatch, (0.0, 0.0), "all_rented")
    mules = mule_accounts(output)

    def incoming(run: Any) -> list[Any]:
        """Lists genuine events paid to a mule by an account that is not a mule."""
        return [e for e in planned_events(run) if e.to_account in mules and e.from_account not in mules]

    assert incoming(baseline)
    assert incoming(output) == incoming(baseline)
    assert not [p for p in output.result.paused_owner_payments if p.to_account in mules and p.from_account not in mules]


def drawn_outcome(event: Any) -> tuple[Any, ...]:
    """
    Reduces an event to what the random failure draws decided: who paid whom, when, how much, and the drawn failure
    (reason and domain) if one was drawn. Running out of balance is not a draw, it follows from money flow, so it
    counts as "no drawn failure" exactly like a success.
    """
    drawn_failure = event.status == "failed" and event.reason != "insufficient_balance"
    reason = (event.reason, event.failure_domain) if drawn_failure else None
    return (event.minute, event.from_account, event.to_account, event.amount_paise, reason)


def test_pausing_does_not_shift_the_failure_draws_of_other_genuine_payments(
    monkeypatch: pytest.MonkeyPatch, baseline: Any
) -> None:
    """
    Compares every genuine payment not sent by a mule with the baseline. Their drawn failures must be identical, which
    proves pausing left Level 11's draws where they were. A payment can still run out of balance because a paused
    payment no longer moves money to its receiver, so insufficient balance is treated as money flow, not as a draw.
    """
    output = run_with_kept_share(monkeypatch, (0.0, 0.0), "all_rented")
    mules = mule_accounts(output)

    with_pausing = [drawn_outcome(e) for e in planned_events(output) if e.from_account not in mules]
    in_baseline = [drawn_outcome(e) for e in planned_events(baseline) if e.from_account not in mules]

    assert with_pausing == in_baseline
    assert len(in_baseline) > 1000
    assert any(outcome[4] is not None for outcome in in_baseline)


def test_real_range_pauses_some_payments_and_keeps_others(baseline: Any) -> None:
    """Checks with the committed range that some owner payments are paused and some are still made."""
    output = run_with_kept_share(None, None, "all_rented")
    mule_spans = {
        role.account_id: (role.valid_from, role.burned_from)
        for role in output.result.truth.roles
        if role.role == "mule"
    }
    kept = [
        e for e in planned_events(output)
        if e.from_account in mule_spans
        and mule_spans[e.from_account][0] <= e.minute < (mule_spans[e.from_account][1] or float("inf"))
    ]

    assert output.result.paused_owner_payments
    assert kept


def test_same_seed_gives_the_same_paused_list() -> None:
    """Checks two runs of the same world and gang pause exactly the same payments."""
    first = run_with_kept_share(None, None, "all_rented")
    second = run_with_kept_share(None, None, "all_rented")

    assert first.result.paused_owner_payments == second.result.paused_owner_payments
