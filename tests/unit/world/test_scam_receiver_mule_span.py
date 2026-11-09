"""
Tests that a scam's receiving mules get a 'mule' role span in the hidden truth (world/gang_planner.py), exactly one
per account, and that rented receivers join the owner-activity pool.

WHY: Before this fix a scam-funded gang wrote only 'victim' spans, so the accounts victims paid looked genuine in truth.

TERM: receiving mule: an acquired mule account that scam victims pay.
TERM: mule span: the truth role span [valid_from, valid_to) in which an account is a mule; valid_from is the minute
      the gang acquired the account.
TERM: owner activity: the share of a rented mule's owner payments that are still made inside its span.
"""

import asyncio
from pathlib import Path
from typing import Any

import numpy as np
import pytest

import world.gang_planner as gang_planner
import world.rented_owner_activity as rented_owner_activity
from core.config_validator import load_yaml
from showcase.level_12_checks_c import run_genuine_shop_downstream_check
from showcase.level_12_support_b import get_showcase_b_settings
from world.build_world import World, build_world
from world.gang_config import BlockStep, GangBudget, GangConfig, gang_config_from_plain
from world.gang_planner import plan_gang
from world.interactive_records import RunnerSettings
from world.interactive_runner import run_interactive_world
from world.rented_owner_activity import plan_owner_activity
from world.truth_writer import write_truth_files

BUDGET = GangBudget(money_paise=50_000_000, time_minutes=500_000, max_risk=0.99)
START_MINUTE = 1440
FIXTURE = Path(__file__).resolve().parents[3] / "showcase" / "fixtures" / "level_12" / "gang_complaints.yaml"


@pytest.fixture
def world() -> World:
    """Builds the small reproducible world (60 people, 12 shops) the scam tests use."""
    return build_world(seed=101, n_people=60, n_shops=12)


def scam_gang(route: bool, origin_mix: str | None = None) -> GangConfig:
    """Builds acquire(4) + scam(2 receivers), plus a two-layer route over the receivers when route is True."""
    acquire: tuple[tuple[str, Any], ...] = (("count", 4),)
    if origin_mix is not None:
        acquire += (("origin_mix", origin_mix),)
    steps = [
        BlockStep("acquire", acquire),
        BlockStep("scam", (("receiving_mules", 2), ("victims_per_receiving_mule", 2))),
    ]
    if route:
        steps.append(BlockStep("route", (("layers", 2), ("accounts_per_layer", 2))))
    return GangConfig("g1", 1, BUDGET, None, tuple(steps), START_MINUTE)


def plan_with_spy(config: GangConfig, world: World, monkeypatch: pytest.MonkeyPatch) -> tuple[Any, list[str], int]:
    """Plans the gang and returns (plan, acquired mules, acquisition minute) read off the acquire step."""
    seen: dict[str, Any] = {}
    original = gang_planner._plan_acquire_step

    def spy(*args: Any, **kwargs: Any) -> Any:
        result = original(*args, **kwargs)
        seen["mules"], seen["acq_min"] = list(result[0]), result[1]
        return result

    monkeypatch.setattr(gang_planner, "_plan_acquire_step", spy)
    plan = plan_gang(config, world, np.random.default_rng(8))
    return plan, seen["mules"], seen["acq_min"]


def mule_spans(plan: Any, account: str) -> list[Any]:
    """Lists the mule spans a plan's truth holds for one account."""
    return [r for r in plan.truth.roles if r.role == "mule" and r.account_id == account]


def test_complaints_fixture_receivers_are_mules_from_acquisition(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Checks each gang_complaints.yaml receiver has one open mule span from its acquisition minute at depth 1."""
    config = gang_config_from_plain(load_yaml(FIXTURE))
    plan, mules, acq_min = plan_with_spy(config, world, monkeypatch)
    for receiver in mules[:2]:
        (span,) = mule_spans(plan, receiver)
        assert (span.valid_from, span.valid_to, span.depth) == (acq_min, None, 1)
        assert span.active_from is not None and span.burned_from is not None


def test_scam_plus_route_gang_has_exactly_one_span_per_receiver(
    world: World, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Checks a gang with both a scam and a route never gets a second mule span for any account."""
    plan, mules, _ = plan_with_spy(scam_gang(route=True), world, monkeypatch)
    for receiver in mules[:2]:
        assert len(mule_spans(plan, receiver)) == 1
    accounts = [r.account_id for r in plan.truth.roles if r.role == "mule"]
    assert len(accounts) == len(set(accounts))


def test_same_seed_twice_gives_identical_truth_files(tmp_path: Path) -> None:
    """Runs the same seed twice and checks roles.jsonl and every other truth file are byte-identical."""
    settings, code_version = get_showcase_b_settings()
    dirs = []
    for name in ("a", "b"):
        output = run_interactive_world(
            fast_settings=settings, runner_settings=RunnerSettings(fraud_enabled=True),
            gang_configs=(scam_gang(route=False),), code_version=code_version,
        )
        dirs.append(write_truth_files(output, tmp_path / name))
    for file_name in ("roles.jsonl", "device_control.jsonl", "payment_stories.jsonl", "gangs.jsonl", "manifest.json"):
        assert (dirs[0] / file_name).read_bytes() == (dirs[1] / file_name).read_bytes()


def test_rented_scam_receivers_now_have_owner_activity(monkeypatch: pytest.MonkeyPatch) -> None:
    """Checks rented scam receivers get a kept share drawn, and with a kept share of 0 their owner payments pause."""
    monkeypatch.setattr(rented_owner_activity, "get_rented_owner_kept_share_range", lambda split: (0.0, 0.0))
    settings, code_version = get_showcase_b_settings()
    output = run_interactive_world(
        fast_settings=settings, runner_settings=RunnerSettings(fraud_enabled=True),
        gang_configs=(scam_gang(route=False, origin_mix="all_rented"),), code_version=code_version,
    )
    roles = output.result.truth.roles
    receivers = {r.account_id for r in roles if r.role == "mule"}
    assert receivers and all(r.origin == "rented" for r in roles if r.role == "mule")
    activity = plan_owner_activity(roles, "training", np.random.default_rng(1))
    assert receivers <= set(activity)
    assert any(p.from_account in receivers for p in output.result.paused_owner_payments)


def test_genuine_shop_that_received_mule_money_stays_genuine(world: World, monkeypatch: pytest.MonkeyPatch) -> None:
    """Checks no shop gets a mule span, and the existing showcase check still sees its cash-out shop as genuine."""
    plan, _, _ = plan_with_spy(scam_gang(route=True), world, monkeypatch)
    assert not [r for r in plan.truth.roles if r.account_id.startswith("acc_shop_")]
    outcome = asyncio.run(run_genuine_shop_downstream_check(None))
    assert outcome.evidence["role_at"] == "genuine"
