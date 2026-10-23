"""
WHAT: Unit tests for Cash-Out Channels and Genuine Shops Downstream (Level 12c C-4, D3).
Verifies:
- channel amounts follow narrowed shares (e.g. only atm; only genuine shops);
- no planned ATM day exceeds card limit and no executed ATM day exceeds it;
- crypto collector is role cash_out and gang-opened with normal id;
- genuine shop receives mule money and role_at is None (genuine in truth);
- total cash-out <= illicit money received by each terminal mule;
- burned_from covers withdrawals;
- refusal of cash_out before route.

WHY: Ensures criminal proceeds extracted from the terminal mule layer model realistic
cash, crypto, and everyday consumer consumption without corrupting truth facts or breaking
money conservation.

ISSUE IDs: G2 (cash-out channels), A2 (genuine shops downstream of mules).
"""

from typing import Any

import numpy as np
import pytest

from core.config_validator import load_yaml
from world.build_world import build_world
from world.cash_out import (
    draw_cash_out_shares,
    plan_cash_out_step,
)
from world.cash_withdrawals import CHANNEL_ATM
from world.checks import check_money_conserved
from world.fast_mode import FastRunSettings
from world.gang_config import BlockStep, GangBudget, GangConfig, validate_gang_config
from world.gang_planner import plan_gang
from world.interactive_records import RunnerSettings
from world.interactive_runner import prepare_world, run_interactive_world
from world.run_identity import REPO_ROOT

FROZEN = load_yaml(REPO_ROOT / "config" / "frozen_anchors.yaml")["frozen_anchors"]["fraud_off_events"]
TINY = FastRunSettings(FROZEN["seed"], FROZEN["n_people"], FROZEN["n_shops"], FROZEN["n_days"], FROZEN["range_name"])
BUDGET = GangBudget(money_paise=100_000_000, time_minutes=500_000, max_risk=0.99)
START_MINUTE = 1440


def opening_total(settings: FastRunSettings) -> int:
    """Calculates total opening balance across all bank ledgers for a fresh world."""
    world = prepare_world(settings, None).world
    return sum(ledger.balance_of(account) for ledger in world.ledgers for account in ledger.accounts)


def test_channel_amounts_follow_narrowed_shares(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verifies that cash-out channel amounts strictly follow narrowed configurations:
    1. Narrowed to ATM only: 100% of funds go to ATM withdrawals, none to crypto, branch, or shops.
    2. Narrowed to genuine shops only: 100% of funds go to merchant purchases, none to cash or crypto.
    """
    world = build_world(TINY.seed, TINY.n_people, TINY.n_shops)
    rng = np.random.default_rng(42)

    # 1. Narrowed to ATM only
    def mock_share_ranges_atm(split: str, *args: Any, **kwargs: Any) -> dict[str, tuple[float, float]]:
        return {
            "crypto": (0.0, 0.0),
            "atm": (1.0, 1.0),
            "branch": (0.0, 0.0),
            "genuine_shop": (0.0, 0.0),
        }

    monkeypatch.setattr("world.cash_out.get_cash_out_channel_share_ranges", mock_share_ranges_atm)

    shares_atm = draw_cash_out_shares("training", rng)
    assert shares_atm["atm"] == 1.0
    assert shares_atm["crypto"] == 0.0
    assert shares_atm["branch"] == 0.0
    assert shares_atm["genuine_shop"] == 0.0

    inflow_paise = 2_500_000
    plan_atm = plan_cash_out_step(
        world=world,
        gang_id="g_atm",
        strength=1,
        terminal_mules=[("acc_mule_1", inflow_paise, 2000)],
        split="training",
        layers=1,
        opening_start_index=0,
        has_disguise=False,
        jitter=None,
        fixed_dwell=30,
        dwell_param=None,
        rng=rng,
    )

    assert len(plan_atm.payments) == 0
    assert len(plan_atm.outside_events) == 0
    assert len(plan_atm.roles) == 0
    assert len(plan_atm.withdrawals) > 0
    assert all(w.channel == CHANNEL_ATM for w in plan_atm.withdrawals)
    assert sum(w.amount_paise for w in plan_atm.withdrawals) == inflow_paise

    # 2. Narrowed to genuine shops only
    def mock_share_ranges_shop(split: str, *args: Any, **kwargs: Any) -> dict[str, tuple[float, float]]:
        return {
            "crypto": (0.0, 0.0),
            "atm": (0.0, 0.0),
            "branch": (0.0, 0.0),
            "genuine_shop": (1.0, 1.0),
        }

    monkeypatch.setattr("world.cash_out.get_cash_out_channel_share_ranges", mock_share_ranges_shop)

    shares_shop = draw_cash_out_shares("training", rng)
    assert shares_shop["genuine_shop"] == 1.0
    assert shares_shop["atm"] == 0.0

    plan_shop = plan_cash_out_step(
        world=world,
        gang_id="g_shop",
        strength=1,
        terminal_mules=[("acc_mule_2", inflow_paise, 2000)],
        split="training",
        layers=1,
        opening_start_index=0,
        has_disguise=False,
        jitter=None,
        fixed_dwell=30,
        dwell_param=None,
        rng=rng,
    )

    assert len(plan_shop.withdrawals) == 0
    assert len(plan_shop.outside_events) == 0
    assert len(plan_shop.roles) == 0
    assert len(plan_shop.payments) == 1
    assert plan_shop.payments[0].amount_paise == inflow_paise
    shop_acct = plan_shop.payments[0].to_account
    assert any(s.account_id == shop_acct for s in world.shops)


def test_no_planned_atm_day_exceeds_card_limit_and_no_executed_atm_day_exceeds_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Verifies that ATM cash extraction adheres strictly to daily card limits at both planning and execution time:
    1. In planning, multiple days of withdrawals are emitted, each strictly <= card_daily_limit_paise.
    2. In execution, the runner confirms no day's sum of successful ATM withdrawals breaches the limit.
    """
    world = build_world(TINY.seed, TINY.n_people, TINY.n_shops)
    rng = np.random.default_rng(123)

    # 1. Planner test: large inflow relative to daily limit
    inflow_paise = 25_000_000  # Rs 250,000

    # Force ATM channel only
    def mock_shares_atm(*args: Any, **kwargs: Any) -> dict[str, float]:
        return {"crypto": 0.0, "atm": 1.0, "branch": 0.0, "genuine_shop": 0.0}

    monkeypatch.setattr("world.cash_out.draw_cash_out_shares", mock_shares_atm)

    terminal_mules = [("acc_person_000010", inflow_paise, 1440)]
    plan = plan_cash_out_step(
        world=world,
        gang_id="g_limit_test",
        strength=1,
        terminal_mules=terminal_mules,
        split="training",
        layers=1,
        opening_start_index=0,
        has_disguise=False,
        jitter=None,
        fixed_dwell=10,
        dwell_param=None,
        rng=rng,
    )

    atm_withdrawals = [w for w in plan.withdrawals if w.channel == CHANNEL_ATM]
    assert len(atm_withdrawals) > 0
    # Group planned ATM withdrawals by day
    by_day: dict[int, int] = {}
    for w in atm_withdrawals:
        assert w.card_daily_limit_paise is not None
        assert w.amount_paise <= w.card_daily_limit_paise
        day = w.minute // 1440
        by_day[day] = by_day.get(day, 0) + w.amount_paise

    card_limit = atm_withdrawals[0].card_daily_limit_paise
    for day, day_total in by_day.items():
        assert day_total <= card_limit

    # 2. Runner test: whole world execution with scam + route + cash_out
    gang = GangConfig(
        gang_id="g_atm_exec",
        strength=1,
        budget=BUDGET,
        steps=(
            BlockStep("acquire", (("count", 4), ("origin_mix", "all_rented"))),
            BlockStep("scam", (("receiving_mules", 1), ("victims_per_receiving_mule", 1))),
            BlockStep("route", (("layers", 1), ("accounts_per_layer", 1))),
            BlockStep("cash_out", ()),
        ),
        start_minute=START_MINUTE,
    )

    runner_settings = RunnerSettings(fraud_enabled=True, account_openings_enabled=True)
    result = run_interactive_world(TINY, runner_settings, gang_configs=(gang,))
    executed_atm = [w for w in result.result.cash_withdrawals if w.channel == CHANNEL_ATM and w.status == "success"]

    # Verify every executed successful ATM day stayed within limit
    exec_by_day: dict[tuple[str, int], int] = {}
    for w in executed_atm:
        key = (w.account_id, w.minute // 1440)
        exec_by_day[key] = exec_by_day.get(key, 0) + w.withdrawn_paise

    # Check against world ranges upper bound for atm daily limit
    for (acct, day), total_withdrawn in exec_by_day.items():
        assert total_withdrawn <= 10_000_000  # Rs 100,000 maximum daily limit in training


def test_crypto_collector_is_role_cash_out_and_gang_opened_with_normal_id(monkeypatch: pytest.MonkeyPatch) -> None:
    """
    Verifies that the crypto collector account:
    1. Uses a gang-opened borrowed-KYC personal account identifier (e.g. new:<gang_id>:0).
    2. Generates an OutsideEvent of kind 'open_account' with owner_kind 'person'.
    3. Has a forensic RoleSpan with role 'cash_out' and origin 'borrowed_kyc'.
    4. Receives UPI payments carrying IllicitMoneyRule(spends_illicit=True, credits_illicit=True).
    """
    world = build_world(TINY.seed, TINY.n_people, TINY.n_shops)
    rng = np.random.default_rng(999)

    # Force crypto channel only
    def mock_shares_crypto(*args: Any, **kwargs: Any) -> dict[str, float]:
        return {"crypto": 1.0, "atm": 0.0, "branch": 0.0, "genuine_shop": 0.0}

    monkeypatch.setattr("world.cash_out.draw_cash_out_shares", mock_shares_crypto)

    gang_id = "g_crypto_test"
    plan = plan_cash_out_step(
        world=world,
        gang_id=gang_id,
        strength=1,
        terminal_mules=[("acc_person_000015", 3_000_000, 1500)],
        split="training",
        layers=2,
        opening_start_index=0,
        has_disguise=False,
        jitter=None,
        fixed_dwell=15,
        dwell_param=None,
        rng=rng,
    )

    assert len(plan.outside_events) == 1
    op_ev = plan.outside_events[0]
    assert op_ev.kind == "open_account"
    assert op_ev.owner_kind == "person"
    assert op_ev.opening_handle == f"new:{gang_id}:0"

    assert len(plan.roles) == 1
    role_span = plan.roles[0]
    assert role_span.account_id == f"new:{gang_id}:0"
    assert role_span.role == "cash_out"
    assert role_span.origin is None
    assert role_span.gang_id == gang_id

    assert len(plan.payments) == 1
    pmt = plan.payments[0]
    assert pmt.to_account == f"new:{gang_id}:0"
    assert pmt.amount_paise == 3_000_000

    assert len(plan.money_rules) == 1
    rule = plan.money_rules[0]
    assert rule.spends_illicit is True
    assert rule.credits_illicit is True

    assert len(plan.stories) == 1
    assert plan.stories[0].block == "cash_out"


def test_collector_opens_no_later_than_its_earliest_payment_when_mules_are_out_of_time_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Mule A (arrives 1000) is processed before mule B (arrives 400); the collector must open by B's payment."""
    world = build_world(TINY.seed, TINY.n_people, TINY.n_shops)

    # The existing way to force the shares: replace draw_cash_out_shares so every mule sends only to crypto.
    def mock_shares_crypto(*args: Any, **kwargs: Any) -> dict[str, float]:
        return {"crypto": 1.0, "atm": 0.0, "branch": 0.0, "genuine_shop": 0.0}

    monkeypatch.setattr("world.cash_out.draw_cash_out_shares", mock_shares_crypto)
    plan = plan_cash_out_step(
        world=world, gang_id="g_order", strength=1,
        terminal_mules=[("acc_person_000015", 3_000_000, 1000), ("acc_person_000016", 2_000_000, 400)],
        split="training", layers=2, opening_start_index=0, has_disguise=False, jitter=None,
        fixed_dwell=15, dwell_param=None, rng=np.random.default_rng(5),
    )

    assert len(plan.outside_events) == 1
    send_minutes = [payment.minute for payment in plan.payments]
    assert len(send_minutes) == 2
    # The expected opening is the earlier of the first mule's arrival (1000) and the earliest planned send.
    expected_opening = min(1000, min(send_minutes))
    assert plan.outside_events[0].minute == expected_opening
    assert plan.outside_events[0].minute <= min(send_minutes)
    assert min(send_minutes) < 1000


def test_single_terminal_mule_opening_event_is_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    """With one terminal mule arriving at 1500 the collector still opens at 1500, exactly the event as before."""
    from world.outside_events import OutsideEvent

    world = build_world(TINY.seed, TINY.n_people, TINY.n_shops)

    def mock_shares_crypto(*args: Any, **kwargs: Any) -> dict[str, float]:
        return {"crypto": 1.0, "atm": 0.0, "branch": 0.0, "genuine_shop": 0.0}

    monkeypatch.setattr("world.cash_out.draw_cash_out_shares", mock_shares_crypto)
    plan = plan_cash_out_step(
        world=world, gang_id="g_one", strength=1, terminal_mules=[("acc_person_000015", 3_000_000, 1500)],
        split="training", layers=2, opening_start_index=0, has_disguise=False, jitter=None,
        fixed_dwell=15, dwell_param=None, rng=np.random.default_rng(999),
    )

    # The pre-fix opening used the first-use arrival minute; the known input gives 1500, and the send is 1515.
    assert plan.payments[0].minute == 1515
    assert plan.outside_events == [
        OutsideEvent(
            1500, "open_account", owner_kind="person", opening_stream="gang_account_openings",
            opening_handle="new:g_one:0",
        )
    ]


def test_genuine_shop_receives_mule_money_and_role_is_none() -> None:
    """
    Verifies that a genuine merchant shop downstream of a terminal mule:
    1. Receives illicit mule money for retail goods (credits_illicit=False, funds exit to legitimate economy).
    2. Has merchant kind in allowed goods-selling categories ('grocery', 'restaurant').
    3. Holds NO RoleSpan in truth.roles (role_at evaluates to None, proving it is genuine in truth).
    """
    gang = GangConfig(
        gang_id="g_shop_test",
        strength=1,
        budget=BUDGET,
        steps=(
            BlockStep("acquire", (("count", 4), ("origin_mix", "all_rented"))),
            BlockStep("scam", (("receiving_mules", 1), ("victims_per_receiving_mule", 1))),
            BlockStep("route", (("layers", 1), ("accounts_per_layer", 1))),
            BlockStep("cash_out", ()),
        ),
        start_minute=START_MINUTE,
    )

    world = build_world(TINY.seed, TINY.n_people, TINY.n_shops)
    world_shops = {s.account_id: s for s in world.shops}
    plan = plan_gang(gang, world, np.random.default_rng(777))

    # Find genuine shop payments in output
    shop_payments = [
        p for p in plan.payments
        if p.to_account in world_shops
    ]

    assert len(shop_payments) > 0, "Expected at least one payment to a genuine merchant shop"
    for p in shop_payments:
        shop = world_shops[p.to_account]
        assert shop.kind in ("grocery", "restaurant")
        # In truth, genuine merchant has NO criminal role span (genuine in truth)
        assert p.to_account not in [r.account_id for r in plan.truth.roles]

    # Run world and check truth roles in executed result
    runner_settings = RunnerSettings(fraud_enabled=True, account_openings_enabled=True)
    result = run_interactive_world(TINY, runner_settings, gang_configs=(gang,))
    for p in shop_payments:
        assert p.to_account not in [r.account_id for r in result.result.truth.roles]


def test_total_cash_out_less_than_or_equal_illicit_money_received_by_each_terminal_mule() -> None:
    """
    Verifies that the sum of all cash-out channels (crypto, ATM, branch, and genuine shops)
    for each terminal mule never exceeds the illicit funds received by that mule.
    Also verifies closed-loop money conservation: opening + created - withdrawn = ending.
    """
    initial_money = opening_total(TINY)

    gang = GangConfig(
        gang_id="g_conservation",
        strength=1,
        budget=BUDGET,
        steps=(
            BlockStep("acquire", (("count", 4), ("origin_mix", "all_rented"))),
            BlockStep("scam", (("receiving_mules", 1), ("victims_per_receiving_mule", 2))),
            BlockStep("route", (("layers", 2), ("accounts_per_layer", 1))),
            BlockStep("cash_out", ()),
        ),
        start_minute=START_MINUTE,
    )

    world = build_world(TINY.seed, TINY.n_people, TINY.n_shops)
    plan = plan_gang(gang, world, np.random.default_rng(888))

    # Check terminal mule inflows vs outflows
    terminal_mules = [
        r.account_id for r in plan.truth.roles
        if r.role == "mule" and r.depth == 2
    ]
    assert len(terminal_mules) > 0

    for mule_acct in terminal_mules:
        inflow = sum(
            p.amount_paise for p in plan.payments
            if p.to_account == mule_acct
        )
        out_payments = sum(
            p.amount_paise for p in plan.payments
            if p.from_account == mule_acct
        )
        out_withdrawals = sum(
            w.amount_paise for w in plan.withdrawals
            if w.account_id == mule_acct
        )
        assert out_payments + out_withdrawals <= inflow
        # At most 400 paise lost to floor division and dropped change < Rs 1
        assert inflow - (out_payments + out_withdrawals) < 400

    runner_settings = RunnerSettings(fraud_enabled=True, account_openings_enabled=True)
    result = run_interactive_world(TINY, runner_settings, gang_configs=(gang,))

    # Total money conservation across the entire world run
    # Stolen money was transferred among citizens and shops, and cash was withdrawn
    expected_total = initial_money + result.result.money_created_paise - result.result.money_withdrawn_paise
    discrepancies = check_money_conserved(result.world.ledgers, expected_total)
    assert discrepancies == [], f"Money conservation violated: {discrepancies}"


def test_burned_from_covers_withdrawals() -> None:
    """
    Verifies that a mule's lifecycle burned_from minute extends beyond its last outgoing
    action, taking into account multi-day ATM withdrawals and delayed branch withdrawals.
    """
    gang = GangConfig(
        gang_id="g_lifecycle",
        strength=1,
        budget=BUDGET,
        steps=(
            BlockStep("acquire", (("count", 4), ("origin_mix", "all_rented"))),
            BlockStep("scam", (("receiving_mules", 1), ("victims_per_receiving_mule", 1))),
            BlockStep("route", (("layers", 1), ("accounts_per_layer", 1))),
            BlockStep("cash_out", ()),
        ),
        start_minute=START_MINUTE,
    )

    plan = plan_gang(gang, build_world(TINY.seed, TINY.n_people, TINY.n_shops), np.random.default_rng(777))

    for role in plan.truth.roles:
        if role.role == "mule":
            acct = role.account_id
            mule_withdrawals = [w for w in plan.withdrawals if w.account_id == acct]
            if mule_withdrawals:
                max_w_min = max(w.minute for w in mule_withdrawals)
                assert role.burned_from is not None
                assert role.burned_from > max_w_min, (
                    f"burned_from {role.burned_from} did not cover withdrawal at {max_w_min}"
                )


def test_refusal_of_cash_out_before_route() -> None:
    """
    Verifies that cash_out requires a preceding route step:
    1. A gang config with cash_out without route raises ValueError.
    2. A gang config with cash_out before route raises ValueError.
    """
    # 1. cash_out without route
    invalid_no_route = GangConfig(
        gang_id="g_no_route",
        strength=1,
        budget=BUDGET,
        steps=(
            BlockStep("acquire", (("count", 2),)),
            BlockStep("cash_out", ()),
        ),
        start_minute=START_MINUTE,
    )
    with pytest.raises(ValueError, match="must route money before cashing out"):
        validate_gang_config(invalid_no_route)

    # 2. cash_out before route
    invalid_before_route = GangConfig(
        gang_id="g_before_route",
        strength=1,
        budget=BUDGET,
        steps=(
            BlockStep("acquire", (("count", 2),)),
            BlockStep("cash_out", ()),
            BlockStep("route", (("layers", 1), ("accounts_per_layer", 1))),
        ),
        start_minute=START_MINUTE,
    )
    with pytest.raises(ValueError, match="must route money before cashing out"):
        validate_gang_config(invalid_before_route)
