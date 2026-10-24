"""
WHAT: Unit tests verifying slow drip route planning, piece splitting, timing dispersion, and money conservation.

WHY: Level 12g slow_drip moves money in small pieces below ₹2,000 spread over calendar days with multi-hour
holding windows. These tests verify mathematical exactness in piece splitting, fail-closed enforcement when piece
limits are breached, and full-chain money conservation across both scam-funded and fixture-funded campaigns.

ISSUE IDs: 12-8
"""

import numpy as np
import pytest

from world.build_world import build_world
from world.fast_mode import FastRunSettings
from world.gang_config import BlockStep, FixtureSource, GangBudget, GangConfig
from world.gang_planner import _plan_drip_chain, plan_gang, split_into_pieces
from world.tactic_config import get_slow_drip_max_pieces

TEST_BUDGET = GangBudget(money_paise=50_000_000, time_minutes=500_000, max_risk=0.99)
TEST_SOURCE = FixtureSource("acc_gang_src_01", 0, 10_000_000, 1440)
TEST_SETTINGS = FastRunSettings(101, 60, 12, 61, "training")
MINUTES_PER_DAY = 1440


def test_split_into_pieces_hand_worked_examples() -> None:
    """Verifies that split_into_pieces partitions sums accurately with exact single-paisa integer conservation."""
    # Hand-worked example from prompt: 500000 paise with max 199900 -> [166667, 166667, 166666].
    pieces = split_into_pieces(500000, 199900)
    assert pieces == [166667, 166667, 166666]
    assert sum(pieces) == 500000
    assert all(p <= 199900 for p in pieces)

    # Clean exact divisor case: 400000 paise with max 200000 -> [200000, 200000].
    clean_pieces = split_into_pieces(400000, 200000)
    assert clean_pieces == [200000, 200000]
    assert sum(clean_pieces) == 400000

    # Single piece case where total is below max.
    single_piece = split_into_pieces(150000, 199900)
    assert single_piece == [150000]

    # Refusal on non-positive amounts.
    with pytest.raises(ValueError, match=r"total_paise must be positive"):
        split_into_pieces(0, 199900)
    with pytest.raises(ValueError, match=r"total_paise must be positive"):
        split_into_pieces(-500, 199900)
    with pytest.raises(ValueError, match=r"piece_max_paise must be positive"):
        split_into_pieces(100000, 0)


def test_planned_scam_funded_drip_gang_in_test_world() -> None:
    """Verifies timing bounds, piece size thresholds, and exact money conservation for scam-funded slow drip."""
    world = build_world(TEST_SETTINGS.seed, TEST_SETTINGS.n_people, TEST_SETTINGS.n_shops)
    rng = np.random.default_rng(42)

    spread_days = 4
    hold_minutes = 300
    piece_max_inr = 1500
    piece_max_paise = piece_max_inr * 100

    steps = (
        BlockStep("acquire", (("count", 6),)),
        BlockStep("scam", (("receiving_mules", 2), ("victims_per_receiving_mule", 2))),
        BlockStep(
            "route",
            (
                ("layers", 3),
                ("accounts_per_layer", 2),
                ("drip_piece_max_inr", piece_max_inr),
                ("drip_spread_days", spread_days),
                ("drip_hold_minutes", hold_minutes),
            ),
        ),
    )
    gang = GangConfig("scam_drip_gang", 1, TEST_BUDGET, None, steps, 1440)
    plan = plan_gang(gang, world, rng, world_split="test")

    # 1. Separate scam payments from route payments by story block.
    scam_pmts = [p for p, s in zip(plan.payments, plan.truth.payment_stories) if s.block == "scam"]
    route_pmts = [p for p, s in zip(plan.payments, plan.truth.payment_stories) if s.block == "route"]
    assert len(route_pmts) > 0

    # 2. Assert every route payment is below Rs 2,000 (amount_paise < 200000).
    assert all(p.amount_paise < 200000 for p in route_pmts)
    assert all(p.amount_paise <= piece_max_paise for p in route_pmts)

    # 3. Assert money conservation: terminal mule inflows equal layer 1 inflows.
    receiving_mules = {
        role.account_id for role in plan.truth.roles if role.role == "mule" and role.depth == 1
    }
    terminal_mules = {
        role.account_id for role in plan.truth.roles if role.role == "mule" and role.depth == 3
    }
    assert len(receiving_mules) == 2
    assert len(terminal_mules) == 2

    for rec_mule in receiving_mules:
        inflow = sum(p.amount_paise for p in scam_pmts if p.to_account == rec_mule) - sum(
            p.amount_paise for p in scam_pmts if p.from_account == rec_mule
        )
        outflow_from_rec = sum(p.amount_paise for p in route_pmts if p.from_account == rec_mule)
        assert outflow_from_rec == inflow

    total_scam_net = sum(
        sum(p.amount_paise for p in scam_pmts if p.to_account == m)
        - sum(p.amount_paise for p in scam_pmts if p.from_account == m)
        for m in receiving_mules
    )
    total_terminal_inflow = sum(
        sum(p.amount_paise for p in route_pmts if p.to_account == t) for t in terminal_mules
    )
    assert total_terminal_inflow == total_scam_net

    # 4. Assert timing rules: first-hop pieces span at least (spread_days - 1) days.
    first_hop_pmts = [p for p in route_pmts if p.from_account in receiving_mules]
    first_send = min(p.minute for p in first_hop_pmts)
    last_first_send = max(p.minute for p in first_hop_pmts)
    assert (last_first_send - first_send) >= (spread_days - 1) * MINUTES_PER_DAY

    # 5. Assert hop dwell and departure sequencing: no piece leaves before it arrived.
    # Trace per-piece forwards from layer 1 -> layer 2 -> layer 3.
    layer2_mules = {
        role.account_id for role in plan.truth.roles if role.role == "mule" and role.depth == 2
    }
    for l2_mule in layer2_mules:
        in_arrivals = [p.minute for p in route_pmts if p.to_account == l2_mule]
        out_departures = [p.minute for p in route_pmts if p.from_account == l2_mule]
        assert len(in_arrivals) == len(out_departures)
        for arr, dep in zip(sorted(in_arrivals), sorted(out_departures)):
            assert dep - arr >= hold_minutes
            assert dep - arr > 60


def test_planned_fixture_funded_drip_gang_in_test_world() -> None:
    """Verifies timing bounds, piece size thresholds, and exact money conservation for fixture-funded slow drip."""
    world = build_world(TEST_SETTINGS.seed, TEST_SETTINGS.n_people, TEST_SETTINGS.n_shops)
    rng = np.random.default_rng(42)

    spread_days = 5
    hold_minutes = 400
    piece_max_inr = 1200
    piece_max_paise = piece_max_inr * 100

    steps = (
        BlockStep("acquire", (("count", 4),)),
        BlockStep(
            "route",
            (
                ("layers", 2),
                ("accounts_per_layer", 2),
                ("drip_piece_max_inr", piece_max_inr),
                ("drip_spread_days", spread_days),
                ("drip_hold_minutes", hold_minutes),
            ),
        ),
    )
    gang = GangConfig("fix_drip_gang", 1, TEST_BUDGET, TEST_SOURCE, steps)
    plan = plan_gang(gang, world, rng, world_split="test")

    route_pmts = plan.payments
    assert len(route_pmts) > 0

    # 1. Assert every route payment is below Rs 2,000.
    assert all(p.amount_paise < 200000 for p in route_pmts)
    assert all(p.amount_paise <= piece_max_paise for p in route_pmts)

    # 2. Assert money conservation: total dispatched from fixture source equals total received at terminal mules.
    terminal_mules = {
        role.account_id for role in plan.truth.roles if role.role == "mule" and role.depth == 2
    }
    source_outflow = sum(p.amount_paise for p in route_pmts if p.from_account == TEST_SOURCE.account_id)
    terminal_inflow = sum(p.amount_paise for p in route_pmts if p.to_account in terminal_mules)
    assert source_outflow == terminal_inflow

    # 3. Assert first-hop disbursement span >= (spread_days - 1) days.
    first_hop_pmts = [p for p in route_pmts if p.from_account == TEST_SOURCE.account_id]
    first_send = min(p.minute for p in first_hop_pmts)
    last_first_send = max(p.minute for p in first_hop_pmts)
    assert (last_first_send - first_send) >= (spread_days - 1) * MINUTES_PER_DAY

    # 4. Assert each mule forward occurs at arrival + hold_minutes.
    layer1_mules = {
        role.account_id for role in plan.truth.roles if role.role == "mule" and role.depth == 1
    }
    for l1_mule in layer1_mules:
        in_arrivals = [p.minute for p in route_pmts if p.to_account == l1_mule]
        out_departures = [p.minute for p in route_pmts if p.from_account == l1_mule]
        assert len(in_arrivals) == len(out_departures)
        for arr, dep in zip(sorted(in_arrivals), sorted(out_departures)):
            assert dep - arr == hold_minutes
            assert dep - arr > 60


def test_too_many_pieces_refused_with_piece_count_and_limit() -> None:
    """Verifies that _plan_drip_chain raises ValueError naming the gang, piece count n, and max_pieces limit."""
    max_pieces = get_slow_drip_max_pieces()
    # 5,000,000 paise with piece_max_inr 50 -> piece_max_paise 5000 -> 1000 pieces > 400.
    with pytest.raises(ValueError) as excinfo:
        _plan_drip_chain(
            gang_id="gang_overflow",
            chain_accounts=["src", "m1", "m2"],
            total_paise=5_000_000,
            first_send_minute=100,
            piece_max_inr=50,
            spread_days=5,
            hold_minutes=300,
            max_pieces=max_pieces,
        )
    message = str(excinfo.value)
    assert "gang_overflow" in message
    assert "1000" in message
    assert str(max_pieces) in message


def test_scam_funded_drip_layer_1_outgoing_leaves_after_last_victim_payment() -> None:
    """Verifies that each layer-1 mule's first outgoing piece leaves >= its last victim payment + drip_hold_minutes."""
    steps = (
        BlockStep("acquire", (("count", 4),)),
        BlockStep("scam", (("receiving_mules", 2), ("victims_per_receiving_mule", 2))),
        BlockStep(
            "route",
            (
                ("layers", 2),
                ("accounts_per_layer", 2),
                ("drip_piece_max_inr", 1500),
                ("drip_spread_days", 3),
                ("drip_hold_minutes", 300),
            ),
        ),
    )
    gang = GangConfig("g_scam_drip", 1, TEST_BUDGET, None, steps, 1440)
    world = build_world(seed=101, n_people=60, n_shops=12)
    rng = np.random.default_rng(42)
    plan = plan_gang(gang, world, rng, world_split="test")

    scam_payments = [p for p, s in zip(plan.payments, plan.truth.payment_stories) if s.block == "scam"]
    layer_1_mules = [r.account_id for r in plan.truth.roles if r.role == "mule" and r.depth == 1]
    assert len(layer_1_mules) == 2

    for mule in layer_1_mules:
        m_victim_pmts = [p for p in scam_payments if p.to_account == mule]
        assert m_victim_pmts, f"Layer-1 mule {mule} received no victim payments"
        last_victim_minute = max(p.minute for p in m_victim_pmts)

        outgoing_pieces = [
            p for p, s in zip(plan.payments, plan.truth.payment_stories)
            if s.block == "route" and p.from_account == mule
        ]
        assert outgoing_pieces, f"Layer-1 mule {mule} has no outgoing route pieces"
        first_outgoing_minute = min(p.minute for p in outgoing_pieces)

        assert first_outgoing_minute >= last_victim_minute + 300


def test_total_smaller_than_one_piece_gives_exactly_one_piece_at_first_send_minute() -> None:
    """Verifies that when total money is smaller than piece_max_inr, exactly 1 piece is planned at first_send_minute."""
    # total_paise = 50,000 (Rs 500) with piece_max_inr = 1500 (Rs 1500)
    chain = ["acc_mule_0", "acc_mule_1"]
    pmts, stories, span = _plan_drip_chain(
        gang_id="g_small",
        chain_accounts=chain,
        total_paise=50_000,
        first_send_minute=1000,
        piece_max_inr=1500,
        spread_days=3,
        hold_minutes=300,
        max_pieces=400,
    )
    assert len(pmts) == 1
    assert pmts[0].amount_paise == 50_000
    assert pmts[0].minute == 1000
    assert len(stories) == 1
    assert span == 0
