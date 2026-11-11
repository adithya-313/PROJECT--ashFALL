# File: tests/unit/world/test_mid_run_account_region.py
# Purpose: Unit tests proving mid-run opened accounts carry regions, inherit correctly, and join connector map.
"""
WHAT: Unit tests verifying mid-run opened accounts in replay and feed projection.
WHY: Guarantees that genuine mid-run accounts inherit regions authentically (person inherits
own home region, shop keeps drawn region) and join the connector map only at or after their opening minute.
"""

import asyncio
from datetime import date
from typing import Any

import numpy as np
import pytest

from world.bank_connectors import MockBankConnectors, load_bank_modes
from world.bank_feed import project_run, resolve_bank_id
from world.bank_feed_config import load_bank_feed_config
from world.interactive_records import PaymentView
from world.interactive_runner import prepare_world
from world.interactive_state import start_runner_state
from world.regions import assign_region
from world.replay_recording import record_sampled_world_for_replay
from world.sampled_world import fast_settings_for, pinned_configs
from world.world_configs import read_world_configs
from world.world_sampler import WorldSpec, load_sampler_config, sample_worlds


class FakeTransport:
    """In-memory fake transport for MockBankConnectors."""

    async def post(self, path: str, body_bytes: bytes, idempotency_key: str) -> Any:
        return None


def test_failing_sampled_small_preset_world_records_and_projects() -> None:
    """
    Test (a): Verifies that the exact sampled small-preset world that previously failed
    (split='training', seed=1000000, world_id='training-0000') records and projects without error.
    """
    sampler_cfg = load_sampler_config()
    configs = read_world_configs()
    specs = sample_worlds(
        split="training",
        count=1,
        sampler_seed=1,
        cfg=sampler_cfg,
        world_ranges=configs.world_ranges,
    )
    spec: WorldSpec = specs[0]
    assert spec.world_id == "training-0000"
    assert spec.seed == 1000000

    preset_name = "small"
    feed_cfg = load_bank_feed_config()
    bank_modes = load_bank_modes()
    spec_configs = pinned_configs(spec, sampler_cfg)
    fast_settings = fast_settings_for(spec, preset_name, sampler_cfg, spec_configs)
    prepared = prepare_world(fast_settings, spec_configs)

    account_bank_map: dict[str, str] = {}
    for ledger in prepared.world.ledgers:
        b_id = resolve_bank_id(ledger.bank_index, feed_cfg)
        for acc in ledger.accounts:
            account_bank_map[acc] = b_id

    loop = asyncio.new_event_loop()
    connectors = MockBankConnectors(
        transports={b: FakeTransport() for b in bank_modes},
        modes=bank_modes,
        account_bank_map=account_bank_map,
        clock_start=prepared.plan.clock.start_date,
        feed_config=feed_cfg,
        decision_timeout_seconds=2.0,
        loop=loop,
    )

    recorded = record_sampled_world_for_replay(
        spec=spec,
        preset_name=preset_name,
        gang_configs=None,
        sampler_cfg=sampler_cfg,
        code_version="test-mid-run-16d4",
        connectors=connectors,
    )

    opened = recorded.output.result.opened_accounts
    assert len(opened) > 0, "World training-0000 must produce >= 1 opened account"
    for row in opened:
        assert len(row) == 7, f"Opened account row must have 7 elements, got {len(row)}: {row}"
        assert isinstance(row[6], int), f"Region must be int, got {type(row[6])}"

    feed = project_run(recorded.output)
    opened_events_by_account = {
        ev.account_id: ev
        for evs in feed.account_events_by_bank.values()
        for ev in evs
        if ev.kind == "account_opened"
    }
    for row in opened:
        acc_id = row[1]
        assert acc_id in opened_events_by_account
        assert opened_events_by_account[acc_id].details["home_region"] == row[6]


def test_person_inherits_own_region_and_shop_keeps_drawn_region() -> None:
    """
    Test (b): Verifies person account opening inherits person's own region,
    and shop account opening retains the drawn region.
    """
    sampler_cfg = load_sampler_config()
    configs = read_world_configs()
    specs = sample_worlds(
        split="training",
        count=1,
        sampler_seed=1,
        cfg=sampler_cfg,
        world_ranges=configs.world_ranges,
    )
    spec: WorldSpec = specs[0]
    spec_configs = pinned_configs(spec, sampler_cfg)
    fast_settings = fast_settings_for(spec, "small", sampler_cfg, spec_configs)
    prepared = prepare_world(fast_settings, spec_configs)
    state = start_runner_state(prepared.world)

    # 1. Person opening: verify inheritance of person.region
    rng_p = np.random.default_rng(12345)
    person_acc = state.open_account("person", "genuine_account_openings", 100, rng_p)
    person_record = state.opened_accounts[-1]
    assert person_record[1] == person_acc
    assert person_record[2] == "person"
    expected_person_region = state.people_by_account[person_acc].region
    assert person_record[6] == expected_person_region

    # 2. Shop opening: verify that drawn region is kept
    rng_s1 = np.random.default_rng(54321)
    # Mirror shop RNG draw sequence:
    # 1: determine_shop_kind(shop_idx, rng)
    # 2: assign_region(n_regions, rng)
    rng_mirror = np.random.default_rng(54321)
    from world.interactive_state import determine_shop_kind
    _ = determine_shop_kind(state.next_shop_index, rng_mirror)
    expected_shop_region = assign_region(int(prepared.world.parameters_drawn["n_regions"]), rng_mirror)

    shop_acc = state.open_account("shop", "genuine_account_openings", 200, rng_s1)
    shop_record = state.opened_accounts[-1]
    assert shop_record[1] == shop_acc
    assert shop_record[2] == "shop"
    assert shop_record[6] == expected_shop_region


def test_payment_refused_before_opening_minute_allowed_at_or_after() -> None:
    """
    Test (c): Verifies that payment to/from a new account before its opening minute is refused,
    and becomes resolvable and allowed at or after its opening minute.
    """
    feed_cfg = load_bank_feed_config()
    bank_modes = load_bank_modes()
    account_map = {"acc_person_initial": "bank_0"}

    opened_records = [
        (100, "acc_person_new", "person", "genuine_account_openings", 0, 500000, 1),
        (200, "acc_shop_new", "shop", "genuine_account_openings", 1, 500000, 2),
    ]

    loop = asyncio.new_event_loop()
    connectors = MockBankConnectors(
        transports={b: FakeTransport() for b in bank_modes},
        modes=bank_modes,
        account_bank_map=account_map,
        clock_start=date(2026, 10, 28),
        feed_config=feed_cfg,
        decision_timeout_seconds=2.0,
        loop=loop,
        opened_accounts=opened_records,
    )

    # 1. Before minute 100: payment to/from acc_person_new must be refused
    pmt_early_receiver = PaymentView("plan_001", 50, "acc_person_initial", "acc_person_new", 10000)
    with pytest.raises(ValueError, match="Receiver account 'acc_person_new' is not in account_bank_map"):
        connectors.build_decision_request(pmt_early_receiver)

    pmt_early_sender = PaymentView("plan_002", 50, "acc_person_new", "acc_person_initial", 10000)
    with pytest.raises(ValueError, match="Sender account 'acc_person_new' is not in account_bank_map"):
        connectors.build_decision_request(pmt_early_sender)

    # 2. At minute 100: acc_person_new joins the connector map and succeeds
    pmt_exact_receiver = PaymentView("plan_003", 100, "acc_person_initial", "acc_person_new", 10000)
    call_recv = connectors.build_decision_request(pmt_exact_receiver)
    assert call_recv.receiver_bank_id == "bank_0"

    pmt_exact_sender = PaymentView("plan_004", 100, "acc_person_new", "acc_person_initial", 10000)
    call_send = connectors.build_decision_request(pmt_exact_sender)
    assert call_send.sender_bank_id == "bank_0"

    # 3. Minute 150: acc_shop_new (opens at 200) is still unknown
    pmt_shop_early = PaymentView("plan_005", 150, "acc_person_initial", "acc_shop_new", 20000)
    with pytest.raises(ValueError, match="Receiver account 'acc_shop_new' is not in account_bank_map"):
        connectors.build_decision_request(pmt_shop_early)

    # 4. Minute 200: acc_shop_new joins connector map and succeeds
    pmt_shop_exact = PaymentView("plan_006", 200, "acc_person_initial", "acc_shop_new", 20000)
    call_shop = connectors.build_decision_request(pmt_shop_exact)
    assert call_shop.receiver_bank_id == resolve_bank_id(1, feed_cfg)
