# File: tests/unit/world/test_sampled_world_hook.py
# Purpose: Unit tests for sampled world recording hooks, gang resolution, and RecordedWorld metadata.
"""
WHAT: Unit tests verifying hook interception and recording for sampled worlds.
WHY: Level 16c Round 2a (Decisions D3, D4):
- resolve_sampled_world_gangs factors gang resolution out of run_sampled_world.
- run_sampled_world accepts an intercepting hook while preserving default event equality.
- record_sampled_world_for_replay captures calls and populates world_id, split, seed, gang_families.
- Live bank modes are strictly refused.
Smallest sampler spec used: WorldSpec drawn from sample_worlds('training', 1, 6, ...) with preset 'tiny'
(from tests/unit/world/test_sampled_world.py:81-83).
"""

import asyncio
import dataclasses
from pathlib import Path
from typing import Any

import pytest

from core.config_validator import load_yaml
from world.bank_connectors import MockBankConnectors, load_bank_modes
from world.bank_feed import resolve_bank_id
from world.bank_feed_config import load_bank_feed_config
from world.gang_config import GangConfig, gang_config_from_plain
from world.interactive_records import HookDecision, PaymentView
from world.interactive_runner import prepare_world
from world.replay_recording import record_sampled_world_for_replay
from world.sampled_world import (
    fast_settings_for,
    pinned_configs,
    resolve_sampled_world_gangs,
    run_sampled_world,
)
from world.world_configs import WorldConfigs, read_world_configs
from world.world_sampler import WorldSpec, load_sampler_config, sample_worlds

REPO_ROOT = Path(__file__).resolve().parents[3]
GANG_PATH = REPO_ROOT / "showcase" / "fixtures" / "level_12" / "gang_complaints.yaml"
PRESET = "tiny"
CODE_VERSION = "test-sampled-world-hook"
SAMPLER_SEED = 6


def _gang() -> GangConfig:
    """Loads the complaints gang fixture matching test_sampled_world.py."""
    return gang_config_from_plain(load_yaml(GANG_PATH))


class FakeTransport:
    """In-memory fake transport for MockBankConnectors."""

    async def post(self, path: str, body_bytes: bytes, idempotency_key: str) -> Any:
        return None


@pytest.fixture(scope="module")
def cfg() -> dict[str, Any]:
    """Reads the sampler config once for the module."""
    return load_sampler_config()


@pytest.fixture(scope="module")
def base() -> WorldConfigs:
    """Reads the world configs once for the module."""
    return read_world_configs()


@pytest.fixture(scope="module")
def spec(cfg: dict[str, Any], base: WorldConfigs) -> WorldSpec:
    """Plans one training world with the sampler using the smallest spec in existing tests."""
    return sample_worlds("training", 1, SAMPLER_SEED, cfg, base.world_ranges)[0]


def test_resolve_sampled_world_gangs(spec: WorldSpec, cfg: dict[str, Any]) -> None:
    """
    Verifies that resolve_sampled_world_gangs returns the same gangs as before
    for an explicit tuple and for a generated mix (captured baseline).
    """
    configs = pinned_configs(spec, cfg)

    # 1. Explicit tuple
    explicit_gang = _gang()
    gangs_explicit = resolve_sampled_world_gangs(spec, PRESET, (explicit_gang,), cfg, configs)
    assert gangs_explicit == (explicit_gang,)

    # 2. Generated mix (captured baseline on unchanged code: 3 gangs, all layered_chain)
    gangs_generated = resolve_sampled_world_gangs(spec, PRESET, None, cfg, configs)
    assert len(gangs_generated) == 3
    assert [g.family for g in gangs_generated] == ["layered_chain", "layered_chain", "layered_chain"]

    # 3. Empty tuple is rejected
    with pytest.raises(ValueError, match="a sampled world is a fraud-on world"):
        resolve_sampled_world_gangs(spec, PRESET, (), cfg, configs)


def test_run_sampled_world_with_recording_hook_matches_default(
    spec: WorldSpec, cfg: dict[str, Any]
) -> None:
    """
    Verifies that run_sampled_world with a recording hook sees every planned payment
    and returns the same output as with the default (full event equality).
    """
    gang = _gang()

    # Default run with allow_all
    default_output = run_sampled_world(spec, PRESET, (gang,), cfg, CODE_VERSION)

    # Run with recording hook
    recorded_payments: list[PaymentView] = []

    def recording_hook(payment: PaymentView) -> HookDecision:
        recorded_payments.append(payment)
        return "allow"

    hook_output = run_sampled_world(
        spec, PRESET, (gang,), cfg, CODE_VERSION, hook=recording_hook
    )

    # Full event equality (as in the 16b-3a fix)
    assert len(recorded_payments) == len(default_output.result.planned_payment_ids)
    assert (
        hook_output.result.run_result.events == default_output.result.run_result.events
    )
    assert (
        hook_output.result.planned_payment_ids
        == default_output.result.planned_payment_ids
    )


def test_record_sampled_world_for_replay_fills_metadata_and_gang_families(
    spec: WorldSpec, cfg: dict[str, Any]
) -> None:
    """
    Verifies that record_sampled_world_for_replay fills world_id, split, seed,
    and gang_families matching resolved gangs.
    """
    configs = pinned_configs(spec, cfg)
    settings = fast_settings_for(spec, PRESET, cfg, configs)
    prepared = prepare_world(settings, configs)
    feed_cfg = load_bank_feed_config()
    bank_modes = load_bank_modes()

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

    explicit_gang = dataclasses.replace(_gang(), family="fast_pass_through")
    rec = record_sampled_world_for_replay(
        spec, PRESET, (explicit_gang,), cfg, CODE_VERSION, connectors
    )

    assert rec.world_id == spec.world_id
    assert rec.split == spec.split
    assert rec.seed == spec.seed
    assert rec.gang_families == ("fast_pass_through",)
    assert len(rec.calls) == len(rec.output.result.planned_payment_ids)


def test_record_sampled_world_for_replay_refuses_live_bank_mode(
    spec: WorldSpec, cfg: dict[str, Any]
) -> None:
    """Verifies that record_sampled_world_for_replay refuses any live bank mode."""
    configs = pinned_configs(spec, cfg)
    settings = fast_settings_for(spec, PRESET, cfg, configs)
    prepared = prepare_world(settings, configs)
    feed_cfg = load_bank_feed_config()

    account_bank_map: dict[str, str] = {}
    for ledger in prepared.world.ledgers:
        b_id = resolve_bank_id(ledger.bank_index, feed_cfg)
        for acc in ledger.accounts:
            account_bank_map[acc] = b_id

    live_modes = {"bank_0": "live", "bank_1": "shadow"}
    loop = asyncio.new_event_loop()
    connectors = MockBankConnectors(
        transports={b: FakeTransport() for b in live_modes},
        modes=live_modes,
        account_bank_map=account_bank_map,
        clock_start=prepared.plan.clock.start_date,
        feed_config=feed_cfg,
        decision_timeout_seconds=2.0,
        loop=loop,
    )

    with pytest.raises(ValueError, match="Live bank mode refused until Level 19"):
        record_sampled_world_for_replay(
            spec, PRESET, (_gang(),), cfg, CODE_VERSION, connectors
        )
