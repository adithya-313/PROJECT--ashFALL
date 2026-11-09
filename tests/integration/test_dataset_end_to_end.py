# File: tests/integration/test_dataset_end_to_end.py
# Purpose: End-to-end integration test verifying that identical replays produce identical datasets.
"""
WHAT: Runs end-to-end historical replays twice on the small world and verifies dataset reproducibility.
WHY: Level 16c-2 / Architect decisions Q2-Q5:
  - Validates that extract_world captures workspace tables before drop.
  - Verifies that Replay A and Replay B produce identical content hashes at all cutoffs.
  - Verifies strict label progression across cutoffs (earlier labels < later labels).
  - Asserts payments table row count equals the total simulation event count.
  - Confirms manifest records registry_wired: true (screener wired) and accurate metadata.
"""

import asyncio
import dataclasses
import os
from collections.abc import Generator
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
import redis

from core.config import config
from core.config_validator import load_yaml
from replay.dataset_builder import WorldExtract, build_dataset, extract_world
from replay.pipeline import feed_with_labels
from replay.runner import IST, ReplayDropContext, run_replay
from tests.integration.conftest import get_test_database_url
from tests.integration.test_replay_recording import RejectingTransport
from world.bank_connectors import MockBankConnectors, load_bank_modes
from world.bank_feed import resolve_bank_id
from world.bank_feed_config import load_bank_feed_config
from world.gang_config import BlockStep, GangConfig, gang_config_from_plain
from world.interactive_records import RunnerSettings
from world.interactive_runner import FastRunSettings, prepare_world
from world.replay_recording import record_world_for_replay
from world.replay_workspace import ReplayWorkspace, workspace_redis_client

pytestmark = [pytest.mark.anyio, pytest.mark.needs_stack]

WORLD_SETTINGS = FastRunSettings(101, 20, 6, 51, "training")


def host_side_redis_url() -> str:
    """Resolve Redis connection string for test runner on host machine."""
    raw = os.environ.get("TEST_REDIS_URL") or config.redis_url
    return raw.replace("://redis:", "://localhost:").replace("@redis:", "@localhost:")


def host_redis_factory(ws: ReplayWorkspace) -> redis.Redis:
    """Constructs a Redis client pointing to host-accessible URL for the workspace DB."""
    return workspace_redis_client(ws, base_redis_url=host_side_redis_url())


@pytest.fixture
def live_redis() -> Generator[redis.Redis, None, None]:
    """Provides real Redis client connected to the host live database."""
    client = redis.Redis.from_url(host_side_redis_url())
    yield client
    client.close()


def _load_adapted_complaint_gang() -> GangConfig:
    """
    WHAT: Loads showcase/fixtures/level_12/gang_complaints.yaml adapted for the 20-person small world.
    WHY: Architect ruling: small world has 14 eligible victims; adapt gang to 1 receiving mule,
    12 victims, and family 'fast_pass_through' without modifying the fixture file on disk.
    """
    repo_root = Path(__file__).resolve().parents[2]
    gang_path = repo_root / "showcase" / "fixtures" / "level_12" / "gang_complaints.yaml"
    base_gang = gang_config_from_plain(load_yaml(gang_path))

    new_steps = []
    for s in base_gang.steps:
        if s.block == "scam":
            p = dict(s.params)
            p["receiving_mules"] = 1
            p["victims_per_receiving_mule"] = 12
            new_steps.append(BlockStep(block="scam", params=tuple(p.items())))
        else:
            new_steps.append(s)

    return dataclasses.replace(base_gang, family="fast_pass_through", steps=tuple(new_steps))


async def test_identical_replays_give_identical_datasets(
    live_redis: redis.Redis,
) -> None:
    """
    WHAT: Replays the small world simulation twice and verifies identical dataset content hashes.
    WHY: Implements Level 16 end-to-end reproducibility contract:
      - Replay A and Replay B produce identical canonical content hashes across cutoffs.
      - Earlier labels form a strict subset of later labels across cutoffs C1, C2, C3.
      - Total payments at C3 equals total simulation event count.
      - Manifest records registry_wired: true.
    """
    cfg = load_bank_feed_config()
    bank_modes = load_bank_modes()
    prepared = prepare_world(WORLD_SETTINGS, None)
    account_bank_map: dict[str, str] = {
        acc: resolve_bank_id(ledger.bank_index, cfg)
        for ledger in prepared.world.ledgers
        for acc in ledger.accounts
    }

    loop = asyncio.get_running_loop()
    connectors = MockBankConnectors(
        transports={b: RejectingTransport() for b in bank_modes},
        modes=bank_modes,
        account_bank_map=account_bank_map,
        clock_start=prepared.plan.clock.start_date,
        feed_config=cfg,
        decision_timeout_seconds=5.0,
        loop=loop,
    )
    runner = RunnerSettings(fraud_enabled=True)
    adapted_gang = _load_adapted_complaint_gang()
    recorded_world = record_world_for_replay(
        WORLD_SETTINGS, runner, connectors, gang_configs=(adapted_gang,)
    )
    feed = feed_with_labels(recorded_world)

    # 1. Derive complaints from projected feed
    all_complaints = [c for cmps in feed.complaints_by_bank.values() for c in cmps]
    same_bank_cmps = sorted(
        [c for c in all_complaints if c.reported_bank_id == c.bank_id],
        key=lambda c: c.received_at,
    )
    assert len(same_bank_cmps) >= 2, f"Expected >= 2 same-bank complaints, found {len(same_bank_cmps)}"
    assert same_bank_cmps[0].received_at < same_bank_cmps[1].received_at

    # 2. Derive three dataset cutoffs per architect ruling 10:
    # C1: Before first same-bank complaint
    # C2: Between the two complaints
    # C3: End of the world days
    t1 = same_bank_cmps[0].received_at
    t2 = same_bank_cmps[1].received_at
    cutoff_1 = t1 - timedelta(days=2)
    cutoff_2 = t1 + (t2 - t1) / 2
    cs = recorded_world.output.plan.clock.start_date
    sim_start_date = cs.astimezone(IST).date() if isinstance(cs, datetime) else cs
    cutoff_3 = datetime(
        sim_start_date.year, sim_start_date.month, sim_start_date.day, 0, 0, 0, tzinfo=IST
    ) + timedelta(days=60)

    # 3. First replay execution (Replay A)
    extract_a: WorldExtract | None = None

    async def hook_a(drop_ctx: ReplayDropContext) -> None:
        nonlocal extract_a
        extract_a = await extract_world(
            drop_ctx,
            recorded_world,
            {
                "decisions_per_event_payment": True,
                "registry_wired": drop_ctx.registry_wired,
                "registry_wired_reason": drop_ctx.registry_wired_reason,
            },
        )

    result_a = await run_replay(
        recorded_world,
        feed,
        admin_pg=get_test_database_url(),
        base_redis_url=host_side_redis_url(),
        redis_client_factory=host_redis_factory,
        decisions_per_event_payment=True,
        before_drop_hook=hook_a,
    )
    assert extract_a is not None

    # 4. Second replay execution (Replay B)
    extract_b: WorldExtract | None = None

    async def hook_b(drop_ctx: ReplayDropContext) -> None:
        nonlocal extract_b
        extract_b = await extract_world(
            drop_ctx,
            recorded_world,
            {
                "decisions_per_event_payment": True,
                "registry_wired": drop_ctx.registry_wired,
                "registry_wired_reason": drop_ctx.registry_wired_reason,
            },
        )

    result_b = await run_replay(
        recorded_world,
        feed,
        admin_pg=get_test_database_url(),
        base_redis_url=host_side_redis_url(),
        redis_client_factory=host_redis_factory,
        decisions_per_event_payment=True,
        before_drop_hook=hook_b,
    )
    assert extract_b is not None

    # 5. Build datasets at the three cutoffs for both replays
    ds_a1 = build_dataset([extract_a], split="training", cutoff=cutoff_1)
    ds_b1 = build_dataset([extract_b], split="training", cutoff=cutoff_1)

    ds_a2 = build_dataset([extract_a], split="training", cutoff=cutoff_2)
    ds_b2 = build_dataset([extract_b], split="training", cutoff=cutoff_2)

    ds_a3 = build_dataset([extract_a], split="training", cutoff=cutoff_3)
    ds_b3 = build_dataset([extract_b], split="training", cutoff=cutoff_3)

    # 6. Verify identical content hashes across replays
    assert ds_a1.manifest["content_hash"] == ds_b1.manifest["content_hash"]
    assert ds_a2.manifest["content_hash"] == ds_b2.manifest["content_hash"]
    assert ds_a3.manifest["content_hash"] == ds_b3.manifest["content_hash"]

    # 7. Verify strict label growth / subset progression
    def complaint_ids(dataset: Any) -> set[str]:
        """Label ids of the complaint-source labels only (registry labels are counted elsewhere)."""
        return {r["label_id"] for r in dataset.labels.to_pylist() if r["source"] == "complaint"}

    labels_c1 = complaint_ids(ds_a1)
    labels_c2 = complaint_ids(ds_a2)
    labels_c3 = complaint_ids(ds_a3)
    registry_c3 = [r for r in ds_a3.labels.to_pylist() if r["source"] == "registry"]
    print("REGISTRY LABELS IN DATASET AT C3:", len(registry_c3))

    assert labels_c1 < labels_c2 < labels_c3
    assert len(labels_c1) == 0
    assert len(labels_c2) == 1
    assert len(labels_c3) == len(same_bank_cmps)

    # 8. Verify payments row count matches simulation events
    events = recorded_world.output.result.run_result.events
    assert len(ds_a3.payments) == len(events)
    assert result_a.decisions_sent == len(events)
    assert result_b.decisions_sent == len(events)

    # 9. Verify manifest registry status
    assert ds_a3.manifest["registry_wired"] is True
    assert ds_b3.manifest["registry_wired"] is True
