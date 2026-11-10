# File: showcase/level_16_support.py
# Purpose: Level 16 showcase - builds the shared state once (recorded world, two replays, datasets, truth folder).
"""
WHAT: Records the committed Level 16 showcase world, replays it twice with the in-process registry wired, builds the
datasets at three cutoffs from replay A (and the matching hashes from replay B) and writes the truth folder. The 12
checks read this state; nothing is replayed twice for them.

WHY: A replay is the slow part of the showcase. Doing it once per check would multiply an hour of work by twelve.

TERM: cutoff: the world time a dataset is built at; it holds only what was known by then.
"""

from __future__ import annotations

import asyncio
import atexit
import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import redis

from core.config import config
from core.event_clock import ManualClock
from replay.dataset_builder import Dataset, WorldExtract, build_dataset, extract_world, write_dataset
from replay.pipeline import feed_with_labels
from replay.registry_driver import RegistryDriver, build_registry_driver
from replay.registry_replay import RegistryReplaySettings
from replay.runner import IST, ReplayDropContext, ReplayResult, run_replay
from showcase.isolation import get_showcase_database_url
from showcase.level_16_world import (
    GuaranteeCounts,
    guarantee_counts,
    level_16_fast_settings,
    level_16_gang,
    level_16_registry_settings,
    level_16_world_configs,
    load_level_16_config,
)
from world.bank_connectors import MockBankConnectors, load_bank_modes
from world.bank_feed import resolve_bank_id
from world.bank_feed_config import load_bank_feed_config
from world.interactive_records import RunnerSettings
from world.interactive_runner import prepare_world
from world.replay_recording import RecordedWorld, record_world_for_replay
from world.replay_workspace import ReplayWorkspace, workspace_redis_client
from world.truth_writer import write_truth_files

__all__ = ["Level16State", "get_state", "host_side_redis_url"]


def host_side_redis_url() -> str:
    """Redis connection string for the host machine (the same rule the replay tests use)."""
    raw = os.environ.get("TEST_REDIS_URL") or config.redis_url
    return raw.replace("://redis:", "://localhost:").replace("@redis:", "@localhost:")


def host_redis_factory(ws: ReplayWorkspace) -> redis.Redis:
    """A Redis client for the replay workspace's own database, reachable from the host."""
    return workspace_redis_client(ws, base_redis_url=host_side_redis_url())


class RejectingTransport:
    """A bank transport that must never be called: recording a world builds requests but sends none."""

    async def post(self, path: str, body_bytes: bytes, idempotency_key: str) -> Any:
        """Always refuses; recording must not reach a bank."""
        raise AssertionError("Transport should not be called")


@dataclass
class Level16State:
    """
    WHAT: Everything the 12 checks share.
    Args:
        cfg: Parsed config/showcase_level_16.yaml.
        recorded: The recorded world.
        registry: The showcase registry settings (bank_2 not in DPIP).
        driver: The registry plan for the world (bulk, holds, seed), used to say what labels are expected.
        counts: The guarantee counts of the planned world.
        extract_a / extract_b: The two replays' workspace extracts.
        result_a / result_b: The two replays' results (feature log hash, screen counts).
        cutoffs: c1 (just after the first label), c2 (middle label) and c3 (end of the world), as aware datetimes.
        datasets: name -> (Dataset, folder) for replay A at each cutoff.
        truth_root: Folder holding the world's truth files (read only through evaluation/dataset_truth_join.py).
        work_dir: Scratch folder for datasets written by checks.
        seconds: Wall seconds the setup took (record, two replays, datasets).
    """

    cfg: dict[str, Any]
    recorded: RecordedWorld
    registry: RegistryReplaySettings
    driver: RegistryDriver
    counts: GuaranteeCounts
    extract_a: WorldExtract
    extract_b: WorldExtract
    result_a: ReplayResult
    result_b: ReplayResult
    cutoffs: dict[str, datetime]
    datasets: dict[str, tuple[Dataset, Path]]
    truth_root: Path
    work_dir: Path
    seconds: float


_STATE: Level16State | None = None


async def _replay_once(recorded: RecordedWorld, registry: RegistryReplaySettings) -> tuple[WorldExtract, ReplayResult]:
    """Replays the recorded world once and extracts the workspace tables before the workspace is dropped."""
    feed = feed_with_labels(recorded)
    holder: list[WorldExtract] = []

    async def hook(ctx: ReplayDropContext) -> None:
        holder.append(
            await extract_world(
                ctx,
                recorded,
                {
                    "decisions_per_event_payment": True,
                    "registry_wired": ctx.registry_wired,
                    "registry_wired_reason": ctx.registry_wired_reason,
                },
            )
        )

    result = await run_replay(
        recorded,
        feed,
        admin_pg=get_showcase_database_url(),
        base_redis_url=host_side_redis_url(),
        redis_client_factory=host_redis_factory,
        decisions_per_event_payment=True,
        before_drop_hook=hook,
        registry_settings=registry,
    )
    return holder[0], result


async def _build_state() -> Level16State:
    """Records the world, replays it twice, builds the datasets and writes the truth folder."""
    began = time.monotonic()
    cfg = load_level_16_config()
    settings = level_16_fast_settings(cfg)
    configs = level_16_world_configs(cfg)
    registry = level_16_registry_settings(cfg)
    feed_cfg = load_bank_feed_config()
    modes = load_bank_modes()
    prepared = prepare_world(settings, configs)
    account_bank_map = {
        acc: resolve_bank_id(ledger.bank_index, feed_cfg)
        for ledger in prepared.world.ledgers
        for acc in ledger.accounts
    }
    connectors = MockBankConnectors(
        transports={b: RejectingTransport() for b in modes},
        modes=modes,
        account_bank_map=account_bank_map,
        clock_start=prepared.plan.clock.start_date,
        feed_config=feed_cfg,
        decision_timeout_seconds=5.0,
        loop=asyncio.get_running_loop(),
    )
    recorded = record_world_for_replay(
        settings,
        RunnerSettings(fraud_enabled=True),
        connectors,
        gang_configs=(level_16_gang(),),
        world_split=settings.range_name,
        configs=configs,
    )
    extract_a, result_a = await _replay_once(recorded, registry)
    extract_b, result_b = await _replay_once(recorded, registry)

    start = recorded.output.plan.clock.start_date
    start_date = start.astimezone(IST).date() if isinstance(start, datetime) else start
    midnight = datetime(start_date.year, start_date.month, start_date.day, tzinfo=IST)
    arrivals = sorted(r["arrival_time"] for r in extract_a.label_rows)
    cutoffs = {
        "c1": arrivals[0] + timedelta(minutes=1),
        "c2": arrivals[len(arrivals) // 2],
        "c3": midnight + timedelta(days=int(cfg["world"]["n_days"])),
    }
    work_dir = Path(tempfile.mkdtemp(prefix="showcase16_"))
    atexit.register(shutil.rmtree, work_dir, ignore_errors=True)
    split = str(cfg["world"]["range_name"])
    datasets: dict[str, tuple[Dataset, Path]] = {}
    for name, cutoff in cutoffs.items():
        dataset = build_dataset([extract_a], split=split, cutoff=cutoff)
        datasets[name] = (dataset, write_dataset(dataset, work_dir / "datasets"))
    truth_root = work_dir / "truth"
    write_truth_files(recorded.output, truth_root)
    registry_plan = build_registry_driver(recorded, ManualClock(), registry)
    return Level16State(
        cfg=cfg,
        recorded=recorded,
        registry=registry,
        driver=registry_plan,
        counts=guarantee_counts(recorded, registry),
        extract_a=extract_a,
        extract_b=extract_b,
        result_a=result_a,
        result_b=result_b,
        cutoffs=cutoffs,
        datasets=datasets,
        truth_root=truth_root,
        work_dir=work_dir,
        seconds=time.monotonic() - began,
    )


async def get_state() -> Level16State:
    """
    WHAT: Returns the shared state, building it on first use.
    WHY: The first check pays for the replays; the other eleven reuse them.
    Returns:
        The Level16State.
    Example:
        state = await get_state()
    """
    global _STATE
    if _STATE is None:
        _STATE = await _build_state()
    return _STATE
