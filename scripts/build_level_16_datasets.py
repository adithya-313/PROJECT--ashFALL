"""
File: scripts/build_level_16_datasets.py
Purpose: Rebuilds immutable point-in-time datasets across all configured simulation splits
         (training, validation, test, stress) using recorded sampled world replays,
         extracts workspace tables, writes datasets with Parquet and manifests, and marks
         any older datasets of the same split as superseded.
"""

import argparse
import asyncio
import os
import time
from collections.abc import Sequence
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import patch

import orjson
import redis
import yaml

from core.config import config
from replay.dataset_builder import (
    Dataset,
    WorldExtract,
    build_dataset,
    extract_world,
    mark_superseded,
    write_dataset,
)
from replay.pipeline import feed_with_labels
from replay.runner import IST, ReplayDropContext, run_replay
from showcase.framework import get_code_version
from showcase.isolation import get_showcase_database_url
from world import interactive_state as ist
from world.bank_connectors import MockBankConnectors, load_bank_modes
from world.bank_feed import resolve_bank_id
from world.bank_feed_config import load_bank_feed_config
from world.config_reader import read_world_ranges
from world.interactive_runner import prepare_world
from world.replay_recording import RecordedWorld, record_sampled_world_for_replay
from world.replay_workspace import ReplayWorkspace, workspace_redis_client
from world.sampled_world import fast_settings_for, pinned_configs
from world.world_sampler import WorldSpec, load_sampler_config, sample_worlds

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "datasets.yaml"
DEFAULT_DATASETS_ROOT = REPO_ROOT / "data" / "datasets"


def host_side_redis_url() -> str:
    """Redis connection string for the host machine."""
    raw = os.environ.get("TEST_REDIS_URL") or config.redis_url
    return raw.replace("://redis:", "://localhost:").replace("@redis:", "@localhost:")


def host_redis_factory(ws: ReplayWorkspace) -> redis.Redis:
    """Creates a host-accessible Redis client for an isolated replay workspace."""
    return workspace_redis_client(ws, base_redis_url=host_side_redis_url())


class RejectingTransport:
    """A bank transport that refuses requests; recording runs in allow-all mode without external bank traffic."""

    async def post(self, path: str, body_bytes: bytes, idempotency_key: str) -> Any:
        raise AssertionError("Bank transport should not be invoked during recording")


def load_datasets_config(path: Path | str | None = None) -> dict[str, int]:
    """
    WHAT: Reads config/datasets.yaml and extracts the configured worlds_per_split dictionary.
    WHY: Keeps dataset split world allocations centralized in configuration.
    """
    target = DEFAULT_CONFIG_PATH if path is None else Path(path)
    with open(target, encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict) or "datasets" not in data:
        raise ValueError(f"{target} missing top-level 'datasets' configuration key")
    datasets_block = data["datasets"]
    if "worlds_per_split" not in datasets_block or "value" not in datasets_block["worlds_per_split"]:
        raise ValueError(f"{target} missing 'datasets.worlds_per_split.value'")
    counts: dict[str, int] = datasets_block["worlds_per_split"]["value"]
    return counts


def plan_datasets(
    datasets_cfg: dict[str, int] | None = None,
    sampler_cfg: dict[str, Any] | None = None,
    world_ranges: dict[str, Any] | None = None,
    sampler_seed: int = 1,
) -> dict[str, list[WorldSpec]]:
    """
    WHAT: Computes the Latin hypercube WorldSpec plan for each configured split.
    WHY: Implements deterministic, reproducible multi-world sampling before execution.
    """
    counts = load_datasets_config() if datasets_cfg is None else datasets_cfg
    s_cfg = load_sampler_config() if sampler_cfg is None else sampler_cfg
    w_ranges = read_world_ranges() if world_ranges is None else world_ranges

    plan: dict[str, list[WorldSpec]] = {}
    for split, count in counts.items():
        specs = sample_worlds(
            split=split,
            count=count,
            sampler_seed=sampler_seed,
            cfg=s_cfg,
            world_ranges=w_ranges,
        )
        plan[split] = specs
    return plan


def derive_dataset_cutoff(recorded_worlds: Sequence[RecordedWorld]) -> datetime:
    """
    WHAT: Derives the point-in-time dataset cutoff timestamp for a multi-world split.
    WHY: Follows the simulation end of world days convention from showcase/level_16_support.py:187
    and tests/integration/test_dataset_end_to_end.py:143-145: midnight IST of simulation clock start
    date plus simulation duration (n_days). For multi-world splits, the maximum world end timestamp
    is chosen so no simulation events prior to planned completion are prematurely excluded.
    """
    if not recorded_worlds:
        raise ValueError("Cannot derive cutoff from empty recorded worlds sequence")
    cutoffs: list[datetime] = []
    for w in recorded_worlds:
        cs = w.output.plan.clock.start_date
        sim_start_date = cs.astimezone(IST).date() if isinstance(cs, datetime) else cs
        start_midnight = datetime(
            sim_start_date.year, sim_start_date.month, sim_start_date.day, 0, 0, 0, tzinfo=IST
        )
        world_end = start_midnight + timedelta(days=w.output.plan.n_days)
        cutoffs.append(world_end)
    return max(cutoffs)


def supersede_prior_datasets(
    split: str,
    new_dir: Path,
    datasets_root: Path = DEFAULT_DATASETS_ROOT,
) -> list[Path]:
    """
    WHAT: Inspects existing dataset directories under datasets_root and marks prior datasets
    of the same split as superseded by new_dir.
    WHY: Guarantees historical lineage without deleting past artifacts, preventing accidental
    usage of superseded data per Rule R8.
    """
    if not datasets_root.exists():
        return []

    superseded: list[Path] = []
    for candidate in sorted(datasets_root.iterdir()):
        if not candidate.is_dir() or candidate.resolve() == new_dir.resolve():
            continue
        manifest_path = candidate / "manifest.json"
        if not manifest_path.is_file():
            continue
        try:
            manifest_bytes = manifest_path.read_bytes()
            manifest_data = orjson.loads(manifest_bytes)
        except Exception:
            continue
        if manifest_data.get("split") == split:
            if manifest_data.get("superseded_by") != new_dir.name:
                mark_superseded(candidate, new_dir)
                superseded.append(candidate)
    return superseded


def summarize_labels_by_source_and_via(dataset: Dataset) -> dict[str, int]:
    """
    WHAT: Aggregates label counts grouped by source and delivery channel (via).
    WHY: Surfaces exact breakdown between direct complaints, hold requests, registry, and police bulk.
    """
    summary: dict[str, int] = {}
    sources = dataset.labels.column("source").to_pylist()
    vias = dataset.labels.column("via").to_pylist()
    for s, v in zip(sources, vias):
        label_key = f"{s} (via {v})" if v is not None else str(s)
        summary[label_key] = summary.get(label_key, 0) + 1
    return summary


class DynamicAccountBankMap(dict[str, str]):
    """
    Mapping of account id to bank id that initializes with starting ledger accounts
    and dynamically resolves mid-run opened accounts from the simulation runner state.
    """

    def __init__(self, initial: dict[str, str], feed_cfg: Any) -> None:
        super().__init__(initial)
        self.feed_cfg = feed_cfg
        self.active_state: Any = None

    def __contains__(self, key: object) -> bool:
        if super().__contains__(key):
            return True
        if self.active_state is not None and key in self.active_state.bank_map:
            b_id = resolve_bank_id(self.active_state.bank_map[str(key)], self.feed_cfg)
            self[str(key)] = b_id
            return True
        return False

    def __getitem__(self, key: str) -> str:
        if not super().__contains__(key):
            if self.active_state is not None and key in self.active_state.bank_map:
                b_id = resolve_bank_id(self.active_state.bank_map[key], self.feed_cfg)
                self[key] = b_id
                return b_id
        return super().__getitem__(key)


def record_world_for_spec(
    spec: WorldSpec,
    preset_name: str,
    sampler_cfg: dict[str, Any],
    code_version: str,
    loop: asyncio.AbstractEventLoop,
) -> RecordedWorld:
    """
    WHAT: Runs simulation in recording mode for a single sampled WorldSpec.
    WHY: Captures connector DecisionCalls and metadata without querying hidden truth tables.
    """
    configs = pinned_configs(spec, sampler_cfg)
    settings = fast_settings_for(spec, preset_name, sampler_cfg, configs)
    prepared = prepare_world(settings, configs)
    feed_cfg = load_bank_feed_config()
    bank_modes = load_bank_modes()

    account_bank_map: dict[str, str] = {}
    for ledger in prepared.world.ledgers:
        b_id = resolve_bank_id(ledger.bank_index, feed_cfg)
        for acc in ledger.accounts:
            account_bank_map[acc] = b_id

    dyn_map = DynamicAccountBankMap(account_bank_map, feed_cfg)

    connectors = MockBankConnectors(
        transports={b: RejectingTransport() for b in bank_modes},
        modes=bank_modes,
        account_bank_map=dyn_map,
        clock_start=prepared.plan.clock.start_date,
        feed_config=feed_cfg,
        decision_timeout_seconds=5.0,
        loop=loop,
    )

    orig_start = ist.start_runner_state

    def custom_start(w: Any) -> Any:
        st = orig_start(w)
        dyn_map.active_state = st
        return st

    with patch("world.interactive_runner.start_runner_state", side_effect=custom_start):
        recorded = record_sampled_world_for_replay(
            spec=spec,
            preset_name=preset_name,
            gang_configs=None,
            sampler_cfg=sampler_cfg,
            code_version=code_version,
            connectors=connectors,
        )
    return recorded


async def replay_and_extract(
    recorded: RecordedWorld,
    admin_pg: str,
    base_redis_url: str,
) -> WorldExtract:
    """
    WHAT: Replays a recorded world through the operational pipeline and extracts workspace tables before teardown.
    WHY: Provides the point-in-time tables needed for dataset construction.
    """
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

    await run_replay(
        recorded,
        feed,
        admin_pg=admin_pg,
        base_redis_url=base_redis_url,
        redis_client_factory=host_redis_factory,
        decisions_per_event_payment=True,
        before_drop_hook=hook,
    )
    if not holder:
        raise ValueError(f"Workspace extract hook was not triggered during replay of {recorded.world_id}")
    return holder[0]


async def build_split_dataset(
    split: str,
    specs: list[WorldSpec],
    preset_name: str,
    sampler_cfg: dict[str, Any],
    code_version: str,
    datasets_root: Path = DEFAULT_DATASETS_ROOT,
    admin_pg: str | None = None,
    base_redis_url: str | None = None,
) -> tuple[Dataset, Path, list[Path]]:
    """
    WHAT: Records, replays, extracts, builds and writes one dataset for a given split across its specs.
    WHY: Implements single-dataset-per-split building and automatic prior version supersession.
    """
    loop = asyncio.get_running_loop()
    db_url = admin_pg if admin_pg is not None else get_showcase_database_url()
    redis_url = base_redis_url if base_redis_url is not None else host_side_redis_url()

    recorded_worlds: list[RecordedWorld] = []
    extracts: list[WorldExtract] = []

    for spec in specs:
        rec = record_world_for_spec(spec, preset_name, sampler_cfg, code_version, loop)
        recorded_worlds.append(rec)
        extract = await replay_and_extract(rec, db_url, redis_url)
        extracts.append(extract)

    cutoff = derive_dataset_cutoff(recorded_worlds)
    dataset = build_dataset(extracts, split=split, cutoff=cutoff, code_version=code_version)
    dataset_dir = write_dataset(dataset, root=datasets_root)
    superseded = supersede_prior_datasets(split, dataset_dir, datasets_root=datasets_root)

    return dataset, dataset_dir, superseded


async def build_all_level_16_datasets(
    splits: Sequence[str] | None = None,
    preset_name: str = "tiny",
    datasets_root: Path = DEFAULT_DATASETS_ROOT,
    sampler_seed: int = 1,
) -> dict[str, dict[str, Any]]:
    """
    WHAT: Builds datasets for all configured splits sequentially, returning execution metrics per split.
    WHY: Executes Level 16 dataset generation unattended while tracking wall time and label counts.
    """
    counts = load_datasets_config()
    sampler_cfg = load_sampler_config()
    world_ranges = read_world_ranges()
    code_ver = get_code_version()

    target_splits = list(splits) if splits is not None else list(counts.keys())
    plan = plan_datasets(counts, sampler_cfg, world_ranges, sampler_seed=sampler_seed)

    results: dict[str, dict[str, Any]] = {}
    for split in target_splits:
        specs = plan[split]
        t0 = time.monotonic()
        dataset, dataset_dir, superseded = await build_split_dataset(
            split=split,
            specs=specs,
            preset_name=preset_name,
            sampler_cfg=sampler_cfg,
            code_version=code_ver,
            datasets_root=datasets_root,
        )
        elapsed = time.monotonic() - t0
        label_summary = summarize_labels_by_source_and_via(dataset)

        split_result: dict[str, Any] = {
            "dataset_id": dataset_dir.name,
            "worlds": [s.world_id for s in specs],
            "world_count": len(specs),
            "rows": {
                "payments": dataset.payments.num_rows,
                "snapshots": dataset.snapshots.num_rows,
                "labels": dataset.labels.num_rows,
            },
            "labels_by_source_and_via": label_summary,
            "superseded_count": len(superseded),
            "superseded_names": [p.name for p in superseded],
            "wall_time_seconds": round(elapsed, 2),
        }
        results[split] = split_result

        print(f"Split: {split}")
        print(f"  Dataset ID: {dataset_dir.name}")
        print(f"  Worlds: {', '.join(split_result['worlds'])}")
        print(
            f"  Rows: payments={split_result['rows']['payments']}, "
            f"snapshots={split_result['rows']['snapshots']}, "
            f"labels={split_result['rows']['labels']}"
        )
        print(f"  Labels breakdown: {label_summary}")
        print(f"  Superseded: {len(superseded)} older datasets")
        print(f"  Wall time: {split_result['wall_time_seconds']}s")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Level 16 point-in-time datasets")
    parser.add_argument(
        "--splits",
        type=str,
        default=None,
        help="Comma-separated splits to build (default: all configured in config/datasets.yaml)",
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="tiny",
        help="World sizing preset (default: 'tiny' per architect ruling)",
    )
    parser.add_argument(
        "--root",
        type=str,
        default=str(DEFAULT_DATASETS_ROOT),
        help="Datasets output root directory (default: data/datasets)",
    )
    args = parser.parse_args()

    selected_splits = [s.strip() for s in args.splits.split(",")] if args.splits else None
    asyncio.run(
        build_all_level_16_datasets(
            splits=selected_splits,
            preset_name=args.preset,
            datasets_root=Path(args.root),
        )
    )


if __name__ == "__main__":
    main()
