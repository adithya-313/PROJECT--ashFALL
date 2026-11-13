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
from world.bank_connectors import MockBankConnectors, load_bank_modes
from world.bank_feed import resolve_bank_id
from world.bank_feed_config import load_bank_feed_config
from world.base_rate import base_rate_for
from world.config_reader import read_world_ranges
from world.interactive_runner import prepare_world
from world.replay_recording import RecordedWorld, record_sampled_world_for_replay
from world.replay_workspace import ReplayWorkspace, workspace_redis_client
from world.sampled_world import fast_settings_for, pinned_configs, resolve_sampled_world_gangs
from world.truth_writer import truth_root_from_env, write_truth_files
from world.world_sampler import WorldSpec, load_sampler_config, sample_worlds

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "datasets.yaml"
DEFAULT_DATASETS_ROOT = REPO_ROOT / "data" / "datasets"
DEFAULT_WORLD_CARDS_ROOT = REPO_ROOT / "runs" / "world_cards"


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


def _gang_mix_fits(spec: WorldSpec, preset_name: str, sampler_cfg: dict[str, Any]) -> bool:
    """
    WHAT: Reports whether one planned world can generate its gang mix from the preset's people.
    WHY: This follows record_sampled_world_for_replay up to resolve_sampled_world_gangs, so the
    scan refuses exactly the worlds the real recording would refuse.
    Raises:
        ValueError: Propagated unless the mix is refused for too few eligible people.
    """
    configs = pinned_configs(spec, sampler_cfg)
    try:
        resolve_sampled_world_gangs(spec, preset_name, None, sampler_cfg, configs)
    except ValueError as error:
        if "eligible people" in str(error):
            return False
        raise
    return True


def find_fitting_sampler_seed(
    splits: Sequence[str],
    preset_name: str,
    start_seed: int,
    max_tries: int,
) -> int:
    """
    WHAT: Returns the first sampler seed whose planned worlds in the given splits all have enough
    eligible people for the gang mix they need.
    WHY: generate_gang_mix refuses a world whose mix needs more mules and victims than the sizing
    preset provides, and the build would otherwise discover that after hours of recording. Trying
    seeds up front turns the refusal into a check that takes seconds.
    Args:
        splits: Split names to check, for example ("training", "validation", "test").
        preset_name: World sizing preset the build will use.
        start_seed: First sampler seed to try.
        max_tries: Number of consecutive seeds to try, starting at start_seed.
    Returns:
        The first seed in [start_seed, start_seed + max_tries) where every planned world fits.
    Raises:
        ValueError: If a split is unknown, or if no seed fits within max_tries.
    """
    counts = load_datasets_config()
    sampler_cfg = load_sampler_config()
    world_ranges = read_world_ranges()

    unknown_splits = [split for split in splits if split not in counts]
    if unknown_splits:
        raise ValueError(f"Unknown split(s) {', '.join(unknown_splits)}; expected some of {sorted(counts)}")

    splits_text = ", ".join(splits)
    for seed in range(start_seed, start_seed + max_tries):
        plan = plan_datasets(counts, sampler_cfg, world_ranges, sampler_seed=seed)
        if all(_gang_mix_fits(spec, preset_name, sampler_cfg) for split in splits for spec in plan[split]):
            return seed
    raise ValueError(
        f"no sampler seed from {start_seed} fits {splits_text} within {max_tries} tries"
    )


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

    connectors = MockBankConnectors(
        transports={b: RejectingTransport() for b in bank_modes},
        modes=bank_modes,
        account_bank_map=account_bank_map,
        clock_start=prepared.plan.clock.start_date,
        feed_config=feed_cfg,
        decision_timeout_seconds=5.0,
        loop=loop,
    )

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
    *,
    profile_dir: Path | None = None,
) -> WorldExtract:
    """
    WHAT: Replays a recorded world through the operational pipeline and extracts workspace tables before teardown.
    WHY: Provides the point-in-time tables needed for dataset construction.
    When profile_dir is set, the replay phase profile is written to profile_dir/<world_id>.json.
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

    # A separate client reads the server-wide commandstats; the value below is shared by every
    # client of this Redis server, so it counts the whole server, not only this replay.
    redis_client: redis.Redis | None = None
    commands_before = 0
    if profile_dir is not None:
        profile_dir.mkdir(parents=True, exist_ok=True)
        redis_client = redis.Redis.from_url(base_redis_url)
        commands_before = redis_command_calls(redis_client)

    result = await run_replay(
        recorded,
        feed,
        admin_pg=admin_pg,
        base_redis_url=base_redis_url,
        redis_client_factory=host_redis_factory,
        decisions_per_event_payment=True,
        before_drop_hook=hook,
    )

    if redis_client is not None:
        commands_after = redis_command_calls(redis_client)
        redis_client.close()
        profile_payload: dict[str, Any] = {
            "profile": result.profile,
            "decisions_sent": result.decisions_sent,
            "longest_drain_seconds": result.longest_drain_seconds,
            "feature_log_sha256": result.feature_log_sha256,
            "redis_commands_during_replay": commands_after - commands_before,
        }
        profile_path = profile_dir / f"{recorded.world_id}.json"
        profile_path.write_bytes(
            orjson.dumps(profile_payload, option=orjson.OPT_SORT_KEYS | orjson.OPT_INDENT_2) + b"\n"
        )

    if not holder:
        raise ValueError(f"Workspace extract hook was not triggered during replay of {recorded.world_id}")
    return holder[0]


def redis_command_calls(client: redis.Redis) -> int:
    """
    WHAT: Sums the `calls` counter over all INFO commandstats sections.

    WHY: The difference between two reads bounds how many Redis commands a replay issued. Redis keeps
    this counter server-wide, so other clients of the same server also contribute.
    """
    commandstats = client.info("commandstats")
    total = 0
    for section in commandstats.values():
        if not isinstance(section, dict):
            raise ValueError(f"Unexpected Redis commandstats section: {section!r}")
        total += int(section["calls"])
    return total


def save_evaluation_inputs(
    recorded: RecordedWorld,
    truth_root: Path,
    world_cards_root: Path,
) -> dict[str, Any]:
    """
    WHAT: Saves the hidden truth files and the world-card base rate for one recorded world so the
    Level 17 evaluation can join them later.
    WHY: The evaluation needs truth and the fraud inflation factor per run, written next to the
    datasets but outside them, so operational tables never carry hidden answers. The truth writer
    already refuses to overwrite a non-empty run folder, which makes a second save for the same
    run fail instead of silently replacing evidence.
    """
    run_id = recorded.output.result.run_id
    truth_dir = write_truth_files(recorded.output, truth_root, write_personas=True)

    base_rate = base_rate_for(recorded.output)
    base_rate_dir = world_cards_root / run_id
    base_rate_dir.mkdir(parents=True, exist_ok=True)
    base_rate_path = base_rate_dir / "base_rate.json"
    base_rate_path.write_bytes(
        orjson.dumps(base_rate, option=orjson.OPT_SORT_KEYS | orjson.OPT_INDENT_2) + b"\n"
    )

    return {
        "run_id": run_id,
        "truth_dir": str(truth_dir),
        "base_rate_path": str(base_rate_path),
        "fraud_inflation_factor": base_rate["fraud_inflation_factor"],
    }


async def build_split_dataset(
    split: str,
    specs: list[WorldSpec],
    preset_name: str,
    sampler_cfg: dict[str, Any],
    code_version: str,
    datasets_root: Path = DEFAULT_DATASETS_ROOT,
    admin_pg: str | None = None,
    base_redis_url: str | None = None,
    write_evaluation_inputs: bool = False,
    truth_root: Path | None = None,
    world_cards_root: Path = DEFAULT_WORLD_CARDS_ROOT,
    replay_profile_dir: Path | None = None,
) -> tuple[Dataset, Path, list[Path]]:
    """
    WHAT: Records, replays, extracts, builds and writes one dataset for a given split across its specs.
    WHY: Implements single-dataset-per-split building and automatic prior version supersession.
    When write_evaluation_inputs is on, each recorded world also saves its truth and base rate.
    When replay_profile_dir is on, each world's replay profile is written there with its record and replay seconds.
    """
    loop = asyncio.get_running_loop()
    db_url = admin_pg if admin_pg is not None else get_showcase_database_url()
    redis_url = base_redis_url if base_redis_url is not None else host_side_redis_url()

    evaluation_truth_root: Path | None = None
    if write_evaluation_inputs:
        if truth_root is None:
            raise ValueError("write_evaluation_inputs requires a truth_root path")
        evaluation_truth_root = truth_root

    recorded_worlds: list[RecordedWorld] = []
    extracts: list[WorldExtract] = []

    for spec in specs:
        record_started = time.perf_counter()
        rec = record_world_for_spec(spec, preset_name, sampler_cfg, code_version, loop)
        record_seconds = time.perf_counter() - record_started
        recorded_worlds.append(rec)
        if evaluation_truth_root is not None:
            save_evaluation_inputs(rec, evaluation_truth_root, world_cards_root)
        replay_started = time.perf_counter()
        if replay_profile_dir is None:
            extract = await replay_and_extract(rec, db_url, redis_url)
        else:
            extract = await replay_and_extract(rec, db_url, redis_url, profile_dir=replay_profile_dir)
        replay_seconds = time.perf_counter() - replay_started
        extracts.append(extract)

        if replay_profile_dir is not None:
            profile_path = replay_profile_dir / f"{rec.world_id}.json"
            profile_payload = orjson.loads(profile_path.read_bytes())
            profile_payload["record_seconds"] = round(record_seconds, 3)
            profile_payload["replay_seconds"] = round(replay_seconds, 3)
            profile_path.write_bytes(
                orjson.dumps(profile_payload, option=orjson.OPT_SORT_KEYS | orjson.OPT_INDENT_2) + b"\n"
            )

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
    write_evaluation_inputs: bool = False,
    world_cards_root: Path = DEFAULT_WORLD_CARDS_ROOT,
    replay_profile_dir: Path | None = None,
) -> dict[str, dict[str, Any]]:
    """
    WHAT: Builds datasets for all configured splits sequentially, returning execution metrics per split.
    WHY: Executes Level 16 dataset generation unattended while tracking wall time and label counts.
    """
    counts = load_datasets_config()
    sampler_cfg = load_sampler_config()
    world_ranges = read_world_ranges()
    code_ver = get_code_version()

    truth_root: Path | None = None
    if write_evaluation_inputs:
        truth_root = truth_root_from_env(os.environ)

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
            write_evaluation_inputs=write_evaluation_inputs,
            truth_root=truth_root,
            world_cards_root=world_cards_root,
            replay_profile_dir=replay_profile_dir,
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
    parser.add_argument(
        "--write-evaluation-inputs",
        action="store_true",
        help="Also save hidden truth and world-card base rates for the Level 17 evaluation (default: off)",
    )
    parser.add_argument(
        "--world-cards-root",
        type=str,
        default=str(DEFAULT_WORLD_CARDS_ROOT),
        help="World cards root for base_rate.json files (default: runs/world_cards)",
    )
    parser.add_argument(
        "--replay-profile-dir",
        type=str,
        default=None,
        help="Directory for per-world replay profile JSON files (default: no profiles written)",
    )
    parser.add_argument(
        "--sampler-seed",
        type=int,
        default=1,
        help="Master sampler seed passed to the plan and the build (default: 1)",
    )
    parser.add_argument(
        "--max-seed-tries",
        type=int,
        # policy: a fit is expected within a few seeds; 200 bounds the scan to seconds and stops
        # it rather than looping forever.
        default=200,
        help="Seeds scanned by --find-sampler-seed before giving up (default: 200)",
    )
    parser.add_argument(
        "--find-sampler-seed",
        action="store_true",
        help="Print the first sampler seed whose planned worlds fit and exit",
    )
    args = parser.parse_args()

    selected_splits = [s.strip() for s in args.splits.split(",")] if args.splits else None
    if args.find_sampler_seed:
        scan_splits = selected_splits if selected_splits is not None else list(load_datasets_config().keys())
        seed = find_fitting_sampler_seed(scan_splits, args.preset, args.sampler_seed, args.max_seed_tries)
        print(f"fitting sampler seed: {seed}")
        return
    asyncio.run(
        build_all_level_16_datasets(
            splits=selected_splits,
            preset_name=args.preset,
            datasets_root=Path(args.root),
            sampler_seed=args.sampler_seed,
            write_evaluation_inputs=args.write_evaluation_inputs,
            world_cards_root=Path(args.world_cards_root),
            replay_profile_dir=Path(args.replay_profile_dir) if args.replay_profile_dir else None,
        )
    )


if __name__ == "__main__":
    main()
