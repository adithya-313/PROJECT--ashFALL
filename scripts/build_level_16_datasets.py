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
import subprocess
import sys
import time
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Protocol

import asyncpg
import orjson
import redis
import yaml

from core.config import config
from core.config_validator import load_yaml
from replay.dataset_builder import (
    Dataset,
    WorldExtract,
    build_dataset,
    extract_world,
    mark_superseded,
    write_dataset,
)
from replay.pipeline import feed_with_labels
from replay.runner import (
    IST,
    ReplayDropContext,
    load_default_local_offset_tracking,
    load_default_registry_poll_on_change,
    load_default_watermark_advance_skip_overlap,
    run_replay,
)
from scripts.dataset_checkpoints import (
    CHECKPOINT_FORMAT_VERSION,
    DatasetCheckpoint,
    checkpoint_path,
    load_checkpoint,
    save_checkpoint,
    verify_checkpoint_identity,
)
from scripts.dataset_shards import (
    EXIT_STOPPED,
    MissingCheckpointsError,
    WorldShard,
    load_plan_checkpoints,
    parse_world_shard,
    shard_specs,
    stop_requested,
)
from showcase.framework import get_code_version
from showcase.isolation import get_showcase_database_url
from world.bank_connectors import MockBankConnectors, load_bank_modes
from world.bank_feed import resolve_bank_id
from world.bank_feed_config import load_bank_feed_config
from world.base_rate import base_rate_for
from world.config_reader import read_world_ranges
from world.interactive_runner import prepare_world
from world.replay_recording import RecordedWorld, record_sampled_world_for_replay
from world.replay_workspace import ReplayWorkspace, load_replay_settings, workspace_redis_client
from world.sampled_world import fast_settings_for, pinned_configs, resolve_sampled_world_gangs
from world.truth_writer import truth_root_from_env, write_truth_files
from world.world_sampler import WorldSpec, load_sampler_config, sample_block, sample_worlds

REPO_ROOT = Path(__file__).resolve().parent.parent

DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "datasets.yaml"
DEFAULT_DATASET_BUILD_CONFIG_PATH = REPO_ROOT / "config" / "dataset_build.yaml"
DEFAULT_DATASETS_ROOT = REPO_ROOT / "data" / "datasets"
DEFAULT_WORLD_CARDS_ROOT = REPO_ROOT / "runs" / "world_cards"

# ConnectionResetError is a built-in subclass of ConnectionError. The redis and asyncpg classes
# do not inherit from the built-in class, so each is named here. A lost connection surfaces as
# one of these; anything else during a replay is a real failure and must not be retried.
CONNECTION_ERROR_TYPES: tuple[type[BaseException], ...] = (
    ConnectionResetError,
    ConnectionError,
    redis.exceptions.ConnectionError,
    asyncpg.exceptions.ConnectionDoesNotExistError,
    asyncpg.exceptions.ConnectionFailureError,
    asyncpg.exceptions.CannotConnectNowError,
    asyncpg.exceptions.ClientCannotConnectError,
)


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


def parse_worlds_per_split(text: str) -> dict[str, int]:
    """
    WHAT: Parses a "training=6,validation=3,test=3" override into split counts.
    WHY: Lets a pilot build change world counts for one run without touching config/datasets.yaml.
    Args:
        text: Comma-separated split=count items.
    Returns:
        The split counts in the order they were given.
    Raises:
        ValueError: On empty text, a malformed item, an unknown split, a repeated split, or a count
        that is not a positive integer.
    """
    allowed_splits = set(load_datasets_config())
    if not text.strip():
        raise ValueError("worlds-per-split text is empty")
    counts: dict[str, int] = {}
    for item in text.split(","):
        split_name, separator, count_text = item.strip().partition("=")
        if not separator:
            raise ValueError(f"worlds-per-split item {item!r} is not split=count")
        split_name = split_name.strip()
        if split_name not in allowed_splits:
            raise ValueError(f"unknown split {split_name!r}; expected some of {sorted(allowed_splits)}")
        if split_name in counts:
            raise ValueError(f"split {split_name!r} appears more than once")
        try:
            count = int(count_text.strip())
        except ValueError:
            raise ValueError(f"count for split {split_name!r} is not an integer: {count_text.strip()!r}") from None
        if count <= 0:
            raise ValueError(f"count for split {split_name!r} must be positive: {count}")
        counts[split_name] = count
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


def parse_blocks(text: str) -> dict[str, list[int]]:
    """
    WHAT: Parses "validation=1-2,test=0-0,training=1-1" into {split: [block numbers]}.
    WHY: The committed Level 12f plan is blocks of block_size worlds, each block its own Latin
    hypercube; the build must name the blocks it wants without a count that changes the plan.
    Args:
        text: Comma-separated split=start-end pieces with inclusive block ranges.
    Returns:
        Block numbers per split, in the order the splits were given.
    Raises:
        ValueError: On an empty piece, a malformed piece, an unknown split, a repeated split, a
        negative block number, or a range whose start is after its end.
    """
    allowed_splits = set(load_sampler_config()["seed_blocks"])
    blocks: dict[str, list[int]] = {}
    for item in text.split(","):
        piece = item.strip()
        if not piece:
            raise ValueError(f"empty block piece in {text!r}")
        split_name, separator, range_text = piece.partition("=")
        if not separator:
            raise ValueError(f"block piece {piece!r} is not split=start-end")
        split_name = split_name.strip()
        if split_name not in allowed_splits:
            raise ValueError(f"unknown split {split_name!r}; expected some of {sorted(allowed_splits)}")
        if split_name in blocks:
            raise ValueError(f"split {split_name!r} appears more than once")
        start_text, dash, end_text = range_text.strip().partition("-")
        if not dash:
            raise ValueError(f"block range {range_text.strip()!r} for split {split_name!r} is not start-end")
        try:
            start = int(start_text.strip())
            end = int(end_text.strip())
        except ValueError:
            raise ValueError(
                f"block range {range_text.strip()!r} for split {split_name!r} is not two integers"
            ) from None
        if start < 0 or end < 0:
            raise ValueError(
                f"block numbers for split {split_name!r} must be at least 0, got {range_text.strip()!r}"
            )
        if start > end:
            raise ValueError(
                f"block range for split {split_name!r} starts at {start} after it ends at {end}"
            )
        blocks[split_name] = list(range(start, end + 1))
    return blocks


def plan_blocks(
    blocks: dict[str, list[int]],
    sampler_seed: int,
    sampler_cfg: dict[str, Any],
    world_ranges: dict[str, Any],
) -> dict[str, list[WorldSpec]]:
    """
    WHAT: Builds the plan of the named blocks: each split gets its blocks concatenated in ascending order.
    WHY: Every committed block is a complete Latin hypercube, so building blocks 0..n covers each key
    parameter's whole range at every n; world ids keep the "<split>-bNN-NNNN" form sample_block gives.
    """
    plan: dict[str, list[WorldSpec]] = {}
    for split, numbers in blocks.items():
        specs: list[WorldSpec] = []
        for number in sorted(numbers):
            specs.extend(sample_block(split, number, sampler_seed, sampler_cfg, world_ranges))
        plan[split] = specs
    return plan


def plan_run(
    counts: dict[str, int],
    blocks: dict[str, list[int]] | None,
    sampler_seed: int,
    sampler_cfg: dict[str, Any],
    world_ranges: dict[str, Any],
) -> dict[str, list[WorldSpec]]:
    """
    WHAT: The plan for one run: the block plan when blocks are named, otherwise the per-split plan.
    WHY: The build and the seed scan must plan the same worlds, and a legacy count above one block is
    refused because such a plan is not the committed Level 12f block plan.
    Raises:
        ValueError: When no blocks are named and a per-split count is larger than block_size.
    """
    if blocks is not None:
        return plan_blocks(blocks, sampler_seed, sampler_cfg, world_ranges)
    block_size = int(sampler_cfg["block_size"]["value"])
    oversized = {split: count for split, count in counts.items() if count > block_size}
    if oversized:
        raise ValueError(
            f"world counts above the {block_size}-world block ({oversized}) are not the committed "
            "Level 12f block plan; build these worlds with --blocks"
        )
    return plan_datasets(counts, sampler_cfg, world_ranges, sampler_seed=sampler_seed)


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
    datasets_cfg: dict[str, int] | None = None,
    blocks: dict[str, list[int]] | None = None,
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
        datasets_cfg: Split counts to plan, or None to read config/datasets.yaml.
        blocks: Block numbers per split to plan instead of the split counts, or None for the legacy plan.
    Returns:
        The first seed in [start_seed, start_seed + max_tries) where every planned world fits.
    Raises:
        ValueError: If a split is unknown or not listed in blocks, or if no seed fits within max_tries.
    """
    counts = load_datasets_config() if datasets_cfg is None else datasets_cfg
    sampler_cfg = load_sampler_config()
    world_ranges = read_world_ranges()

    unknown_splits = [split for split in splits if split not in counts]
    if unknown_splits:
        raise ValueError(f"Unknown split(s) {', '.join(unknown_splits)}; expected some of {sorted(counts)}")
    if blocks is not None:
        missing_blocks = [split for split in splits if split not in blocks]
        if missing_blocks:
            raise ValueError(
                f"--blocks does not list split(s) {', '.join(missing_blocks)}; "
                f"it lists {', '.join(sorted(blocks))}"
            )

    splits_text = ", ".join(splits)
    for seed in range(start_seed, start_seed + max_tries):
        plan = plan_run(counts, blocks, seed, sampler_cfg, world_ranges)
        if all(_gang_mix_fits(spec, preset_name, sampler_cfg) for split in splits for spec in plan[split]):
            return seed
    raise ValueError(
        f"no sampler seed from {start_seed} fits {splits_text} within {max_tries} tries"
    )


def world_end_for_recorded(recorded: RecordedWorld) -> datetime:
    """
    WHAT: Returns the simulation end of one recorded world as midnight IST of its last simulated day.
    WHY: Shared by the split cutoff and the per-world checkpoint, so a resumed build derives the
    same cutoff it would derive from the recorded world.
    """
    cs = recorded.output.plan.clock.start_date
    sim_start_date = cs.astimezone(IST).date() if isinstance(cs, datetime) else cs
    start_midnight = datetime(
        sim_start_date.year, sim_start_date.month, sim_start_date.day, 0, 0, 0, tzinfo=IST
    )
    return start_midnight + timedelta(days=recorded.output.plan.n_days)


def cutoff_from_world_ends(world_ends: Sequence[datetime]) -> datetime:
    """
    WHAT: Returns the latest world end of a multi-world split.
    WHY: A resumed build has a saved world end instead of a recorded world, and needs the same
    cutoff rule: no simulation events before the planned completion of any world are excluded.
    """
    if not world_ends:
        raise ValueError("Cannot derive cutoff from empty world end sequence")
    return max(world_ends)


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
    return cutoff_from_world_ends([world_end_for_recorded(w) for w in recorded_worlds])


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


def load_dataset_build_config(path: Path = DEFAULT_DATASET_BUILD_CONFIG_PATH) -> dict[str, Any]:
    """
    WHAT: Reads config/dataset_build.yaml and returns its labelled parameters.
    WHY: Retry policy is a choice, so it lives in config and not in code; the file is read only
    here, never by the recording or run-id path (amendment A-041).
    """
    data = load_yaml(path)
    parameters: dict[str, Any] = data["parameters"]
    return parameters


def validate_replay_redis_urls(urls: Any) -> list[str]:
    """
    WHAT: Checks raw replay_redis_urls content and returns it as a list of addresses.
    WHY: The config switch and the --replay-redis-urls flag must accept exactly the same addresses,
    so both paths share these checks instead of growing two copies.
    Raises:
        ValueError: On a non-list value, a non-redis:// entry, or a repeated entry.
    """
    if not isinstance(urls, list):
        raise ValueError(f"replay_redis_urls must be a list, got {urls!r}")
    seen: set[str] = set()
    for url in urls:
        if not isinstance(url, str) or not url.startswith("redis://"):
            raise ValueError(f"replay_redis_urls entry {url!r} is not a redis:// URL")
        if url in seen:
            raise ValueError(f"replay_redis_urls entry {url!r} appears more than once")
        seen.add(url)
    return list(urls)


def load_replay_redis_urls(config: dict[str, Any]) -> list[str]:
    """
    WHAT: Reads and validates the replay_redis_urls switch from the dataset build config.
    WHY: An empty list is the OFF state, so the switch must be checked here rather than trusted, and a
    bad entry would otherwise reach a child as a broken Redis address.
    Raises:
        ValueError: On a non-list value, a non-redis:// entry, or a repeated entry.
    """
    return validate_replay_redis_urls(config["replay_redis_urls"]["value"])


def resolve_replay_redis_urls(flag_text: str | None, config: dict[str, Any]) -> list[str]:
    """
    WHAT: The Redis address list for this run: the flag text when given, otherwise the config switch.
    WHY: The flag turns multi-Redis on for one lab run without editing a tracked file, so the tree
    stays clean and the run's code_version does not gain a "-dirty" suffix.
    Raises:
        ValueError: On an empty comma-separated piece, or any entry the config rules refuse.
    """
    if flag_text is None:
        return load_replay_redis_urls(config)
    pieces = [piece.strip() for piece in flag_text.split(",")]
    if any(not piece for piece in pieces):
        raise ValueError(f"--replay-redis-urls contains an empty URL: {flag_text!r}")
    return validate_replay_redis_urls(pieces)


def redis_url_for_worker(worker_index: int, urls: Sequence[str]) -> str | None:
    """
    WHAT: The Redis address for worker worker_index, or None when the switch is off.
    WHY: Round-robin (index % server count) spreads neighbouring workers over different servers, so two
    workers started together do not land on the same server; with one URL every worker keeps today's address.
    """
    if not urls:
        return None
    return urls[worker_index % len(urls)]


async def replay_world_with_retry(
    recorded: RecordedWorld,
    admin_pg: str,
    base_redis_url: str,
    *,
    config: dict[str, Any],
    profile_dir: Path | None = None,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> WorldExtract:
    """
    WHAT: Replays one world, retrying from scratch only when Redis or Postgres dropped the connection.
    WHY: run_replay drops its workspace in a finally block, so a fresh call starts clean. Any other
    exception is a real failure and is raised at once instead of burning the remaining attempts.
    """
    retry_enabled = config["retry_world_on_connection_error"]["value"]
    if retry_enabled:
        attempts = int(config["max_world_attempts"]["value"])
        waits = list(config["retry_wait_seconds"]["value"])
        if attempts < 1:
            raise ValueError(f"max_world_attempts must be at least 1, got {attempts}")
        if len(waits) != attempts - 1:
            raise ValueError(
                f"retry_wait_seconds has {len(waits)} entries but {attempts} attempts need {attempts - 1}"
            )
    else:
        attempts = 1
        waits = []

    for attempt in range(1, attempts + 1):
        try:
            if profile_dir is None:
                return await replay_and_extract(recorded, admin_pg, base_redis_url)
            return await replay_and_extract(recorded, admin_pg, base_redis_url, profile_dir=profile_dir)
        except CONNECTION_ERROR_TYPES as error:
            if attempt == attempts:
                raise
            wait_seconds = waits[attempt - 1]
            print(
                f"Replay attempt {attempt}/{attempts} for world {recorded.world_id} failed with "
                f"{type(error).__name__}: {error}; retrying in {wait_seconds}s"
            )
            await sleep(wait_seconds)

    raise RuntimeError(f"replay of world {recorded.world_id} ended without a result or an error")


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


class BuildStopped(Exception):
    """WHAT: A STOP file ended the build before every planned world was built. WHY: main exits with EXIT_STOPPED."""


class WorkerFailure(Exception):
    """WHAT: At least one child build exited non-zero.
    WHY: The parent must not assemble, and main exits with exit_code."""

    def __init__(self, exit_code: int) -> None:
        self.exit_code = exit_code
        super().__init__(f"a child build failed; exiting {exit_code}")


@dataclass(frozen=True)
class WorkerCommand:
    """WHAT: One child build: its shard, the worlds it will build, its command, log file and Redis address."""

    index: int
    world_ids: tuple[str, ...]
    argv: tuple[str, ...]
    log_path: Path
    redis_url: str | None


class ProcessHandle(Protocol):
    """WHAT: The part of subprocess.Popen the launcher uses, so tests can supply a fake process."""

    def wait(self) -> int: ...


def validate_parallel_workers(count: int, redis_urls: Sequence[str] | None = None) -> int:
    """
    WHAT: Checks a worker count against the replay workspace pool read from config/replay.yaml,
    multiplied by the number of Redis servers in use.
    WHY: Each replay holds one Redis workspace DB for its whole run; more workers than the pool would
    wait on each other's locks instead of running, so the count is refused up front. Every server has
    its own pool, so the cap grows with the server count.
    """
    if count < 1:
        raise ValueError(f"--parallel-workers must be at least 1, got {count}")
    settings = load_replay_settings()
    pool_size = settings.redis_workspace_db_max - settings.redis_workspace_db_min + 1
    urls = load_replay_redis_urls(load_dataset_build_config()) if redis_urls is None else redis_urls
    capacity = pool_size * max(1, len(urls))
    if count > capacity:
        servers = "" if len(urls) <= 1 else f" over {len(urls)} redis servers"
        raise ValueError(
            f"--parallel-workers {count} exceeds the replay workspace pool of {capacity}{servers} "
            f"(redis_workspace_db_min={settings.redis_workspace_db_min}, "
            f"redis_workspace_db_max={settings.redis_workspace_db_max})"
        )
    return count


def worker_argv(base_argv: Sequence[str], shard: WorldShard, split: str) -> list[str]:
    """
    WHAT: Builds one child's command: this script under this interpreter with the parent's flags,
    its own split and its world shard.
    WHY: The parent runs one split at a time, so the child is told that single split. --parallel-workers
    and an earlier --splits are dropped so the child cannot start a nested pool or a second split, and
    --replay-redis-urls is dropped because the child's address arrives only through TEST_REDIS_URL.
    """
    stripped: list[str] = []
    skip_value = False
    for arg in base_argv:
        if skip_value:
            skip_value = False
            continue
        if arg in ("--parallel-workers", "--splits", "--replay-redis-urls"):
            skip_value = True
            continue
        if arg.startswith("--parallel-workers=") or arg.startswith("--splits="):
            continue
        if arg.startswith("--replay-redis-urls="):
            continue
        stripped.append(arg)
    return [
        sys.executable,
        str(Path(__file__).resolve()),
        *stripped,
        "--splits",
        split,
        "--world-shard",
        f"{shard.index}/{shard.count}",
    ]


def split_worker_commands(
    base_argv: Sequence[str],
    split: str,
    specs: Sequence[WorldSpec],
    checkpoint_dir: Path,
    workers: int,
    redis_urls: Sequence[str],
) -> list[WorkerCommand]:
    """WHAT: One WorkerCommand per worker K of N. WHY: The parent prints and starts exactly these."""
    commands: list[WorkerCommand] = []
    for index in range(workers):
        shard = WorldShard(index, workers)
        selected = shard_specs(specs, shard)
        commands.append(
            WorkerCommand(
                index=index,
                world_ids=tuple(spec.world_id for spec in selected),
                argv=tuple(worker_argv(base_argv, shard, split)),
                log_path=checkpoint_dir / split / f"worker-{index}.log",
                redis_url=redis_url_for_worker(index, redis_urls),
            )
        )
    return commands


def start_worker_process(
    argv: Sequence[str], log_path: Path, redis_url: str | None = None
) -> subprocess.Popen[bytes]:
    """
    WHAT: Starts one child with an explicit argument list and its own log file.
    WHY: No shell, so flags such as K/N reach argparse exactly as built, and each worker's output
    stays in its own file. A redis_url reaches the child as TEST_REDIS_URL in a copy of the parent
    environment, so the parent's own environment (and its other children) keeps its address.
    """
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("wb") as log_handle:
        if redis_url is None:
            return subprocess.Popen(list(argv), stdout=log_handle, stderr=subprocess.STDOUT)
        env = {**os.environ, "TEST_REDIS_URL": redis_url}
        return subprocess.Popen(list(argv), stdout=log_handle, stderr=subprocess.STDOUT, env=env)


def run_worker_commands(
    commands: Sequence[WorkerCommand],
    *,
    spawn: Callable[[Sequence[str], Path, str | None], ProcessHandle] = start_worker_process,
) -> list[int]:
    """
    WHAT: Starts every worker first, then waits for all of them, returning their exit codes in worker order.
    WHY: A failing worker must not kill a healthy one; every healthy worker is left to finish and checkpoint.
    """
    processes = [spawn(command.argv, command.log_path, command.redis_url) for command in commands]
    return [process.wait() for process in processes]


def worker_failure_exit_code(codes: Sequence[int]) -> int:
    """WHAT: EXIT_STOPPED when every failed worker stopped or was incomplete, otherwise 1."""
    failed = [code for code in codes if code != 0]
    if failed and all(code == EXIT_STOPPED for code in failed):
        return EXIT_STOPPED
    return 1


def log_tail(log_path: Path, lines: int = 5) -> list[str]:
    """
    WHAT: The last lines of a worker log.
    WHY: Child output can contain bytes that are not UTF-8, and a failure report must still print the tail.
    """
    text = log_path.read_bytes().decode("utf-8", errors="replace")
    return text.splitlines()[-lines:]


async def checkpoint_split_worlds(
    split: str,
    specs: Sequence[WorldSpec],
    preset_name: str,
    sampler_cfg: dict[str, Any],
    code_version: str,
    *,
    sampler_seed: int,
    checkpoint_dir: Path,
    db_url: str,
    redis_url: str,
    evaluation_truth_root: Path | None,
    world_cards_root: Path,
    replay_profile_dir: Path | None,
) -> bool:
    """
    WHAT: Resumes each world that has a valid checkpoint and builds and checkpoints the rest, in plan order.
    WHY: A worker writes only checkpoints, so the dataset is assembled by one process from plan order.
    Returns:
        True when a STOP file was found before a world that still needed building; that world is not started.
    """
    loop = asyncio.get_running_loop()
    retry_config = load_dataset_build_config()

    for spec in specs:
        checkpoint_file = checkpoint_path(checkpoint_dir, split, spec.world_id)
        if checkpoint_file.is_file():
            checkpoint = load_checkpoint(checkpoint_file)
            verify_checkpoint_identity(
                checkpoint,
                split=split,
                world_id=spec.world_id,
                seed=spec.seed,
                preset_name=preset_name,
                sampler_seed=sampler_seed,
                code_version=code_version,
            )
            print(
                f"Resumed world {spec.world_id} from {checkpoint_file} "
                f"(record {checkpoint.record_seconds:.2f}s, replay {checkpoint.replay_seconds:.2f}s earlier)"
            )
            continue

        if stop_requested(checkpoint_dir, split):
            print(f"STOP file found; not starting {spec.world_id}")
            return True

        record_started = time.perf_counter()
        rec = record_world_for_spec(spec, preset_name, sampler_cfg, code_version, loop)
        record_seconds = time.perf_counter() - record_started
        replay_started = time.perf_counter()
        extract = await replay_world_with_retry(
            rec,
            db_url,
            redis_url,
            config=retry_config,
            profile_dir=replay_profile_dir,
        )
        replay_seconds = time.perf_counter() - replay_started

        world_end = world_end_for_recorded(rec)
        if evaluation_truth_root is not None:
            truth_folder = evaluation_truth_root / rec.output.result.run_id
            if truth_folder.exists():
                raise FileExistsError(
                    f"truth folder '{truth_folder}' already exists from an earlier attempt; "
                    f"delete it before resuming world {spec.world_id}, so a checkpoint never "
                    "claims work that was not completed"
                )
            save_evaluation_inputs(rec, evaluation_truth_root, world_cards_root)
        save_checkpoint(
            checkpoint_file,
            DatasetCheckpoint(
                format_version=CHECKPOINT_FORMAT_VERSION,
                split=split,
                world_id=spec.world_id,
                seed=spec.seed,
                preset_name=preset_name,
                sampler_seed=sampler_seed,
                code_version=code_version,
                run_id=rec.output.result.run_id,
                world_end=world_end,
                record_seconds=record_seconds,
                replay_seconds=replay_seconds,
                extract=extract,
            ),
        )

        if replay_profile_dir is not None:
            profile_path = replay_profile_dir / f"{rec.world_id}.json"
            profile_payload = orjson.loads(profile_path.read_bytes())
            profile_payload["record_seconds"] = round(record_seconds, 3)
            profile_payload["replay_seconds"] = round(replay_seconds, 3)
            profile_path.write_bytes(
                orjson.dumps(profile_payload, option=orjson.OPT_SORT_KEYS | orjson.OPT_INDENT_2) + b"\n"
            )

    return False


async def build_split_dataset(
    split: str,
    specs: list[WorldSpec],
    preset_name: str,
    sampler_cfg: dict[str, Any],
    code_version: str,
    sampler_seed: int = 1,
    datasets_root: Path = DEFAULT_DATASETS_ROOT,
    admin_pg: str | None = None,
    base_redis_url: str | None = None,
    write_evaluation_inputs: bool = False,
    truth_root: Path | None = None,
    world_cards_root: Path = DEFAULT_WORLD_CARDS_ROOT,
    replay_profile_dir: Path | None = None,
    checkpoint_dir: Path | None = None,
    assemble_only: bool = False,
) -> tuple[Dataset, Path, list[Path]]:
    """
    WHAT: Records, replays, extracts, builds and writes one dataset for a given split across its specs.
    WHY: Implements single-dataset-per-split building and automatic prior version supersession.
    When write_evaluation_inputs is on, each recorded world also saves its truth and base rate.
    When replay_profile_dir is on, each world's replay profile is written there with its record and replay seconds.
    When checkpoint_dir is on, each world saves a checkpoint after its truth and replay, and a rerun
    resumes a world whose checkpoint matches the current spec and settings.
    When assemble_only is on, the dataset is built purely from checkpoints in plan order and no world runs.
    """
    db_url = admin_pg if admin_pg is not None else get_showcase_database_url()
    redis_url = base_redis_url if base_redis_url is not None else host_side_redis_url()

    print(
        "Replay switches: "
        f"watermark_advance_skip_overlap={load_default_watermark_advance_skip_overlap()} "
        f"local_offset_tracking={load_default_local_offset_tracking()} "
        f"registry_poll_on_change={load_default_registry_poll_on_change()}"
    )

    evaluation_truth_root: Path | None = None
    if write_evaluation_inputs:
        if truth_root is None:
            raise ValueError("write_evaluation_inputs requires a truth_root path")
        evaluation_truth_root = truth_root

    extracts: list[WorldExtract] = []
    if checkpoint_dir is None:
        loop = asyncio.get_running_loop()
        recorded_worlds: list[RecordedWorld] = []
        retry_config = load_dataset_build_config()

        for spec in specs:
            record_started = time.perf_counter()
            rec = record_world_for_spec(spec, preset_name, sampler_cfg, code_version, loop)
            record_seconds = time.perf_counter() - record_started
            recorded_worlds.append(rec)
            if evaluation_truth_root is not None:
                save_evaluation_inputs(rec, evaluation_truth_root, world_cards_root)
            replay_started = time.perf_counter()
            extract = await replay_world_with_retry(
                rec,
                db_url,
                redis_url,
                config=retry_config,
                profile_dir=replay_profile_dir,
            )
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
    else:
        if not assemble_only:
            stopped = await checkpoint_split_worlds(
                split,
                specs,
                preset_name,
                sampler_cfg,
                code_version,
                sampler_seed=sampler_seed,
                checkpoint_dir=checkpoint_dir,
                db_url=db_url,
                redis_url=redis_url,
                evaluation_truth_root=evaluation_truth_root,
                world_cards_root=world_cards_root,
                replay_profile_dir=replay_profile_dir,
            )
            if stopped:
                raise BuildStopped(f"split {split} stopped before every planned world was built")
        extracts, world_ends = load_plan_checkpoints(
            checkpoint_dir,
            split,
            specs,
            preset_name=preset_name,
            sampler_seed=sampler_seed,
            code_version=code_version,
        )
        cutoff = cutoff_from_world_ends(world_ends)

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
    worlds_per_split: dict[str, int] | None = None,
    checkpoint_dir: Path | None = None,
    world_shard: WorldShard | None = None,
    assemble_only: bool = False,
    parallel_workers: int | None = None,
    base_argv: Sequence[str] = (),
    redis_urls_flag: str | None = None,
    blocks: dict[str, list[int]] | None = None,
) -> dict[str, dict[str, Any]]:
    """
    WHAT: Builds datasets for all configured splits sequentially, returning execution metrics per split.
    WHY: Executes Level 16 dataset generation unattended while tracking wall time and label counts.
    When world_shard is on, this process only writes checkpoints for its share of each split and
    writes no dataset.
    When assemble_only is on, every split's dataset is built from checkpoints and no world runs.
    When parallel_workers is above 1, each split runs that many child builds and is then assembled here.
    redis_urls_flag overrides the config's replay_redis_urls for this run, so a multi-Redis build needs
    no tracked-file edit.
    blocks names the committed Level 12f blocks to build; it replaces the per-split counts.
    """
    if blocks is not None and worlds_per_split is not None:
        raise ValueError("--blocks and --worlds-per-split cannot be used together")
    if redis_urls_flag is not None and (parallel_workers is None or parallel_workers <= 1):
        raise ValueError("--replay-redis-urls requires --parallel-workers above 1")
    redis_urls: Sequence[str] = []
    if parallel_workers is not None:
        redis_urls = resolve_replay_redis_urls(redis_urls_flag, load_dataset_build_config())
        validate_parallel_workers(parallel_workers, redis_urls)
    counts = load_datasets_config() if worlds_per_split is None else worlds_per_split
    if blocks is not None:
        print(f"World blocks per split: {blocks} (source: flag)")
    else:
        print(f"Worlds per split: {counts} (source: {'flag' if worlds_per_split is not None else 'config'})")
    sampler_cfg = load_sampler_config()
    world_ranges = read_world_ranges()
    code_ver = get_code_version()

    truth_root: Path | None = None
    if write_evaluation_inputs:
        truth_root = truth_root_from_env(os.environ)

    target_splits = list(splits) if splits is not None else list(blocks if blocks is not None else counts)
    if blocks is not None:
        missing_blocks = [split for split in target_splits if split not in blocks]
        if missing_blocks:
            raise ValueError(
                f"--splits names {', '.join(missing_blocks)}, but --blocks lists {', '.join(sorted(blocks))}"
            )
    plan = plan_run(counts, blocks, sampler_seed, sampler_cfg, world_ranges)

    results: dict[str, dict[str, Any]] = {}
    for split in target_splits:
        specs = plan[split]
        t0 = time.monotonic()
        if world_shard is not None:
            if checkpoint_dir is None:
                raise ValueError("world_shard requires checkpoint_dir")
            stopped = await checkpoint_split_worlds(
                split,
                shard_specs(specs, world_shard),
                preset_name,
                sampler_cfg,
                code_ver,
                sampler_seed=sampler_seed,
                checkpoint_dir=checkpoint_dir,
                db_url=get_showcase_database_url(),
                redis_url=host_side_redis_url(),
                evaluation_truth_root=truth_root if write_evaluation_inputs else None,
                world_cards_root=world_cards_root,
                replay_profile_dir=replay_profile_dir,
            )
            if stopped:
                raise BuildStopped(f"split {split} stopped before every planned world was built")
            continue
        run_parallel = False
        if parallel_workers is not None and parallel_workers > 1:
            run_parallel = True
            if checkpoint_dir is None:
                raise ValueError("parallel_workers requires checkpoint_dir")
            commands = split_worker_commands(
                base_argv, split, specs, checkpoint_dir, parallel_workers, redis_urls
            )
            for command in commands:
                line = f"Worker {command.index}: worlds {', '.join(command.world_ids)}; log {command.log_path}"
                if command.redis_url is not None:
                    line += f"; redis {command.redis_url}"
                print(line)
            codes = run_worker_commands(commands)
            if any(code != 0 for code in codes):
                for command, code in zip(commands, codes):
                    print(f"Worker {command.index} exited {code}")
                    if code != 0:
                        print("  last log lines:")
                        for line in log_tail(command.log_path):
                            print(f"    {line}")
                raise WorkerFailure(worker_failure_exit_code(codes))
        dataset, dataset_dir, superseded = await build_split_dataset(
            split=split,
            specs=specs,
            preset_name=preset_name,
            sampler_cfg=sampler_cfg,
            code_version=code_ver,
            sampler_seed=sampler_seed,
            datasets_root=datasets_root,
            write_evaluation_inputs=write_evaluation_inputs,
            truth_root=truth_root,
            world_cards_root=world_cards_root,
            replay_profile_dir=replay_profile_dir,
            checkpoint_dir=checkpoint_dir,
            assemble_only=assemble_only or run_parallel,
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


def world_shard_argument(text: str) -> WorldShard:
    """WHAT: argparse type for --world-shard. WHY: argparse shows ValueError text only through ArgumentTypeError."""
    try:
        return parse_world_shard(text)
    except ValueError as error:
        raise argparse.ArgumentTypeError(str(error)) from None


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
        "--checkpoint-dir",
        type=str,
        default=None,
        help="Directory for one resumable checkpoint per world (default: no checkpoints)",
    )
    parser.add_argument(
        "--world-shard",
        type=world_shard_argument,
        default=None,
        help="Only build the worlds at plan positions where position %% N == K, given as K/N (needs --checkpoint-dir)",
    )
    parser.add_argument(
        "--assemble-only",
        action="store_true",
        help="Build the dataset from existing checkpoints without running any world (needs --checkpoint-dir)",
    )
    parser.add_argument(
        "--parallel-workers",
        type=int,
        default=None,
        help="Run this many child builds per split, then assemble here (needs --checkpoint-dir; default: one process)",
    )
    parser.add_argument(
        "--replay-redis-urls",
        type=str,
        default=None,
        help="Comma-separated Redis URLs to spread parallel workers over; overrides replay_redis_urls in "
        "config/dataset_build.yaml (default: use the config)",
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
    parser.add_argument(
        "--worlds-per-split",
        type=str,
        default=None,
        help='Override worlds per split for this run, for example "training=6,validation=3,test=3"',
    )
    parser.add_argument(
        "--blocks",
        type=str,
        default=None,
        help='Build committed Level 12f blocks, for example "validation=1-2,test=0-0,training=1-1" '
        "(cannot be combined with --worlds-per-split)",
    )
    args = parser.parse_args()

    checkpoint_dir = Path(args.checkpoint_dir) if args.checkpoint_dir else None
    if (args.world_shard is not None or args.assemble_only) and checkpoint_dir is None:
        parser.error("--world-shard and --assemble-only require --checkpoint-dir")
    if args.world_shard is not None and args.assemble_only:
        parser.error("--world-shard and --assemble-only cannot be used together")
    if args.parallel_workers is not None and checkpoint_dir is None:
        parser.error("--parallel-workers requires --checkpoint-dir")
    if args.parallel_workers is not None and args.world_shard is not None:
        parser.error("--parallel-workers starts the shard workers itself, so --world-shard cannot be given")
    if args.parallel_workers is not None and args.assemble_only:
        parser.error("--parallel-workers already assembles after its workers, so --assemble-only cannot be given")
    if args.blocks is not None and args.worlds_per_split is not None:
        parser.error("--blocks and --worlds-per-split cannot be used together")

    blocks = parse_blocks(args.blocks) if args.blocks is not None else None
    selected_splits = [s.strip() for s in args.splits.split(",")] if args.splits else None
    if blocks is not None and selected_splits is None:
        selected_splits = list(blocks)
    worlds_per_split = parse_worlds_per_split(args.worlds_per_split) if args.worlds_per_split is not None else None
    if args.find_sampler_seed:
        if blocks is not None:
            default_splits = list(blocks)
        elif worlds_per_split is None:
            default_splits = list(load_datasets_config().keys())
        else:
            default_splits = list(worlds_per_split)
        scan_splits = selected_splits if selected_splits is not None else default_splits
        seed = find_fitting_sampler_seed(
            scan_splits,
            args.preset,
            args.sampler_seed,
            args.max_seed_tries,
            datasets_cfg=worlds_per_split,
            blocks=blocks,
        )
        print(f"fitting sampler seed: {seed}")
        return
    try:
        asyncio.run(
            build_all_level_16_datasets(
                splits=selected_splits,
                preset_name=args.preset,
                datasets_root=Path(args.root),
                sampler_seed=args.sampler_seed,
                write_evaluation_inputs=args.write_evaluation_inputs,
                world_cards_root=Path(args.world_cards_root),
                replay_profile_dir=Path(args.replay_profile_dir) if args.replay_profile_dir else None,
                checkpoint_dir=checkpoint_dir,
                worlds_per_split=worlds_per_split,
                world_shard=args.world_shard,
                assemble_only=args.assemble_only,
                parallel_workers=args.parallel_workers,
                base_argv=sys.argv[1:],
                redis_urls_flag=args.replay_redis_urls,
                blocks=blocks,
            )
        )
    except BuildStopped:
        sys.exit(EXIT_STOPPED)
    except WorkerFailure as error:
        sys.exit(error.exit_code)
    except MissingCheckpointsError as error:
        print(f"ERROR: {error}")
        sys.exit(1)


if __name__ == "__main__":
    main()
