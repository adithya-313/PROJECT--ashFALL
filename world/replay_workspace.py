# File: world/replay_workspace.py
# Purpose: Level 16b isolated replay workspaces across PostgreSQL, Redis, and Kafka.
"""
WHAT: Provides lifecycle management (allocation, migration, and teardown) for isolated replay
workspaces across PostgreSQL schemas, Redis database numbers, and Kafka topics.

WHY: Simulation replays must evaluate models and policies against historical data without
polluting live Redis feature states, corrupting PostgreSQL tables, or interleaving Kafka streams.
Each replay runs in a dedicated workspace bounded by strict database and schema isolation guards.
"""

import asyncio
import logging
import os
import re
import time
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from aiokafka.admin import AIOKafkaAdminClient, NewTopic

from core.config import config
from core.config_validator import load_yaml
from scripts.run_migrations import apply_pending_migrations

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parent.parent
CONFIG_PATH = REPO_ROOT / "config" / "replay.yaml"

UNLOCK_LUA = """
if redis.call("get", KEYS[1]) == ARGV[1] then
    return redis.call("del", KEYS[1])
else
    return 0
end
"""

__all__ = [
    "ReplaySettings",
    "ReplayWorkspace",
    "allocate_workspace",
    "drop_workspace",
    "get_default_replay_settings",
    "get_live_redis_db",
    "load_replay_settings",
    "parse_redis_db_index",
    "workspace_redis_client",
    "workspace_redis_url",
]


@dataclass(frozen=True)
class ReplaySettings:
    """Configuration parameters for isolated replay workspaces."""

    redis_workspace_db_min: int
    redis_workspace_db_max: int
    registry_key: str
    registry_lock_seconds: int
    drain_timeout_seconds: int
    kafka_topic_partitions: int


@dataclass(frozen=True)
class ReplayWorkspace:
    """An isolated operational workspace for historical simulation replay."""

    name: str
    postgres_schema: str
    redis_db: int
    kafka_topic: str
    kafka_group_id: str


def load_replay_settings(config_path: Path | str | None = None) -> ReplaySettings:
    """
    WHAT: Loads replay workspace settings from config/replay.yaml with strict boundary validation.

    WHY: Prevents hardcoded magic numbers and ensures workspace allocation parameters adhere
    to validated policy constraints.
    """
    path = Path(config_path) if config_path is not None else CONFIG_PATH
    data = load_yaml(path)
    if not isinstance(data, dict) or "replay_workspace" not in data:
        raise ValueError(f"Missing 'replay_workspace' root block in {path}")

    block = data["replay_workspace"]
    db_min = int(block["redis_workspace_db_min"]["value"])
    db_max = int(block["redis_workspace_db_max"]["value"])
    reg_key = str(block["registry_key"]["value"]).strip()
    lock_sec = int(block["registry_lock_seconds"]["value"])
    drain_sec = int(block["drain_timeout_seconds"]["value"])
    partitions = int(block["kafka_topic_partitions"]["value"])

    if db_min < 1:
        raise ValueError(f"redis_workspace_db_min must be >= 1, got {db_min}")
    if db_max < db_min:
        raise ValueError(f"redis_workspace_db_max ({db_max}) must be >= redis_workspace_db_min ({db_min})")
    if not reg_key:
        raise ValueError("registry_key must be a non-empty string")
    if lock_sec <= 0:
        raise ValueError(f"registry_lock_seconds must be positive, got {lock_sec}")
    if drain_sec <= 0:
        raise ValueError(f"drain_timeout_seconds must be positive, got {drain_sec}")
    if partitions < 1:
        raise ValueError(f"kafka_topic_partitions must be >= 1, got {partitions}")

    return ReplaySettings(
        redis_workspace_db_min=db_min,
        redis_workspace_db_max=db_max,
        registry_key=reg_key,
        registry_lock_seconds=lock_sec,
        drain_timeout_seconds=drain_sec,
        kafka_topic_partitions=partitions,
    )


@lru_cache(maxsize=1)
def get_default_replay_settings() -> ReplaySettings:
    """Return cached default ReplaySettings loaded from config/replay.yaml."""
    return load_replay_settings()


def parse_redis_db_index(redis_url: str) -> int:
    """
    WHAT: Extracts the integer database index from a Redis connection URL path.

    WHY: Live DB index must be parsed dynamically from config.redis_url rather than hardcoded
    to ensure live traffic and replay traffic are strictly partitioned.
    """
    parsed = urlparse(redis_url)
    path = parsed.path.lstrip("/")
    if not path:
        return 0
    try:
        return int(path)
    except ValueError as exc:
        raise ValueError(f"Invalid Redis DB index in URL {redis_url!r}: {exc}") from exc


def get_live_redis_db(live_redis: Any = None) -> int:
    """
    WHAT: Resolves the live Redis database index from a live client or application configuration.

    WHY: Ensures workspace allocation and teardown never target the live database.
    """
    if live_redis is not None and hasattr(live_redis, "connection_pool"):
        db = live_redis.connection_pool.connection_kwargs.get("db")
        if db is not None:
            return int(db)
    return parse_redis_db_index(config.redis_url)


def workspace_redis_url(redis_url: str, db_index: int) -> str:
    """
    WHAT: Returns a Redis connection URL with the path replaced by the specified DB index.

    WHY: Generates isolated connection strings for workspace-specific Redis clients.
    """
    parsed = urlparse(redis_url)
    return parsed._replace(path=f"/{db_index}").geturl()


def workspace_redis_client(workspace: ReplayWorkspace, base_redis_url: str | None = None) -> Any:
    """
    WHAT: Constructs a Redis client pointing exclusively to the workspace's assigned DB number.

    WHY: Enforces that workspace Redis clients are built ONLY through this factory.
    """
    import redis

    base_url = base_redis_url if base_redis_url is not None else config.redis_url
    target_url = workspace_redis_url(base_url, workspace.redis_db)
    return redis.Redis.from_url(target_url)


async def _acquire_registry_lock(live_redis: Any, lock_key: str, token: str, lock_seconds: int) -> None:
    """Acquire a distributed lock on live Redis using SET NX with timeout."""
    deadline = time.monotonic() + lock_seconds
    while time.monotonic() < deadline:
        acquired = live_redis.set(lock_key, token, nx=True, ex=lock_seconds)
        if acquired:
            return
        await asyncio.sleep(0.05)
    raise TimeoutError(f"Could not acquire workspace registry lock {lock_key!r} within {lock_seconds}s")


def _release_registry_lock(live_redis: Any, lock_key: str, token: str) -> None:
    """Release distributed lock safely using token verification."""
    try:
        live_redis.eval(UNLOCK_LUA, 1, lock_key, token)
    except Exception as exc:
        logger.warning("Error releasing registry lock %s: %s", lock_key, exc)


async def allocate_workspace(
    admin_pg: Any,
    live_redis: Any,
    base_name: str,
    settings: ReplaySettings | None = None,
    live_db_override: int | None = None,
) -> ReplayWorkspace:
    """
    WHAT: Allocates an isolated ReplayWorkspace with a dedicated Redis DB, Postgres schema, and Kafka topic.

    WHY: Guarantees zero state collision with live traffic and concurrent replay sessions.
    Acquires a distributed lock in live Redis, allocates the lowest available DB index in [min, max]
    excluding the live DB, creates and migrates the PostgreSQL schema, and registers the Kafka topic.
    """
    cfg = settings if settings is not None else get_default_replay_settings()
    if not isinstance(base_name, str) or not base_name.strip():
        raise ValueError(f"Invalid base_name {base_name!r}: must be non-empty string")

    name = base_name.strip().lower()
    if not re.match(r"^[a-z0-9_]+$", name):
        raise ValueError(
            f"Invalid workspace name {base_name!r}: must contain only lowercase letters, digits, and underscores"
        )

    lock_key = f"{cfg.registry_key}:lock"
    token = os.urandom(16).hex()
    await _acquire_registry_lock(live_redis, lock_key, token, cfg.registry_lock_seconds)

    try:
        live_db = live_db_override if live_db_override is not None else get_live_redis_db(live_redis)
        raw_registry = live_redis.hgetall(cfg.registry_key)
        allocated: dict[str, int] = {}
        for k, v in raw_registry.items():
            holder = k.decode("utf-8") if isinstance(k, bytes) else str(k)
            db_num = int(v.decode("utf-8") if isinstance(v, bytes) else v)
            allocated[holder] = db_num

        if name in allocated:
            raise ValueError(f"Workspace {name!r} is already allocated (DB {allocated[name]})")

        used_dbs = set(allocated.values())
        chosen_db: int | None = None
        for candidate in range(cfg.redis_workspace_db_min, cfg.redis_workspace_db_max + 1):
            if candidate == live_db:
                continue
            if candidate not in used_dbs:
                chosen_db = candidate
                break

        if chosen_db is None:
            holders_desc = ", ".join(f"{h}: DB {d}" for h, d in sorted(allocated.items()))
            db_range_str = f"[{cfg.redis_workspace_db_min}, {cfg.redis_workspace_db_max}]"
            raise RuntimeError(
                f"No free Redis databases available in range {db_range_str} "
                f"(live DB is {live_db}). Active holders: {holders_desc}"
            )

        live_redis.hset(cfg.registry_key, name, str(chosen_db))
    finally:
        _release_registry_lock(live_redis, lock_key, token)

    workspace = ReplayWorkspace(
        name=name,
        postgres_schema=name,
        redis_db=chosen_db,
        kafka_topic=f"{name}.payments",
        kafka_group_id=f"{name}.updater",
    )

    # If database and messaging are supplied, provision Postgres schema and Kafka topic
    if admin_pg is not None:
        try:
            # 1. Postgres schema creation and migrations
            if isinstance(admin_pg, str):
                import asyncpg

                admin_conn = await asyncpg.connect(admin_pg)
                try:
                    await admin_conn.execute(f"CREATE SCHEMA {workspace.postgres_schema}")
                finally:
                    await admin_conn.close()

                mig_conn = await asyncpg.connect(admin_pg, server_settings={"search_path": workspace.postgres_schema})
                try:
                    await apply_pending_migrations(mig_conn)
                finally:
                    await mig_conn.close()
            elif hasattr(admin_pg, "execute"):
                await admin_pg.execute(f"CREATE SCHEMA {workspace.postgres_schema}")
                orig_path = await admin_pg.fetchval("SHOW search_path")
                try:
                    await admin_pg.execute(f"SET search_path = {workspace.postgres_schema}")
                    await apply_pending_migrations(admin_pg)
                finally:
                    await admin_pg.execute(f"SET search_path = {orig_path}")

            # 2. Kafka topic creation
            admin_client = AIOKafkaAdminClient(bootstrap_servers=config.kafka_bootstrap_servers)
            await admin_client.start()
            try:
                topic = NewTopic(
                    name=workspace.kafka_topic,
                    num_partitions=cfg.kafka_topic_partitions,
                    replication_factor=1,
                )
                await admin_client.create_topics([topic])
            finally:
                await admin_client.close()

        except BaseException as exc:
            # Rollback allocation on setup failure
            logger.error("Failed provisioning resources for workspace %s: %s; cleaning up", name, exc)
            await drop_workspace(admin_pg, live_redis, workspace, settings=cfg, live_db_override=live_db_override)
            raise

    return workspace


async def drop_workspace(
    admin_pg: Any,
    live_redis: Any,
    workspace: ReplayWorkspace,
    settings: ReplaySettings | None = None,
    live_db_override: int | None = None,
) -> None:
    """
    WHAT: Tears down an isolated ReplayWorkspace and frees its resources.

    WHY: Enforces strict safety guards: refuses to FLUSHDB on the live Redis database, refuses
    to drop public or system schemas, purges throwaway Redis state and Kafka topics, and unregisters
    the workspace under a distributed lock.
    """
    cfg = settings if settings is not None else get_default_replay_settings()
    live_db = live_db_override if live_db_override is not None else get_live_redis_db(live_redis)

    # GUARD 1: Refuse to FLUSHDB on live Redis DB
    if workspace.redis_db == live_db:
        raise ValueError(f"Refusing to FLUSHDB on live Redis database {live_db}")

    # GUARD: Refuse to drop public or protected schemas
    schema = workspace.postgres_schema.lower()
    if schema in ("public", "information_schema", "pg_catalog"):
        raise ValueError(f"Refusing to drop protected schema {schema!r}")

    # 1. Flush Redis workspace DB
    fake_server = None
    if hasattr(live_redis, "connection_pool"):
        fake_server = live_redis.connection_pool.connection_kwargs.get("server")

    if fake_server is not None:
        import fakeredis

        ws_redis = fakeredis.FakeRedis(server=fake_server, db=workspace.redis_db)
        ws_redis.flushdb()
    else:
        ws_redis = workspace_redis_client(workspace)
        try:
            ws_redis.flushdb()
        finally:
            ws_redis.close()

    # 2. Drop PostgreSQL schema
    if admin_pg is not None:
        if isinstance(admin_pg, str):
            import asyncpg

            admin_conn = await asyncpg.connect(admin_pg)
            try:
                await admin_conn.execute(f"DROP SCHEMA IF EXISTS {workspace.postgres_schema} CASCADE")
            finally:
                await admin_conn.close()
        elif hasattr(admin_pg, "execute"):
            await admin_pg.execute(f"DROP SCHEMA IF EXISTS {workspace.postgres_schema} CASCADE")

    # 3. Delete Kafka topic
    if admin_pg is not None:
        admin_client = AIOKafkaAdminClient(bootstrap_servers=config.kafka_bootstrap_servers)
        try:
            await admin_client.start()
            await admin_client.delete_topics([workspace.kafka_topic])
        except Exception as exc:
            logger.warning("Could not delete kafka topic %s: %s", workspace.kafka_topic, exc)
        finally:
            try:
                await admin_client.close()
            except Exception:
                pass

    # 4. Remove from registry under distributed lock
    lock_key = f"{cfg.registry_key}:lock"
    token = os.urandom(16).hex()
    await _acquire_registry_lock(live_redis, lock_key, token, cfg.registry_lock_seconds)
    try:
        live_redis.hdel(cfg.registry_key, workspace.name)
    finally:
        _release_registry_lock(live_redis, lock_key, token)
