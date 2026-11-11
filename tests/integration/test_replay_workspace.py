# File: tests/integration/test_replay_workspace.py
# Purpose: Integration tests for replay workspace schema creation, Kafka topic lifecycle, and isolation guards.
import asyncio
import hashlib
import os
import threading
import uuid
from typing import AsyncGenerator, Generator

import asyncpg
import pytest
import redis
from aiokafka.admin import AIOKafkaAdminClient

from core.config import config
from scripts.run_migrations import find_migration_files
from tests.integration.conftest import get_test_database_url
from world.replay_workspace import (
    ReplayWorkspace,
    allocate_workspace,
    drop_workspace,
    get_default_replay_settings,
    workspace_redis_client,
)

pytestmark = [pytest.mark.anyio, pytest.mark.needs_stack]


def host_side_redis_url() -> str:
    """Resolve Redis connection string for test runner on host machine."""
    raw = os.environ.get("TEST_REDIS_URL") or config.redis_url
    return raw.replace("://redis:", "://localhost:").replace("@redis:", "@localhost:")


def host_redis_factory(ws: ReplayWorkspace) -> redis.Redis:
    """Constructs a Redis client pointing to host-accessible URL for the workspace DB."""
    return workspace_redis_client(ws, base_redis_url=host_side_redis_url())


def snapshot_redis_db(client: redis.Redis) -> tuple[int, str]:
    """Snapshot total DB key count and SHA-256 over sorted key names and their dump values."""
    dbsize = client.dbsize()
    keys = sorted(client.keys("*"))
    hasher = hashlib.sha256()
    for k in keys:
        dump_val = client.dump(k) or b""
        hasher.update(k if isinstance(k, bytes) else k.encode("utf-8"))
        hasher.update(dump_val)
    return dbsize, hasher.hexdigest()


@pytest.fixture
async def pg_conn() -> AsyncGenerator[asyncpg.Connection, None]:
    """Provides a fresh direct asyncpg connection to the test database."""
    conn = await asyncpg.connect(get_test_database_url())
    try:
        yield conn
    finally:
        await conn.close()


@pytest.fixture
def live_redis() -> Generator[redis.Redis, None, None]:
    """Provides real Redis client connected to the host live database."""
    client = redis.Redis.from_url(host_side_redis_url())
    yield client
    client.close()


async def test_guard_1_live_db_isolation(
    pg_conn: asyncpg.Connection,
    live_redis: redis.Redis,
) -> None:
    """
    GUARD 1: Snapshot the live DB before; allocate, write keys into the workspace DB,
    and drop; snapshot after is byte-for-byte identical.
    """
    # Seed key in live DB
    live_redis.set("live_anchor_key", "anchor_payload_value")
    dbsize_before, digest_before = snapshot_redis_db(live_redis)

    run_id = uuid.uuid4().hex[:12]
    ws = await allocate_workspace(
        pg_conn,
        live_redis,
        f"rpl_g1_{run_id}",
        base_redis_url=host_side_redis_url(),
        redis_client_factory=host_redis_factory,
    )

    # Write keys into the workspace DB
    ws_redis = workspace_redis_client(ws, base_redis_url=host_side_redis_url())
    try:
        ws_redis.set("workspace_only_key_1", "replay_state_1")
        ws_redis.set("workspace_only_key_2", "replay_state_2")
        assert ws_redis.get("workspace_only_key_1") == b"replay_state_1"
        assert ws_redis.get("workspace_only_key_2") == b"replay_state_2"
    finally:
        ws_redis.close()

    # Drop workspace
    await drop_workspace(
        pg_conn,
        live_redis,
        ws,
        base_redis_url=host_side_redis_url(),
        redis_client_factory=host_redis_factory,
    )

    # Snapshot live DB after
    dbsize_after, digest_after = snapshot_redis_db(live_redis)
    assert dbsize_after == dbsize_before
    assert digest_after == digest_before


async def test_guard_2_concurrent_allocations_and_registry(
    live_redis: redis.Redis,
) -> None:
    """
    GUARD 2: Allocate twice concurrently -> different DB numbers;
    both drop cleanly; registry entry is removed.
    """
    db_url = get_test_database_url()
    run_id = uuid.uuid4().hex[:12]
    name1 = f"rpl_c1_{run_id}"
    name2 = f"rpl_c2_{run_id}"

    barrier = threading.Barrier(2)

    def _alloc_worker(name: str) -> ReplayWorkspace:
        barrier.wait()
        return asyncio.run(
            allocate_workspace(
                db_url,
                live_redis,
                name,
                base_redis_url=host_side_redis_url(),
                redis_client_factory=host_redis_factory,
            )
        )

    ws1, ws2 = await asyncio.gather(
        asyncio.to_thread(_alloc_worker, name1),
        asyncio.to_thread(_alloc_worker, name2),
    )

    assert ws1.redis_db != ws2.redis_db
    cfg = get_default_replay_settings()
    assert cfg.redis_workspace_db_min <= ws1.redis_db <= cfg.redis_workspace_db_max
    assert cfg.redis_workspace_db_min <= ws2.redis_db <= cfg.redis_workspace_db_max

    # Both drop cleanly
    await drop_workspace(
        db_url,
        live_redis,
        ws1,
        base_redis_url=host_side_redis_url(),
        redis_client_factory=host_redis_factory,
    )
    await drop_workspace(
        db_url,
        live_redis,
        ws2,
        base_redis_url=host_side_redis_url(),
        redis_client_factory=host_redis_factory,
    )

    # Registry has unmapped both
    assert not live_redis.hexists(cfg.registry_key, name1)
    assert not live_redis.hexists(cfg.registry_key, name2)


async def test_postgres_schema_lifecycle_and_migration_count(
    pg_conn: asyncpg.Connection,
    live_redis: redis.Redis,
) -> None:
    """
    Schema exists with all migrations applied (count rows in tracking table = number of migration files);
    after drop the schema is gone and public tables are untouched.
    """
    num_files = len(find_migration_files())
    run_id = uuid.uuid4().hex[:12]
    ws = await allocate_workspace(
        pg_conn,
        live_redis,
        f"rpl_pg_{run_id}",
        base_redis_url=host_side_redis_url(),
        redis_client_factory=host_redis_factory,
    )

    # Schema exists
    exists = await pg_conn.fetchval(
        "SELECT 1 FROM information_schema.schemata WHERE schema_name = $1",
        ws.postgres_schema,
    )
    assert exists == 1

    # Migration count in workspace schema equals number of migration files
    applied_count = await pg_conn.fetchval(
        f"SELECT count(*) FROM {ws.postgres_schema}.schema_migrations"
    )
    assert applied_count == num_files

    # Drop workspace
    await drop_workspace(
        pg_conn,
        live_redis,
        ws,
        base_redis_url=host_side_redis_url(),
        redis_client_factory=host_redis_factory,
    )

    # Schema is completely gone
    exists_after = await pg_conn.fetchval(
        "SELECT 1 FROM information_schema.schemata WHERE schema_name = $1",
        ws.postgres_schema,
    )
    assert exists_after is None


async def test_kafka_topic_lifecycle(
    pg_conn: asyncpg.Connection,
    live_redis: redis.Redis,
) -> None:
    """Kafka topic exists after allocate and is deleted after drop."""
    admin = AIOKafkaAdminClient(bootstrap_servers=config.kafka_bootstrap_servers)
    await admin.start()
    try:
        run_id = uuid.uuid4().hex[:12]
        ws = await allocate_workspace(
            pg_conn,
            live_redis,
            f"rpl_k_{run_id}",
            base_redis_url=host_side_redis_url(),
            redis_client_factory=host_redis_factory,
        )

        # Topic exists
        metadata = await admin.describe_topics([ws.kafka_topic])
        assert any(t["topic"] == ws.kafka_topic for t in metadata)

        # Drop workspace
        await drop_workspace(
            pg_conn,
            live_redis,
            ws,
            base_redis_url=host_side_redis_url(),
            redis_client_factory=host_redis_factory,
        )

        # Topic is gone
        all_topics = await admin.list_topics()
        assert ws.kafka_topic not in all_topics
    finally:
        await admin.close()


async def test_public_and_protected_schemas_cannot_be_dropped(
    pg_conn: asyncpg.Connection,
    live_redis: redis.Redis,
) -> None:
    """Refuse to drop 'public' or protected schemas."""
    ws_pub = ReplayWorkspace(
        name="pub_ws",
        postgres_schema="public",
        redis_db=2,
        kafka_topic="pub_ws.payments",
        kafka_group_id="pub_ws.updater",
    )
    with pytest.raises(ValueError, match="Refusing to drop protected schema 'public'"):
        await drop_workspace(
            pg_conn,
            live_redis,
            ws_pub,
            base_redis_url=host_side_redis_url(),
            redis_client_factory=host_redis_factory,
        )
