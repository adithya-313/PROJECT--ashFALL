# File: tests/unit/test_replay_workspace.py
# Purpose: Unit tests for replay workspace config loading, URL parsing, and Redis DB allocation with live-DB guards.
import fakeredis
import pytest

from world.replay_workspace import (
    ReplaySettings,
    ReplayWorkspace,
    allocate_workspace,
    drop_workspace,
    get_default_replay_settings,
    load_replay_settings,
    parse_redis_db_index,
    workspace_redis_url,
)

pytestmark = pytest.mark.anyio


def test_parse_redis_db_index_from_urls() -> None:
    """Live DB number is parsed correctly from URLs with and without explicit /N path."""
    assert parse_redis_db_index("redis://localhost:6379/0") == 0
    assert parse_redis_db_index("redis://localhost:6379/3") == 3
    assert parse_redis_db_index("redis://localhost:6379") == 0
    assert parse_redis_db_index("redis://localhost:6379/") == 0
    assert parse_redis_db_index("rediss://user:secret@cache.internal:6380/14") == 14

    with pytest.raises(ValueError, match="Invalid Redis DB index"):
        parse_redis_db_index("redis://localhost:6379/invalid_num")


def test_workspace_redis_url_replaces_db_index() -> None:
    """workspace_redis_url swaps the path to the assigned DB number."""
    assert workspace_redis_url("redis://localhost:6379/0", 5) == "redis://localhost:6379/5"
    assert workspace_redis_url("redis://localhost:6379", 3) == "redis://localhost:6379/3"
    assert workspace_redis_url("rediss://user:pass@remote:6380/1", 12) == "rediss://user:pass@remote:6380/12"


def test_load_replay_settings_validates_bounds() -> None:
    """Config loader enforces min >= 1 and max >= min."""
    cfg = load_replay_settings()
    assert cfg.redis_workspace_db_min == 1
    assert cfg.redis_workspace_db_max == 15
    assert cfg.registry_key == "replay:workspace_registry"
    assert cfg.registry_lock_seconds == 10
    assert cfg.drain_timeout_seconds == 30
    assert cfg.kafka_topic_partitions == 1
    assert get_default_replay_settings() == cfg


async def test_allocation_never_returns_live_db() -> None:
    """
    Allocation never returns the live DB number, even when the live DB
    lies inside the [redis_workspace_db_min, redis_workspace_db_max] range.
    """
    server = fakeredis.FakeServer()
    # Live DB is 1 (inside range [1, 5])
    client = fakeredis.FakeRedis(server=server, db=1)
    custom_settings = ReplaySettings(
        redis_workspace_db_min=1,
        redis_workspace_db_max=5,
        registry_key="replay:test_registry",
        registry_lock_seconds=5,
        drain_timeout_seconds=30,
        kafka_topic_partitions=1,
    )

    ws = await allocate_workspace(
        admin_pg=None,
        live_redis=client,
        base_name="rpl_run01_1",
        settings=custom_settings,
        live_db_override=1,
    )
    # 1 is live DB, so lowest free DB must be 2
    assert ws.redis_db != 1
    assert ws.redis_db == 2
    assert ws.name == "rpl_run01_1"
    assert ws.postgres_schema == "rpl_run01_1"
    assert ws.kafka_topic == "rpl_run01_1.payments"
    assert ws.kafka_group_id == "rpl_run01_1.updater"


async def test_two_allocations_get_different_numbers_and_release_frees_number() -> None:
    """
    Two allocations get different DB numbers; releasing one frees its number
    to be reused by the next allocation.
    """
    server = fakeredis.FakeServer()
    client = fakeredis.FakeRedis(server=server, db=0)
    settings = get_default_replay_settings()

    ws1 = await allocate_workspace(None, client, "rpl_run02_1", settings=settings)
    ws2 = await allocate_workspace(None, client, "rpl_run02_2", settings=settings)

    assert ws1.redis_db == 1
    assert ws2.redis_db == 2

    # Drop ws1 to free DB 1
    await drop_workspace(None, client, ws1, settings=settings)

    # Next allocation reclaims lowest free DB (1)
    ws3 = await allocate_workspace(None, client, "rpl_run02_3", settings=settings)
    assert ws3.redis_db == 1


async def test_full_pool_raises_error_naming_holders() -> None:
    """When all DB numbers in range are exhausted, raises RuntimeError listing active holders."""
    server = fakeredis.FakeServer()
    client = fakeredis.FakeRedis(server=server, db=0)
    tight_settings = ReplaySettings(
        redis_workspace_db_min=1,
        redis_workspace_db_max=2,
        registry_key="replay:tight_registry",
        registry_lock_seconds=5,
        drain_timeout_seconds=30,
        kafka_topic_partitions=1,
    )

    await allocate_workspace(None, client, "ws_alice", settings=tight_settings)
    await allocate_workspace(None, client, "ws_bob", settings=tight_settings)

    with pytest.raises(RuntimeError, match="No free Redis databases available") as exc:
        await allocate_workspace(None, client, "ws_charlie", settings=tight_settings)

    msg = str(exc.value)
    assert "Active holders:" in msg
    assert "ws_alice" in msg
    assert "ws_bob" in msg


async def test_drop_on_live_db_raises_value_error_and_flushes_nothing() -> None:
    """
    GUARD 1: Calling drop_workspace with workspace.redis_db == live_db raises
    ValueError immediately and flushes nothing from the live database.
    """
    server = fakeredis.FakeServer()
    live_client = fakeredis.FakeRedis(server=server, db=0)
    live_client.set("customer_account_state:live", "do_not_flush")

    ws_live = ReplayWorkspace(
        name="ws_fake_live",
        postgres_schema="ws_fake_live",
        redis_db=0,
        kafka_topic="ws_fake_live.payments",
        kafka_group_id="ws_fake_live.updater",
    )

    with pytest.raises(ValueError, match="Refusing to FLUSHDB on live Redis database"):
        await drop_workspace(None, live_client, ws_live, live_db_override=0)

    # Assert live key is completely intact
    assert live_client.get("customer_account_state:live") == b"do_not_flush"


async def test_lock_released_when_allocation_raises() -> None:
    """Distributed registry lock is released in finally even when allocation raises."""
    server = fakeredis.FakeServer()
    client = fakeredis.FakeRedis(server=server, db=0)
    settings = get_default_replay_settings()
    lock_key = f"{settings.registry_key}:lock"

    # Allocate a valid workspace
    await allocate_workspace(None, client, "ws_existing", settings=settings)

    # Attempting to allocate duplicate name raises ValueError
    with pytest.raises(ValueError, match="already allocated"):
        await allocate_workspace(None, client, "ws_existing", settings=settings)

    # Lock must be released (None in Redis)
    assert client.get(lock_key) is None
