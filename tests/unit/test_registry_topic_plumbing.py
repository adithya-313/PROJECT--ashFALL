"""
Tests for the registry topic plumbing: the workspace registry topic (created and dropped with the workspace) and
the national outbox relay's optional topic_map.

WHY: Registry events travel national_outbox -> a workspace-only registry topic; the live relay must stay byte-identical.

TERM: topic_map: optional mapping from a national_outbox row's topic to the Kafka topic it is published to.
"""

import json
from typing import Any

import fakeredis
import pytest

import world.replay_workspace as replay_workspace
from nationalsim.national_outbox import relay_national_outbox
from world.replay_workspace import (
    ReplayWorkspace,
    allocate_workspace,
    drop_workspace,
    get_default_replay_settings,
)

pytestmark = pytest.mark.anyio


class FakeTxn:
    """Async context manager standing in for a database transaction."""

    async def __aenter__(self) -> "FakeTxn":
        return self

    async def __aexit__(self, *exc: object) -> None:
        return None


class FakeConn:
    """Minimal asyncpg-like connection holding national_outbox rows."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        self.executed: list[Any] = []

    def transaction(self) -> FakeTxn:
        return FakeTxn()

    async def fetch(self, *args: Any) -> list[dict[str, Any]]:
        return self.rows

    async def execute(self, *args: Any) -> None:
        self.executed.append(args)


class FakeProducer:
    """Records every (topic, key, value) send."""

    def __init__(self) -> None:
        self.sent: list[tuple[str, bytes, bytes]] = []

    async def send_and_wait(self, topic: str, key: bytes, value: bytes) -> None:
        self.sent.append((topic, key, value))


def outbox_rows() -> list[dict[str, Any]]:
    """Two staged rows: one registry event, one hold request."""
    return [
        {"id": 1, "event_id": "e1", "run_id": "r", "topic": "registry.events", "bank_key": "r:bank_0",
         "schema_version": "v1", "payload": json.dumps({"a": 1})},
        {"id": 2, "event_id": "e2", "run_id": "r", "topic": "hold.requests", "bank_key": "r:bank_1",
         "schema_version": "v1", "payload": json.dumps({"b": 2})},
    ]


async def test_topic_map_none_publishes_each_row_to_its_own_topic() -> None:
    """Checks a relay without topic_map sends to the row topics, as before."""
    producer = FakeProducer()
    published = await relay_national_outbox(FakeConn(outbox_rows()), producer)  # type: ignore[arg-type]
    assert published == 2
    assert [t for t, _, _ in producer.sent] == ["registry.events", "hold.requests"]


async def test_topic_map_redirects_only_mapped_topics() -> None:
    """Checks a mapped topic is used when given and an unmapped row keeps its own topic."""
    producer = FakeProducer()
    await relay_national_outbox(
        FakeConn(outbox_rows()), producer,  # type: ignore[arg-type]
        topic_map={"registry.events": "ws1.registry.events"},
    )
    assert [t for t, _, _ in producer.sent] == ["ws1.registry.events", "hold.requests"]
    assert producer.sent[0][1:] == (b"r:bank_0", b'{"a": 1}')


def test_registry_topic_is_named_after_the_payments_topic() -> None:
    """Checks the registry topic name is derived from kafka_topic, with no new required field."""
    ws = ReplayWorkspace("ws1", "ws1", 1, "ws1.payments", "ws1.updater")
    assert ws.registry_topic == "ws1.registry.events"


class FakeAdminClient:
    """Records topic creation and deletion requests."""

    created: list[str] = []
    deleted: list[str] = []

    def __init__(self, **kwargs: Any) -> None:
        pass

    async def start(self) -> None:
        return None

    async def close(self) -> None:
        return None

    async def create_topics(self, topics: list[Any]) -> None:
        FakeAdminClient.created.extend(t.name for t in topics)

    async def delete_topics(self, names: list[str]) -> None:
        FakeAdminClient.deleted.extend(names)


class FakePg:
    """admin_pg stand-in that accepts the schema statements."""

    async def execute(self, *args: Any) -> None:
        return None

    async def fetchval(self, *args: Any) -> str:
        return "public"


async def test_workspace_create_and_drop_include_the_registry_topic(monkeypatch: pytest.MonkeyPatch) -> None:
    """Checks provisioning creates both topics and dropping deletes both."""
    async def no_migrations(conn: Any) -> None:
        return None

    monkeypatch.setattr(replay_workspace, "AIOKafkaAdminClient", FakeAdminClient)
    monkeypatch.setattr(replay_workspace, "apply_pending_migrations", no_migrations)
    FakeAdminClient.created, FakeAdminClient.deleted = [], []
    server = fakeredis.FakeServer()
    client = fakeredis.FakeRedis(server=server, db=0)
    settings = get_default_replay_settings()

    def factory(ws: ReplayWorkspace) -> Any:
        return fakeredis.FakeRedis(server=server, db=ws.redis_db)

    ws = await allocate_workspace(FakePg(), client, "rpl_reg_1", settings=settings, redis_client_factory=factory)
    assert FakeAdminClient.created == ["rpl_reg_1.payments", "rpl_reg_1.registry.events"]
    await drop_workspace(FakePg(), client, ws, settings=settings, redis_client_factory=factory)
    assert FakeAdminClient.deleted == ["rpl_reg_1.payments", "rpl_reg_1.registry.events"]
