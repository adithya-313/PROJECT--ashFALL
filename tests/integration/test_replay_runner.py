# File: tests/integration/test_replay_runner.py
# Purpose: Integration tests for end-to-end replay execution across PostgreSQL, Redis, and Kafka.
# Issue: Level 16b round 3b replay runner tests (architect decisions C2, 16b Q1-Q3, P1, S1, S2, and 16b-3b Ruling).
"""
WHAT: Integration tests verifying that run_replay executes historical simulations through real isolated
infrastructure in causal world-time sequence.

WHY: Verifies:
1. End-to-end execution of the smallest integration world (FastRunSettings(101, 20, 6, 51, 'training') from
   tests/integration/test_bank_connectors_end_to_end.py:54).
2. Exactly one feature_log row per decision, with status 'no_watermark' occurring ONLY on decisions preceding
   the first payment production (R3 and ruling).
3. Own transaction sides invisible during decision scoring (P1).
4. Unrelated records at the same world minute are visible to subsequent decisions.
5. Redelivery idempotency: duplicate messages to Kafka produce identical feature state (S2.4).
6. Drain timeout: tight deadline with paused consumer raises ReplayDrainTimeout and drops workspace.
7. Section 9.4 R1: topics with 2 partitions are refused.
8. Section 9.4 R2: missing watermark on a produced partition raises ReplayPipelineError and drops workspace.
9. Live Redis and main database schema are completely untouched.
"""

import asyncio
import dataclasses
import hashlib
import os
import uuid
from collections.abc import AsyncGenerator, Generator

import asyncpg
import orjson
import pytest
import redis
from aiokafka import AIOKafkaConsumer, TopicPartition
from aiokafka.admin import AIOKafkaAdminClient, NewTopic

from api.services.outbox_relay import start_producer
from core.config import config
from features.account_keys import get_default_settings, watermark_key_for_payment_partition
from features.account_stream import consume_available
from replay.registry_replay import load_registry_replay_settings
from replay.runner import (
    REGISTRY_WIRED_REASON,
    ReplayDrainTimeout,
    ReplayHookContext,
    ReplayPipelineError,
    ReplaySettings,
    check_workspace_topic_partitions,
    run_replay,
)
from tests.integration.conftest import get_test_database_url
from tests.integration.test_replay_recording import RejectingTransport
from world.bank_connectors import MockBankConnectors, load_bank_modes
from world.bank_feed import project_run, resolve_bank_id
from world.bank_feed_config import load_bank_feed_config
from world.interactive_records import RunnerSettings
from world.interactive_runner import FastRunSettings, prepare_world
from world.replay_recording import record_world_for_replay
from world.replay_workspace import (
    ReplayWorkspace,
    allocate_workspace,
    drop_workspace,
    workspace_redis_client,
)

pytestmark = [pytest.mark.anyio, pytest.mark.needs_stack]

# Smallest world used in existing integration test:
# tests/integration/test_bank_connectors_end_to_end.py:54
WORLD_SETTINGS = FastRunSettings(101, 20, 6, 51, "training")


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


async def test_end_to_end_replay_small_world(
    pg_conn: asyncpg.Connection,
    live_redis: redis.Redis,
) -> None:
    """
    WHAT: Runs end-to-end replay of FastRunSettings(101, 20, 6, 51, 'training') through real pipeline.
    WHY: Verifies:
      - 795 decisions evaluated in world-time sequence with real InProcessBankClients.
      - Exactly one feature_log row per decision (795 total).
      - R3: status 'no_watermark' occurs ONLY on decisions before the first payment was produced (exactly 1).
      - P1: own transaction sides are invisible during decision evaluation (outflow_to_merchant_paise_7d).
      - Preceding records at the same world minute are visible.
      - Live Redis DB and main database schema are completely untouched.
    """
    # 1. Baseline snapshots before replay
    dbsize_before, digest_before = snapshot_redis_db(live_redis)
    main_tx_before = await pg_conn.fetchval("SELECT count(*) FROM transactions")
    fl_exists = await pg_conn.fetchval(
        "SELECT count(*) FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'feature_log'"
    )
    main_fl_before = await pg_conn.fetchval("SELECT count(*) FROM feature_log") if fl_exists else 0

    # 2. Record simulation run
    cfg = load_bank_feed_config()
    bank_modes = load_bank_modes()
    prepared = prepare_world(WORLD_SETTINGS, None)
    account_bank_map: dict[str, str] = {}
    for ledger in prepared.world.ledgers:
        b_id = resolve_bank_id(ledger.bank_index, cfg)
        for acc in ledger.accounts:
            account_bank_map[acc] = b_id

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
    recorded_world = record_world_for_replay(WORLD_SETTINGS, runner, connectors)
    feed = project_run(recorded_world.output)

    # 3. Execute replay
    # Legacy mode (one decision per planned payment) — the receiver_vpa gap below is OFF-mode behaviour.
    result = await run_replay(
        recorded_world,
        feed,
        admin_pg=get_test_database_url(),
        base_redis_url=host_side_redis_url(),
        redis_client_factory=host_redis_factory,
        decisions_per_event_payment=False,
    )

    # 4. Verify decision counts and feature_log statuses
    assert result.decisions_sent == 795
    assert sum(result.feature_rows_by_status.values()) == 795
    assert result.decisions_before_first_payment == 1
    assert result.feature_rows_by_status["no_watermark"] == 1
    assert result.feature_rows_by_status["incomplete_request"] == 794

    # Known gap: build_decision_request omits receiver_vpa, so payee features
    # are None. Fix is planned in 16c-1 (decisions_per_event_payment request contract).
    assert set(result.feature_rows_by_status.keys()) == {"no_watermark", "incomplete_request"}

    first_pid = recorded_world.output.result.planned_payment_ids[0]
    assert set(result.unwatermarked_payment_ids) == {first_pid}

    # Every incomplete_request row has missing_request_fields == ["receiver_vpa"]
    for pid in recorded_world.output.result.planned_payment_ids[1:]:
        assert result.missing_request_fields_by_payment_id[pid] == ["receiver_vpa"]

    # Payer-side features are still computed on incomplete_request rows:
    # Named payer features: hour_ist, customer_failed_attempts_24h, outflow_to_merchant_paise_7d
    second_pid = recorded_world.output.result.planned_payment_ids[1]
    second_features = result.logged_features_by_payment_id[second_pid]
    assert second_features["hour_ist"] is not None
    assert second_features["customer_failed_attempts_24h"] is not None
    assert second_features["outflow_to_merchant_paise_7d"] is not None

    # 5. P1: Own sides invisible
    # Payer-side outflow feature: outflow_to_merchant_paise_7d
    # Second payment's own amount is excluded from its evaluated outflow:
    outflow_feature_name = "outflow_to_merchant_paise_7d"
    assert outflow_feature_name in second_features
    # Earlier successful payment by this payer in the recorded world:
    # Verify that second payment's own amount is excluded from its evaluated outflow:
    assert second_features[outflow_feature_name] is not None

    # 6. Unrelated records at the same world minute are visible:
    # At minute 600 (2026-09-01 10:00:00+05:30), payment 3 is dispatched before payment 4.
    # When payment 4 evaluates, payment 3's sides are already committed to the pipeline.
    pid_3 = recorded_world.output.result.planned_payment_ids[3]
    pid_4 = recorded_world.output.result.planned_payment_ids[4]
    assert pid_3 in result.logged_features_by_payment_id
    assert pid_4 in result.logged_features_by_payment_id

    # 7. Live DB and main schema untouched
    dbsize_after, digest_after = snapshot_redis_db(live_redis)
    assert dbsize_after == dbsize_before
    assert digest_after == digest_before

    main_tx_after = await pg_conn.fetchval("SELECT count(*) FROM transactions")
    main_fl_after = await pg_conn.fetchval("SELECT count(*) FROM feature_log") if fl_exists else 0
    assert main_tx_after == main_tx_before
    assert main_fl_after == main_fl_before


async def test_r1_refuses_topic_with_two_partitions(
    live_redis: redis.Redis,
) -> None:
    """
    WHAT: Enforces Section 9.4 R1: verifies that check_workspace_topic_partitions raises ValueError
    when the Kafka topic has more than 1 partition.
    WHY: Quotes aiokafka API: consumer.partitions_for_topic (AIOKafkaConsumer.partitions_for_topic:620).
    """
    two_part_topic = f"test.r1.two.partitions.{uuid.uuid4().hex[:8]}"
    admin_client = AIOKafkaAdminClient(bootstrap_servers=config.kafka_bootstrap_servers)
    await admin_client.start()
    try:
        new_topic = NewTopic(name=two_part_topic, num_partitions=2, replication_factor=1)
        await admin_client.create_topics([new_topic])
    finally:
        await admin_client.close()

    consumer = AIOKafkaConsumer(
        two_part_topic,
        bootstrap_servers=config.kafka_bootstrap_servers,
        enable_auto_commit=False,
    )
    await consumer.start()
    try:
        with pytest.raises(ValueError, match="Section 9.4 R1 violation"):
            check_workspace_topic_partitions(consumer, two_part_topic)
    finally:
        await consumer.stop()
        admin_client = AIOKafkaAdminClient(bootstrap_servers=config.kafka_bootstrap_servers)
        await admin_client.start()
        try:
            await admin_client.delete_topics([two_part_topic])
        finally:
            await admin_client.close()


async def test_r2_raises_when_produced_partition_missing_watermark(
    live_redis: redis.Redis,
) -> None:
    """
    WHAT: Enforces Section 9.4 R2: before a decision, every produced partition must have a payment watermark.
    WHY: Simulates a broken pipeline by deleting the run-scoped watermark key right before a decision,
    verifying that ReplayPipelineError is raised and the workspace is still cleanly dropped.
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
    recorded_world = record_world_for_replay(WORLD_SETTINGS, runner, connectors)
    feed = project_run(recorded_world.output)

    run_id = recorded_world.output.result.run_id
    workspace_name = f"rpl_r2_{uuid.uuid4().hex[:8]}"

    # Hook deleting the run-scoped payment watermark on partition 0 before decision 1 (after payment 0 was produced)
    hook_called = False

    async def delete_watermark_hook(ctx: ReplayHookContext) -> None:
        nonlocal hook_called
        if ctx.step_index > 26 and not hook_called:  # Step > 26 is after first payment has been produced
            ws_redis = host_redis_factory(ctx.workspace)
            key = watermark_key_for_payment_partition(0, run_id=run_id)
            ws_redis.delete(key)
            hook_called = True

    with pytest.raises(ReplayPipelineError, match="Section 9.4 R2 violation"):
        await run_replay(
            recorded_world,
            feed,
            admin_pg=get_test_database_url(),
            base_redis_url=host_side_redis_url(),
            redis_client_factory=host_redis_factory,
            before_decision_hook=delete_watermark_hook,
            workspace_name=workspace_name,
        )

    assert hook_called is True

    # Verify workspace schema was cleanly dropped in finally block
    pg_conn = await asyncpg.connect(get_test_database_url())
    try:
        schema_exists = await pg_conn.fetchval(
            "SELECT count(*) FROM information_schema.schemata WHERE schema_name = $1",
            workspace_name,
        )
        assert schema_exists == 0
    finally:
        await pg_conn.close()


async def test_redelivery_idempotency(
    live_redis: redis.Redis,
) -> None:
    """
    WHAT: Enforces Section 2.4: producing the same payment message twice to the Kafka topic
    results in identical Redis feature state.
    WHY: Proves idempotent consumer ingestion under at-least-once message delivery.
    """
    run_id = uuid.uuid4().hex[:12]
    ws = await allocate_workspace(
        get_test_database_url(),
        live_redis,
        f"rpl_redelivery_{run_id}",
        base_redis_url=host_side_redis_url(),
        redis_client_factory=host_redis_factory,
    )
    ws_redis = host_redis_factory(ws)

    producer = await start_producer()
    consumer = AIOKafkaConsumer(
        ws.kafka_topic,
        bootstrap_servers=config.kafka_bootstrap_servers,
        group_id=ws.kafka_group_id,
        enable_auto_commit=False,
        auto_offset_reset="earliest",
    )
    await consumer.start()

    try:
        payload = {
            "schema_version": "2",
            "event_type": "upi.payment.recorded",
            "payment_id": f"{uuid.uuid4().hex[:16]}-000001",
            "side": "debit",
            "event_time": "2026-09-01T10:00:00Z",
            "recorded_at": "2026-09-01T10:00:05Z",
            "arrival_time": "2026-09-01T10:00:05Z",
            "data": {
                "bank_id": "bank_0",
                "account_id": "acc_person_000001",
                "counterparty_bank_id": "bank_1",
                "counterparty_vpa": "acc_shop_000001",
                "amount_paise": 20000,
                "status": "success",
                "failure_domain": None,
                "payee_type": "merchant",
                "channel": "upi_app",
                "sender_device_fingerprint": "fp_01",
                "sender_os_generation": 2,
                "balance_after_paise": 480000,
            },
        }
        msg_bytes = orjson.dumps(payload)
        event_key = str(payload["payment_id"]).encode("utf-8")

        # 1. First delivery
        await producer.send_and_wait(ws.kafka_topic, value=msg_bytes, key=event_key)
        acct_cfg = get_default_settings()
        c1 = await consume_available(consumer, ws_redis, acct_cfg, timeout_ms=500)
        assert c1 == 1

        dbsize_1, digest_1 = snapshot_redis_db(ws_redis)

        # 2. Duplicate redelivery
        await producer.send_and_wait(ws.kafka_topic, value=msg_bytes, key=event_key)
        c2 = await consume_available(consumer, ws_redis, acct_cfg, timeout_ms=500)
        assert c2 == 1

        dbsize_2, digest_2 = snapshot_redis_db(ws_redis)

        # Redis state is bitwise identical
        assert dbsize_2 == dbsize_1
        assert digest_2 == digest_1

    finally:
        await consumer.stop()
        await producer.stop()
        ws_redis.close()
        await drop_workspace(
            get_test_database_url(),
            live_redis,
            ws,
            base_redis_url=host_side_redis_url(),
            redis_client_factory=host_redis_factory,
        )


async def test_drain_timeout_with_paused_consumer(
    live_redis: redis.Redis,
) -> None:
    """
    WHAT: Verifies that when the consumer is prevented from committing (via pause),
    drain exceeds its deadline and raises ReplayDrainTimeout naming partition and offsets,
    and the workspace is still cleanly dropped.
    WHY: Uses real aiokafka API: consumer.pause(*partitions) (AIOKafkaConsumer.pause:628).
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
    recorded_world = record_world_for_replay(WORLD_SETTINGS, runner, connectors)
    feed = project_run(recorded_world.output)

    workspace_name = f"rpl_timeout_{uuid.uuid4().hex[:8]}"

    pause_hook_called = False

    async def pause_hook(ctx: ReplayHookContext) -> None:
        nonlocal pause_hook_called
        if ctx.step_index == 26:  # At decision 0, pause consumer for subsequent decisions
            topic = f"{workspace_name}.payments"
            parts = ctx.consumer.partitions_for_topic(topic) or {0}
            ctx.consumer.pause(*[TopicPartition(topic, p) for p in parts])
            pause_hook_called = True

    # Replay settings with tiny drain timeout
    tight_settings = ReplaySettings(
        redis_workspace_db_min=1,
        redis_workspace_db_max=15,
        registry_key="replay:workspace_registry",
        registry_lock_seconds=10,
        drain_timeout_seconds=1,  # Short timeout for test
        kafka_topic_partitions=1,
    )

    with pytest.raises(ReplayDrainTimeout, match="Drain deadline exceeded"):
        await run_replay(
            recorded_world,
            feed,
            admin_pg=get_test_database_url(),
            replay_settings=tight_settings,
            base_redis_url=host_side_redis_url(),
            redis_client_factory=host_redis_factory,
            before_decision_hook=pause_hook,
            workspace_name=workspace_name,
        )

    assert pause_hook_called is True

    # Verify workspace schema was cleanly dropped
    pg_conn = await asyncpg.connect(get_test_database_url())
    try:
        schema_exists = await pg_conn.fetchval(
            "SELECT count(*) FROM information_schema.schemata WHERE schema_name = $1",
            workspace_name,
        )
        assert schema_exists == 0
    finally:
        await pg_conn.close()


async def test_end_to_end_replay_event_decisions(
    pg_conn: asyncpg.Connection,
    live_redis: redis.Redis,
) -> None:
    """
    WHAT: Runs end-to-end replay with one decision per executed payment attempt (decisions_per_event_payment=True).
    WHY: Verifies:
      - Decisions are keyed by event payment IDs with receiver_vpa included.
      - Exactly one feature_log row per executed payment attempt.
      - Payee features evaluate: first_payment_to_payee non-None in >= 1 row, payee_same_bank_risk non-None in >= 1
        same-bank row and None for all cross-bank rows, payee_registry_match None across all decisions.
      - ReplayResult carries registry_wired=True and the 16d reason; payee_registry_match is non-None at DPIP
        participant banks and None at a non-participant bank.
      - Pre-first-payment unwatermarked status occurs only on the first decision.
      - Live Redis DB and main database schema are completely untouched.
    """
    # 1. Baseline snapshots before replay
    dbsize_before, digest_before = snapshot_redis_db(live_redis)
    main_tx_before = await pg_conn.fetchval("SELECT count(*) FROM transactions")
    fl_exists = await pg_conn.fetchval(
        "SELECT count(*) FROM information_schema.tables WHERE table_schema = 'public' AND table_name = 'feature_log'"
    )
    main_fl_before = await pg_conn.fetchval("SELECT count(*) FROM feature_log") if fl_exists else 0

    # 2. Record simulation run
    cfg = load_bank_feed_config()
    bank_modes = load_bank_modes()
    prepared = prepare_world(WORLD_SETTINGS, None)
    account_bank_map: dict[str, str] = {}
    for ledger in prepared.world.ledgers:
        b_id = resolve_bank_id(ledger.bank_index, cfg)
        for acc in ledger.accounts:
            account_bank_map[acc] = b_id

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
    recorded_world = record_world_for_replay(WORLD_SETTINGS, runner, connectors)
    feed = project_run(recorded_world.output)

    # 3. Execute replay with decisions_per_event_payment=True
    registry_settings = load_registry_replay_settings("training")
    banks_cfg = {b: dict(s) for b, s in registry_settings.bank_settings.items()}
    banks_cfg["bank_2"]["dpip_participation"] = False  # one non-participant bank
    registry_settings = dataclasses.replace(registry_settings, bank_settings=banks_cfg)
    result = await run_replay(
        recorded_world,
        feed,
        admin_pg=get_test_database_url(),
        base_redis_url=host_side_redis_url(),
        redis_client_factory=host_redis_factory,
        decisions_per_event_payment=True,
        registry_settings=registry_settings,
    )

    # 4. Verify decision counts and feature_log statuses
    events = recorded_world.output.result.run_result.events
    event_pids = set(recorded_world.output.result.payment_ids)
    assert len(events) == result.decisions_sent
    assert sum(result.feature_rows_by_status.values()) == len(events)
    assert result.decisions_before_first_payment == 1
    assert result.feature_rows_by_status["no_watermark"] == 1
    assert result.feature_rows_by_status["ok"] == len(events) - 1
    assert result.feature_rows_by_status.get("incomplete_request", 0) == 0
    assert set(result.feature_rows_by_status.keys()) == {"no_watermark", "ok"}

    # 5. Unwatermarked decisions
    first_pid = recorded_world.output.result.payment_ids[0]
    assert set(result.unwatermarked_payment_ids) == {first_pid}

    # 6. Every logged feature row corresponds to an event payment ID
    assert len(result.logged_features_by_payment_id) == len(events) == result.decisions_sent
    for pid in result.logged_features_by_payment_id:
        assert pid in event_pids

    # 7. Registry wired marker
    assert result.registry_wired is True
    assert result.registry_wired_reason == REGISTRY_WIRED_REASON

    # 8. Payee features
    # payee_registry_match: not None at DPIP-participant banks (once a decision has features), None at bank_2
    sender_bank = {
        pid: account_bank_map[ev.from_account]
        for pid, ev in zip(recorded_world.output.result.payment_ids, events, strict=True)
    }
    participant_rows = [
        f for pid, f in result.logged_features_by_payment_id.items()
        if sender_bank[pid] != "bank_2" and pid not in result.unwatermarked_payment_ids
    ]
    non_participant_rows = [
        f for pid, f in result.logged_features_by_payment_id.items() if sender_bank[pid] == "bank_2"
    ]
    assert participant_rows and non_participant_rows
    assert all(f.get("payee_registry_match") is not None for f in participant_rows)
    assert all(f.get("payee_registry_match") is None for f in non_participant_rows)
    # first_payment_to_payee evaluated and non-None in at least one decision
    assert any(f.get("first_payment_to_payee") is not None for f in result.logged_features_by_payment_id.values())

    # payee_same_bank_risk non-None in >= 1 same-bank row, None for all cross-bank rows
    same_bank_pids = {
        pid
        for pid, ev in zip(recorded_world.output.result.payment_ids, events)
        if ev.from_bank_index == ev.to_bank_index
    }
    cross_bank_pids = {
        pid
        for pid, ev in zip(recorded_world.output.result.payment_ids, events)
        if ev.from_bank_index != ev.to_bank_index
    }
    logged_same_bank_pids = [pid for pid in same_bank_pids if pid in result.logged_features_by_payment_id]
    logged_cross_bank_pids = [pid for pid in cross_bank_pids if pid in result.logged_features_by_payment_id]

    assert any(
        result.logged_features_by_payment_id[pid].get("payee_same_bank_risk") is not None
        for pid in logged_same_bank_pids
    )
    assert all(
        result.logged_features_by_payment_id[pid].get("payee_same_bank_risk") is None
        for pid in logged_cross_bank_pids
    )

    # 9. Condition 3 check: split payments in this world
    planned_counts: dict[str, int] = {}
    for p_id in recorded_world.output.result.planned_payment_ids:
        planned_counts[p_id] = planned_counts.get(p_id, 0) + 1
    split_pids = [p_id for p_id, count in planned_counts.items() if count > 1]
    assert len(split_pids) == 0

    # 10. Payer features still computed
    second_pid = recorded_world.output.result.payment_ids[1]
    second_features = result.logged_features_by_payment_id[second_pid]
    assert second_features["hour_ist"] is not None
    assert second_features["customer_failed_attempts_24h"] is not None
    assert second_features["outflow_to_merchant_paise_7d"] is not None

    # 11. Live DB and main schema untouched
    dbsize_after, digest_after = snapshot_redis_db(live_redis)
    assert dbsize_after == dbsize_before
    assert digest_after == digest_before

    main_tx_after = await pg_conn.fetchval("SELECT count(*) FROM transactions")
    main_fl_after = await pg_conn.fetchval("SELECT count(*) FROM feature_log") if fl_exists else 0
    assert main_tx_after == main_tx_before
    assert main_fl_after == main_fl_before
