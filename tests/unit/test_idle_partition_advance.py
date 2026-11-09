# File: tests/unit/test_idle_partition_advance.py
# Purpose: Unit tests for idle-partition payment watermark advance in replay.
"""
WHAT: Fast unit tests verifying idle_advance_target logic and replay switch behavior.
WHY: Architect 16c-3 (b):
  - A quiet payment partition stalls the watermark; the replay proves the drain, so it may advance.
  - Advance when drained and nothing in flight (next_unsent_arrival is None).
  - No advance when not drained (partition_drained=False).
  - No advance when next unsent record's arrival <= target (record in flight).
  - Target equals clock_now - commit_grace exactly.
  - Switch-off path never calls record_payment_watermark.
"""

from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, MagicMock

import pytest

from replay.runner import (
    idle_advance_target,
    load_default_idle_partition_advance,
    run_replay,
)
from world.replay_workspace import ReplayWorkspace

UTC = timezone.utc


def test_default_reads_true_from_real_config() -> None:
    """Verifies that load_default_idle_partition_advance reads True from config/replay.yaml without mocks."""
    val = load_default_idle_partition_advance()
    assert val is True


def test_idle_advance_target_when_drained_and_nothing_in_flight() -> None:
    """Verifies that a drained partition with no remaining unsent steps advances to clock_now - commit_grace."""
    clock_now = datetime(2026, 9, 2, 0, 0, 0, tzinfo=UTC)
    commit_grace = 5
    target, reason = idle_advance_target(
        clock_now=clock_now,
        commit_grace_seconds=commit_grace,
        partition_drained=True,
        next_unsent_arrival=None,
    )
    expected_target = clock_now - timedelta(seconds=commit_grace)
    assert target == expected_target
    assert reason == "advance"


def test_idle_advance_target_no_advance_when_not_drained() -> None:
    """Break-it target 1: Refuses advance when partition is not drained."""
    clock_now = datetime(2026, 9, 2, 0, 0, 0, tzinfo=UTC)
    target, reason = idle_advance_target(
        clock_now=clock_now,
        commit_grace_seconds=5,
        partition_drained=False,
        next_unsent_arrival=None,
    )
    assert target is None
    assert reason == "partition not drained"


def test_idle_advance_target_no_advance_when_next_unsent_in_flight() -> None:
    """Break-it target 1 / in-flight: Refuses advance when next unsent step arrives at or before target."""
    clock_now = datetime(2026, 9, 2, 0, 0, 0, tzinfo=UTC)
    commit_grace = 5
    expected_target = clock_now - timedelta(seconds=commit_grace)

    # 1. Next unsent step arrives strictly before target
    target_early, reason_early = idle_advance_target(
        clock_now=clock_now,
        commit_grace_seconds=commit_grace,
        partition_drained=True,
        next_unsent_arrival=expected_target - timedelta(seconds=1),
    )
    assert target_early is None
    assert "in flight" in reason_early

    # 2. Next unsent step arrives exactly at target
    target_exact, reason_exact = idle_advance_target(
        clock_now=clock_now,
        commit_grace_seconds=commit_grace,
        partition_drained=True,
        next_unsent_arrival=expected_target,
    )
    assert target_exact is None
    assert "in flight" in reason_exact


def test_idle_advance_target_equals_clock_now_minus_commit_grace_exactly() -> None:
    """Break-it target 2: Target must equal clock_now - commit_grace exactly, never clock_now."""
    clock_now = datetime(2026, 9, 5, 14, 30, 22, tzinfo=UTC)
    commit_grace = 5
    next_unsent = clock_now + timedelta(seconds=10)

    target, reason = idle_advance_target(
        clock_now=clock_now,
        commit_grace_seconds=commit_grace,
        partition_drained=True,
        next_unsent_arrival=next_unsent,
    )
    assert target is not None
    assert target == clock_now - timedelta(seconds=5)
    assert target != clock_now
    assert reason == "advance"


def test_idle_advance_target_requires_timezone_aware() -> None:
    """Verifies that naive datetime raises ValueError."""
    naive_clock = datetime(2026, 9, 2, 0, 0, 0)
    with pytest.raises(ValueError, match="clock_now must be timezone-aware"):
        idle_advance_target(
            clock_now=naive_clock,
            commit_grace_seconds=5,
            partition_drained=True,
            next_unsent_arrival=None,
        )


@pytest.mark.anyio
async def test_switch_off_path_never_calls_record_payment_watermark(monkeypatch: pytest.MonkeyPatch) -> None:
    """Break-it target 3: When idle_partition_advance=False, record_payment_watermark is never called."""
    from replay.ordering import DecisionStep, FeedStep
    from world.bank_connectors import DecisionCall

    # Mock steps: 1 FeedStep (transaction) followed by 1 DecisionStep
    step_time = datetime(2026, 9, 1, 10, 0, 0, tzinfo=UTC)
    t_feed = step_time - timedelta(seconds=10)
    feed_step = FeedStep(
        world_time=t_feed,
        kind="transaction",
        bank_id="bank_0",
        record_id="rec_1",
        record=MagicMock(),
    )
    call_view = DecisionCall(
        sender_bank_id="bank_0",
        receiver_bank_id="bank_1",
        body_bytes=b"{}",
        idempotency_key="key_1",
    )
    dummy_decision_step = DecisionStep(
        world_time=step_time,
        planned_payment_id="p1",
        call=call_view,
    )

    monkeypatch.setattr(
        "replay.runner.build_replay_steps",
        lambda *args, **kwargs: [feed_step, dummy_decision_step],
    )

    # Workspace mocking
    ws = ReplayWorkspace(
        name="test_ws",
        postgres_schema="test_schema",
        redis_db=0,
        kafka_topic="test_topic",
        kafka_group_id="test_group",
    )
    monkeypatch.setattr("replay.runner.allocate_workspace", AsyncMock(return_value=ws))
    monkeypatch.setattr("replay.runner.drop_workspace", AsyncMock(return_value=None))

    mock_conn = AsyncMock()
    mock_conn.fetch.return_value = []
    mock_pool = MagicMock()
    mock_pool.acquire.return_value.__aenter__ = AsyncMock(return_value=mock_conn)
    mock_pool.acquire.return_value.__aexit__ = AsyncMock(return_value=None)
    mock_pool.close = AsyncMock()
    monkeypatch.setattr("asyncpg.create_pool", AsyncMock(return_value=mock_pool))

    mock_redis = MagicMock()
    monkeypatch.setattr("replay.runner.workspace_redis_client", MagicMock(return_value=mock_redis))
    monkeypatch.setattr("replay.runner.start_producer", AsyncMock(return_value=AsyncMock()))
    mock_consumer = AsyncMock()
    monkeypatch.setattr("replay.runner.AIOKafkaConsumer", MagicMock(return_value=mock_consumer))
    monkeypatch.setattr("replay.runner.check_workspace_topic_partitions", MagicMock())
    monkeypatch.setattr("replay.runner.drain", AsyncMock(return_value=0.01))
    monkeypatch.setattr("api.v1.bank_service.InProcessBankClient.post", AsyncMock())
    monkeypatch.setattr("world.bank_connectors.MockBankConnectors.send_feed_record", AsyncMock())
    monkeypatch.setattr("replay.runner.get_payment_watermark", MagicMock(return_value=123456))

    # Spy on record_payment_watermark
    spy_watermark = MagicMock()
    monkeypatch.setattr("replay.runner.record_payment_watermark", spy_watermark)

    dummy_recorded = MagicMock()
    dummy_recorded.output.result.run_id = "test_run_12345"
    dummy_recorded.output.world.ledgers = []
    dummy_recorded.output.result.opened_accounts = []
    dummy_recorded.output.plan.clock.start_date = step_time.date()
    dummy_feed = MagicMock()

    # Run with idle_partition_advance=False
    await run_replay(
        dummy_recorded,
        dummy_feed,
        admin_pg="postgresql://dummy",
        base_redis_url="redis://dummy",
        idle_partition_advance=False,
    )

    # Spy must confirm record_payment_watermark was NEVER called
    assert spy_watermark.call_count == 0, (
        f"Expected 0 calls to record_payment_watermark when switch is False, got {spy_watermark.call_count}"
    )
