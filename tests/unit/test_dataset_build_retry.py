"""
File: tests/unit/test_dataset_build_retry.py
Purpose: Unit tests for the Level 16 world replay retry: one dropped connection is retried, a
         persistent one stops at the configured attempt count, other errors are never retried,
         the switch off means one attempt, and the new config file passes the config validator.
"""

import asyncio
from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import Any, cast
from unittest.mock import MagicMock

import pytest

import scripts.build_level_16_datasets as build_module
from core.config_validator import validate_file
from replay.dataset_builder import WorldExtract
from scripts.build_level_16_datasets import (
    load_dataset_build_config,
    replay_world_with_retry,
)
from world.replay_recording import RecordedWorld

REPO_ROOT = Path(__file__).resolve().parents[2]


def _recorded() -> RecordedWorld:
    recorded = MagicMock()
    recorded.world_id = "test-0000"
    return cast(RecordedWorld, recorded)


def _extract() -> WorldExtract:
    return WorldExtract(
        world_id="test-0000",
        split="test",
        seed=1,
        run_id="run-1",
        gang_families=(),
        decisions_per_event_payment=True,
        registry_wired=False,
        registry_wired_reason="",
        payment_rows=[],
        snapshot_rows=[],
        label_rows=[],
    )


def _config(retry_enabled: bool = True) -> dict[str, Any]:
    config = load_dataset_build_config()
    config["retry_world_on_connection_error"] = {
        **config["retry_world_on_connection_error"],
        "value": retry_enabled,
    }
    return config


def _install_fake_replay(
    monkeypatch: pytest.MonkeyPatch, outcomes: list[BaseException | WorldExtract]
) -> list[str]:
    """Replaces the replay with one that raises or returns each outcome in turn, and returns the call log."""
    calls: list[str] = []

    async def fake_replay(
        recorded: RecordedWorld,
        admin_pg: str,
        base_redis_url: str,
        *,
        profile_dir: Path | None = None,
    ) -> WorldExtract:
        if recorded.world_id is None:
            raise ValueError("fake replay needs a world id")
        calls.append(recorded.world_id)
        outcome = outcomes.pop(0) if len(outcomes) > 1 else outcomes[0]
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    monkeypatch.setattr(build_module, "replay_and_extract", fake_replay)
    return calls


def _fake_sleep(waits: list[float]) -> Callable[[float], Awaitable[None]]:
    """Returns a sleep stand-in that records each requested wait instead of sleeping."""

    async def sleep(seconds: float) -> None:
        waits.append(seconds)

    return sleep


def test_one_connection_error_is_retried_and_the_result_returned(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies a single ConnectionResetError leads to a second replay and the successful result."""
    config = _config()
    extract = _extract()
    calls = _install_fake_replay(monkeypatch, [ConnectionResetError("reset"), extract])
    waits: list[float] = []

    result = asyncio.run(
        replay_world_with_retry(
            _recorded(), "postgresql://unused", "redis://unused", config=config, sleep=_fake_sleep(waits)
        )
    )

    assert result == extract
    assert calls == ["test-0000", "test-0000"]
    assert waits == [config["retry_wait_seconds"]["value"][0]]


def test_persistent_connection_error_stops_at_the_configured_attempts(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies a world that always loses its connection is replayed exactly max_world_attempts times."""
    config = _config()
    calls = _install_fake_replay(monkeypatch, [ConnectionResetError("reset")])
    waits: list[float] = []

    with pytest.raises(ConnectionResetError, match="reset"):
        asyncio.run(
            replay_world_with_retry(
                _recorded(), "postgresql://unused", "redis://unused", config=config, sleep=_fake_sleep(waits)
            )
        )

    assert len(calls) == config["max_world_attempts"]["value"]
    assert waits == config["retry_wait_seconds"]["value"]


def test_other_errors_are_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies a ValueError from the replay is raised at once after one attempt."""
    calls = _install_fake_replay(monkeypatch, [ValueError("unexpected state")])
    waits: list[float] = []

    with pytest.raises(ValueError, match="unexpected state"):
        asyncio.run(
            replay_world_with_retry(
                _recorded(), "postgresql://unused", "redis://unused", config=_config(), sleep=_fake_sleep(waits)
            )
        )

    assert calls == ["test-0000"]
    assert waits == []


def test_switch_off_means_one_attempt(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verifies the retry switch off leaves one attempt, even for a connection error."""
    calls = _install_fake_replay(monkeypatch, [ConnectionResetError("reset")])
    waits: list[float] = []

    with pytest.raises(ConnectionResetError, match="reset"):
        asyncio.run(
            replay_world_with_retry(
                _recorded(),
                "postgresql://unused",
                "redis://unused",
                config=_config(retry_enabled=False),
                sleep=_fake_sleep(waits),
            )
        )

    assert calls == ["test-0000"]
    assert waits == []


def test_dataset_build_config_passes_the_config_validator() -> None:
    """Verifies the new config file's labelled blocks are valid, as the takeover realism file is checked."""
    assert validate_file(REPO_ROOT / "config" / "dataset_build.yaml") == []
