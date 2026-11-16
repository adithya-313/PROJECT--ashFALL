"""
File: tests/unit/test_multi_redis_routing.py
Purpose: Unit tests for the replay_redis_urls switch: the off state, the round-robin mapping, refused
         entries, and the worker cap that grows with the number of Redis servers.

WHAT: Every test calls the real production functions; only the config loader is replaced, and only in
the cap tests, so the switch carries three URLs without editing the committed file.

WHY: An empty list must reproduce today's single-Redis behaviour exactly, and a bad switch must be
refused by name instead of reaching a child build as a broken address.
"""

from pathlib import Path
from typing import Any

import pytest

import scripts.build_level_16_datasets as build_module
from core.config_validator import validate_file
from scripts.build_level_16_datasets import (
    load_dataset_build_config,
    load_replay_redis_urls,
    redis_url_for_worker,
    validate_parallel_workers,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
THREE_URLS = ["redis://localhost:6379/0", "redis://localhost:6380/0", "redis://localhost:6381/0"]


def _config_with_urls(urls: list[str]) -> dict[str, Any]:
    """Reads the committed config and replaces only the switch value, so the rest stays real."""
    config = load_dataset_build_config()
    config["replay_redis_urls"] = {**config["replay_redis_urls"], "value": urls}
    return config


@pytest.mark.parametrize("worker_index", [0, 1, 13, 99])
def test_empty_list_routes_every_worker_to_the_parent_address(worker_index: int) -> None:
    """With the switch off every index maps to None, meaning the child keeps the parent's address."""
    assert redis_url_for_worker(worker_index, []) is None


def test_three_urls_route_fourteen_workers_round_robin() -> None:
    """Fourteen workers over three URLs follow index % 3, so neighbours never share a server."""
    mapping = [redis_url_for_worker(index, THREE_URLS) for index in range(14)]
    expected = [THREE_URLS[index] for index in (0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1)]

    assert mapping == expected


def test_a_duplicate_url_is_refused_by_name() -> None:
    """The same address twice would send two workers to one server and leave the count untrustworthy."""
    config = {"replay_redis_urls": {"value": [THREE_URLS[0], THREE_URLS[0]]}}

    with pytest.raises(ValueError, match=f"replay_redis_urls entry {THREE_URLS[0]!r} appears more than once"):
        load_replay_redis_urls(config)


@pytest.mark.parametrize("entry", ["http://localhost:6379/0", ""])
def test_a_non_redis_entry_is_refused_by_name(entry: str) -> None:
    """Anything that is not a non-empty redis:// address is refused and named."""
    config = {"replay_redis_urls": {"value": [entry]}}

    with pytest.raises(ValueError, match=f"replay_redis_urls entry {entry!r} is not a redis:// URL"):
        load_replay_redis_urls(config)


def test_the_cap_is_fifteen_with_the_switch_off() -> None:
    """The committed empty switch keeps today's one-server cap of the replay workspace pool."""
    assert load_replay_redis_urls(load_dataset_build_config()) == []
    assert validate_parallel_workers(15) == 15
    with pytest.raises(ValueError, match="exceeds the replay workspace pool of 15 "):
        validate_parallel_workers(16)


def test_the_cap_is_forty_five_with_three_urls(monkeypatch: pytest.MonkeyPatch) -> None:
    """Three servers multiply the pool of 15 by three, and 46 is refused naming the server count."""
    monkeypatch.setattr(build_module, "load_dataset_build_config", lambda: _config_with_urls(THREE_URLS))

    assert validate_parallel_workers(45) == 45
    with pytest.raises(ValueError, match="exceeds the replay workspace pool of 45 over 3 redis servers"):
        validate_parallel_workers(46)


def test_the_committed_config_passes_the_validator() -> None:
    """The new labelled block must keep config/dataset_build.yaml at zero validator problems."""
    assert validate_file(REPO_ROOT / "config" / "dataset_build.yaml") == []
