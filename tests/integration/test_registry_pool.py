# File: tests/integration/test_registry_pool.py
# Purpose: The API's shared registry connection pool is reused by every decision and closed at shutdown.
# Issue: Level 15 decision latency (pooled registry client).
"""
WHAT: Starts the real mock registry on a localhost port and drives 20 decisions' registry questions through one
RegistryGateway on a single event loop. Checks that no new httpx.AsyncClient is created after startup, that the
number of TCP connections the server accepted stays within the pool's max_connections (and far below the number
of requests: connections are reused), and that closing the gateway closes the pool.

WHY: A new client and TCP connection per call is what made every registry call slow. One pooled client with
explicit limits must serve all decisions, and be closed with aclose() at shutdown.
"""

import asyncio
from typing import Any

import httpx
import pytest
import uvicorn.protocols.http.h11_impl as h11_impl
import uvicorn.protocols.http.httptools_impl as httptools_impl

from benchmarks.decision_latency import BENCHMARK_DPIP_KEY, start_mock_registry
from features.registry_prefetch import RegistryGateway, prefetch_registry_answers
from nationalsim.bank_client import create_shared_http_client, load_registry_pool_settings

RUN_ID = "66db9aa439b092ee"
DECISIONS = 20
QUESTIONS = [("account", "a" * 64), ("device", "b" * 64), ("account", "c" * 64)]
MAX_CONNECTIONS = 8


def test_twenty_decisions_share_one_pool_and_it_is_closed_at_shutdown(monkeypatch: pytest.MonkeyPatch) -> None:
    """No AsyncClient is built per decision, connections are reused, and aclose() closes the pool."""
    accepted: list[int] = []
    for protocol in (h11_impl.H11Protocol, httptools_impl.HttpToolsProtocol):
        real_connection_made = protocol.connection_made

        def counting_connection_made(self: Any, transport: Any, _real: Any = real_connection_made) -> None:
            accepted.append(1)
            _real(self, transport)

        monkeypatch.setattr(protocol, "connection_made", counting_connection_made)

    registry = start_mock_registry(RUN_ID, BENCHMARK_DPIP_KEY, [], None)
    try:
        gateway = RegistryGateway(
            http_client=create_shared_http_client(MAX_CONNECTIONS, MAX_CONNECTIONS, 2.0),
            base_url=registry.base_url,
            client_timeout_seconds=2.0,
            retry_backoff_seconds=0.2,
            registry_timeout_seconds=5.0,
        )

        constructed: list[int] = []
        real_init = httpx.AsyncClient.__init__

        def counting_init(self: httpx.AsyncClient, *args: Any, **kwargs: Any) -> None:
            constructed.append(1)
            real_init(self, *args, **kwargs)

        monkeypatch.setattr(httpx.AsyncClient, "__init__", counting_init)

        async def drive() -> int:
            answered = 0
            for _ in range(DECISIONS):
                client = gateway.client_for("bank_0", RUN_ID)
                answers = await prefetch_registry_answers(client, QUESTIONS, gateway.registry_timeout_seconds)
                answered += sum(1 for outcome in answers.values() if not outcome.timed_out)
            assert not gateway.http_client.is_closed
            await gateway.aclose()
            assert gateway.http_client.is_closed
            return answered

        answered = asyncio.run(drive())
    finally:
        registry.stop()

    assert answered == DECISIONS * len(QUESTIONS), "every question must have been answered by the registry"
    assert constructed == [], "a decision created its own AsyncClient instead of using the shared pool"
    assert 1 <= len(accepted) <= MAX_CONNECTIONS, f"{len(accepted)} connections were opened"
    assert len(accepted) < DECISIONS * len(QUESTIONS) / 4, "connections should be reused, not opened per request"


def test_configured_pool_settings_are_positive_and_consistent() -> None:
    """The repository config gives the registry URL and a pool whose keep-alive size fits inside max_connections."""
    base_url, max_connections, max_keepalive = load_registry_pool_settings()
    assert base_url == "http://nationalsim:8080"
    assert 0 < max_keepalive <= max_connections
