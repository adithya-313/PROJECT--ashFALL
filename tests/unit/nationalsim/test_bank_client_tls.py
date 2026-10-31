# File: tests/unit/nationalsim/test_bank_client_tls.py
# Purpose: The bank client builds its TLS context once per client, not once per request.
# Issue: Level 15 decision latency (bank client TLS setup).
"""
WHAT: Makes five screen requests (each in its own event loop, as screen_sync does) through one
NationalsimBankClient and checks the TLS context is created exactly once, is the very object every request's
httpx.AsyncClient receives, and still verifies certificates.

WHY: Building the context loads the CA bundle (about 250 ms on some machines). Doing it per request made every
registry call slower than the decision's registry timeout. Verification must stay on.
"""

import asyncio
import ssl
from pathlib import Path
from typing import Any

import httpx
import pytest

from nationalsim.bank_client import NationalsimBankClient, load_registry_pool_settings

SCREEN_ANSWER = {
    "match": False,
    "status": "no_match",
    "distinct_accounts_linked": None,
    "risk_score": None,
    "first_visible_at": None,
}


def _handler(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json=SCREEN_ANSWER)


def test_tls_context_is_built_once_per_client_and_verifies(monkeypatch: pytest.MonkeyPatch) -> None:
    """Five requests, one context creation; every AsyncClient gets that same context; verify mode stays on."""
    created: list[ssl.SSLContext] = []
    real_create = httpx.create_ssl_context

    def counting_create(*args: Any, **kwargs: Any) -> ssl.SSLContext:
        context = real_create(*args, **kwargs)
        created.append(context)
        return context

    monkeypatch.setattr(httpx, "create_ssl_context", counting_create)

    passed_verify: list[Any] = []
    real_init = httpx.AsyncClient.__init__

    def spying_init(self: httpx.AsyncClient, *args: Any, **kwargs: Any) -> None:
        passed_verify.append(kwargs.get("verify"))
        real_init(self, *args, **kwargs)

    monkeypatch.setattr(httpx.AsyncClient, "__init__", spying_init)

    client = NationalsimBankClient(
        "http://registry.test", bank_id="bank_0", run_id="run_tls", transport=httpx.MockTransport(_handler)
    )
    for _ in range(5):
        result = asyncio.run(client.screen("a" * 64, "account"))
        assert result.status == "no_match"

    assert len(created) == 1
    assert len(passed_verify) == 5
    assert all(verify is created[0] for verify in passed_verify)
    assert created[0].verify_mode == ssl.CERT_REQUIRED
    assert created[0].check_hostname is True


def test_shared_http_client_is_used_for_every_request_and_never_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    """With a shared AsyncClient: no TLS context is built, no per-request client is made, the pool stays open."""
    created_contexts: list[ssl.SSLContext] = []
    real_create = httpx.create_ssl_context
    monkeypatch.setattr(
        httpx,
        "create_ssl_context",
        lambda *args, **kwargs: created_contexts.append(real_create(*args, **kwargs)) or created_contexts[-1],
    )
    constructed: list[int] = []
    real_init = httpx.AsyncClient.__init__

    def counting_init(self: httpx.AsyncClient, *args: Any, **kwargs: Any) -> None:
        constructed.append(1)
        real_init(self, *args, **kwargs)

    shared = httpx.AsyncClient(transport=httpx.MockTransport(_handler))
    monkeypatch.setattr(httpx.AsyncClient, "__init__", counting_init)

    client = NationalsimBankClient(
        "http://registry.test",
        bank_id="bank_0",
        run_id="run_pool",
        timeout_seconds=1.0,
        retry_backoff_seconds=0.1,
        http_client=shared,
    )

    async def five_calls() -> None:
        for _ in range(5):
            assert (await client.screen("a" * 64, "account")).status == "no_match"
        assert not shared.is_closed
        await shared.aclose()

    asyncio.run(five_calls())
    assert created_contexts == []
    assert constructed == []


def test_registry_pool_settings_errors_are_loud(tmp_path: Path) -> None:
    """A missing key or a keep-alive size above max_connections is an error, not a default."""
    bad = tmp_path / "bad.yaml"
    bad.write_text("nationalsim:\n  registry_base_url:\n    value: 'http://x:1'\n", encoding="utf-8")
    with pytest.raises(ValueError, match="client_max_connections.value"):
        load_registry_pool_settings(bad)
    oversized = tmp_path / "oversized.yaml"
    oversized.write_text(
        "nationalsim:\n  registry_base_url: {value: 'http://x:1'}\n  client_max_connections: {value: 4}\n"
        "  client_max_keepalive_connections: {value: 9}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="keepalive connections exceed max connections"):
        load_registry_pool_settings(oversized)
