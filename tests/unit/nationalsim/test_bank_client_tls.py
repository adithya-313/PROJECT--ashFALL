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
from typing import Any

import httpx
import pytest

from nationalsim.bank_client import NationalsimBankClient

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
