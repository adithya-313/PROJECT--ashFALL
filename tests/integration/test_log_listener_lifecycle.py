"""
File: tests/integration/test_log_listener_lifecycle.py
Purpose: Verifies that the QueueListener background thread is properly started and stopped.

CRITICAL LIMITATION: Like `test_pool_initialized_once`, this test executes completely
in-process using an ASGI client. It does not touch the real multi-worker Gunicorn
container deployment. It proves that the FastAPI lifespan logic successfully creates,
starts, and stops the background thread in a single Python process, but does not
definitively prove Gunicorn's worker forking semantics in the live Docker environment.
"""

import pytest
import httpx

from api.main import app

@pytest.mark.anyio
async def test_queue_listener_lifecycle() -> None:
    """
    Asserts that the QueueListener is created, its thread is alive during a request,
    and it is properly cleaned up after the lifespan exits.
    """
    # Using ASGITransport to trigger the FastAPI lifespan
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        # Inside the context, lifespan has run startup.
        listener = getattr(app.state, "log_listener", None)
        assert listener is not None, "log_listener was not stored on app.state"
        
        # Verify the background thread is alive
        assert hasattr(listener, "_thread") and listener._thread is not None
        assert listener._thread.is_alive(), "QueueListener background thread is not alive"
        
        # Send a request to ensure logging occurs while the thread is alive
        response = await client.get("/health")
        assert response.status_code == 200

    # Outside the context, lifespan has run shutdown.
    # The stop_log_listener function calls listener.stop(), which joins the thread.
    assert not listener._thread.is_alive(), "QueueListener thread should be dead after lifespan exits"
