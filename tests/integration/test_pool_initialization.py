import pytest
from fastapi.testclient import TestClient

from api.main import app
from core.config import config
from tests.db_skip import host_side_database_url

# This test starts the whole app in-process, and the app connects to Postgres. The configured address uses the
# Docker-only host name "postgres", which does not exist on a laptop or a CI runner, so we point the app at localhost
# for this test only (monkeypatch undoes the change afterwards). It still needs a running Postgres, hence the
# needs_stack marker.
pytestmark = pytest.mark.needs_stack


@pytest.fixture(autouse=True)
def use_host_side_database(monkeypatch: pytest.MonkeyPatch) -> None:
    """Point the app's database setting at localhost for this test."""
    monkeypatch.setattr(config, "database_url", host_side_database_url())

def test_pool_initialized_once(caplog):
    """
    Test that the asyncpg database pool is initialized exactly once per worker process.
    
    Nuance stated for the spec: The literal spec wording asks for "exactly once", but in a
    multi-worker Gunicorn setup, the lifespan runs once per worker process. Therefore, the 
    pool initializes N times globally (where N is the number of Gunicorn workers). 
    
    CRITICAL LIMITATION: This test is run via an in-process TestClient. It never touches 
    the real multi-worker Gunicorn container. It only proves the lifecycle logic for a 
    single, unrepresentative test process. To truly prove initialization behavior in 
    the live multi-worker deployment, one would need to parse the stdout of the 
    running Docker container and assert the initialization log line appears N times.
    This test verifies that within a single worker's lifespan (the test process), 
    the pool initializes exactly once, regardless of how many requests are handled.
    """
    caplog.set_level("INFO")
    
    with TestClient(app) as client:
        # Send 100 requests to /health
        for _ in range(100):
            response = client.get("/health")
            assert response.status_code == 200

    # Count how many times the pool initialization was logged
    init_logs = [record for record in caplog.text.split("\n") if "Database pool initialized" in record]
    
    assert len(init_logs) == 1, f"Expected exactly 1 pool initialization log, found {len(init_logs)}"
