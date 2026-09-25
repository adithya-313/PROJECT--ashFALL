import pytest
from fastapi.testclient import TestClient
from api.main import app

def test_pool_initialized_once(caplog):
    """
    Test that the asyncpg database pool is initialized exactly once per worker process.
    
    Nuance stated for the spec: The literal spec wording asks for "exactly once", but in a
    multi-worker Gunicorn setup, the lifespan runs once per worker process. Therefore, the 
    pool initializes N times globally (where N is the number of Gunicorn workers). 
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
