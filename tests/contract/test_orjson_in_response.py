import orjson
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

def test_orjson_in_response():
    """
    Test that the /health response uses the correct content-type and deserializes via orjson.
    """
    with TestClient(app) as client:
        response = client.get("/health")
        
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"
        
        # Should deserialize correctly using orjson
        data = orjson.loads(response.content)
        assert data["status"] == "ok"
