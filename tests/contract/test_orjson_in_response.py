import orjson
from fastapi.testclient import TestClient
from api.main import app

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
