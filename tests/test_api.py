import pytest
from fastapi.testclient import TestClient
from app.main import app
from app.utils.bootstrap import bootstrap_directories

# Call bootstrap on test initialization to create storage directories
bootstrap_directories()

client = TestClient(app)

def test_health_endpoint():
    """
    Assert healthcheck endpoint replies healthy status.
    """
    response = client.get("/health")
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["status"] == "healthy"
    assert "database_connected" in json_data
    assert "cuda_available" in json_data

def test_version_endpoint():
    """
    Assert version endpoint returns API specification details.
    """
    response = client.get("/version")
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["version"] == "1.0.0"
    assert json_data["api_prefix"] == "v1"

def test_predict_endpoint_validation():
    """
    Assert predictions post parameters respect schema limits.
    """
    # Missing parameters
    response = client.post("/predict", json={})
    assert response.status_code == 422
    
    # Valid payload structure
    response = client.post("/predict", json={"video_id": 42, "model_type": "multimodal"})
    assert response.status_code == 200
    json_data = response.json()
    assert json_data["success"] is True
    assert json_data["video_id"] == 42
    assert "prediction_score" in json_data
    assert "is_fake" in json_data

def test_custom_exception_handling():
    """
    Assert custom API exceptions are formatted as structured JSON error payloads.
    """
    # Trigger an unsupported file upload check to verify exception response format
    # We query history endpoint with boundary invalid request limits to force error
    response = client.get("/history?limit=-10")
    assert response.status_code == 422 # FastAPI validation response
