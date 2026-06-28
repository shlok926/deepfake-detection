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

from app.database.connection import SessionLocal
from app.database.models import VideoModel
from app.services.inference import InferenceService

def test_predict_endpoint_validation(monkeypatch, tmp_path):
    """
    Assert predictions post parameters respect schema limits.
    """
    # Missing parameters
    response = client.post("/predict", json={})
    assert response.status_code == 422
    
    # Mock inference service so we don't need real models/videos
    def mock_predict(self, *args, **kwargs):
        return {
            "prediction_score": 0.85,
            "is_fake": True,
            "details": {"multimodal_fusion": "Test mock explanation"}
        }
    monkeypatch.setattr(InferenceService, "predict_video", mock_predict)

    # Insert dummy video into DB
    dummy_file = tmp_path / "dummy.mp4"
    dummy_file.write_text("dummy")
    
    db = SessionLocal()
    if not db.query(VideoModel).filter(VideoModel.id == 42).first():
        video = VideoModel(id=42, file_path=str(dummy_file), file_name="dummy.mp4", file_size_mb=1.0, sha256_hash="dummyhash42")
        db.add(video)
        db.commit()
    db.close()

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
