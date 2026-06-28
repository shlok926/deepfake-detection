import pytest
from fastapi.testclient import TestClient
from app.main import app

from app.database.connection import SessionLocal
from app.database.models import VideoModel
from app.services.inference import InferenceService

client = TestClient(app)

def test_live_predict_endpoint_fallback(monkeypatch, tmp_path):
    # 1. Mock inference service so we don't need real models/videos
    def mock_predict(self, *args, **kwargs):
        return {
            "prediction_score": 0.85,
            "is_fake": True,
            "details": {"multimodal_fusion": "Test mock explanation"}
        }
    monkeypatch.setattr(InferenceService, "predict_video", mock_predict)

    # 2. Insert dummy video into DB so endpoint doesn't 404
    dummy_file = tmp_path / "dummy.mp4"
    dummy_file.write_text("dummy")
    
    db = SessionLocal()
    # Check if video 9999 already exists from other tests
    if not db.query(VideoModel).filter(VideoModel.id == 9999).first():
        video = VideoModel(id=9999, file_path=str(dummy_file), file_name="dummy.mp4", file_size_mb=1.0, sha256_hash="dummyhash9999")
        db.add(video)
        db.commit()
    db.close()

    # Make sure we hit the endpoint using fallback mechanics (or database fallback)
    response = client.post(
        "/predict",
        json={"video_id": 9999, "model_type": "multimodal"}
    )
    
    assert response.status_code == 200
    data = response.json()
    assert data["success"] is True
    assert "prediction_score" in data
    assert "is_fake" in data
    assert isinstance(data["prediction_score"], float)
    assert isinstance(data["is_fake"], bool)
    assert "explanation" in data
    assert "multimodal_fusion" in data["explanation"]
