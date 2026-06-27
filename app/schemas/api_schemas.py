from datetime import datetime
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

# 1. Health & Version Schemas
class HealthResponse(BaseModel):
    status: str = Field(default="healthy")
    database_connected: bool
    cuda_available: bool
    gpu_devices_count: int
    timestamp: datetime

class VersionResponse(BaseModel):
    version: str
    api_prefix: str
    environment: str

# 2. Upload Schemas
class UploadResponse(BaseModel):
    success: bool
    video_id: int
    file_name: str
    file_size_mb: float
    sha256_hash: str
    message: str

# 3. Predict Schemas
class PredictRequest(BaseModel):
    video_id: int
    model_type: str = Field(default="multimodal", description="video, audio, or multimodal")

class PredictResponse(BaseModel):
    success: bool
    prediction_id: int
    video_id: int
    model_type: str
    model_version: str
    prediction_score: float
    is_fake: bool
    explanation: Optional[Dict[str, Any]] = None
    created_at: datetime

# 4. Report Schemas
class ReportResponse(BaseModel):
    success: bool
    prediction_id: int
    report_id: int
    report_path: str
    download_url: str
    created_at: datetime

# 5. History Schemas
class PredictionHistoryItem(BaseModel):
    prediction_id: int
    video_name: str
    model_type: str
    prediction_score: float
    is_fake: bool
    created_at: datetime

class HistoryListResponse(BaseModel):
    success: bool
    total_records: int
    data: List[PredictionHistoryItem]

# 6. Model Schemas
class ModelRegistryItem(BaseModel):
    model_id: int
    name: str
    version: str
    status: str
    accuracy: Optional[float] = None
    last_updated: datetime

class ModelListResponse(BaseModel):
    success: bool
    available_models: List[ModelRegistryItem]
