import os
from typing import Dict, Any, Optional
from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict
from app.config.constants import EnvironmentType, FusionStrategyType, LogLevelType

class AppSettings(BaseSettings):
    """
    Enterprise-grade configuration system using Pydantic Settings.
    Loads variables from environment or .env file.
    Includes strict model validation for secure secrets and configurations.
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )

    # 1. Environment & General Config
    ENV: EnvironmentType = Field(default=EnvironmentType.DEVELOPMENT, description="Current environment: development, testing, production")
    PROJECT_NAME: str = Field(default="Deepfake Forensics Platform", description="Platform name")
    API_VERSION: str = Field(default="v1", description="FastAPI prefix version")
    DEBUG: bool = Field(default=True, description="Enable debugging modes")
    SECRET_KEY: str = Field(default="super-secret-key-change-in-production-123456", description="JWT and session signature secret")

    # 2. Database Connection Strings
    DATABASE_URL: str = Field(
        default="sqlite:///./storage/forensics.db", 
        description="SQL Database connection string (PostgreSQL/SQLite)"
    )
    MONGO_URI: Optional[str] = Field(default=None, description="MongoDB connection string for metadata/unstructured log cache")

    # 3. Security configurations
    ALLOWED_HOSTS: str = Field(default="*", description="Allowed hosts list (comma-separated)")
    MAX_UPLOAD_SIZE_MB: int = Field(default=100, description="Max allowed file upload size in MB")
    ALLOWED_EXTENSIONS: str = Field(
        default="mp4,avi,mov,mkv,mp3,wav", 
        description="Allowed file extensions (comma-separated)"
    )
    JWT_EXPIRY_MINUTES: int = Field(default=60, description="JWT lifespan in minutes")

    # 4. Storage & Directory configurations
    STORAGE_ROOT: str = Field(default="storage", description="Root storage directory")
    UPLOADS_DIR: str = Field(default="storage/uploads", description="Folder to buffer user video/audio uploads")
    REPORTS_DIR: str = Field(default="storage/reports", description="Folder to write forensic PDF/JSON reports")
    LOGS_DIR: str = Field(default="logs", description="Folder to maintain daily logs")
    MODELS_DIR: str = Field(default="ai_engine/models", description="Trained weights check-point path")
    DATASET_CACHE_DIR: str = Field(default="storage/dataset_cache", description="Where downloaded benchmark datasets are cached")

    # 5. Model Inference & GPU Configuration
    USE_GPU: bool = Field(default=True, description="Utilize CUDA if available")
    CUDA_DEVICE_ID: int = Field(default=0, description="Index of Target GPU")
    FUSION_STRATEGY: FusionStrategyType = Field(default=FusionStrategyType.LATE_AVERAGE, description="Multimodal fusion strategy")

    # 6. Third Party & MLOps API Keys
    MLFLOW_TRACKING_URI: Optional[str] = Field(default=None, description="MLflow experiment server endpoint")
    WANDB_API_KEY: Optional[str] = Field(default=None, description="Weights & Biases API Key")
    PROMETHEUS_METRICS_ENABLED: bool = Field(default=True, description="Expose metrics for Prometheus scraping")

    # 7. Logging level
    LOG_LEVEL: LogLevelType = Field(default=LogLevelType.INFO, description="Standard logging output level")

    def get_allowed_extensions_list(self) -> list[str]:
        return [ext.strip().lower() for ext in self.ALLOWED_EXTENSIONS.split(",")]

    def get_allowed_hosts_list(self) -> list[str]:
        return [host.strip() for host in self.ALLOWED_HOSTS.split(",")]

    @model_validator(mode="after")
    def validate_production_security(self) -> 'AppSettings':
        """
        Enforce strict security validation checks on configuration startup.
        """
        # 1. Enforce secure SECRET_KEY in production mode
        if self.ENV == EnvironmentType.PRODUCTION:
            default_keys = [
                "super-secret-key-change-in-production-123456",
                "development-security-secret-change-this-string-in-production-nodes"
            ]
            if self.SECRET_KEY in default_keys:
                raise ValueError("SECURITY RISK: Default SECRET_KEY is not allowed in production environment profiles.")
            if len(self.SECRET_KEY) < 16:
                raise ValueError("SECURITY RISK: SECRET_KEY must contain at least 16 characters in production.")
                
        # 2. Enforce valid database url scheme
        valid_prefixes = ["sqlite://", "postgresql://", "postgresql+psycopg2://"]
        if not any(self.DATABASE_URL.startswith(p) for p in valid_prefixes):
            raise ValueError(f"DATABASE CONFIG ERROR: Invalid URL scheme. Must start with one of: {', '.join(valid_prefixes)}")

        return self

# Instantiate singleton settings
settings = AppSettings()
