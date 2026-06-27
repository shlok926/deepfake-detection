import logging
import os

from app.config.config import settings

logger = logging.getLogger("bootstrap")


def bootstrap_directories() -> None:
    """
    Bootstrap the project directory hierarchy on startup.
    Ensures all storage, logging, model checkpoints, and artifact output paths exist.
    """
    directories = [
        settings.STORAGE_ROOT,
        settings.UPLOADS_DIR,
        settings.REPORTS_DIR,
        settings.LOGS_DIR,
        settings.MODELS_DIR,
        settings.DATASET_CACHE_DIR,
        os.path.join(settings.STORAGE_ROOT, "predictions"),
        os.path.join(settings.STORAGE_ROOT, "artifacts"),
        os.path.join(settings.STORAGE_ROOT, "heatmaps"),
        os.path.join("ai_engine", "checkpoints"),
        "monitoring/prometheus",
        "monitoring/grafana",
        "mlops/dvc",
        "mlops/mlflow",
    ]

    print("=== Platform Initialization: Directory Bootstrapping ===")
    for directory in directories:
        if not os.path.exists(directory):
            try:
                os.makedirs(directory, exist_ok=True)
                print(f"[BOOTSTRAP] Created directory: {directory}")
            except Exception as e:
                print(f"[BOOTSTRAP] ERROR: Could not create directory {directory}: {e}")
        else:
            print(f"[BOOTSTRAP] Directory already exists: {directory}")
    print("=======================================================\n")


if __name__ == "__main__":
    bootstrap_directories()
