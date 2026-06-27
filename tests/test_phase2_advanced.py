import os
import json
import pytest
import cv2
from ai_engine.datasets.duplicates import VideoDuplicateDetector
from ai_engine.datasets.cache import PreprocessingCache
from ai_engine.datasets.reporter import DatasetReporter
from ai_engine.datasets.parser import BaseDatasetParser
from ai_engine.datasets.registry import DatasetRegistry, DatasetMetadata, DatasetConfig

class MockVideoParser(BaseDatasetParser):
    """
    Subclass implementation demonstrating Extensibility (Phase 2.10)
    """
    def parse_metadata(self, raw_dataset_dir: str) -> list:
        return [
            {"video_path": "clip1.mp4", "label": "real"},
            {"video_path": "clip2.mp4", "label": "fake"}
        ]

class MockVideoCapture:
    def __init__(self, path) -> None:
        self.opened = True
    def isOpened(self) -> bool:
        return True
    def get(self, prop_id: int) -> float:
        if prop_id == cv2.CAP_PROP_FRAME_WIDTH:
            return 640.0
        elif prop_id == cv2.CAP_PROP_FRAME_HEIGHT:
            return 480.0
        elif prop_id == cv2.CAP_PROP_FPS:
            return 30.0
        elif prop_id == cv2.CAP_PROP_FRAME_COUNT:
            return 150.0
        elif prop_id == cv2.CAP_PROP_FOURCC:
            # MP4V
            return float(ord('M') | (ord('P') << 8) | (ord('4') << 16) | (ord('V') << 24))
        return 0.0
    def read(self) -> tuple:
        import numpy as np
        # Return a mock BGR frame (3 channel)
        return True, np.zeros((480, 640, 3), dtype=np.uint8)
    def release(self) -> None:
        self.opened = False

@pytest.fixture
def advanced_setup_env(tmp_path, monkeypatch):
    """
    Prepares mock files and mocks OpenCV and Librosa loaders
    """
    # Create files for duplicates testing
    file1 = tmp_path / "vid1.mp4"
    file2 = tmp_path / "vid2.mp4"
    
    with open(file1, "w") as f:
        f.write("standard video data content")
        
    with open(file2, "w") as f:
        f.write("standard video data content") # exact duplicate

    # Mock OpenCV
    monkeypatch.setattr(cv2, "VideoCapture", MockVideoCapture)
    
    # Mock Librosa
    import librosa
    def mock_load(path, sr=None, duration=None):
        import numpy as np
        return np.ones(100), 16000
    monkeypatch.setattr(librosa, "load", mock_load)

    return {
        "file1": str(file1),
        "file2": str(file2),
        "cache_dir": str(tmp_path / "cache"),
        "reports_dir": str(tmp_path / "reports"),
        "metadata_csv": str(tmp_path / "metadata.csv")
    }

def test_duplicate_detection(advanced_setup_env):
    detector = VideoDuplicateDetector(hamming_threshold=4)
    
    # Scan files
    res = detector.scan_for_duplicates([advanced_setup_env["file1"], advanced_setup_env["file2"]])
    
    assert res["total_files_scanned"] == 2
    assert res["duplicate_groups_found"] == 1
    
    # Check duplicate details
    dup_group = res["duplicates"][0]
    assert dup_group["original_file"] == advanced_setup_env["file1"]
    assert dup_group["copies"][0]["file_path"] == advanced_setup_env["file2"]
    assert dup_group["copies"][0]["type"] == "sha256_exact"

def test_preprocessing_cache(advanced_setup_env):
    cache = PreprocessingCache(advanced_setup_env["cache_dir"])
    
    mock_hash = "abc123hash"
    
    # Verify miss
    assert cache.get(mock_hash) is None
    
    # Create fake files to prevent stale cleanups
    frames_dir = os.path.join(advanced_setup_env["cache_dir"], "frames", mock_hash)
    os.makedirs(frames_dir, exist_ok=True)
    
    # Save hit
    cache.put(
        file_hash=mock_hash,
        extracted_frames_dir=frames_dir,
        face_crops_dir=frames_dir,
        audio_track_path=None,
        spectrogram_path=None,
        metadata={"width": 640}
    )
    
    # Verify hit
    hit = cache.get(mock_hash)
    assert hit is not None
    assert hit["extracted_frames_dir"] == frames_dir
    assert hit["metadata"]["width"] == 640

def test_dataset_reporter(advanced_setup_env):
    # Setup mock registry configuration
    import csv
    with open(advanced_setup_env["metadata_csv"], "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["video_path", "label"])
        writer.writerow(["vid1.mp4", "real"])

    registry = DatasetRegistry()
    meta = DatasetMetadata(
        name="Advanced Test Dataset",
        version="1.0",
        release_year=2026,
        modality="video",
        total_samples=1,
        real_samples=1,
        fake_samples=0,
        file_size_gb=0.1,
        description="Dataset for testing reports",
        download_source="http://localhost",
        labels=["real", "fake"],
        classes=["real", "fake"],
        expected_folder_structure={"raw": "raw/"},
        citation="N/A"
    )
    # Use parent directory of file1 as raw video dir
    config = DatasetConfig(
        dataset_id="adv_dataset",
        raw_video_dir=os.path.dirname(advanced_setup_env["file1"]),
        processed_faces_dir="N/A",
        audio_features_dir="N/A",
        metadata_csv_path=advanced_setup_env["metadata_csv"]
    )
    registry.register_dataset("adv_dataset", meta, config)

    reporter = DatasetReporter(registry)
    res = reporter.generate_report("adv_dataset", advanced_setup_env["reports_dir"])

    assert res["success"] is True
    assert os.path.exists(res["json_report_path"])
    assert os.path.exists(res["markdown_report_path"])

    # Check parsed details
    data = res["data"]
    assert data["statistics"]["total_videos"] == 1
    assert data["statistics"]["codec_distribution"] != {}

def test_pipeline_extensibility():
    parser = MockVideoParser()
    items = parser.parse_metadata("/path/to/raw")
    
    assert len(items) == 2
    assert items[0]["video_path"] == "clip1.mp4"
    assert items[0]["label"] == "real"
