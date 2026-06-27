import os
import csv
import pytest
from ai_engine.datasets.registry import DatasetRegistry, DatasetMetadata, DatasetConfig
from ai_engine.datasets.validator import DatasetValidator

@pytest.fixture
def temp_dataset_env(tmp_path):
    """
    Creates a mock dataset folder structure with:
    - 1 valid video (using a tiny valid MP4 block or mock video)
    - 1 duplicate of the valid video
    - 1 corrupt video (0 bytes)
    - 1 unsupported file extension video (.txt)
    - A metadata CSV listing all these files with various valid and invalid annotations.
    """
    raw_video_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    audio_dir = tmp_path / "audio"
    raw_video_dir.mkdir()
    processed_dir.mkdir()
    audio_dir.mkdir()
    
    # 1. Create a minimal valid video file. We can create a tiny file block.
    # Note: OpenCV VideoCapture requires a real valid header, but we can write a mock block 
    # to trigger corrupt check or use a small stub. For testing, we'll write:
    # - valid_video.mp4 (mocked, but since we check cap.isOpened(), a mock block will raise corrupt,
    # so we'll test the corrupt video branch and missing video branch explicitly)
    
    # Create missing video path in CSV but don't write to disk
    missing_filename = "missing.mp4"
    
    # Create unsupported format file on disk
    unsupported_filename = "doc.txt"
    with open(raw_video_dir / unsupported_filename, "w") as f:
        f.write("unsupported file content")
        
    # Create corrupt/0-byte video on disk
    corrupt_filename = "corrupt.mp4"
    with open(raw_video_dir / corrupt_filename, "wb") as f:
        f.write(b"") # empty
        
    # Create a simple mock video on disk (since it's not a real mp4, OpenCV will treat it as corrupt, which is great)
    mock_filename = "mock.mp4"
    with open(raw_video_dir / mock_filename, "wb") as f:
        f.write(b"mock video data header block 12345")

    # Create a duplicate of mock video (same content/hash)
    duplicate_filename = "dup.mp4"
    with open(raw_video_dir / duplicate_filename, "wb") as f:
        f.write(b"mock video data header block 12345")

    csv_path = tmp_path / "metadata.csv"
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["video_path", "label"])
        writer.writerow([missing_filename, "real"])
        writer.writerow([unsupported_filename, "real"])
        writer.writerow([corrupt_filename, "real"])
        writer.writerow([mock_filename, "real"])
        writer.writerow([duplicate_filename, "real"])
        writer.writerow(["valid_path.mp4", "invalid_label_xyz"]) # invalid label check
        writer.writerow(["", "real"]) # missing path check
        
    return {
        "raw_video_dir": str(raw_video_dir),
        "processed_dir": str(processed_dir),
        "audio_dir": str(audio_dir),
        "metadata_csv": str(csv_path)
    }

def test_dataset_validator_rules(temp_dataset_env):
    # 1. Setup mock registry
    registry = DatasetRegistry()
    
    meta = DatasetMetadata(
        name="Mock Test Dataset",
        version="1.0",
        release_year=2026,
        modality="video",
        total_samples=7,
        real_samples=3,
        fake_samples=4,
        file_size_gb=0.1,
        description="Dataset for testing validations",
        download_source="http://localhost",
        labels=["real", "fake"],
        classes=["real", "fake"],
        expected_folder_structure={"raw": "raw/"},
        citation="N/A"
    )
    
    config = DatasetConfig(
        dataset_id="mock_dataset",
        raw_video_dir=temp_dataset_env["raw_video_dir"],
        processed_faces_dir=temp_dataset_env["processed_dir"],
        audio_features_dir=temp_dataset_env["audio_dir"],
        metadata_csv_path=temp_dataset_env["metadata_csv"]
    )
    
    registry.register_dataset("mock_dataset", meta, config)
    
    # 2. Run validator
    validator = DatasetValidator(registry)
    report = validator.run_full_validation("mock_dataset")
    
    assert report["success"] is True
    assert report["errors_found"] is True
    
    # Verify missing files caught
    assert "missing.mp4" in report["files"]["missing_videos"]
    
    # Verify unsupported formats caught
    assert "doc.txt" in report["files"]["unsupported_formats"]
    
    # Verify corrupt files caught (empty file and mock video both will fail OpenCV check)
    assert "corrupt.mp4" in report["files"]["corrupt_videos"]
    assert "mock.mp4" in report["files"]["corrupt_videos"]
    
    # Verify duplicate files caught (dup.mp4 is duplicate of mock.mp4)
    # The duplicate tracker will list dup.mp4 pointing to mock.mp4
    assert "dup.mp4" in report["files"]["duplicate_files"]
    assert report["files"]["duplicate_files"]["dup.mp4"] == "mock.mp4"
    
    # Verify invalid labels caught
    invalid_labels = [row["reason"] for row in report["metadata"]["invalid_rows"]]
    assert any("invalid_label_xyz" in r for r in invalid_labels)
    assert any("Missing video_path" in r for r in invalid_labels)
