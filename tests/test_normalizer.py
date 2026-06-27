import os
import csv
import pytest
from ai_engine.datasets.registry import DatasetRegistry, DatasetMetadata, DatasetConfig
from ai_engine.datasets.normalizer import DatasetNormalizer

@pytest.fixture
def temp_normalization_env(tmp_path):
    """
    Creates a mock dataset containing:
    - 10 video files (5 labeled real, 5 labeled fake)
    - A master metadata.csv
    """
    raw_dir = tmp_path / "raw"
    processed_dir = tmp_path / "processed"
    audio_dir = tmp_path / "audio"
    raw_dir.mkdir()
    processed_dir.mkdir()
    audio_dir.mkdir()

    csv_rows = []
    # Create 10 files on disk
    for i in range(1, 11):
        filename = f"video_{i}.mp4"
        label = "real" if i <= 5 else "fake"
        
        with open(raw_dir / filename, "w") as f:
            f.write(f"video data content {i}")
            
        csv_rows.append([filename, label])

    csv_path = tmp_path / "metadata.csv"
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["video_path", "label"])
        writer.writerows(csv_rows)

    return {
        "raw_dir": str(raw_dir),
        "processed_dir": str(processed_dir),
        "audio_dir": str(audio_dir),
        "metadata_csv": str(csv_path),
        "output_dir": str(tmp_path / "normalized_output")
    }

def test_dataset_normalizer_splits(temp_normalization_env):
    # 1. Setup registry
    registry = DatasetRegistry()
    meta = DatasetMetadata(
        name="Normalization Test Dataset",
        version="1.0",
        release_year=2026,
        modality="video",
        total_samples=10,
        real_samples=5,
        fake_samples=5,
        file_size_gb=0.1,
        description="Dataset for testing splits",
        download_source="http://localhost",
        labels=["real", "fake"],
        classes=["real", "fake"],
        expected_folder_structure={"raw": "raw/"},
        citation="N/A"
    )
    config = DatasetConfig(
        dataset_id="norm_dataset",
        raw_video_dir=temp_normalization_env["raw_dir"],
        processed_faces_dir=temp_normalization_env["processed_dir"],
        audio_features_dir=temp_normalization_env["audio_dir"],
        metadata_csv_path=temp_normalization_env["metadata_csv"],
        validation_split=0.2, # 2 samples
        test_split=0.2       # 2 samples
    )
    registry.register_dataset("norm_dataset", meta, config)

    # 2. Run normalizer
    normalizer = DatasetNormalizer(registry)
    res = normalizer.normalize("norm_dataset", temp_normalization_env["output_dir"])

    assert res["success"] is True
    # total samples is 10. val_split=0.2 (2), test_split=0.2 (2), train_split=0.6 (6)
    assert res["train_samples"] == 6
    assert res["validation_samples"] == 2
    assert res["test_samples"] == 2

    output_dir = temp_normalization_env["output_dir"]
    
    # 3. Check physical layout
    assert os.path.exists(os.path.join(output_dir, "real"))
    assert os.path.exists(os.path.join(output_dir, "fake"))
    assert os.path.exists(os.path.join(output_dir, "metadata"))
    assert os.path.exists(os.path.join(output_dir, "train"))
    assert os.path.exists(os.path.join(output_dir, "validation"))
    assert os.path.exists(os.path.join(output_dir, "test"))

    # Check split csv files are written
    assert os.path.exists(os.path.join(output_dir, "metadata", "train.csv"))
    assert os.path.exists(os.path.join(output_dir, "metadata", "validation.csv"))
    assert os.path.exists(os.path.join(output_dir, "metadata", "test.csv"))

    # Verify real folder has exactly 5 files (since we have 5 real videos total)
    real_files = os.listdir(os.path.join(output_dir, "real"))
    assert len(real_files) == 5
    for rf in real_files:
        # Check that we linked video_1 to video_5 (the real ones)
        idx = int(rf.split("_")[1].split(".")[0])
        assert idx <= 5

    # Verify fake folder has exactly 5 files
    fake_files = os.listdir(os.path.join(output_dir, "fake"))
    assert len(fake_files) == 5
    for ff in fake_files:
        idx = int(ff.split("_")[1].split(".")[0])
        assert idx > 5
