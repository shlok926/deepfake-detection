import os
import json
import pytest
from ai_engine.datasets.registry import DatasetRegistry, DatasetMetadata, DatasetConfig
from ai_engine.datasets.versioning import DatasetVersionManager

@pytest.fixture
def temp_versioning_env(tmp_path):
    """
    Creates a mock normalized folder structure.
    """
    norm_dir = tmp_path / "normalized"
    norm_dir.mkdir()
    
    # Add mock folders and files
    (norm_dir / "real").mkdir()
    (norm_dir / "fake").mkdir()
    
    with open(norm_dir / "real" / "v1.mp4", "w") as f:
        f.write("mock data")
        
    return {
        "norm_dir": str(norm_dir),
        "versions_root": str(tmp_path / "versions")
    }

def test_dataset_version_increment(temp_versioning_env):
    # 1. Setup registry
    registry = DatasetRegistry()
    meta = DatasetMetadata(
        name="Versioning Test Dataset",
        version="1.0",
        release_year=2026,
        modality="video",
        total_samples=1,
        real_samples=1,
        fake_samples=0,
        file_size_gb=0.1,
        description="Dataset for testing versions",
        download_source="http://localhost",
        labels=["real", "fake"],
        classes=["real", "fake"],
        expected_folder_structure={"raw": "raw/"},
        citation="N/A"
    )
    config = DatasetConfig(
        dataset_id="ver_dataset",
        raw_video_dir="N/A",
        processed_faces_dir="N/A",
        audio_features_dir="N/A",
        metadata_csv_path="N/A"
    )
    registry.register_dataset("ver_dataset", meta, config)

    manager = DatasetVersionManager(registry)
    
    # 2. Run versioning archives creator three times
    p_config = {"face_crop_size": 224, "fps": 10}
    
    # Run 1 -> v1
    res1 = manager.create_versioned_archive(
        "ver_dataset", 
        temp_versioning_env["norm_dir"], 
        p_config, 
        temp_versioning_env["versions_root"]
    )
    assert res1["success"] is True
    assert res1["version_id"] == "v1"
    assert os.path.exists(os.path.join(temp_versioning_env["versions_root"], "v1", "version_metadata.json"))
    
    # Run 2 -> v2
    res2 = manager.create_versioned_archive(
        "ver_dataset", 
        temp_versioning_env["norm_dir"], 
        p_config, 
        temp_versioning_env["versions_root"]
    )
    assert res2["success"] is True
    assert res2["version_id"] == "v2"
    assert os.path.exists(os.path.join(temp_versioning_env["versions_root"], "v2", "version_metadata.json"))
    
    # Run 3 -> v3
    res3 = manager.create_versioned_archive(
        "ver_dataset", 
        temp_versioning_env["norm_dir"], 
        p_config, 
        temp_versioning_env["versions_root"]
    )
    assert res3["success"] is True
    assert res3["version_id"] == "v3"
    assert os.path.exists(os.path.join(temp_versioning_env["versions_root"], "v3", "version_metadata.json"))

    # Verify JSON structure in v3
    with open(os.path.join(temp_versioning_env["versions_root"], "v3", "version_metadata.json"), "r") as f:
        meta_json = json.load(f)
        assert meta_json["version_id"] == "v3"
        assert meta_json["dataset_id"] == "ver_dataset"
        assert meta_json["preprocessing_parameters"]["face_crop_size"] == 224
        assert "created_at" in meta_json
