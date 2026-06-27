import os
import csv
import pytest
from ai_engine.datasets.registry import DatasetRegistry, DatasetMetadata, DatasetConfig
from ai_engine.datasets.splitter import DatasetSplitter

@pytest.fixture
def temp_splitter_env(tmp_path):
    """
    Creates a mock dataset containing:
    - 10 real videos
    - 20 fake videos
    (Total = 30 videos, 2:1 fake-to-real ratio)
    """
    csv_rows = []
    # 10 Real
    for i in range(1, 11):
        csv_rows.append([f"real_{i}.mp4", "real"])
    # 20 Fake
    for i in range(1, 21):
        csv_rows.append([f"fake_{i}.mp4", "fake"])

    csv_path = tmp_path / "metadata.csv"
    with open(csv_path, mode="w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["video_path", "label"])
        writer.writerows(csv_rows)

    return {
        "metadata_csv": str(csv_path),
        "output_dir": str(tmp_path / "splits_out")
    }

def test_dataset_stratified_splitting(temp_splitter_env):
    # 1. Setup registry
    registry = DatasetRegistry()
    meta = DatasetMetadata(
        name="Splitting Test Dataset",
        version="1.0",
        release_year=2026,
        modality="video",
        total_samples=30,
        real_samples=10,
        fake_samples=20,
        file_size_gb=0.1,
        description="Dataset for testing splits",
        download_source="http://localhost",
        labels=["real", "fake"],
        classes=["real", "fake"],
        expected_folder_structure={"raw": "raw/"},
        citation="N/A"
    )
    config = DatasetConfig(
        dataset_id="split_dataset",
        raw_video_dir="N/A",
        processed_faces_dir="N/A",
        audio_features_dir="N/A",
        metadata_csv_path=temp_splitter_env["metadata_csv"],
        validation_split=0.2, # 30 * 0.2 = 6 samples
        test_split=0.1       # 30 * 0.1 = 3 samples
    )
    registry.register_dataset("split_dataset", meta, config)

    splitter = DatasetSplitter(registry)

    # 2. Run stratified split
    splits1 = splitter.get_stratified_splits("split_dataset", random_seed=42)
    splits2 = splitter.get_stratified_splits("split_dataset", random_seed=42)
    splits3 = splitter.get_stratified_splits("split_dataset", random_seed=999) # different seed

    # 3. Verify Reproducibility (same seed = identical lists)
    assert splits1["train"] == splits2["train"]
    assert splits1["validation"] == splits2["validation"]
    assert splits1["test"] == splits2["test"]

    # Verify Configurable Seed (different seed = different list orders/structures)
    assert splits1["train"] != splits3["train"]

    # 4. Verify Stratification Proportions (exactly 2:1 fake-to-real ratio in each split)
    # Total train count: 30 - 6 - 3 = 21. Real should be 7, Fake should be 14.
    train_reals = [r for r in splits1["train"] if r["label"] == "real"]
    train_fakes = [r for r in splits1["train"] if r["label"] == "fake"]
    assert len(splits1["train"]) == 21
    assert len(train_reals) == 7
    assert len(train_fakes) == 14

    # Total val count: 6. Real should be 2, Fake should be 4.
    val_reals = [r for r in splits1["validation"] if r["label"] == "real"]
    val_fakes = [r for r in splits1["validation"] if r["label"] == "fake"]
    assert len(splits1["validation"]) == 6
    assert len(val_reals) == 2
    assert len(val_fakes) == 4

    # Total test count: 3. Real should be 1, Fake should be 2.
    test_reals = [r for r in splits1["test"] if r["label"] == "real"]
    test_fakes = [r for r in splits1["test"] if r["label"] == "fake"]
    assert len(splits1["test"]) == 3
    assert len(test_reals) == 1
    assert len(test_fakes) == 2

    # 5. Verify CSV write capabilities
    report = splitter.write_splits("split_dataset", temp_splitter_env["output_dir"], random_seed=42)
    assert report["success"] is True
    assert os.path.exists(os.path.join(temp_splitter_env["output_dir"], "train.csv"))
    assert os.path.exists(os.path.join(temp_splitter_env["output_dir"], "validation.csv"))
    assert os.path.exists(os.path.join(temp_splitter_env["output_dir"], "test.csv"))
