import os
import csv
import shutil
import random
import logging
from typing import Dict, Any, List, Optional
from ai_engine.datasets.registry import DatasetRegistry, DatasetConfig

logger = logging.getLogger("system")

class DatasetNormalizer:
    """
    Normalizes diverse deepfake dataset layout schemes into a unified research-level architecture:
    dataset/
      ├── real/
      ├── fake/
      ├── metadata/
      │     ├── train.csv
      │     ├── validation.csv
      │     └── test.csv
      ├── train/
      ├── validation/
      └── test/
    """
    def __init__(self, registry: Optional[DatasetRegistry] = None) -> None:
        self.registry = registry or DatasetRegistry()

    def _create_link_or_copy(self, source_path: str, dest_path: str) -> None:
        """
        Attempts to create a symbolic link to prevent heavy storage duplication.
        Falls back to hardlink or direct file copy if symlink creation is disallowed by OS policies.
        """
        if os.path.exists(dest_path):
            os.remove(dest_path)
            
        try:
            # Attempt Symlink creation
            os.symlink(os.path.abspath(source_path), dest_path)
        except (OSError, NotImplementedError):
            try:
                # Fallback to Hardlink
                os.link(source_path, dest_path)
            except OSError:
                # Fallback to Copy
                shutil.copy2(source_path, dest_path)

    def normalize(self, dataset_id: str, custom_output_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Processes dataset and generates normalized layout splits under storage root.
        """
        cfg = self.registry.get_config(dataset_id)
        meta = self.registry.get_metadata(dataset_id)
        
        if not cfg or not meta:
            return {"success": False, "error": f"Dataset '{dataset_id}' not found in registry."}

        # Setup paths
        out_root = custom_output_dir or f"storage/datasets/{dataset_id}_normalized"
        
        real_dir = os.path.join(out_root, "real")
        fake_dir = os.path.join(out_root, "fake")
        meta_dir = os.path.join(out_root, "metadata")
        train_dir = os.path.join(out_root, "train")
        val_dir = os.path.join(out_root, "validation")
        test_dir = os.path.join(out_root, "test")

        # Create target folders
        for folder in [real_dir, fake_dir, meta_dir, train_dir, val_dir, test_dir]:
            os.makedirs(folder, exist_ok=True)

        if not os.path.exists(cfg.metadata_csv_path):
            return {"success": False, "error": f"Metadata CSV not found: {cfg.metadata_csv_path}"}

        # 1. Read rows from metadata CSV
        rows: List[Dict[str, str]] = []
        try:
            with open(cfg.metadata_csv_path, mode="r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for r in reader:
                    if r.get("video_path") and r.get("label"):
                        rows.append(r)
        except Exception as e:
            return {"success": False, "error": f"Failed reading metadata: {e}"}

        if not rows:
            return {"success": False, "error": "No valid rows found in metadata CSV."}

        # Deterministic shuffle for splitting
        random.seed(42)
        random.shuffle(rows)

        # 2. Compute train, validation, and test split indices
        total = len(rows)
        val_count = int(total * cfg.validation_split)
        test_count = int(total * cfg.test_split)
        train_count = total - val_count - test_count

        train_rows = rows[:train_count]
        val_rows = rows[train_count : train_count + val_count]
        test_rows = rows[train_count + val_count :]

        splits_mapping = [
            ("train", train_rows, train_dir),
            ("validation", val_rows, val_dir),
            ("test", test_rows, test_dir)
        ]

        normalized_report = {
            "success": True,
            "output_directory": out_root,
            "train_samples": len(train_rows),
            "validation_samples": len(val_rows),
            "test_samples": len(test_rows),
            "linked_files": 0
        }

        # 3. Process each split and copy/link files
        for split_name, split_rows, split_target_dir in splits_mapping:
            split_csv_path = os.path.join(meta_dir, f"{split_name}.csv")
            
            with open(split_csv_path, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["original_path", "normalized_path", "label"])
                
                for item in split_rows:
                    rel_path = item["video_path"]
                    label = item["label"] # 'real' or 'fake'
                    
                    src_abs = os.path.join(cfg.raw_video_dir, rel_path)
                    if not os.path.exists(src_abs):
                        # Skip missing files, but warn
                        logger.warning(f"Skipping missing video: {src_abs}")
                        continue

                    filename = os.path.basename(rel_path)
                    
                    # Target path in either real/ or fake/ folder
                    class_folder = real_dir if label == "real" else fake_dir
                    class_dest = os.path.join(class_folder, filename)
                    
                    # Link to class folder
                    self._create_link_or_copy(src_abs, class_dest)
                    
                    # Link to split folder (train/validation/test)
                    split_dest = os.path.join(split_target_dir, filename)
                    self._create_link_or_copy(src_abs, split_dest)
                    
                    # Record row in split metadata CSV
                    writer.writerow([rel_path, os.path.join(split_name, filename), label])
                    normalized_report["linked_files"] += 1

        return normalized_report
