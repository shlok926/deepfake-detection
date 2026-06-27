import os
import csv
import random
import logging
from typing import Dict, Any, List, Optional, Tuple
from ai_engine.datasets.registry import DatasetRegistry

logger = logging.getLogger("system")

class DatasetSplitter:
    """
    Splits forensic datasets into reproducible Train, Validation, and Test partitions.
    Guarantees class stratification (preserves real/fake balance in each split)
    using configurable random seeds.
    """
    def __init__(self, registry: Optional[DatasetRegistry] = None) -> None:
        self.registry = registry or DatasetRegistry()

    def get_stratified_splits(
        self, 
        dataset_id: str, 
        random_seed: int = 42
    ) -> Dict[str, List[Dict[str, str]]]:
        """
        Loads metadata CSV and returns a dictionary with stratified 'train', 'validation', and 'test' rows.
        """
        cfg = self.registry.get_config(dataset_id)
        
        if not cfg:
            raise ValueError(f"Dataset '{dataset_id}' not found in registry config.")

        if not os.path.exists(cfg.metadata_csv_path):
            raise FileNotFoundError(f"Metadata CSV not found: {cfg.metadata_csv_path}")

        # 1. Read metadata rows
        rows: List[Dict[str, str]] = []
        with open(cfg.metadata_csv_path, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for r in reader:
                if r.get("video_path") and r.get("label"):
                    rows.append({
                        "video_path": r["video_path"],
                        "label": r["label"]
                    })

        if not rows:
            raise ValueError("No valid rows found in metadata CSV for splitting.")

        # 2. Group samples by their labels (stratification groups)
        groups: Dict[str, List[Dict[str, str]]] = {}
        for row in rows:
            lbl = row["label"]
            groups.setdefault(lbl, []).append(row)

        train_set: List[Dict[str, str]] = []
        val_set: List[Dict[str, str]] = []
        test_set: List[Dict[str, str]] = []

        # Setup deterministic random generator
        rng = random.Random(random_seed)

        # 3. For each label group, split and distribute
        for lbl, lbl_rows in groups.items():
            # Shuffle items inside the class group
            rng.shuffle(lbl_rows)
            
            total_lbl = len(lbl_rows)
            val_count = int(total_lbl * cfg.validation_split)
            test_count = int(total_lbl * cfg.test_split)
            train_count = total_lbl - val_count - test_count

            # Distribute into stratified splits
            train_set.extend(lbl_rows[:train_count])
            val_set.extend(lbl_rows[train_count : train_count + val_count])
            test_set.extend(lbl_rows[train_count + val_count :])

        # 4. Shuffle splits again to mix class labels inside each split
        rng.shuffle(train_set)
        rng.shuffle(val_set)
        rng.shuffle(test_set)

        return {
            "train": train_set,
            "validation": val_set,
            "test": test_set
        }

    def write_splits(
        self, 
        dataset_id: str, 
        output_metadata_dir: str, 
        random_seed: int = 42
    ) -> Dict[str, Any]:
        """
        Retrieves stratified splits and saves them as train.csv, validation.csv, and test.csv.
        """
        splits = self.get_stratified_splits(dataset_id, random_seed)
        os.makedirs(output_metadata_dir, exist_ok=True)

        report = {
            "success": True,
            "random_seed": random_seed,
            "train_count": len(splits["train"]),
            "validation_count": len(splits["validation"]),
            "test_count": len(splits["test"]),
            "files_written": {}
        }

        for split_name, rows in splits.items():
            split_filepath = os.path.join(output_metadata_dir, f"{split_name}.csv")
            with open(split_filepath, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["video_path", "label"])
                for r in rows:
                    writer.writerow([r["video_path"], r["label"]])
            
            report["files_written"][split_name] = split_filepath

        return report
