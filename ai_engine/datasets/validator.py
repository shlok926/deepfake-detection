import csv
import logging
import os
from typing import Any, Dict, List, Optional, Set, Tuple

import cv2

from ai_engine.datasets.registry import DatasetConfig, DatasetRegistry
from app.config.config import settings
from app.utils.crypto import calculate_sha256

logger = logging.getLogger("system")


class DatasetValidator:
    """
    Automated Dataset Validator for Deepfake Forensics Benchmarks.
    Checks directory properties, metadata CSV compliance, corruption, format limits, and duplicates.
    """

    def __init__(self, registry: Optional[DatasetRegistry] = None) -> None:
        self.registry = registry or DatasetRegistry()

    def run_full_validation(self, dataset_id: str) -> Dict[str, Any]:
        """
        Executes complete verification pipeline:
        1. Validate folder structure (empty dirs check)
        2. Validate metadata CSV headers and rows format
        3. Scan and verify individual video files integrity
        """
        cfg = self.registry.get_config(dataset_id)
        meta = self.registry.get_metadata(dataset_id)

        if not cfg or not meta:
            return {"success": False, "error": f"Dataset '{dataset_id}' not found in active registry."}

        report = {
            "dataset_id": dataset_id,
            "success": True,
            "errors_found": False,
            "empty_directories": [],
            "metadata": {"status": "unverified", "total_rows": 0, "invalid_rows": [], "missing_labels": 0},
            "files": {"missing_videos": [], "corrupt_videos": [], "unsupported_formats": [], "duplicate_files": {}},
        }

        # Step 1: Check Directories and scan for empty folders
        target_dirs = [cfg.raw_video_dir, cfg.processed_faces_dir, cfg.audio_features_dir]
        for path in target_dirs:
            if not os.path.exists(path):
                # Don't fail validation just because processed directories aren't built yet,
                # but warn if they are missing
                continue
            if len(os.listdir(path)) == 0:
                report["empty_directories"].append(path)
                report["errors_found"] = True

        # Step 2: Validate metadata CSV if it exists
        if not os.path.exists(cfg.metadata_csv_path):
            report["success"] = False
            report["errors_found"] = True
            report["metadata"]["status"] = f"CSV not found: {cfg.metadata_csv_path}"
            return report

        # Track file hashes to find duplicates, and verify rows
        seen_hashes: Dict[str, str] = {}  # sha256 -> relative_path
        unique_videos: Set[str] = set()

        try:
            with open(cfg.metadata_csv_path, mode="r", encoding="utf-8") as f:
                # Use Sniffer to check dialect if possible, fallback to standard dict reader
                sample = f.read(2048)
                f.seek(0)
                try:
                    dialect = csv.Sniffer().sniff(sample)
                    reader = csv.DictReader(f, dialect=dialect)
                except Exception:
                    reader = csv.DictReader(f)

                headers = reader.fieldnames or []
                required_cols = {"video_path", "label"}

                # Check for incorrect annotations structure / invalid headers
                if not required_cols.issubset(set(headers)):
                    report["success"] = False
                    report["errors_found"] = True
                    report["metadata"][
                        "status"
                    ] = f"Invalid CSV schema. Missing columns: {required_cols - set(headers)}"
                    return report

                report["metadata"]["status"] = "valid_headers"

                for idx, row in enumerate(reader, start=2):  # 1-indexed plus header row
                    report["metadata"]["total_rows"] += 1

                    video_rel_path = row.get("video_path")
                    label = row.get("label")

                    # 2a. Validate label
                    if not label:
                        report["metadata"]["missing_labels"] += 1
                        report["errors_found"] = True
                        report["metadata"]["invalid_rows"].append({"row": idx, "reason": "Missing label column value"})
                        continue

                    if label not in meta.labels:
                        report["errors_found"] = True
                        report["metadata"]["invalid_rows"].append(
                            {
                                "row": idx,
                                "reason": f"Label '{label}' is not valid for this dataset. Allowed: {meta.labels}",
                            }
                        )
                        continue

                    # 2b. Validate video path references
                    if not video_rel_path:
                        report["errors_found"] = True
                        report["metadata"]["invalid_rows"].append(
                            {"row": idx, "reason": "Missing video_path column value"}
                        )
                        continue

                    video_abs_path = os.path.join(cfg.raw_video_dir, video_rel_path)

                    # 3. Check for Missing videos
                    if not os.path.exists(video_abs_path):
                        report["files"]["missing_videos"].append(video_rel_path)
                        report["errors_found"] = True
                        continue

                    # 4. Check for Duplicate Files using SHA256 (run before format/corruption checks)
                    try:
                        file_hash = calculate_sha256(video_abs_path)
                        if file_hash in seen_hashes:
                            orig_path = seen_hashes[file_hash]
                            report["files"]["duplicate_files"][video_rel_path] = orig_path
                            report["errors_found"] = True
                        else:
                            seen_hashes[file_hash] = video_rel_path
                    except Exception as e:
                        logger.error(f"Failed hashing file: {video_abs_path} - {e}")

                    # 5. Check for Unsupported Formats
                    ext = video_rel_path.split(".")[-1].lower() if "." in video_rel_path else ""
                    allowed_exts = settings.get_allowed_extensions_list()
                    if ext not in allowed_exts:
                        report["files"]["unsupported_formats"].append(video_rel_path)
                        report["errors_found"] = True
                        continue

                    # 6. Check for Corrupt Videos using OpenCV
                    try:
                        cap = cv2.VideoCapture(video_abs_path)
                        if not cap.isOpened():
                            report["files"]["corrupt_videos"].append(video_rel_path)
                            report["errors_found"] = True
                            cap.release()
                            continue

                        fps = cap.get(cv2.CAP_PROP_FPS)
                        frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
                        cap.release()

                        if fps <= 0 or frames <= 0:
                            report["files"]["corrupt_videos"].append(video_rel_path)
                            report["errors_found"] = True
                            continue
                    except Exception as e:
                        report["files"]["corrupt_videos"].append(video_rel_path)
                        report["errors_found"] = True
                        continue

        except Exception as e:
            report["success"] = False
            report["errors_found"] = True
            report["metadata"]["status"] = f"CRITICAL read error: {e}"
            return report

        if report["errors_found"]:
            report["metadata"]["status"] = "completed_with_errors"
        else:
            report["metadata"]["status"] = "all_checks_passed"

        return report
