import os
import csv
import logging
from ai_engine.datasets.registry import DatasetRegistry
from ai_engine.datasets.validator import DatasetValidator
from ai_engine.datasets.statistics import DatasetStatsAnalyzer
from ai_engine.datasets.splitter import DatasetSplitter
from ai_engine.datasets.normalizer import DatasetNormalizer
from ai_engine.datasets.reporter import DatasetReporter

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("bootstrap")

def bootstrap():
    dataset_id = "deepfake_detection"
    registry = DatasetRegistry()
    cfg = registry.get_config(dataset_id)
    
    if not cfg:
        logger.error(f"Config for {dataset_id} not found in registry.")
        return

    raw_real_dir = os.path.join(cfg.raw_video_dir, "real")
    raw_fake_dir = os.path.join(cfg.raw_video_dir, "fake")

    # 1. Automatically generate metadata.csv if missing
    if not os.path.exists(cfg.metadata_csv_path):
        logger.info(f"Metadata CSV not found. Scanning raw video paths to generate: {cfg.metadata_csv_path}")
        os.makedirs(os.path.dirname(cfg.metadata_csv_path), exist_ok=True)
        
        rows = []
        # Add real videos
        if os.path.exists(raw_real_dir):
            for file in os.listdir(raw_real_dir):
                if file.lower().endswith(".mp4"):
                    rows.append({
                        "video_path": os.path.join("real", file).replace("\\", "/"),
                        "label": "real"
                    })
        # Add fake videos
        if os.path.exists(raw_fake_dir):
            for file in os.listdir(raw_fake_dir):
                if file.lower().endswith(".mp4"):
                    rows.append({
                        "video_path": os.path.join("fake", file).replace("\\", "/"),
                        "label": "fake"
                    })

        if not rows:
            logger.error("No raw video files found in real/ or fake/ directories.")
            return

        with open(cfg.metadata_csv_path, mode="w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["video_path", "label"])
            for r in rows:
                writer.writerow([r["video_path"], r["label"]])
                
        logger.info(f"Successfully generated metadata CSV with {len(rows)} entries.")

    # 2. Run Dataset Validator
    logger.info("Executing Dataset Validator...")
    validator = DatasetValidator(registry)
    val_res = validator.run_full_validation(dataset_id)
    logger.info(f"Validation Result Status: {val_res.get('metadata', {}).get('status')}")

    # 3. Run Dataset Stats Analyzer
    logger.info("Executing Dataset Stats Analyzer...")
    analyzer = DatasetStatsAnalyzer(registry)
    stats_res = analyzer.analyze(dataset_id)
    logger.info(f"Stats Result - Total Videos: {stats_res.get('total_videos')}, Real/Fake Ratio: {stats_res.get('real_fake_ratio')}")

    # 4. Run Dataset Splitter
    logger.info("Executing Dataset Splitter...")
    splitter = DatasetSplitter(registry)
    splits_out_dir = "storage/datasets/deepfake_detection_splits"
    split_res = splitter.write_splits(dataset_id, splits_out_dir, random_seed=42)
    logger.info(f"Split Result - Train: {split_res.get('train_count')}, Val: {split_res.get('validation_count')}, Test: {split_res.get('test_count')}")

    # 5. Run Dataset Normalizer
    logger.info("Executing Dataset Normalizer...")
    normalizer = DatasetNormalizer(registry)
    norm_out_dir = "storage/datasets/deepfake_detection_normalized"
    norm_res = normalizer.normalize(dataset_id, norm_out_dir)
    logger.info(f"Normalization Result - Linked Files: {norm_res.get('linked_files')}")

    # 6. Generate Forensic Audit Report
    logger.info("Generating final forensic audit report...")
    reporter = DatasetReporter(registry)
    report_res = reporter.generate_report(dataset_id)
    logger.info(f"Audit report generated successfully at: {report_res.get('markdown_report_path')}")

if __name__ == "__main__":
    bootstrap()
