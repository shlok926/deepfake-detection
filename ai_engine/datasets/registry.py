import os
import json
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field

class DatasetMetadata(BaseModel):
    """
    Detailed metadata representation of a digital forensics dataset,
    fully compliant with scientific and research requirements.
    """
    name: str
    version: str
    release_year: int
    modality: str = Field(description="video, audio, or multimodal")
    total_samples: int
    real_samples: int
    fake_samples: int
    file_size_gb: float
    description: str
    download_source: str = Field(description="Official download page/command config source URL")
    labels: List[str] = Field(default_factory=list, description="Classification labels (e.g. ['real', 'fake'])")
    classes: List[str] = Field(default_factory=list, description="Specific subclasses or manipulation methods")
    expected_folder_structure: Dict[str, str] = Field(
        default_factory=dict, 
        description="Dictionary mapping semantic parts of the dataset to expected directory sub-paths"
    )
    citation: str = Field(description="Academic citation information (BibTeX format)")

class DatasetConfig(BaseModel):
    """
    Individual configuration variables for directories and schemas.
    """
    dataset_id: str
    raw_video_dir: str
    processed_faces_dir: str
    audio_features_dir: str
    metadata_csv_path: str
    validation_split: float = 0.2
    test_split: float = 0.1

class DatasetRegistry:
    """
    Enterprise-level Dataset Registry engine.
    Maintains configurations, versions, validations, and statistics for deepfake benchmarks.
    """
    def __init__(self) -> None:
        self.datasets: Dict[str, Dict[str, Any]] = {}
        self.configs: Dict[str, DatasetConfig] = {}
        self._initialize_default_registry()

    def _initialize_default_registry(self) -> None:
        """
        Populate defaults for all required forensic datasets with full academic metadata and paths.
        """
        default_benchmarks = [
            {
                "id": "faceforensics",
                "meta": DatasetMetadata(
                    name="FaceForensics++",
                    version="1.0",
                    release_year=2019,
                    modality="multimodal",
                    total_samples=1000000,
                    real_samples=1000,
                    fake_samples=4000,
                    file_size_gb=450.0,
                    description="Facial manipulations containing Deepfakes, Face2Face, FaceSwap and NeuralTextures.",
                    download_source="https://github.com/ondyari/FaceForensics",
                    labels=["real", "fake"],
                    classes=["youtube", "Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures"],
                    expected_folder_structure={
                        "original_sequences": "original_sequences/youtube/c23/videos",
                        "manipulated_sequences": "manipulated_sequences/{manipulation_method}/c23/videos",
                        "metadata": "metadata.json"
                    },
                    citation=r"""@inproceedings{roessler2019faceforensics,
  title={Faceforensics++: Learning to detect manipulated facial images},
  author={Rossler, Andreas and Cozzolino, Davide and Verdoliva, Luisa and Riess, Christian and Nie{\ss}ner, Justus and Niessner, Matthias},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={1--11},
  year={2019}
}"""
                ),
                "config": DatasetConfig(
                    dataset_id="faceforensics",
                    raw_video_dir="storage/datasets/faceforensics/raw_videos",
                    processed_faces_dir="storage/datasets/faceforensics/face_crops",
                    audio_features_dir="storage/datasets/faceforensics/audio_data",
                    metadata_csv_path="storage/datasets/faceforensics/metadata.csv"
                )
            },
            {
                "id": "celeb-df",
                "meta": DatasetMetadata(
                    name="Celeb-DF",
                    version="2.0",
                    release_year=2020,
                    modality="video",
                    total_samples=6229,
                    real_samples=890,
                    fake_samples=5339,
                    file_size_gb=35.0,
                    description="High-quality Deepfake dataset containing celebrity YouTube videos.",
                    download_source="https://github.com/yuezunli/Celeb-DF-v2",
                    labels=["real", "fake"],
                    classes=["Celeb-real", "Celeb-synthesis"],
                    expected_folder_structure={
                        "real_videos": "Celeb-real/videos",
                        "fake_videos": "Celeb-synthesis/videos",
                        "test_list": "List_of_testing_videos.txt"
                    },
                    citation=r"""@inproceedings{li2020celeb,
  title={Celeb-df: A large-scale challenging dataset for deepfake forensics},
  author={Li, Yuezun and Yang, Xin and Sun, Pu and Qi, Honggang and Lyu, Siwei},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={3207--3216},
  year={2020}
}"""
                ),
                "config": DatasetConfig(
                    dataset_id="celeb-df",
                    raw_video_dir="storage/datasets/celeb-df/raw_videos",
                    processed_faces_dir="storage/datasets/celeb-df/face_crops",
                    audio_features_dir="storage/datasets/celeb-df/audio_data",
                    metadata_csv_path="storage/datasets/celeb-df/metadata.csv"
                )
            },
            {
                "id": "dfdc",
                "meta": DatasetMetadata(
                    name="Deepfake Detection Challenge",
                    version="1.0",
                    release_year=2020,
                    modality="multimodal",
                    total_samples=128000,
                    real_samples=24000,
                    fake_samples=104000,
                    file_size_gb=4000.0,
                    description="Kaggle deepfake challenge dataset funded by Meta, Microsoft, and others.",
                    download_source="https://ai.facebook.com/datasets/dfdc/",
                    labels=["real", "fake"],
                    classes=["real", "fake"],
                    expected_folder_structure={
                        "dataset_chunks": "dfdc_train_part_*/",
                        "metadata_file": "metadata.json"
                    },
                    citation=r"""@article{dolhansky2020deepfake,
  title={The deepfake detection challenge (dfdc) dataset},
  author={Dolhansky, Brian and Bitton, Joanna and Pflaum, Ben and Lu, Jiahui and Howes, Russ and Wang, Menglin and Canton Ferrero, Cristian},
  journal={arXiv preprint arXiv:2006.07397},
  year={2020}
}"""
                ),
                "config": DatasetConfig(
                    dataset_id="dfdc",
                    raw_video_dir="storage/datasets/dfdc/raw_videos",
                    processed_faces_dir="storage/datasets/dfdc/face_crops",
                    audio_features_dir="storage/datasets/dfdc/audio_data",
                    metadata_csv_path="storage/datasets/dfdc/metadata.csv"
                )
            },
            {
                "id": "fakeavceleb",
                "meta": DatasetMetadata(
                    name="FakeAVCeleb",
                    version="1.0",
                    release_year=2021,
                    modality="multimodal",
                    total_samples=20000,
                    real_samples=5000,
                    fake_samples=15000,
                    file_size_gb=120.0,
                    description="Multimodal dataset with deepfake audio and faces corresponding to celebrities.",
                    download_source="https://github.com/DASH-Lab/FakeAVCeleb",
                    labels=["real", "fake"],
                    classes=["RealVideo-RealAudio", "FakeVideo-RealAudio", "RealVideo-FakeAudio", "FakeVideo-FakeAudio"],
                    expected_folder_structure={
                        "raw_clips": "FakeAVCeleb_v1/FakeAVCeleb/",
                        "meta_csv": "meta.csv"
                    },
                    citation=r"""@inproceedings{khalid2021fakeavceleb,
  title={FakeAVCeleb: a novel audio-video multimodal deepfake dataset},
  author={Khalid, Hasam and Tariq, Shahroz and Kim, Minha and Woo, Simon S},
  booktitle={Proceedings of the 32nd ACM Workshop on Network and Operating Systems Support for Digital Audio and Video},
  pages={15--21},
  year={2021}
}"""
                ),
                "config": DatasetConfig(
                    dataset_id="fakeavceleb",
                    raw_video_dir="storage/datasets/fakeavceleb/raw_videos",
                    processed_faces_dir="storage/datasets/fakeavceleb/face_crops",
                    audio_features_dir="storage/datasets/fakeavceleb/audio_data",
                    metadata_csv_path="storage/datasets/fakeavceleb/metadata.csv"
                )
            },
            {
                "id": "deeperforensics",
                "meta": DatasetMetadata(
                    name="DeeperForensics-1.0",
                    version="1.0",
                    release_year=2020,
                    modality="video",
                    total_samples=60000,
                    real_samples=50000,
                    fake_samples=10000,
                    file_size_gb=300.0,
                    description="Large scale deepfake dataset with various perturbations and natural compression.",
                    download_source="https://github.com/EndlessSora/DeeperForensics-1.0",
                    labels=["real", "fake"],
                    classes=["real", "fake"],
                    expected_folder_structure={
                        "source_videos": "deeperforensics_v1/videos/source/",
                        "manipulated_videos": "deeperforensics_v1/videos/manipulated/",
                        "meta_json": "meta.json"
                    },
                    citation=r"""@inproceedings{jiang2020deeperforensics,
  title={Deeperforensics-1.0: A large-scale dataset for real-world face forgery detection},
  author={Jiang, Liming and Li, Ruotian and Wu, Wayne and Qian, Chen and Loy, Chen Change},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={2895--2904},
  year={2020}
}"""
                ),
                "config": DatasetConfig(
                    dataset_id="deeperforensics",
                    raw_video_dir="storage/datasets/deeperforensics/raw_videos",
                    processed_faces_dir="storage/datasets/deeperforensics/face_crops",
                    audio_features_dir="storage/datasets/deeperforensics/audio_data",
                    metadata_csv_path="storage/datasets/deeperforensics/metadata.csv"
                )
            },
            {
                "id": "forgerynet",
                "meta": DatasetMetadata(
                    name="ForgeryNet",
                    version="1.0",
                    release_year=2021,
                    modality="multimodal",
                    total_samples=2900000,
                    real_samples=1400000,
                    fake_samples=1500000,
                    file_size_gb=20000.0,
                    description="Extremely large-scale digital forgery detection benchmark dataset.",
                    download_source="https://yotamgigi.github.io/forgerynet/",
                    labels=["real", "fake"],
                    classes=["real", "fake"],
                    expected_folder_structure={
                        "training_images": "forgerynet_v1/images/train/",
                        "validation_images": "forgerynet_v1/images/val/",
                        "meta_csv": "train.csv"
                    },
                    citation=r"""@inproceedings{he2021forgerynet,
  title={Forgerynet: A versatile benchmark for face forgery detection and localization},
  author={He, Yinan and Gan, Bei Gan and Chen, Jingdong and Zhou, Yafeng and Chi, Guanying and Yu, Jiaming and Han, Jinzhao and Zhang, Shouyu and Ji, Pei and Lyu, Siwei and others},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={4360--4369},
  year={2021}
}"""
                ),
                "config": DatasetConfig(
                    dataset_id="forgerynet",
                    raw_video_dir="storage/datasets/forgerynet/raw_videos",
                    processed_faces_dir="storage/datasets/forgerynet/face_crops",
                    audio_features_dir="storage/datasets/forgerynet/audio_data",
                    metadata_csv_path="storage/datasets/forgerynet/metadata.csv"
                )
            },
            {
                "id": "wilddeepfake",
                "meta": DatasetMetadata(
                    name="WildDeepfake",
                    version="1.0",
                    release_year=2020,
                    modality="video",
                    total_samples=7307,
                    real_samples=3805,
                    fake_samples=3502,
                    file_size_gb=10.0,
                    description="Deepfake videos extracted directly from the wild (internet/social media platforms).",
                    download_source="https://github.com/chengshuozhang/WildDeepfake",
                    labels=["real", "fake"],
                    classes=["real", "fake"],
                    expected_folder_structure={
                        "real_clips": "real/*.mp4",
                        "fake_clips": "fake/*.mp4",
                        "metadata": "metadata.csv"
                    },
                    citation=r"""@inproceedings{zi2020wilddeepfake,
  title={Wilddeepfake: A challenging real-world dataset for deepfake detection},
  author={Zi, Bo and Chang, Minghao and Chen, Jing and Ma, Xingjun and Jiang, Yu-Gang},
  booktitle={Proceedings of the 28th ACM International Conference on Multimedia},
  pages={2382--2390},
  year={2020}
}"""
                ),
                "config": DatasetConfig(
                    dataset_id="wilddeepfake",
                    raw_video_dir="storage/datasets/wilddeepfake/raw_videos",
                    processed_faces_dir="storage/datasets/wilddeepfake/face_crops",
                    audio_features_dir="storage/datasets/wilddeepfake/audio_data",
                    metadata_csv_path="storage/datasets/wilddeepfake/metadata.csv"
                )
            },
            {
                "id": "deepfake_detection",
                "meta": DatasetMetadata(
                    name="Deepfake Detection Local Dataset",
                    version="1.0",
                    release_year=2026,
                    modality="video",
                    total_samples=32,
                    real_samples=17,
                    fake_samples=15,
                    file_size_gb=0.1,
                    description="Local Deepfake Detection benchmark dataset.",
                    download_source="Local Workspace",
                    labels=["real", "fake"],
                    classes=["real", "fake"],
                    expected_folder_structure={
                        "real_videos": "real/",
                        "fake_videos": "fake/"
                    },
                    citation="Local Dataset"
                ),
                "config": DatasetConfig(
                    dataset_id="deepfake_detection",
                    raw_video_dir="deepfake_detection/data/raw_videos",
                    processed_faces_dir="deepfake_detection/face_crops",
                    audio_features_dir="deepfake_detection/audio_data",
                    metadata_csv_path="deepfake_detection/data/metadata.csv"
                )
            }
        ]
        
        for item in default_benchmarks:
            self.datasets[item["id"]] = {
                "metadata": item["meta"],
                "config": item["config"]
            }
            self.configs[item["id"]] = item["config"]

    def get_dataset(self, dataset_id: str) -> Optional[Dict[str, Any]]:
        return self.datasets.get(dataset_id.lower())

    def get_metadata(self, dataset_id: str) -> Optional[DatasetMetadata]:
        data = self.get_dataset(dataset_id)
        return data["metadata"] if data else None

    def get_config(self, dataset_id: str) -> Optional[DatasetConfig]:
        return self.configs.get(dataset_id.lower())

    def register_dataset(self, dataset_id: str, metadata: DatasetMetadata, config: DatasetConfig) -> None:
        """
        Dynamically register new custom forensic datasets at runtime.
        """
        self.datasets[dataset_id.lower()] = {
            "metadata": metadata,
            "config": config
        }
        self.configs[dataset_id.lower()] = config

    def validate_dataset_directories(self, dataset_id: str) -> Dict[str, Any]:
        """
        Verifies if paths configured for a dataset physically exist on disk.
        """
        cfg = self.get_config(dataset_id)
        if not cfg:
            return {"valid": False, "error": f"Dataset '{dataset_id}' not found in registry."}
            
        status = {
            "raw_video_dir_exists": os.path.exists(cfg.raw_video_dir),
            "processed_faces_dir_exists": os.path.exists(cfg.processed_faces_dir),
            "audio_features_dir_exists": os.path.exists(cfg.audio_features_dir),
            "metadata_csv_exists": os.path.exists(cfg.metadata_csv_path)
        }
        
        valid = all(status.values())
        return {
            "valid": valid,
            "paths_status": status
        }
        
if __name__ == "__main__":
    registry = DatasetRegistry()
    print("Registered datasets:", list(registry.datasets.keys()))
