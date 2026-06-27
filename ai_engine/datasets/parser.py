from abc import ABC, abstractmethod
from typing import Any, Dict, List


class BaseDatasetParser(ABC):
    """
    Abstract Base Dataset Parser / Adapter.
    Adding a new dataset to the platform requires only:
    1. Implementing a subclass of BaseDatasetParser to map custom raw metadata files (JSON, TXT, XML)
       into standard list layout: [{"video_path": "...", "label": "real/fake"}]
    2. Registering the dataset metadata and directory config in the DatasetRegistry class.

    No core preprocessing or model inference code needs to be modified.
    """

    @abstractmethod
    def parse_metadata(self, raw_dataset_dir: str) -> List[Dict[str, str]]:
        """
        Parses raw dataset files and returns a standard list of dictionaries:
        [
          {"video_path": "relative/path/to/video.mp4", "label": "real"},
          {"video_path": "relative/path/to/video2.mp4", "label": "fake"}
        ]
        """
        pass
