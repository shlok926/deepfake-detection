import os
import json
import logging
from typing import Dict, Any, Optional

logger = logging.getLogger("system")

class PreprocessingCache:
    """
    MLOps Caching System to skip redundant video framing, face detections, 
    vocal extractions, or spectrogram computations.
    Persists mappings of video SHA256 hashes to artifact folders.
    """
    def __init__(self, cache_dir: str = "storage/cache") -> None:
        self.cache_dir = cache_dir
        self.manifest_path = os.path.join(cache_dir, "cache_manifest.json")
        
        # Ensure directories exist
        os.makedirs(os.path.join(cache_dir, "frames"), exist_ok=True)
        os.makedirs(os.path.join(cache_dir, "crops"), exist_ok=True)
        os.makedirs(os.path.join(cache_dir, "audio"), exist_ok=True)
        os.makedirs(os.path.join(cache_dir, "spectrograms"), exist_ok=True)
        
        self.manifest: Dict[str, Dict[str, Any]] = self._load_manifest()

    def _load_manifest(self) -> Dict[str, Dict[str, Any]]:
        """
        Loads local cache catalog.
        """
        if os.path.exists(self.manifest_path):
            try:
                with open(self.manifest_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"Failed to read cache manifest: {e}")
                return {}
        return {}

    def _save_manifest(self) -> None:
        """
        Saves local cache catalog.
        """
        try:
            with open(self.manifest_path, "w", encoding="utf-8") as f:
                json.dump(self.manifest, f, indent=4)
        except Exception as e:
            logger.error(f"Failed to write cache manifest: {e}")

    def get(self, file_hash: str) -> Optional[Dict[str, Any]]:
        """
        Retrieves preprocessed directory paths if hit occurs, else None.
        Checks folder presence on disk before yielding hit.
        """
        entry = self.manifest.get(file_hash)
        if not entry:
            return None
            
        # Verify physical directories exist on disk to guarantee hits are safe
        keys_to_check = ["extracted_frames_dir", "face_crops_dir", "audio_track_path", "spectrogram_path"]
        for key in keys_to_check:
            path = entry.get(key)
            if path and not os.path.exists(path):
                # Cache entry is stale, clean it
                logger.warning(f"Cache stale for hash '{file_hash}', path not found: {path}")
                self.manifest.pop(file_hash, None)
                self._save_manifest()
                return None
                
        return entry

    def put(
        self, 
        file_hash: str, 
        extracted_frames_dir: Optional[str] = None,
        face_crops_dir: Optional[str] = None,
        audio_track_path: Optional[str] = None,
        spectrogram_path: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> None:
        """
        Dumps new preprocessing references into the manifest catalog.
        """
        self.manifest[file_hash] = {
            "extracted_frames_dir": extracted_frames_dir,
            "face_crops_dir": face_crops_dir,
            "audio_track_path": audio_track_path,
            "spectrogram_path": spectrogram_path,
            "metadata": metadata or {}
        }
        self._save_manifest()

    def clear(self) -> None:
        """
        Purges manifest catalog.
        """
        self.manifest = {}
        self._save_manifest()
