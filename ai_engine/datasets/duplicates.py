import os
import cv2
import logging
from typing import Dict, Any, List, Set, Tuple, Optional
from app.utils.crypto import calculate_sha256

logger = logging.getLogger("system")

class VideoDuplicateDetector:
    """
    Forensic Duplicate and Near-Duplicate Video Detector.
    Uses:
    1. Exact File Hashes (SHA-256)
    2. Perceptual Hashing (aHash/Average Hash of key frames)
    3. Metadata properties matching (duration, FPS, resolution, file size)
    """
    def __init__(self, hamming_threshold: int = 4) -> None:
        self.hamming_threshold = hamming_threshold

    def calculate_ahash(self, frame) -> str:
        """
        Calculates 64-bit Average Hash (aHash) for a video frame image.
        1. Resize to 8x8 pixels.
        2. Convert to grayscale.
        3. Compute average color intensity.
        4. Output 64-bit binary representation.
        """
        if frame is None:
            return "0" * 64
        # Resize to 8x8 and convert to grayscale
        resized = cv2.resize(frame, (8, 8), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        
        # Calculate average
        avg = gray.mean()
        
        # Generate hash bitstring
        bits = "".join(["1" if pixel > avg else "0" for row in gray for pixel in row])
        
        # Convert binary string to hex
        hex_val = f"{int(bits, 2):016x}"
        return hex_val

    def get_hamming_distance(self, hash1: str, hash2: str) -> int:
        """
        Returns Hamming distance between two 16-character hexadecimal perceptual hashes.
        """
        try:
            val1 = int(hash1, 16)
            val2 = int(hash2, 16)
            # XOR to find bit differences, then count set bits
            xor_val = val1 ^ val2
            return bin(xor_val).count("1")
        except Exception:
            return 64 # Max distance on error

    def extract_video_signature(self, video_abs_path: str) -> Dict[str, Any]:
        """
        Extracts exact hash, metadata signature, and perceptual hash for a video file.
        """
        signature = {
            "sha256": "",
            "file_size_bytes": 0,
            "duration": 0.0,
            "fps": 0.0,
            "resolution": "0x0",
            "perceptual_hash": ""
        }

        if not os.path.exists(video_abs_path):
            return signature

        # 1. Exact SHA-256
        signature["sha256"] = calculate_sha256(video_abs_path)
        signature["file_size_bytes"] = os.path.getsize(video_abs_path)

        # 2. Metadata Properties & Perceptual Hash from first frame
        cap = cv2.VideoCapture(video_abs_path)
        if cap.isOpened():
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            signature["resolution"] = f"{w}x{h}"
            signature["fps"] = round(fps, 2) if fps > 0 else 0.0
            signature["duration"] = round(frames / fps, 2) if (fps > 0 and frames > 0) else 0.0

            # Read first frame for perceptual hash
            ret, frame = cap.read()
            if ret:
                signature["perceptual_hash"] = self.calculate_ahash(frame)
            cap.release()

        return signature

    def scan_for_duplicates(self, video_paths: List[str]) -> Dict[str, Any]:
        """
        Scans a list of absolute video paths and clusters duplicates and near-duplicates.
        """
        signatures: Dict[str, Dict[str, Any]] = {}
        
        # Extract signatures
        for path in video_paths:
            if os.path.exists(path):
                signatures[path] = self.extract_video_signature(path)

        duplicates: List[Dict[str, Any]] = []
        visited: Set[str] = set()

        path_list = list(signatures.keys())
        
        for i in range(len(path_list)):
            p1 = path_list[i]
            if p1 in visited:
                continue
                
            sig1 = signatures[p1]
            cluster = []

            for j in range(i + 1, len(path_list)):
                p2 = path_list[j]
                if p2 in visited:
                    continue
                    
                sig2 = signatures[p2]
                
                # Check 1: Exact Match (SHA-256)
                is_exact = (sig1["sha256"] == sig2["sha256"]) and (sig1["sha256"] != "")
                
                # Check 2: Near-Duplicate Perceptual Hash distance
                h_dist = self.get_hamming_distance(sig1["perceptual_hash"], sig2["perceptual_hash"])
                is_near_p = (h_dist <= self.hamming_threshold) and (sig1["perceptual_hash"] != "")

                # Check 3: Metadata match (extreme case of structural similarities)
                is_meta_match = (
                    sig1["resolution"] == sig2["resolution"] and
                    sig1["duration"] == sig2["duration"] and
                    sig1["fps"] == sig2["fps"] and
                    sig1["file_size_bytes"] == sig2["file_size_bytes"] and
                    sig1["file_size_bytes"] > 0
                )

                if is_exact or is_near_p or is_meta_match:
                    reason = "sha256_exact"
                    if is_near_p and not is_exact:
                        reason = f"perceptual_near_match (distance={h_dist})"
                    elif is_meta_match and not is_exact:
                        reason = "metadata_exact_match"
                        
                    cluster.append({
                        "file_path": p2,
                        "type": reason,
                        "hamming_distance": h_dist
                    })
                    visited.add(p2)

            if cluster:
                duplicates.append({
                    "original_file": p1,
                    "signatures": sig1,
                    "copies": cluster
                })
                visited.add(p1)

        return {
            "total_files_scanned": len(video_paths),
            "duplicate_groups_found": len(duplicates),
            "duplicates": duplicates
        }
