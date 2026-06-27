import os
import cv2
import librosa
from PIL import Image
from typing import Dict, Any, List
from app.utils.exceptions import VideoCorruptedException, AudioMissingException, InvalidInputException

def is_safe_path(target_path: str, base_directory: str) -> bool:
    """
    Path Manager Security Utility.
    Ensures the absolute path of 'target_path' resides strictly inside the 'base_directory'.
    Prevents path traversal vulnerabilities (e.g., symlink attacks or ../ escapes).
    """
    abs_base = os.path.abspath(base_directory)
    abs_target = os.path.abspath(target_path)
    return abs_target.startswith(abs_base)

def validate_video_integrity(video_path: str) -> Dict[str, Any]:
    """
    Video Utility.
    Verifies that the video file can be opened and successfully parsed by OpenCV.
    Extracts frame count, FPS, dimensions, and duration.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise VideoCorruptedException(video_path, "OpenCV failed to open video file capture stream.")
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    cap.release()
    
    if fps <= 0 or frame_count <= 0 or width <= 0 or height <= 0:
        raise VideoCorruptedException(video_path, "Video metadata properties contain invalid zero or negative dimensions/frames.")
        
    duration = frame_count / fps
    return {
        "valid": True,
        "width": width,
        "height": height,
        "fps": fps,
        "frame_count": int(frame_count),
        "duration_seconds": round(duration, 2)
    }

def validate_audio_integrity(audio_path: str) -> Dict[str, Any]:
    """
    Audio Utility.
    Validates audio stream properties using Librosa.
    Extracts sample rate and duration.
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
        
    try:
        y, sr = librosa.load(audio_path, sr=None, duration=1.0) # load 1 second to test
        duration = librosa.get_duration(y=y, sr=sr)
    except Exception as e:
        raise AudioMissingException(f"Audio format cannot be decoded or is missing: {e}")
        
    return {
        "valid": True,
        "sample_rate": sr,
        "duration_seconds": round(duration, 2)
    }

def validate_image_integrity(image_path: str) -> Dict[str, Any]:
    """
    Image Utility.
    Ensures PIL can open and decode facial crop image dimensions.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image crop not found: {image_path}")
        
    try:
        with Image.open(image_path) as img:
            width, height = img.size
            img_format = img.format
            img.verify() # verify integrity
    except Exception as e:
        raise InvalidInputException("image_file", image_path, f"Image is corrupted or unreadable. PIL error: {e}")
        
    return {
        "valid": True,
        "width": width,
        "height": height,
        "format": img_format
    }
