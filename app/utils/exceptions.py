from typing import Optional, Dict, Any
from app.utils.logging import get_error_logger

logger = get_error_logger()

class ForensicsPlatformException(Exception):
    """
    Base exception class for all custom errors inside the platform.
    Ensures every error is structured with a Code, Description, and Solution.
    """
    def __init__(
        self,
        error_code: str,
        description: str,
        solution: str,
        status_code: int = 500,
        extra_details: Optional[Dict[str, Any]] = None
    ):
        self.error_code = error_code
        self.description = description
        self.solution = solution
        self.status_code = status_code
        self.extra_details = extra_details or {}
        
        super().__init__(self.description)
        self.log_error()

    def log_error(self) -> None:
        """
        Automatically logs this exception to the enterprise error logging pipeline.
        """
        log_payload = {
            "error_code": self.error_code,
            "solution": self.solution,
            "extra_details": self.extra_details
        }
        logger.error(
            f"[{self.error_code}] {self.description} | Solution: {self.solution}",
            extra={"extra_info": log_payload}
        )

    def to_dict(self) -> Dict[str, Any]:
        """
        Formats error response as a serialized dictionary.
        """
        return {
            "success": False,
            "error": {
                "code": self.error_code,
                "description": self.description,
                "solution": self.solution,
                "details": self.extra_details
            }
        }

# Concrete Exception Types

class DatasetNotFoundException(ForensicsPlatformException):
    def __init__(self, dataset_name: str, path: str):
        super().__init__(
            error_code="ERR_DATASET_NOT_FOUND",
            description=f"Dataset '{dataset_name}' not found at path '{path}'.",
            solution="Ensure dataset path config matches your folder hierarchy, or check file permissions.",
            status_code=404,
            extra_details={"dataset_name": dataset_name, "expected_path": path}
        )

class GPUUnavailableException(ForensicsPlatformException):
    def __init__(self, requested_device_id: int):
        super().__init__(
            error_code="ERR_GPU_UNAVAILABLE",
            description=f"CUDA GPU device index {requested_device_id} is requested but not available in PyTorch.",
            solution="Verify NVIDIA drivers are installed correctly and check output of 'nvidia-smi'. Otherwise, switch setting to CPU.",
            status_code=503,
            extra_details={"requested_device_id": requested_device_id}
        )

class ModelMissingException(ForensicsPlatformException):
    def __init__(self, model_name: str, expected_path: str):
        super().__init__(
            error_code="ERR_MODEL_MISSING",
            description=f"Trained weight file for model '{model_name}' is missing at: {expected_path}.",
            solution=f"Download correct pretrained weight binaries or train the model first using train.py.",
            status_code=404,
            extra_details={"model_name": model_name, "expected_path": expected_path}
        )

class VideoCorruptedException(ForensicsPlatformException):
    def __init__(self, video_path: str, details: str):
        super().__init__(
            error_code="ERR_VIDEO_CORRUPTED",
            description=f"The video at {video_path} is corrupted or cannot be read. OpenCV/MoviePy error: {details}",
            solution="Verify file format is valid, check for corrupted video byte structures, and try transcoding using FFmpeg first.",
            status_code=400,
            extra_details={"video_path": video_path, "lib_error": details}
        )

class AudioMissingException(ForensicsPlatformException):
    def __init__(self, video_path: str):
        super().__init__(
            error_code="ERR_AUDIO_MISSING",
            description=f"No audio tracks found inside the media file: {video_path}.",
            solution="If running in multimodal mode, use a video that contains active audio stream. Or run prediction with single-modality Video model.",
            status_code=400,
            extra_details={"video_path": video_path}
        )

class FaceDetectionFailureException(ForensicsPlatformException):
    def __init__(self, frame_info: str, details: str):
        super().__init__(
            error_code="ERR_FACE_DETECTION_FAILED",
            description=f"Face detection engine failed to execute. Details: {details}",
            solution="Check if MTCNN model weights are downloaded properly and that image dimensions are correct.",
            status_code=500,
            extra_details={"frame_info": frame_info, "details": details}
        )

class InvalidInputException(ForensicsPlatformException):
    def __init__(self, parameter: str, value: Any, message: str):
        super().__init__(
            error_code="ERR_INVALID_INPUT",
            description=f"Invalid parameter '{parameter}' with value '{value}'. {message}",
            solution="Review request query/body variables and respect type annotations and validations.",
            status_code=422,
            extra_details={"parameter": parameter, "value": str(value)}
        )

class UnsupportedFormatException(ForensicsPlatformException):
    def __init__(self, file_ext: str, allowed_list: list[str]):
        super().__init__(
            error_code="ERR_UNSUPPORTED_FORMAT",
            description=f"File extension '.{file_ext}' is not supported.",
            solution=f"Please upload media files with extensions in the allowed list: {', '.join(allowed_list)}",
            status_code=400,
            extra_details={"file_extension": file_ext, "allowed": allowed_list}
        )

class TimeoutException(ForensicsPlatformException):
    def __init__(self, operation: str, seconds: float):
        super().__init__(
            error_code="ERR_OPERATION_TIMEOUT",
            description=f"Operation '{operation}' timed out after {seconds} seconds.",
            solution="Increase system timeouts or allocate more CPU/GPU resources to process heavy media volumes.",
            status_code=504,
            extra_details={"operation": operation, "limit_seconds": seconds}
        )

class MemoryErrorException(ForensicsPlatformException):
    def __init__(self, context: str):
        super().__init__(
            error_code="ERR_OUT_OF_MEMORY",
            description=f"System exhausted available resources during task: {context}",
            solution="Reduce batch size, trim long video files, or allocate swap memory/CUDA VRAM.",
            status_code=507,
            extra_details={"context": context}
        )
