import logging

import torch

from app.config.config import settings
from app.utils.exceptions import GPUUnavailableException

logger = logging.getLogger("system")


def get_optimal_device() -> torch.device:
    """
    Auto-detects and returns the optimal hardware computation device (CUDA/CPU).
    Validates CUDA requirements against settings and logs GPU diagnostics.
    """
    cuda_available = torch.cuda.is_available()

    if settings.USE_GPU:
        if not cuda_available:
            # Fallback to CPU with a warning, or raise if in strict production modes
            logger.warning("CUDA requested in settings but not supported by PyTorch. Falling back to CPU execution.")
            return torch.device("cpu")

        device_count = torch.cuda.device_count()
        if settings.CUDA_DEVICE_ID >= device_count:
            raise GPUUnavailableException(settings.CUDA_DEVICE_ID)

        gpu_name = torch.cuda.get_device_name(settings.CUDA_DEVICE_ID)
        logger.info(f"Target GPU execution context selected: Device [{settings.CUDA_DEVICE_ID}] - {gpu_name}")
        return torch.device(f"cuda:{settings.CUDA_DEVICE_ID}")

    logger.info("CPU execution context selected (USE_GPU is disabled).")
    return torch.device("cpu")


def get_device_diagnostics() -> dict:
    """
    Gathers detailed hardware telemetry details.
    """
    cuda_available = torch.cuda.is_available()
    diagnostics = {
        "cuda_available": cuda_available,
        "device_count": torch.cuda.device_count() if cuda_available else 0,
        "current_device_id": settings.CUDA_DEVICE_ID if cuda_available else None,
        "device_name": (
            torch.cuda.get_device_name(settings.CUDA_DEVICE_ID)
            if cuda_available and settings.CUDA_DEVICE_ID < torch.cuda.device_count()
            else "N/A"
        ),
        "pytorch_version": torch.__version__,
    }
    return diagnostics


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    dev = get_optimal_device()
    print("Optimal device selected:", dev)
    print("Diagnostics:", get_device_diagnostics())
