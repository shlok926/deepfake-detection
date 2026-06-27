from enum import Enum

class EnvironmentType(str, Enum):
    DEVELOPMENT = "development"
    TESTING = "testing"
    PRODUCTION = "production"

class ModelType(str, Enum):
    VIDEO = "video"
    AUDIO = "audio"
    MULTIMODAL = "multimodal"

class FusionStrategyType(str, Enum):
    LATE_AVERAGE = "late_average"
    LATE_CONCAT = "late_concat"
    EARLY = "early"

class LogLevelType(str, Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"

# Global Constants
DEFAULT_MAX_AUDIO_LEN_STEPS = 300
DEFAULT_FACE_CROP_SIZE = (224, 224)
SHA256_HASH_LENGTH = 64
DEFAULT_JWT_ALGORITHM = "HS256"
