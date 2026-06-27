import hashlib
import hmac
import logging
from typing import Optional

from app.config.config import settings

logger = logging.getLogger("system")


def calculate_sha256(file_path: str) -> str:
    """
    Calculates the SHA-256 hash of a file on disk using buffered streaming chunks
    to avoid heavy memory usage.
    """
    sha256 = hashlib.sha256()
    try:
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(65536), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
    except Exception as e:
        logger.error(f"Error calculating SHA256 checksum for {file_path}: {e}")
        raise IOError(f"Could not hash file: {e}")


def hash_password(password: str) -> str:
    """
    Creates a secure SHA-256 password hash salted with the application secret key.
    Note: In fully-featured deployment, use bcrypt or argon2.
    """
    salt = settings.SECRET_KEY.encode("utf-8")
    hashed = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 100000)  # iterations
    return hashed.hex()


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Verifies if plain password salt match matches original password string.
    Prevents timing attacks using constant-time comparison.
    """
    target_hash = hash_password(plain_password)
    return hmac.compare_digest(target_hash, hashed_password)
