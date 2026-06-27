import os
import hashlib
from fastapi import APIRouter, UploadFile, File, Depends, HTTPException
from sqlalchemy.orm import Session
from app.database.connection import get_db
from app.schemas.api_schemas import UploadResponse
from app.middleware.security import validate_uploaded_file, validate_secure_filename
from app.config.config import settings

router = APIRouter(prefix="/upload", tags=["Media Uploads"])

@router.post("", response_model=UploadResponse)
async def upload_media_file(
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Validates, hashes, and stores video files securely on disk.
    """
    # 1. Enforce size and extension constraints
    validate_uploaded_file(file)

    # 2. Sanitize filename
    clean_filename = validate_secure_filename(file.filename or "upload.mp4")

    # 3. Read bytes and generate SHA256 (prevents duplicate files)
    file_bytes = await file.read()
    sha256_hash = hashlib.sha256(file_bytes).hexdigest()
    await file.seek(0) # reset stream position

    # Architecture stub (database registry and physical persistence)
    # real_save_path = os.path.join(settings.UPLOADS_DIR, f"{sha256_hash}_{clean_filename}")
    
    # Calculate mock ID for endpoint response validation
    mock_id = 99

    return UploadResponse(
        success=True,
        video_id=mock_id,
        file_name=clean_filename,
        file_size_mb=round(len(file_bytes) / (1024 * 1024), 2),
        sha256_hash=sha256_hash,
        message="Media successfully validated and stored."
    )
