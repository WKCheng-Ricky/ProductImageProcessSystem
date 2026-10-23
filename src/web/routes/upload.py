"""
Upload Route - Page 1: Upload & Collage Detection

Author: W Cheng
Co-Author: Claude (Anthropic AI Assistant)
Created: 2026-01-09
Project: Product Image Processing System
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Request
from PIL import Image
import io
import logging
from pathlib import Path

from ..models.schemas import UploadResponse, CollageInfoResponse
from ..identifiers import QueryPlainId
from ..services.session_manager import SessionManager
from ..services.pipeline_service import get_pipeline
from src.pipeline.collage_detector import CollageInfo

logger = logging.getLogger(__name__)
router = APIRouter()

# Allowed file extensions
ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB


def get_session_manager(request: Request) -> SessionManager:
    """Dependency for session manager"""
    return request.app.state.session_manager


@router.post("/upload", response_model=UploadResponse)
def upload_image(
    file: UploadFile = File(...),
    skip_watermark_removal: bool = False,
    skip_collage_detection: bool = False,
    id: QueryPlainId = None,
    session_manager: SessionManager = Depends(get_session_manager)
):
    """
    Upload an image and detect collage

    This is Page 1 of the pipeline:
    - Validates uploaded file
    - Saves original image
    - Detects if image is a collage
    - Creates session for pipeline state
    """
    # Validate file extension
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )

    # Read file content
    try:
        content = file.file.read()
        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=400,
                detail=f"File too large. Maximum size: {MAX_FILE_SIZE // 1024 // 1024}MB"
            )

        # Open image with PIL
        img = Image.open(io.BytesIO(content)).convert('RGB')
    except Exception as e:
        logger.error(f"Error reading image: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

    # Generate session ID
    session_id = session_manager.generate_session_id()
    session_path = session_manager.get_session_path(session_id)

    # Create session directory
    session_path.mkdir(parents=True, exist_ok=True)

    # Save original image
    original_path = session_path / f"original{file_ext}"
    try:
        img.save(original_path, quality=95)
        logger.info(f"Saved original image to {original_path}")
    except Exception as e:
        logger.error(f"Error saving image: {e}")
        raise HTTPException(status_code=500, detail="Error saving image")

    # Get pipeline
    pipeline = get_pipeline()

    # Step 0: Remove watermarks (before collage detection) - CONDITIONAL
    watermark_result = None
    watermarks_removed = 0
    watermark_skipped = skip_watermark_removal

    if not skip_watermark_removal:
        try:
            watermark_result = pipeline.watermark_remover.remove_watermark(img)
            watermarks_removed = watermark_result.watermarks_removed
            if watermarks_removed > 0:
                logger.info(f"Removed {watermarks_removed} watermarks before collage detection")
                img_for_detection = watermark_result.image  # Use cleaned image

                # Save watermark-removed image
                watermark_removed_path = session_path / f"watermark_removed{file_ext}"
                watermark_result.image.save(watermark_removed_path, quality=95)
                logger.info(f"Saved watermark-removed image to {watermark_removed_path}")
            else:
                logger.info("No watermarks detected")
                img_for_detection = img  # Use original
        except Exception as e:
            logger.warning(f"Watermark removal failed, using original: {e}")
            img_for_detection = img
            watermarks_removed = 0
    else:
        logger.info("Watermark removal skipped by user")
        img_for_detection = img

    # Step 1: Run collage detection (on watermark-cleaned image) - CONDITIONAL
    collage_skipped = skip_collage_detection

    if not skip_collage_detection:
        try:
            collage_info = pipeline.collage_detector.is_collage(img_for_detection)
            logger.info(f"Collage detection: {collage_info}")
        except Exception as e:
            logger.error(f"Error in collage detection: {e}")
            raise HTTPException(
                status_code=500,
                detail="Error detecting collage"
            )
    else:
        # Create a default "not collage" result
        collage_info = CollageInfo(
            is_collage=False,
            num_images=1,
            layout="single",
            grid_size=None,
            confidence=1.0
        )
        logger.info("Collage detection skipped by user")

    # Create session
    session_data = {
        "current_step": 1,
        "original_path": str(original_path),
        "original_filename": file.filename,
        "id": id.plainId if id is not None else None,
        "watermark_skipped": watermark_skipped,
        "watermarks_removed": watermarks_removed,
        "collage_skipped": collage_skipped,
        "collage_info": {
            "is_collage": collage_info.is_collage,
            "num_images": collage_info.num_images,
            "layout": collage_info.layout,
            "grid_size": collage_info.grid_size,
            "confidence": collage_info.confidence
        }
    }

    try:
        session_manager.create_session(session_id, session_data)
    except Exception as e:
        logger.error(f"Error creating session: {e}")
        raise HTTPException(status_code=500, detail="Error creating session")

    # Prepare response
    collage_response = CollageInfoResponse(
        is_collage=collage_info.is_collage,
        num_images=collage_info.num_images,
        layout=collage_info.layout,
        grid_size=collage_info.grid_size,
        confidence=collage_info.confidence
    )

    return UploadResponse(
        session_id=session_id,
        collage_info=collage_response,
        original_image_url=f"/uploads/sessions/{session_id}/original{file_ext}",
        watermark_skipped=watermark_skipped,
        collage_skipped=collage_skipped
    )


@router.get("/session/{session_id}/collage", response_model=CollageInfoResponse)
def get_collage_info(
    session_id: str,
    session_manager: SessionManager = Depends(get_session_manager)
):
    """Get collage detection results for a session"""
    try:
        session = session_manager.get_session(session_id)
        collage_info = session["collage_info"]

        return CollageInfoResponse(
            is_collage=collage_info["is_collage"],
            num_images=collage_info["num_images"],
            layout=collage_info["layout"],
            grid_size=tuple(collage_info["grid_size"]) if collage_info.get("grid_size") else None,
            confidence=collage_info["confidence"]
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error retrieving collage info: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
