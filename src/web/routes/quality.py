"""
Quality Route - Page 2: Aesthetic Scoring

Aesthetic scoring is done on the original/raw image BEFORE any processing.
This allows evaluating the inherent quality of the uploaded image.

Author: W Cheng
Co-Author: Claude (Anthropic AI Assistant)
Created: 2026-01-09
Updated: 2026-01-11
Project: Product Image Processing System
"""

from fastapi import APIRouter, HTTPException, Depends, Request
from PIL import Image
import logging
import os
from pathlib import Path

from ..models.schemas import QualityScoreResponse
from ..services.session_manager import SessionManager
from ..services.pipeline_service import get_pipeline

logger = logging.getLogger(__name__)
router = APIRouter()


def get_session_manager(request: Request) -> SessionManager:
    """Dependency for session manager"""
    return request.app.state.session_manager


@router.post("/session/{session_id}/quality", response_model=QualityScoreResponse)
async def score_quality(
    session_id: str,
    session_manager: SessionManager = Depends(get_session_manager)
):
    """
    Score image aesthetic quality

    This is Page 2 of the pipeline:
    - Evaluates sharpness, brightness, contrast on the ORIGINAL image
    - Checks for watermarks and text overlays
    - Calculates overall aesthetic score

    Aesthetic scoring is done BEFORE any preprocessing to evaluate
    the raw image quality.
    """
    # Get session
    try:
        session = session_manager.get_session(session_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    # Always use original image for aesthetic scoring (before any processing)
    image_path = Path(session["original_path"])

    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")

    # Load image
    try:
        img = Image.open(image_path).convert('RGB')
        logger.info(f"Scoring quality for: {image_path}")
    except Exception as e:
        logger.error(f"Error loading image: {e}")
        raise HTTPException(status_code=500, detail=f"Error loading image: {str(e)}")

    # Run quality scoring
    try:
        pipeline = get_pipeline()
        quality_score = pipeline.quality_scorer.score(img)
        logger.info(f"Quality score: {quality_score.overall:.2f}")
    except Exception as e:
        logger.error(f"Error in quality scoring: {e}")
        raise HTTPException(status_code=500, detail=f"Error scoring quality: {str(e)}")

    # Get original image URL (with correct extension)
    original_filename = image_path.name
    original_image_url = f"/uploads/sessions/{session_id}/{original_filename}"

    # Extract file extension
    _, file_extension = os.path.splitext(original_filename)

    # Update session
    quality_data = {
        "overall": quality_score.overall,
        "sharpness": quality_score.sharpness,
        "brightness": quality_score.brightness,
        "contrast": quality_score.contrast,
        "resolution_score": quality_score.resolution_score,
        "product_ratio": quality_score.product_ratio,
        "has_watermark": quality_score.has_watermark,
        "has_text_overlay": quality_score.has_text_overlay,
        "aesthetic": quality_score.aesthetic, # aesthetic overall score
        "clarity_score": quality_score.clarity_score,
        "clarity_score_est": quality_score.clarity_score_est,
        "composition_score": quality_score.composition_score,
        "composition_score_est": quality_score.composition_score_est,
        "defect_score": quality_score.defect_score,
        "defect_score_est": quality_score.defect_score_est,
        "original_image_url": original_image_url,
        "file_extension": file_extension,
        "resolution": (img.width, img.height)
    }

    try:
        session_manager.update_session(session_id, {
            "current_step": 2,
            "quality_score": quality_data # to be used in embedding.py
        })
    except Exception as e:
        logger.error(f"Error updating session: {e}")
        raise HTTPException(status_code=500, detail=f"Error updating session: {str(e)}")

    return QualityScoreResponse(**quality_data)


@router.get("/session/{session_id}/quality", response_model=QualityScoreResponse)
async def get_quality_score(
    session_id: str,
    session_manager: SessionManager = Depends(get_session_manager)
):
    """Get quality score for a session"""
    try:
        session = session_manager.get_session(session_id)

        if "quality_score" not in session:
            raise HTTPException(status_code=404, detail="Quality scoring not performed yet")

        quality_data = session["quality_score"]
        return QualityScoreResponse(**quality_data)

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving quality score: {e}")
        raise HTTPException(status_code=500, detail=str(e))
