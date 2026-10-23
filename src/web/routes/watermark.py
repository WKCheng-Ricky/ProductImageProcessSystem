"""
Watermark Removal Route - Page 2: Watermark Removal

Author: W Cheng
Co-Author: Claude (Anthropic AI Assistant)
Created: 2026-01-09
Project: Product Image Processing System
"""

from fastapi import APIRouter, HTTPException, Depends, Request
from fastapi.templating import Jinja2Templates
from PIL import Image
import logging
from pathlib import Path

from ..models.schemas import WatermarkRemovalResponse, WatermarkRegionInfo
from ..services.session_manager import SessionManager
from ..services.pipeline_service import get_pipeline

logger = logging.getLogger(__name__)
router = APIRouter()
templates = Jinja2Templates(directory="web/templates")


def get_session_manager(request: Request) -> SessionManager:
    """Dependency for session manager"""
    return request.app.state.session_manager


@router.get("/watermark/{session_id}")
async def watermark_page(request: Request, session_id: str):
    """Render watermark removal page"""
    return templates.TemplateResponse(request, "watermark.html", {
        "session_id": session_id
    })


@router.post("/session/{session_id}/watermark", response_model=WatermarkRemovalResponse)
def remove_watermarks(
    session_id: str,
    session_manager: SessionManager = Depends(get_session_manager)
):
    """
    Remove watermarks from uploaded image

    This is Page 2 of the pipeline:
    - Retrieves session and original image
    - Detects and removes text watermarks
    - Saves watermark-removed image
    - Updates session with watermark removal results
    """
    # Get session
    try:
        session = session_manager.get_session(session_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    # Get original image
    original_path = Path(session["original_path"])
    if not original_path.exists():
        raise HTTPException(status_code=404, detail="Original image not found")

    try:
        img = Image.open(original_path).convert('RGB')
    except Exception as e:
        logger.error(f"Error loading image: {e}")
        raise HTTPException(status_code=500, detail="Error loading image")

    # Get pipeline and remove watermarks
    try:
        pipeline = get_pipeline()
        watermark_result = pipeline.watermark_remover.remove_watermark(img)

        logger.info(f"Watermark removal: detected={len(watermark_result.watermarks_detected)}, "
                   f"removed={watermark_result.watermarks_removed}")
    except Exception as e:
        logger.error(f"Error removing watermarks: {e}")
        raise HTTPException(status_code=500, detail="Error removing watermarks")

    # Save watermark-removed image
    session_path = session_manager.get_session_path(session_id)
    file_ext = original_path.suffix
    watermark_removed_path = session_path / f"watermark_removed{file_ext}"

    watermark_removed_url = None
    if watermark_result.watermarks_removed > 0:
        try:
            watermark_result.image.save(watermark_removed_path, quality=95)
            watermark_removed_url = f"/uploads/sessions/{session_id}/watermark_removed{file_ext}"
            logger.info(f"Saved watermark-removed image to {watermark_removed_path}")
        except Exception as e:
            logger.error(f"Error saving watermark-removed image: {e}")
            raise HTTPException(status_code=500, detail="Error saving image")

    # Convert watermark regions to response format
    regions = [
        WatermarkRegionInfo(
            bbox=wm.bbox,
            text=wm.text,
            confidence=wm.confidence,
            type=wm.type
        )
        for wm in watermark_result.watermarks_detected
    ]

    # Update session
    watermark_data = {
        "watermarks_detected": len(watermark_result.watermarks_detected),
        "watermarks_removed": watermark_result.watermarks_removed,
        "watermark_removed_path": str(watermark_removed_path) if watermark_removed_url else None
    }

    try:
        session_manager.update_session(session_id, {
            "current_step": 2,
            "watermark_result": watermark_data
        })
    except Exception as e:
        logger.error(f"Error updating session: {e}")
        raise HTTPException(status_code=500, detail="Error updating session")

    # Get original image URL
    original_url = f"/uploads/sessions/{session_id}/original{file_ext}"

    return WatermarkRemovalResponse(
        watermarks_detected=len(watermark_result.watermarks_detected),
        watermarks_removed=watermark_result.watermarks_removed,
        success=watermark_result.success,
        original_image_url=original_url,
        watermark_removed_image_url=watermark_removed_url,
        regions=regions
    )


@router.get("/session/{session_id}/watermark", response_model=WatermarkRemovalResponse)
def get_watermark_results(
    session_id: str,
    session_manager: SessionManager = Depends(get_session_manager)
):
    """Get watermark removal results for a session"""
    try:
        session = session_manager.get_session(session_id)

        if "watermark_result" not in session:
            raise HTTPException(status_code=404, detail="Watermark removal not performed yet")

        watermark_data = session["watermark_result"]
        file_ext = Path(session["original_path"]).suffix
        original_url = f"/uploads/sessions/{session_id}/original{file_ext}"

        watermark_removed_url = None
        if watermark_data["watermark_removed_path"]:
            watermark_removed_url = f"/uploads/sessions/{session_id}/watermark_removed{file_ext}"

        return WatermarkRemovalResponse(
            watermarks_detected=watermark_data["watermarks_detected"],
            watermarks_removed=watermark_data["watermarks_removed"],
            success=True,
            original_image_url=original_url,
            watermark_removed_image_url=watermark_removed_url,
            regions=[]  # Regions not stored in session currently
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving watermark results: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
