"""
Preprocessing Route - Page 3: Image Preprocessing

Author: W Cheng
Co-Author: Claude (Anthropic AI Assistant)
Created: 2026-01-09
Project: Product Image Processing System
"""

from fastapi import APIRouter, HTTPException, Depends, Request
from PIL import Image
import logging
from pathlib import Path

from ..models.schemas import PreprocessingResultResponse
from ..services.session_manager import SessionManager
from ..services.pipeline_service import get_pipeline

logger = logging.getLogger(__name__)
router = APIRouter()


def get_session_manager(request: Request) -> SessionManager:
    """Dependency for session manager"""
    return request.app.state.session_manager


@router.post("/session/{session_id}/preprocess", response_model=PreprocessingResultResponse)
def preprocess_image(
    session_id: str,
    remove_bg: bool = True,
    crop: bool = True,
    remove_watermark: bool = True,
    remove_text: bool = False,
    resize: bool = True,
    session_manager: SessionManager = Depends(get_session_manager)
):
    """
    Preprocess image (background removal, cropping, etc.)

    This is Page 3 of the pipeline:
    - Removes background (optional)
    - Crops to product (optional)
    - Removes watermarks (optional)
    - Removes text overlays (optional)
    - Resizes to standard size (optional)
    """
    # Get session
    try:
        session = session_manager.get_session(session_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    # Determine which image to process
    # If split was performed, use the selected split image
    # Otherwise, use the original
    if "split_result" in session:
        split_data = session["split_result"]
        selected_index = session.get("selected_image_index", 0)
        if selected_index is None or selected_index >= len(split_data["split_paths"]):
            selected_index = 0
        image_path = Path(split_data["split_paths"][selected_index])
    else:
        image_path = Path(session["original_path"])

    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")

    # Load image
    try:
        img = Image.open(image_path).convert('RGB')
        original_size = img.size
        logger.info(f"Processing image: {image_path}, size: {original_size}")
    except Exception as e:
        logger.error(f"Error loading image: {e}")
        raise HTTPException(status_code=500, detail=f"Error loading image: {str(e)}")

    # Run preprocessing
    try:
        pipeline = get_pipeline()

        # Call standardize() method with correct parameters
        result = pipeline.preprocessor.standardize(
            img,
            remove_bg=remove_bg,
            crop=crop,
            remove_watermark=remove_watermark,
            remove_text=remove_text,
            resize=resize
        )

        logger.info(f"Preprocessing complete: bg_removed={result.background_removed}")
    except Exception as e:
        logger.error(f"Error in preprocessing: {e}")
        raise HTTPException(status_code=500, detail=f"Error preprocessing image: {str(e)}")

    # Save processed image
    session_path = session_manager.get_session_path(session_id)
    processed_path = session_path / "preprocessed.jpg"

    try:
        result.image.save(processed_path, quality=95)
        logger.info(f"Saved processed image to {processed_path}")
    except Exception as e:
        logger.error(f"Error saving processed image: {e}")
        raise HTTPException(status_code=500, detail=f"Error saving image: {str(e)}")

    # Update session
    preprocessing_data = {
        "processed_path": str(processed_path),
        "original_size": original_size,
        "background_removed": result.background_removed,
        "watermark_removed": result.watermark_removed,
        "text_removed": result.text_removed,
        "product_bbox": result.product_bbox
    }

    try:
        session_manager.update_session(session_id, {
            "current_step": 3,
            "preprocessing_result": preprocessing_data
        })
    except Exception as e:
        logger.error(f"Error updating session: {e}")
        raise HTTPException(status_code=500, detail=f"Error updating session: {str(e)}")

    return PreprocessingResultResponse(
        processed_image_url=f"/uploads/sessions/{session_id}/preprocessed.jpg",
        original_size=original_size,
        background_removed=result.background_removed,
        watermark_removed=result.watermark_removed,
        text_removed=result.text_removed,
        product_bbox=result.product_bbox
    )


@router.post("/session/{session_id}/preprocess/bypass")
def bypass_preprocessing(
    session_id: str,
    session_manager: SessionManager = Depends(get_session_manager)
):
    """
    Bypass preprocessing and use original image directly.
    Copies the source image as the preprocessed image without modifications.
    """
    try:
        session = session_manager.get_session(session_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    # Determine source image
    if "split_result" in session:
        split_data = session["split_result"]
        selected_index = session.get("selected_image_index", 0)
        if selected_index is None or selected_index >= len(split_data["split_paths"]):
            selected_index = 0
        image_path = Path(split_data["split_paths"][selected_index])
    else:
        image_path = Path(session["original_path"])

    if not image_path.exists():
        raise HTTPException(status_code=404, detail="Image not found")

    # Load and save as preprocessed (no modifications)
    try:
        img = Image.open(image_path).convert('RGB')
        original_size = img.size

        session_path = session_manager.get_session_path(session_id)
        processed_path = session_path / "preprocessed.jpg"
        img.save(processed_path, quality=95)

        logger.info(f"Bypass preprocessing: copied {image_path} to {processed_path}")
    except Exception as e:
        logger.error(f"Error in bypass preprocessing: {e}")
        raise HTTPException(status_code=500, detail=str(e))

    # Update session
    preprocessing_data = {
        "processed_path": str(processed_path),
        "original_size": original_size,
        "background_removed": False,
        "watermark_removed": False,
        "text_removed": False,
        "product_bbox": None,
        "bypassed": True
    }

    session_manager.update_session(session_id, {
        "current_step": 3,
        "preprocessing_result": preprocessing_data
    })

    return {
        "success": True,
        "message": "Preprocessing bypassed",
        "processed_image_url": f"/uploads/sessions/{session_id}/preprocessed.jpg"
    }


@router.get("/session/{session_id}/preprocess", response_model=PreprocessingResultResponse)
def get_preprocessing_results(
    session_id: str,
    session_manager: SessionManager = Depends(get_session_manager)
):
    """Get preprocessing results for a session"""
    try:
        session = session_manager.get_session(session_id)

        if "preprocessing_result" not in session:
            raise HTTPException(status_code=404, detail="Preprocessing not performed yet")

        prep_data = session["preprocessing_result"]

        return PreprocessingResultResponse(
            processed_image_url=f"/uploads/sessions/{session_id}/preprocessed.jpg",
            original_size=tuple(prep_data["original_size"]),
            background_removed=prep_data["background_removed"],
            watermark_removed=prep_data["watermark_removed"],
            text_removed=prep_data["text_removed"],
            product_bbox=tuple(prep_data["product_bbox"]) if prep_data.get("product_bbox") else None
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving preprocessing results: {e}")
        raise HTTPException(status_code=500, detail=str(e))
