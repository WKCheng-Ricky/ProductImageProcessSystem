"""
Split Route - Page 2: Collage Split

Author: W Cheng
Co-Author: Claude (Anthropic AI Assistant)
Created: 2026-01-09
Project: Product Image Processing System
"""

from fastapi import APIRouter, HTTPException, Depends, Request
from PIL import Image
from typing import Optional, List
import logging
from pathlib import Path

from ..models.schemas import CollageSplitResultResponse
from ..services.session_manager import SessionManager
from ..services.pipeline_service import get_pipeline
from src.pipeline.collage_detector import CollageInfo

logger = logging.getLogger(__name__)
router = APIRouter()


def get_session_manager(request: Request) -> SessionManager:
    """Dependency for session manager"""
    return request.app.state.session_manager


@router.post("/session/{session_id}/split", response_model=CollageSplitResultResponse)
def split_collage(
    session_id: str,
    selected_indices: Optional[List[int]] = None,
    session_manager: SessionManager = Depends(get_session_manager)
):
    """
    Split collage into individual images

    This is Page 2 of the pipeline:
    - Retrieves session and original image
    - Splits collage based on detected layout
    - Saves individual split images
    - Updates session with split results
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
        raise HTTPException(status_code=500, detail=f"Error loading image: {str(e)}")

    # Get collage info
    collage_data = session["collage_info"]
    collage_info = CollageInfo(
        is_collage=collage_data["is_collage"],
        num_images=collage_data["num_images"],
        layout=collage_data["layout"],
        grid_size=tuple(collage_data["grid_size"]) if collage_data.get("grid_size") else None,
        confidence=collage_data["confidence"]
    )

    # If not a collage, return the original as single image
    if not collage_info.is_collage:
        session_path = session_manager.get_session_path(session_id)
        split_dir = session_path / "split_images"
        split_dir.mkdir(exist_ok=True)

        # Copy original as the only split image
        split_path = split_dir / "image_0.jpg"
        img.save(split_path, quality=95)

        split_result = {
            "num_images": 1,
            "split_paths": [str(split_path)],
            "bboxes": [[0, 0, img.width, img.height]]
        }

        session_manager.update_session(session_id, {
            "current_step": 2,
            "split_result": split_result,
            "selected_image_index": 0
        })

        return CollageSplitResultResponse(
            num_images=1,
            split_image_urls=[f"/uploads/sessions/{session_id}/split_images/image_0.jpg"],
            bboxes=[(0, 0, img.width, img.height)]
        )

    # Perform smart split
    try:
        pipeline = get_pipeline()
        split_result = pipeline.collage_detector.smart_split(img, collage_info)
        logger.info(f"Split collage into {len(split_result.images)} images")
    except Exception as e:
        logger.error(f"Error splitting collage: {e}")
        raise HTTPException(status_code=500, detail=f"Error splitting collage: {str(e)}")

    # Save split images
    session_path = session_manager.get_session_path(session_id)
    split_dir = session_path / "split_images"
    split_dir.mkdir(exist_ok=True)

    split_paths = []
    split_urls = []

    for i, split_img in enumerate(split_result.images):
        # Skip if not in selected indices (if specified)
        if selected_indices is not None and i not in selected_indices:
            continue

        split_path = split_dir / f"image_{i}.jpg"
        try:
            split_img.save(split_path, quality=95)
            split_paths.append(str(split_path))
            split_urls.append(f"/uploads/sessions/{session_id}/split_images/image_{i}.jpg")
        except Exception as e:
            logger.error(f"Error saving split image {i}: {e}")
            raise HTTPException(status_code=500, detail=f"Error saving split image: {str(e)}")

    # Update session
    split_data = {
        "num_images": len(split_paths),
        "split_paths": split_paths,
        "bboxes": split_result.bboxes,
        "selected_image_index": 0 if split_paths else None  # Default to first image
    }

    try:
        session_manager.update_session(session_id, {
            "current_step": 2,
            "split_result": split_data
        })
    except Exception as e:
        logger.error(f"Error updating session: {e}")
        raise HTTPException(status_code=500, detail=f"Error updating session: {str(e)}")

    return CollageSplitResultResponse(
        num_images=len(split_paths),
        split_image_urls=split_urls,
        bboxes=[tuple(bbox) for bbox in split_result.bboxes]
    )


@router.get("/session/{session_id}/split", response_model=CollageSplitResultResponse)
def get_split_results(
    session_id: str,
    session_manager: SessionManager = Depends(get_session_manager)
):
    """Get split results for a session"""
    try:
        session = session_manager.get_session(session_id)

        if "split_result" not in session:
            raise HTTPException(status_code=404, detail="Split not performed yet")

        split_data = session["split_result"]

        # Generate URLs from paths
        split_urls = []
        for i in range(split_data["num_images"]):
            split_urls.append(f"/uploads/sessions/{session_id}/split_images/image_{i}.jpg")

        return CollageSplitResultResponse(
            num_images=split_data["num_images"],
            split_image_urls=split_urls,
            bboxes=[tuple(bbox) for bbox in split_data["bboxes"]]
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving split results: {e}")
        raise HTTPException(status_code=500, detail=str(e))
