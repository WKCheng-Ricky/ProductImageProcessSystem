"""
Embedding Route - Page 5: Embedding Generation

Author: W Cheng
Co-Author: Claude (Anthropic AI Assistant)
Created: 2026-01-09
Project: Product Image Processing System
"""

from fastapi import APIRouter, HTTPException, Depends, Request
from PIL import Image
import numpy as np
import time
import logging
import uuid
from pathlib import Path
from typing import Optional

from ..models.schemas import EmbeddingResultResponse
from ..services.session_manager import SessionManager
from ..services.pipeline_service import get_pipeline

logger = logging.getLogger(__name__)
router = APIRouter()


def get_session_manager(request: Request) -> SessionManager:
    return request.app.state.session_manager


@router.post("/session/{session_id}/embedding", response_model=EmbeddingResultResponse)
def generate_embedding(
    session_id: str,
    id: Optional[str] = None,
    session_manager: SessionManager = Depends(get_session_manager)
):
    """
    Generate CLIP embedding for processed image

    This is Page 5 of the pipeline:
    - Generates embedding using CLIP model
    - Saves embedding to file
    - Optionally inserts into vector database
    """
    # Get session
    try:
        session = session_manager.get_session(session_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    # Use id from parameter, fallback to session
    if id is None:
        id = session.get("id")

    # if id is not provided by user
    if id is None:
        id = str(uuid.uuid4())

    # Get preprocessed image (if available), otherwise use best available
    if "preprocessing_result" in session:
        image_path = Path(session["preprocessing_result"]["processed_path"])
    elif "split_result" in session:
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
        logger.info(f"Generating embedding for: {image_path}")
    except Exception as e:
        logger.error(f"Error loading image: {e}")
        raise HTTPException(status_code=500, detail=f"Error loading image: {str(e)}")

    # Generate embedding (may take time on first request)
    try:
        pipeline = get_pipeline()

        # Property access auto-initializes (lazy loading)
        start_time = time.time()
        embedding_result = pipeline.embedding_generator.generate(img)
        generation_time = time.time() - start_time

        logger.info(f"Embedding generated in {generation_time:.2f}s, model: {embedding_result.model_name}, dimension: {embedding_result.dimension}")
    except Exception as e:
        logger.error(f"Error generating embedding: {e}")
        raise HTTPException(status_code=500, detail=f"Error generating embedding: {str(e)}")

    # Save embedding
    session_path = session_manager.get_session_path(session_id)
    embedding_path = session_path / "embedding.npy"

    try:
        np.save(embedding_path, embedding_result.embedding)
        logger.info(f"Saved embedding to {embedding_path}")
    except Exception as e:
        logger.error(f"Error saving embedding: {e}")
        raise HTTPException(status_code=500, detail=f"Error saving embedding: {str(e)}")

    # Check for duplicate before inserting
    is_duplicate = False
    duplicate_of = None
    duplicate_score = None
    DUPLICATE_THRESHOLD = 0.999

    try:
        existing_results = pipeline.vector_db.search(
            embedding=embedding_result.embedding,
            top_k=1,
            threshold=DUPLICATE_THRESHOLD
        )
        if existing_results and existing_results[0].score >= DUPLICATE_THRESHOLD:
            is_duplicate = True
            duplicate_of = existing_results[0].id
            duplicate_score = existing_results[0].score
            logger.info(f"Duplicate detected: {duplicate_score:.4f} match with {duplicate_of}")
    except Exception as e:
        logger.warning(f"Failed to check for duplicates: {e}")

    # Optionally insert into vector database (skip if duplicate)
    saved_to_db = False
    if id and not is_duplicate:
        try:
            # Property access auto-initializes (lazy loading)
            image_url = f"/uploads/sessions/{session_id}/preprocessed.jpg"

            # Get original image URL
            original_path = Path(session["original_path"])
            original_image_url = f"/uploads/sessions/{session_id}/original{original_path.suffix}"

            # Build metadata with quality scores if available
            metadata = {"session_id": session_id, "image_url": image_url, "original_image_url": original_image_url}
            quality_score = session.get("quality_score")
            if quality_score:
                metadata["aesthetic"] = quality_score.get("aesthetic")
                metadata["overall"] = quality_score.get("overall")

            pipeline.vector_db.insert(
                id=id,
                embedding=embedding_result.embedding,
                metadata=metadata
            )
            saved_to_db = True
            db_type = type(pipeline.vector_db).__name__
            db_size = pipeline.vector_db.size()
            logger.info(f"Inserted embedding for ID: {id} into {db_type}, dimension: {embedding_result.dimension}, total embeddings: {db_size}")
        except Exception as e:
            logger.warning(f"Failed to insert into vector DB: {e}")
            # Don't fail the request if DB insertion fails
    elif is_duplicate:
        logger.info(f"Skipped insertion for duplicate image (matches {duplicate_of})")

    # Update session
    embedding_data = {
        "model_name": embedding_result.model_name,
        "dimension": embedding_result.dimension,
        "normalized": embedding_result.normalized,
        "generation_time": generation_time,
        "embedding_path": str(embedding_path),
        "id": id,
        "saved_to_db": saved_to_db,
        "is_duplicate": is_duplicate,
        "duplicate_of": duplicate_of,
        "duplicate_score": duplicate_score
    }

    try:
        session_manager.update_session(session_id, {
            "current_step": 5,
            "embedding_result": embedding_data
        })
    except Exception as e:
        logger.error(f"Error updating session: {e}")
        raise HTTPException(status_code=500, detail=f"Error updating session: {str(e)}")

    return EmbeddingResultResponse(
        model_name=embedding_result.model_name,
        dimension=embedding_result.dimension,
        normalized=embedding_result.normalized,
        generation_time=generation_time,
        id=id,
        saved_to_db=saved_to_db,
        is_duplicate=is_duplicate,
        duplicate_of=duplicate_of,
        duplicate_score=duplicate_score
    )


@router.get("/session/{session_id}/embedding", response_model=EmbeddingResultResponse)
def get_embedding_info(
    session_id: str,
    session_manager: SessionManager = Depends(get_session_manager)
):
    """Get embedding info for a session"""
    try:
        session = session_manager.get_session(session_id)

        if "embedding_result" not in session:
            raise HTTPException(status_code=404, detail="Embedding not generated yet")

        emb_data = session["embedding_result"]

        return EmbeddingResultResponse(
            model_name=emb_data["model_name"],
            dimension=emb_data["dimension"],
            normalized=emb_data["normalized"],
            generation_time=emb_data["generation_time"],
            id=emb_data.get("id"),
            saved_to_db=emb_data.get("saved_to_db", False),
            is_duplicate=emb_data.get("is_duplicate", False),
            duplicate_of=emb_data.get("duplicate_of"),
            duplicate_score=emb_data.get("duplicate_score")
        )

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving embedding info: {e}")
        raise HTTPException(status_code=500, detail=str(e))
