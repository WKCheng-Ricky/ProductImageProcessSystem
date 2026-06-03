"""
Search Route - Page 6: Similar Product Search

Author: W Cheng
Co-Author: Claude (Anthropic AI Assistant)
Created: 2026-01-09
Project: Product Image Processing System
"""

import logging
from pathlib import Path

import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Request

from src.pipeline.embedding_generator import TextImageEmbeddingGenerator
from ..models.schemas import SearchResultItem, SearchResultsResponse, TextSearchRequest, TextSearchResultsResponse
from ..services.pipeline_service import get_pipeline
from ..services.session_manager import SessionManager

logger = logging.getLogger(__name__)
router = APIRouter()


def get_session_manager(request: Request) -> SessionManager:
    """Dependency for session manager"""
    return request.app.state.session_manager


@router.post("/session/{session_id}/search", response_model=SearchResultsResponse)
async def search_similar_products(
    session_id: str,
    top_k: int = 10,
    threshold: float = 0.70,
    session_manager: SessionManager = Depends(get_session_manager)
):
    """
    Search for similar products

    This is Page 6 of the pipeline:
    - Searches vector database for similar embeddings
    - Returns top-k most similar products
    - Filters by similarity threshold
    """
    # Validate parameters
    if top_k < 1 or top_k > 100:
        raise HTTPException(status_code=400, detail="top_k must be between 1 and 100")
    if threshold < 0 or threshold > 1:
        raise HTTPException(status_code=400, detail="threshold must be between 0 and 1")

    # Get session
    try:
        session = session_manager.get_session(session_id)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    # Check if embedding exists
    if "embedding_result" not in session:
        raise HTTPException(status_code=400, detail="Embedding not generated yet")

    # Load embedding
    embedding_path = Path(session["embedding_result"]["embedding_path"])
    if not embedding_path.exists():
        raise HTTPException(status_code=404, detail="Embedding file not found")

    try:
        embedding = np.load(embedding_path)
        logger.info(f"Loaded embedding from {embedding_path}, shape: {embedding.shape}")
    except Exception as e:
        logger.error(f"Error loading embedding: {e}")
        raise HTTPException(status_code=500, detail=f"Error loading embedding: {str(e)}")

    # Search for similar products
    try:
        pipeline = get_pipeline()

        # Property access auto-initializes (lazy loading)
        db_size = pipeline.vector_db.size()
        db_type = type(pipeline.vector_db).__name__
        logger.info(f"Searching for similar products: query_dimension={embedding.shape[0]}, top_k={top_k}, threshold={threshold}, db_size={db_size}, db_type={db_type}")

        # Perform search
        search_results = pipeline.product_grouper.find_similar_products(
            embedding=embedding,
            top_k=top_k,
            threshold=threshold
        )

        logger.info(f"Found {len(search_results)} similar products")
        for i, result in enumerate(search_results[:5], 1):  # Log top 5
            logger.info(f"  {i}. ID: {result.id}, Score: {result.score:.4f}")
        if len(search_results) == 0:
            logger.warning(f"No similar products found (database has {db_size} embeddings)")
    except Exception as e:
        logger.error(f"Error searching for similar products: {e}")
        raise HTTPException(status_code=500, detail=f"Error searching: {str(e)}")

    # Convert results to response format
    result_items = []
    for result in search_results:
        # Try to get image URLs if available in metadata
        image_url = None
        original_image_url = None
        if result.metadata:
            if "image_url" in result.metadata:
                image_url = result.metadata["image_url"]
            if "original_image_url" in result.metadata:
                original_image_url = result.metadata["original_image_url"]

        result_items.append(SearchResultItem(
            id=result.id,
            score=float(result.score),
            image_url=image_url,
            original_image_url=original_image_url,
            metadata=result.metadata
        ))

    # Get query image URL
    if "preprocessing_result" in session:
        query_image_url = f"/uploads/sessions/{session_id}/preprocessed.jpg"
    elif "split_result" in session and session["split_result"]["num_images"] > 0:
        selected_index = session.get("selected_image_index", 0)
        query_image_url = f"/uploads/sessions/{session_id}/split_images/image_{selected_index}.jpg"
    else:
        # Get original image extension
        original_path = Path(session["original_path"])
        query_image_url = f"/uploads/sessions/{session_id}/original{original_path.suffix}"

    # Update session
    search_data = {
        "num_results": len(result_items),
        "top_k": top_k,
        "threshold": threshold,
        "results": [
            {
                "id": item.id,
                "score": item.score,
                "confidence_level": item.confidence_level
            }
            for item in result_items
        ]
    }

    try:
        session_manager.update_session(session_id, {
            "current_step": 6,
            "search_results": search_data
        })
    except Exception as e:
        logger.error(f"Error updating session: {e}")
        raise HTTPException(status_code=500, detail=f"Error updating session: {str(e)}")

    return SearchResultsResponse(
        results=result_items,
        total_count=len(result_items),
        query_image_url=query_image_url
    )


@router.get("/search/by-image/{image_id}", response_model=SearchResultsResponse)
async def search_by_image_id(
    image_id: str,
    top_k: int = 10,
    threshold: float = 0.70
):
    """
    Search for similar products by image ID.

    This endpoint retrieves the embedding for the specified image from the database
    and searches for similar products.
    """
    if top_k < 1 or top_k > 100:
        raise HTTPException(status_code=400, detail="top_k must be between 1 and 100")
    if threshold < 0 or threshold > 1:
        raise HTTPException(status_code=400, detail="threshold must be between 0 and 1")

    try:
        pipeline = get_pipeline()

        # Get the embedding for this image from the vector database
        query_result = pipeline.vector_db.get(image_id)
        if query_result is None:
            raise HTTPException(status_code=404, detail=f"Image {image_id} not found in database")

        embedding = query_result.embedding
        query_image_url = query_result.metadata.get("image_url") if query_result.metadata else None
        logger.info(f"Loaded embedding for image {image_id}, shape: {embedding.shape}")

        # Search for similar products (excluding the query image itself)
        search_results = pipeline.product_grouper.find_similar_products(
            embedding=embedding,
            top_k=top_k + 1,  # Get one extra to exclude self
            threshold=threshold
        )

        # Filter out the query image from results
        search_results = [r for r in search_results if r.id != image_id][:top_k]

        logger.info(f"Found {len(search_results)} similar products for image {image_id}")

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error searching for similar products: {e}")
        raise HTTPException(status_code=500, detail=f"Error searching: {str(e)}")

    # Convert results to response format
    result_items = []
    for result in search_results:
        image_url = None
        original_image_url = None
        if result.metadata:
            if "image_url" in result.metadata:
                image_url = result.metadata["image_url"]
            if "original_image_url" in result.metadata:
                original_image_url = result.metadata["original_image_url"]

        result_items.append(SearchResultItem(
            id=result.id,
            score=float(result.score),
            image_url=image_url,
            original_image_url=original_image_url,
            metadata=result.metadata
        ))

    return SearchResultsResponse(
        results=result_items,
        total_count=len(result_items),
        query_image_url=query_image_url
    )


@router.get("/session/{session_id}/search", response_model=SearchResultsResponse)
async def get_search_results(
    session_id: str,
    session_manager: SessionManager = Depends(get_session_manager)
):
    """Get search results for a session"""
    try:
        session = session_manager.get_session(session_id)

        if "search_results" not in session:
            raise HTTPException(status_code=404, detail="Search not performed yet")

        search_data = session["search_results"]

        # Reconstruct result items
        result_items = [
            SearchResultItem(
                id=r["id"],
                score=r["score"],
                image_url=None,
                metadata=None
            )
            for r in search_data["results"]
        ]

        # Get query image URL
        if "preprocessing_result" in session:
            query_image_url = f"/uploads/sessions/{session_id}/preprocessed.jpg"
        elif "split_result" in session and session["split_result"]["num_images"] > 0:
            selected_index = session.get("selected_image_index", 0)
            query_image_url = f"/uploads/sessions/{session_id}/split_images/image_{selected_index}.jpg"
        else:
            original_path = Path(session["original_path"])
            query_image_url = f"/uploads/sessions/{session_id}/original{original_path.suffix}"

        return SearchResultsResponse(
            results=result_items,
            total_count=search_data["num_results"],
            query_image_url=query_image_url
        )

    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error retrieving search results: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/search/by-text", response_model=TextSearchResultsResponse)
async def search_by_text(request: TextSearchRequest):
    """
    Search for products by text description.

    Uses CLIP's text encoder to generate an embedding from the text query,
    then searches the vector database for visually similar images.
    """
    try:
        pipeline = get_pipeline()

        # Generate text embedding using CLIP
        text_generator = TextImageEmbeddingGenerator(
            model_name=pipeline.embedding_generator.model_name,
            device=pipeline.embedding_generator.device,
            normalize=True
        )

        logger.info(f"Generating text embedding for query: '{request.query}'")
        text_embedding_result = text_generator.generate_text_embedding(request.query)
        text_embedding = text_embedding_result.embedding

        logger.info(f"Text embedding generated, shape: {text_embedding.shape}")

        # Search vector database
        db_size = pipeline.vector_db.size()
        logger.info(f"Searching for products: query='{request.query}', top_k={request.top_k}, threshold={request.threshold}, db_size={db_size}")

        search_results = pipeline.product_grouper.find_similar_products(
            embedding=text_embedding,
            top_k=request.top_k,
            threshold=request.threshold
        )

        logger.info(f"Found {len(search_results)} products matching text query")
        for i, result in enumerate(search_results[:5], 1):
            logger.info(f"  {i}. ID: {result.id}, Score: {result.score:.4f}")

    except ImportError as e:
        logger.error(f"Import error: {e}")
        raise HTTPException(status_code=500, detail="Text search not available: embedding generator not found")
    except Exception as e:
        logger.error(f"Error in text search: {e}")
        raise HTTPException(status_code=500, detail=f"Error searching: {str(e)}")

    # Convert results to response format
    result_items = []
    for result in search_results:
        image_url = None
        original_image_url = None
        if result.metadata:
            if "image_url" in result.metadata:
                image_url = result.metadata["image_url"]
            if "original_image_url" in result.metadata:
                original_image_url = result.metadata["original_image_url"]

        result_items.append(SearchResultItem(
            id=result.id,
            score=float(result.score),
            image_url=image_url,
            original_image_url=original_image_url,
            metadata=result.metadata
        ))

    return TextSearchResultsResponse(
        results=result_items,
        total_count=len(result_items),
        query_text=request.query
    )
