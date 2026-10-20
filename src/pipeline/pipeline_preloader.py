#!/usr/bin/env python3
"""
Pipeline Preloader - Warm up models by simulating frontend single file upload

This script creates a dummy test image and runs it through the entire
frontend single file upload pipeline to preload all models into memory.

Author: W Cheng
Co-Author: Claude (Anthropic AI Assistant)
Created: 2026-01-13
Project: Product Image Processing System

Usage:
    python pipeline_preloader.py

    or

    source venv/bin/activate && python pipeline_preloader.py
"""

import logging
import sys
import tempfile
from pathlib import Path

import numpy as np
from PIL import Image

from .config import DEFAULT_TARGET_SIZE, PIPELINE_SIMILARITY_THRESHOLD
from .product_image_pipeline import ProductImagePipeline

logger = logging.getLogger(__name__)


def create_dummy_product_image() -> Image.Image:
    """
    Create a dummy product image for preloading

    Returns:
        PIL Image with a simple product-like shape
    """
    logger.info("Creating dummy product image...")

    img = Image.new('RGB', (800, 600), color=(255, 255, 255))

    img_array = np.array(img)
    img_array[150:450, 250:550] = [200, 100, 50]
    img_array[200:250, 300:500] = [180, 80, 30]
    img_array[400:420, 280:570] = [180, 80, 30]

    return Image.fromarray(img_array)


def simulate_frontend_upload(pipeline: ProductImagePipeline, test_image: Image.Image) -> dict:
    """
    Simulate the frontend single file upload process

    This follows the same workflow as web/routes/upload.py:
    1. Watermark removal (optional)
    2. Collage detection
    3. Quality scoring
    4. Preprocessing
    5. Embedding generation

    Args:
        pipeline: ProductImagePipeline instance
        test_image: Test image to process

    Returns:
        dict with processing results
    """
    logger.info("=== Simulating Frontend Single File Upload Process ===")

    result = {
        "success": False,
        "steps_completed": []
    }

    try:
        logger.info("\nStep 0: Watermark Removal")
        watermark_result = pipeline.watermark_remover.remove_watermark(test_image)
        watermarks_removed = watermark_result.watermarks_removed

        if watermarks_removed > 0:
            logger.info(f"Removed {watermarks_removed} watermarks")
            img_for_detection = watermark_result.image
        else:
            logger.info("No watermarks detected")
            img_for_detection = test_image

        result["steps_completed"].append("watermark_removal")
        result["watermarks_removed"] = watermarks_removed

        logger.info("\nStep 1: Collage Detection")
        collage_info = pipeline.collage_detector.is_collage(img_for_detection)
        logger.info(f"Is collage: {collage_info.is_collage}")
        logger.info(f"Num images: {collage_info.num_images}")
        logger.info(f"Layout: {collage_info.layout}")
        logger.info(f"Confidence: {collage_info.confidence:.3f}")

        result["steps_completed"].append("collage_detection")
        result["collage_info"] = {
            "is_collage": collage_info.is_collage,
            "num_images": collage_info.num_images,
            "layout": collage_info.layout,
            "confidence": collage_info.confidence
        }

        logger.info("\nStep 2: Aesthetic Quality Scoring (on raw image)")
        quality = pipeline.quality_scorer.score(test_image)
        logger.info(f"Overall quality score: {quality.overall:.3f}")
        logger.info(f"Sharpness: {quality.sharpness:.1f}")
        logger.info(f"Brightness: {quality.brightness:.1f}")
        logger.info(f"Contrast: {quality.contrast:.1f}")
        logger.info(f"Aesthetic score: {quality.aesthetic:.3f}")

        result["steps_completed"].append("quality_scoring")
        result["quality_score"] = quality.overall

        if collage_info.is_collage:
            logger.info("\nStep 3: Collage Split")
            split_result = pipeline.collage_detector.smart_split(img_for_detection, collage_info)
            images_to_process = split_result.images
            logger.info(f"Split into {len(images_to_process)} images")
            result["steps_completed"].append("collage_split")
            result["num_split_images"] = len(images_to_process)
        else:
            logger.info("\nStep 3: Collage Split (skipped - not a collage)")
            images_to_process = [img_for_detection]
            result["num_split_images"] = 1

        logger.info("\nStep 4: Image Preprocessing (background removal, standardization)")
        if len(images_to_process) > 1:
            best_idx, processed_img, _ = pipeline.image_selector.select_best(
                images_to_process,
                return_all_scores=False
            )
            logger.info(f"Selected best image: index {best_idx}")
            result["best_image_index"] = best_idx
        else:
            preprocess_result = pipeline.preprocessor.standardize(images_to_process[0])
            processed_img = preprocess_result.image
            logger.info(f"Preprocessed image size: {processed_img.size}")
            logger.info(f"Background removed: {preprocess_result.background_removed}")

        result["steps_completed"].append("preprocessing")
        result["processed_size"] = processed_img.size

        logger.info("\nStep 5: Embedding Generation (CLIP)")
        embedding_result = pipeline.embedding_generator.generate(processed_img)
        embedding = embedding_result.embedding
        logger.info(f"Generated embedding: dimension={len(embedding)}")
        logger.info(f"Model: {embedding_result.model_name}")

        result["steps_completed"].append("embedding_generation")
        result["embedding_dimension"] = len(embedding)
        result["embedding_model"] = embedding_result.model_name

        logger.info("\nStep 6: Vector Database Check")
        db_type = type(pipeline.vector_db).__name__
        db_size = pipeline.vector_db.size()
        logger.info(f"Database type: {db_type}")
        logger.info(f"Database size: {db_size} embeddings")

        result["steps_completed"].append("vector_db_check")
        result["database_type"] = db_type
        result["database_size"] = db_size

        if db_size > 0:
            logger.info("\nStep 7: Similar Product Search")
            similar = pipeline.product_grouper.find_similar_products(
                embedding,
                top_k=5
            )
            logger.info(f"Found {len(similar)} similar products")
            result["steps_completed"].append("similarity_search")
            result["similar_products_found"] = len(similar)
        else:
            logger.info("\nStep 7: Similar Product Search (skipped - empty database)")

        result["success"] = True
        logger.info("\n=== Frontend Upload Simulation Complete ===")

    except Exception as e:
        logger.error(f"Error during simulation: {e}", exc_info=True)
        result["error"] = str(e)

    return result


def preload_pipeline():
    """
    Main preloading function

    Creates a dummy image and runs it through the frontend upload pipeline
    to preload all models into memory.
    """
    logger.info("="*60)
    logger.info("PIPELINE PRELOADER - Frontend Upload Simulation")
    logger.info("="*60)

    logger.info("\n[1/3] Initializing ProductImagePipeline...")
    pipeline = ProductImagePipeline(
        target_size=DEFAULT_TARGET_SIZE,
        similarity_threshold=PIPELINE_SIMILARITY_THRESHOLD,
        quality_threshold=0.3,
        use_gpu=False,
        output_dir=None
    )
    logger.info("Pipeline initialized")

    logger.info("\n[2/3] Creating dummy product image...")
    test_image = create_dummy_product_image()

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        test_image.save(tmp.name, quality=95)
        test_path = tmp.name
        logger.info(f"Saved test image to: {test_path}")

    logger.info("\n[3/3] Running frontend upload simulation...")
    result = simulate_frontend_upload(pipeline, test_image)

    Path(test_path).unlink()
    logger.info("Cleaned up test image")

    if result["success"]:
        logger.info("\n" + "="*60)
        logger.info("PRELOAD SUCCESSFUL")
        logger.info("="*60)
        logger.info(f"Steps completed: {', '.join(result['steps_completed'])}")
        logger.info(f"Embedding model: {result.get('embedding_model', 'N/A')}")
        logger.info(f"Embedding dimension: {result.get('embedding_dimension', 'N/A')}")
        logger.info(f"Database: {result.get('database_type', 'N/A')} ({result.get('database_size', 0)} embeddings)")
        logger.info("="*60)
        logger.info("\nAll models are now loaded and ready for frontend requests!")
        return 0
    else:
        logger.error("\n" + "="*60)
        logger.error("PRELOAD FAILED")
        logger.error("="*60)
        logger.error(f"Error: {result.get('error', 'Unknown error')}")
        logger.error("="*60)
        return 1


if __name__ == "__main__":
    exit_code = preload_pipeline()
    sys.exit(exit_code)
