#!/usr/bin/env python3
"""
Product Image Processing System - Main Program

Complete Pipeline: Image Input → Collage Detection → Cleaning → Quality Scoring → Embedding → Search/Grouping

Author: W Cheng
Co-Author: Claude (Anthropic AI Assistant)
Created: 2026-01-07
Project: Product Image Processing System

Usage:
    # Process single image
    python main.py --image path/to/image.jpg

    # Batch processing
    python main.py --batch path/to/images/

    # Start API service
    python main.py --serve --port 8000

    # Test mode
    python main.py --test
"""

import argparse
import json
import logging
import os

import numpy as np
import uvicorn
from PIL import Image

from src.logging_config import configure_logging
from src.pipeline.model_cache import get_model_cache
from src.pipeline.product_grouper import ProductGrouper
from src.pipeline.product_image_pipeline import ProductImagePipeline
from src.web.app import app

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

configure_logging()
logger = logging.getLogger(__name__)


def preload_models():
    """
    Preload all ML models using configuration from .env file.
    This ensures models are cached before the web server starts.
    """
    logger.info("Preloading models from .env configuration...")
    cache = get_model_cache()

    # Get CLIP model configuration from environment
    clip_model = os.getenv('CLIP_MODEL', 'ViT-B-32')
    clip_pretrained = os.getenv('CLIP_PRETRAINED', 'openai')

    logger.info(f"CLIP_MODEL={clip_model}, CLIP_PRETRAINED={clip_pretrained}")

    # Preload CLIP model (used by both embedding generator and aesthetic scorer)
    logger.info("Loading CLIP model...")
    cache.get_clip_model(clip_model, pretrained=clip_pretrained, device="auto")
    logger.info("CLIP model loaded and cached")

    # Preload rembg model
    logger.info("Loading rembg (U2Net) model...")
    try:
        cache.get_rembg_session("u2net", providers=("CPUExecutionProvider",))
        logger.info("rembg model loaded and cached")
    except ImportError:
        logger.warning("rembg not installed, skipping")

    logger.info(f"Preloaded models: {cache.get_cached_models()}")
    logger.info("Model preloading complete")


def run_test():
    """Run tests"""
    logger.info("Running pipeline test...")

    # Initialize pipeline first to trigger model downloads
    logger.info("Initializing pipeline and loading models...")
    pipeline = ProductImagePipeline(
        output_dir="/tmp/processed",
    )

    # Force load all lazy-loaded models (important for first run on GCP)
    print("\n=== Warming Up Models ===")

    # Create a small test image for warm-up
    warmup_img = Image.new('RGB', (100, 100), color=(200, 100, 50))

    logger.info("Loading rembg model (U2Net) via preprocessor...")
    _ = pipeline.preprocessor.standardize(warmup_img)  # Force rembg model download/load
    logger.info("Rembg model ready")

    logger.info("Loading quality scorer (LAION aesthetic model)...")
    _ = pipeline.quality_scorer.score(warmup_img)  # Force LAION aesthetic model download/load
    logger.info("Quality scorer ready")

    logger.info("Loading embedding model (CLIP)...")
    _ = pipeline.embedding_generator.generate(warmup_img)  # Force CLIP model download/load
    logger.info("Embedding model ready")

    logger.info("Loading vector database...")
    _ = pipeline.vector_db  # Force vector DB initialization
    logger.info("Vector database ready")

    logger.info("Loading product grouper...")
    _ = pipeline.product_grouper  # Force grouper initialization
    logger.info("Product grouper ready")

    print("=== All Models Loaded ===\n")

    # Create test image
    test_img = Image.new('RGB', (800, 600), color=(255, 255, 255))

    # Draw a simple rectangle on the image as a product
    img_array = np.array(test_img)
    img_array[100:500, 200:600] = [200, 100, 50]  # Brown rectangle
    test_img = Image.fromarray(img_array)

    # Save test image
    test_path = "/tmp/test_product.jpg"
    test_img.save(test_path)

    # Test each component
    print("\n=== Test Collage Detection ===")
    collage_info = pipeline.collage_detector.is_collage(test_img)
    print(f"Is collage: {collage_info.is_collage}")
    print(f"Num images: {collage_info.num_images}")

    print("\n=== Test Quality Scoring ===")
    quality = pipeline.quality_scorer.score(test_img)
    print(f"Overall score: {quality.overall:.3f}")
    print(f"Sharpness: {quality.sharpness:.1f}")
    print(f"Brightness: {quality.brightness:.1f}")

    print("\n=== Test Preprocessing ===")
    result = pipeline.preprocessor.standardize(test_img)
    print(f"Original size: {result.original_size}")
    print(f"Processed size: {result.image.size}")
    print(f"Background removed: {result.background_removed}")

    print("\n=== Test Complete Pipeline ===")
    processing_result = pipeline.process_image(
        test_path,
        id="TEST_ID_001",
        search_similar=False,
    )
    print(f"Success: {processing_result.success}")
    print(f"Quality score: {processing_result.quality_score:.3f}")
    print(f"Embedding shape: {processing_result.embedding.shape if processing_result.embedding is not None else None}")

    print("\n=== Test Vector Database ===")
    # Test 1: Insert multiple embeddings
    # Use actual dimension from embedding generator
    test_dim = pipeline.embedding_generator.dimension
    test_embeddings = {
        "ID_001": np.random.rand(test_dim),
        "ID_002": np.random.rand(test_dim),
        "ID_003": np.random.rand(test_dim),
    }

    print(f"Vector DB type: {type(pipeline.vector_db).__name__}")
    print(f"Initial DB size: {pipeline.vector_db.size()}")

    # Insert test embeddings
    for id, emb in test_embeddings.items():
        pipeline.vector_db.insert(id, emb)
        print(f"  Inserted {id}")

    print(f"After insert DB size: {pipeline.vector_db.size()}")

    # Test 2: Search for similar products
    print("\n=== Test Similarity Search ===")
    search_results = pipeline.vector_db.search(
        test_embeddings["ID_001"],
        top_k=3,
        threshold=0.0
    )
    print(f"Found {len(search_results)} similar products:")
    for result in search_results:
        print(f"  ID: {result.id}, Score: {result.score:.4f}")

    # Test 3: Retrieve specific ID
    print("\n=== Test ID Retrieval ===")
    retrieved = pipeline.vector_db.get("ID_002")
    if retrieved:
        print(f"Retrieved {retrieved.id}, embedding shape: {retrieved.embedding.shape}")
    else:
        print("ID not found")

    # Test 4: Duplicate detection via ProductGrouper
    print("\n=== Test Duplicate Detection ===")
    grouper = ProductGrouper(pipeline.vector_db, similarity_threshold=0.95)
    is_dup, match_id = grouper.is_duplicate(test_embeddings["ID_001"], threshold=0.95)
    print(f"Is duplicate: {is_dup}, Match: {match_id}")

    # Test 5: Cleanup
    print("\n=== Cleanup Test Data ===")
    for id in test_embeddings.keys():
        pipeline.vector_db.delete(id)
        print(f"  Deleted {id}")
    print(f"Final DB size: {pipeline.vector_db.size()}")

    print("\n=== Tests Completed ===")

    # Cleanup
    os.remove(test_path)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="Product Image Processing System"
    )

    parser.add_argument(
        "--image",
        type=str,
        help="Process single image",
    )
    parser.add_argument(
        "--batch",
        type=str,
        help="Batch process images in directory",
    )
    parser.add_argument(
        "--search-image",
        type=str,
        help="Search for similar products using image",
    )
    parser.add_argument(
        "--search-text",
        type=str,
        help="Search for products using text",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./processed",
        help="Output directory (default: ./processed)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.85,
        help="Similarity threshold (default: 0.85)",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU",
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run tests",
    )
    parser.add_argument(
        "--preload-model",
        action="store_true",
        help="Preload model before --serve",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Start API service",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="API service port (default: 8000)",
    )

    args = parser.parse_args()

    # Test mode
    if args.test:
        run_test()
        return

    # Initialize pipeline
    pipeline = ProductImagePipeline(
        similarity_threshold=args.threshold,
        use_gpu=args.gpu,
        output_dir=args.output,
    )

    # Process single image
    if args.image:
        result = pipeline.process_image(args.image)
        print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))

    # Batch processing
    elif args.batch:
        results = pipeline.process_batch(args.batch)
        output = [r.to_dict() for r in results]
        print(json.dumps(output, indent=2, ensure_ascii=False))

    # Image search
    elif args.search_image:
        results = pipeline.search_by_image(args.search_image)
        for r in results:
            print(f"ID: {r.id}, Score: {r.score:.3f}")

    # Text search
    elif args.search_text:
        results = pipeline.search_by_text(args.search_text)
        for r in results:
            print(f"ID: {r.id}, Score: {r.score:.3f}")

    # API service
    elif args.serve:
        if args.preload_model:
            preload_models()

        logger.info(f"Starting Web interface...")
        logger.info(f"Visit: http://localhost:{args.port}")
        uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
