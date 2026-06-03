"""
Pipeline Service Wrapper

Singleton wrapper for ProductImagePipeline to ensure models are loaded once.

Author: W Cheng
Co-Author: Claude (Anthropic AI Assistant)
Created: 2026-01-09
Project: Product Image Processing System
"""

import logging

from src.pipeline.product_image_pipeline import ProductImagePipeline

logger = logging.getLogger(__name__)


class PipelineService:
    """Singleton service for pipeline management"""

    _instance = None
    _pipeline = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @classmethod
    def get_pipeline(cls) -> ProductImagePipeline:
        """
        Get or create the pipeline instance (lazy loading)
        """
        if cls._pipeline is None:
            logger.info("Initializing ProductImagePipeline (first time - may take ~30s)")
            cls._pipeline = ProductImagePipeline(
                target_size=(512, 512),
                similarity_threshold=0.85,
                quality_threshold=0.3,
                use_gpu=False,  # Change to True if GPU available
                output_dir="uploads/processed"
            )
            # Log database type after initialization
            db_type = type(cls._pipeline.vector_db).__name__
            logger.info(f"Pipeline initialized successfully with {db_type}")
        return cls._pipeline

    @classmethod
    def reset_pipeline(cls):
        """Reset the pipeline (useful for testing)"""
        cls._pipeline = None
        logger.info("Pipeline reset")


# Convenience function for dependency injection
def get_pipeline() -> ProductImagePipeline:
    """Get the pipeline instance for use in routes"""
    return PipelineService.get_pipeline()
