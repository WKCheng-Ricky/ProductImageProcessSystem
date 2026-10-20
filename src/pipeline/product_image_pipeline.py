from dataclasses import asdict, dataclass
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PIL import Image

from .best_image_selector import BestImageSelector
from .collage_detector import CollageDetector
from .config import CONFIG, DEFAULT_TARGET_SIZE, PIPELINE_SIMILARITY_THRESHOLD, QUALITY_THRESHOLD
from .embedding_generator import EmbeddingGenerator, TextImageEmbeddingGenerator
from .image_preprocessor import ImagePreprocessor
# product_grouper must load before milvus_db (circular import)
from .product_grouper import InMemoryVectorDB, ProductGroup, ProductGrouper, SearchResult, VectorDatabase
from .milvus_db import MilvusVectorLiteDB
from .quality_scorer import ImageQualityScorer
from .watermark_remover import WatermarkRemover

logger = logging.getLogger(__name__)


@dataclass
class ProcessingResult:
    """Single image processing result"""
    success: bool
    image_path: Optional[str]
    processed_image_path: Optional[str]
    quality_score: Optional[float]
    embedding: Optional[np.ndarray]
    is_collage: bool
    num_split_images: int
    similar_products: List[Dict[str, Any]]
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (for JSON serialization)"""
        result = asdict(self)
        # Convert numpy array to list
        if result['embedding'] is not None:
            result['embedding'] = result['embedding'].tolist()
        return result


class ProductImagePipeline:
    """Product Image Processing Pipeline"""

    def __init__(
        self,
        target_size: tuple = DEFAULT_TARGET_SIZE,
        similarity_threshold: float = PIPELINE_SIMILARITY_THRESHOLD,
        quality_threshold: float = QUALITY_THRESHOLD,
        use_gpu: bool = False,
        output_dir: Optional[str] = None,
    ):
        """
        Initialize Pipeline

        Args:
            target_size: Target image size
            similarity_threshold: Similarity search threshold
            quality_threshold: Quality score threshold
            use_gpu: Whether to use GPU
            output_dir: Output directory
        """
        self.target_size = target_size
        self.similarity_threshold = similarity_threshold
        self.quality_threshold = quality_threshold
        self.output_dir = output_dir

        logger.info("Initializing pipeline components...")

        # Initialize all modules
        self.watermark_remover = WatermarkRemover(use_ocr=False)  # Use simple method first to avoid OCR dependency
        self.collage_detector = CollageDetector()
        self.quality_scorer = ImageQualityScorer()
        self.preprocessor = ImagePreprocessor(
            target_size=target_size,
            use_gpu=use_gpu,
        )
        self.image_selector = BestImageSelector(
            quality_scorer=self.quality_scorer,
            preprocessor=self.preprocessor,
        )

        # Embedding and vector database (lazy loading)
        self._embedding_generator = None
        self._vector_db = None
        self._product_grouper = None

        logger.info("Pipeline initialized successfully")

    @property
    def embedding_generator(self) -> EmbeddingGenerator:
        """Lazy load Embedding generator"""
        if self._embedding_generator is None:
            logger.info("Loading embedding model...")
            self._embedding_generator = TextImageEmbeddingGenerator()
        return self._embedding_generator

    @property
    def vector_db(self) -> VectorDatabase:
        """Lazy load vector database"""
        if self._vector_db is None:
            if CONFIG["use_milvus"]:
                try:
                    # Get actual dimension and model name from embedding model
                    actual_dimension = self.embedding_generator.dimension
                    model_name = self.embedding_generator.model_name
                    logger.info(f"Detected embedding model: {model_name}, dimension: {actual_dimension}")

                    logger.info(f"Initializing Milvus Lite: db_path={CONFIG['milvus_lite_path']}, collection={CONFIG['milvus_collection']}")
                    self._vector_db = MilvusVectorLiteDB(
                        collection_name=CONFIG["milvus_collection"],
                        db_path=CONFIG["milvus_lite_path"],
                        dimension=actual_dimension,
                        model_name=model_name,
                    )
                    logger.info("Milvus Lite vector database initialized successfully")
                except Exception as e:
                    logger.error(
                        f"USE_MILVUS=true but Milvus failed to initialize ({e}); "
                        "falling back to InMemoryVectorDB - embeddings will NOT be persisted",
                        exc_info=True,
                    )
                    self._vector_db = InMemoryVectorDB()
            else:
                logger.info("Using InMemoryVectorDB (USE_MILVUS=false)")
                self._vector_db = InMemoryVectorDB()
        return self._vector_db

    @property
    def product_grouper(self) -> ProductGrouper:
        """Lazy load product grouper"""
        if self._product_grouper is None:
            self._product_grouper = ProductGrouper(
                vector_db=self.vector_db,
                similarity_threshold=self.similarity_threshold,
            )
        return self._product_grouper

    @staticmethod
    def _load_rgb_image(image_path: str) -> Image.Image:
        img = Image.open(image_path)
        return img if img.mode == 'RGB' else img.convert('RGB')

    def _remove_watermark(self, img: Image.Image) -> Image.Image:
        """Runs before collage detection so watermarks do not look like cell borders."""
        watermark_result = self.watermark_remover.remove_watermark(img)
        if watermark_result.watermarks_removed > 0:
            logger.info(f"Removed {watermark_result.watermarks_removed} watermarks before collage detection")
            return watermark_result.image
        logger.info("No watermarks detected")
        return img

    def _detect_and_split_collage(self, img: Image.Image):
        collage_info = self.collage_detector.is_collage(img)
        logger.info(f"Collage detection: is_collage={collage_info.is_collage}, num={collage_info.num_images}")
        if not collage_info.is_collage:
            return collage_info, [img]
        return collage_info, self.collage_detector.smart_split(img, collage_info).images

    def _clean_and_select_best(self, images: List[Image.Image]) -> Image.Image:
        if len(images) == 1:
            return self.preprocessor.standardize(images[0]).image
        best_idx, processed_img, _ = self.image_selector.select_best(images, return_all_scores=False)
        logger.info(f"Selected best image: index {best_idx}")
        return processed_img

    def _score_quality(self, processed_img: Image.Image):
        quality = self.quality_scorer.score(processed_img)
        logger.info(f"Quality score: {quality.overall:.3f}")
        if quality.overall < self.quality_threshold:
            logger.warning(f"Image quality below threshold: {quality.overall:.3f}")
        return quality

    def _save_processed_image(self, processed_img: Image.Image, image_path: str) -> str:
        os.makedirs(self.output_dir, exist_ok=True)
        processed_image_path = os.path.join(self.output_dir, Path(image_path).stem + "_processed.jpg")
        processed_img.save(processed_image_path, quality=95)
        logger.info(f"Saved processed image: {processed_image_path}")
        return processed_image_path

    def _find_similar(self, embedding: np.ndarray) -> List[Dict[str, Any]]:
        if self.vector_db.size() == 0:
            return []
        similar = self.product_grouper.find_similar_products(embedding, top_k=10)
        logger.info(f"Found {len(similar)} similar products")
        return [{"id": r.id, "score": r.score} for r in similar]

    def _store_embedding(self, id: str, embedding: np.ndarray, image_path: str, quality_score: float) -> None:
        self.vector_db.insert(
            id=id,
            embedding=embedding,
            metadata={"image_path": image_path, "quality_score": quality_score},
        )
        logger.info(f"Inserted ID {id} into database")

    def process_image(
        self,
        image_path: str,
        id: Optional[str] = None,
        save_processed: bool = True,
        search_similar: bool = True,
    ) -> ProcessingResult:
        """
        Process single image

        Args:
            image_path: Image file path
            id: ID (for storing in database)
            save_processed: Whether to save processed image
            search_similar: Whether to search for similar products

        Returns:
            ProcessingResult - processing result
        """
        try:
            logger.info(f"Processing image: {image_path}")

            img = self._load_rgb_image(image_path)
            img = self._remove_watermark(img)
            collage_info, images_to_process = self._detect_and_split_collage(img)
            processed_img = self._clean_and_select_best(images_to_process)
            quality = self._score_quality(processed_img)

            processed_image_path = None
            if save_processed and self.output_dir:
                processed_image_path = self._save_processed_image(processed_img, image_path)

            embedding = self.embedding_generator.generate(processed_img).embedding
            logger.info(f"Generated embedding: dim={len(embedding)}")

            similar_products = self._find_similar(embedding) if search_similar else []

            if id:
                self._store_embedding(id, embedding, image_path, quality.overall)

            return ProcessingResult(
                success=True,
                image_path=image_path,
                processed_image_path=processed_image_path,
                quality_score=quality.overall,
                embedding=embedding,
                is_collage=collage_info.is_collage,
                num_split_images=len(images_to_process),
                similar_products=similar_products,
            )

        except Exception as e:
            logger.error(f"Error processing image {image_path}: {e}")
            return ProcessingResult(
                success=False,
                image_path=image_path,
                processed_image_path=None,
                quality_score=None,
                embedding=None,
                is_collage=False,
                num_split_images=0,
                similar_products=[],
                error_message=str(e),
            )

    def process_batch(
        self,
        image_dir: str,
        extensions: tuple = ('.jpg', '.jpeg', '.png', '.webp'),
        save_processed: bool = True,
    ) -> List[ProcessingResult]:
        """
        Batch process images in directory

        Args:
            image_dir: Image directory
            extensions: Supported file extensions
            save_processed: Whether to save processed images

        Returns:
            List of ProcessingResult
        """
        image_paths = []
        for ext in extensions:
            image_paths.extend(Path(image_dir).glob(f"*{ext}"))
            image_paths.extend(Path(image_dir).glob(f"*{ext.upper()}"))

        logger.info(f"Found {len(image_paths)} images in {image_dir}")

        results = []
        for i, image_path in enumerate(image_paths):
            logger.info(f"Processing {i+1}/{len(image_paths)}: {image_path}")

            # Use filename as ID
            id = image_path.stem

            result = self.process_image(
                str(image_path),
                id=id,
                save_processed=save_processed,
                search_similar=True,
            )
            results.append(result)

        # Statistics
        success_count = sum(1 for r in results if r.success)
        logger.info(f"Batch processing completed: {success_count}/{len(results)} successful")

        return results

    def search_by_image(
        self,
        image_path: str,
        top_k: int = 10,
    ) -> List[SearchResult]:
        """
        Search for similar products using image

        Args:
            image_path: Query image path
            top_k: Number of results to return

        Returns:
            List of SearchResult
        """
        # Load and process image
        img = Image.open(image_path)
        preprocess_result = self.preprocessor.standardize(img)

        # Generate embedding
        embedding_result = self.embedding_generator.generate(preprocess_result.image)

        # Search
        results = self.product_grouper.find_similar_products(
            embedding_result.embedding,
            top_k=top_k,
        )

        return results

    def search_by_text(
        self,
        query_text: str,
        top_k: int = 10,
    ) -> List[SearchResult]:
        """
        Search for products using text

        Args:
            query_text: Search text query
            top_k: Number of results to return

        Returns:
            List of SearchResult
        """
        # Generate text embedding
        text_embedding = self.embedding_generator.generate_text_embedding(query_text)

        # Search
        results = self.product_grouper.find_similar_products(
            text_embedding.embedding,
            top_k=top_k,
        )

        return results

    def group_products(self) -> List[ProductGroup]:
        """
        Group all products in database

        Returns:
            List of ProductGroup
        """
        return self.product_grouper.group_skus_clustering()

    def select_best_from_sku(
        self,
        image_paths: List[str],
    ) -> tuple:
        """
        Select best main image from multiple images of the same SKU

        Args:
            image_paths: List of image paths

        Returns:
            (best index, processed image, all candidate scores)
        """
        images = [Image.open(p) for p in image_paths]

        best_idx, best_img, candidates = self.image_selector.select_best(
            images,
            return_all_scores=True,
        )

        return best_idx, best_img, candidates