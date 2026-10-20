"""
Best Main Image Selection Module

Select the most suitable main image from multiple images of the same SKU.
"""

import numpy as np
from PIL import Image
from typing import List, Tuple, Optional
from dataclasses import dataclass
import logging

from .quality_scorer import ImageQualityScorer, QualityScore
from .image_preprocessor import ImagePreprocessor, PreprocessingResult

logger = logging.getLogger(__name__)


@dataclass
class ImageCandidate:
    """Image candidate"""
    index: int                          # Original index
    original_image: Image.Image         # Original image
    processed_image: Image.Image        # Processed image
    quality_score: QualityScore         # Quality score
    visibility_score: float             # Product visibility score
    final_score: float                  # Final composite score


class BestImageSelector:
    """Best image selector"""

    def __init__(
        self,
        quality_scorer: Optional[ImageQualityScorer] = None,
        preprocessor: Optional[ImagePreprocessor] = None,
        quality_weight: float = 0.5,
        visibility_weight: float = 0.3,
        composition_weight: float = 0.2,
        min_quality_threshold: float = 0.2,
    ):
        """
        Initialize selector

        Args:
            quality_scorer: Quality scorer
            preprocessor: Preprocessor
            quality_weight: Quality score weight
            visibility_weight: Visibility weight
            composition_weight: Composition weight
            min_quality_threshold: Minimum quality threshold
        """
        self.scorer = quality_scorer or ImageQualityScorer()
        self.preprocessor = preprocessor or ImagePreprocessor()

        self.weights = {
            'quality': quality_weight,
            'visibility': visibility_weight,
            'composition': composition_weight,
        }
        self.min_quality_threshold = min_quality_threshold

    def calculate_visibility_score(
        self,
        img: Image.Image,
        mask: Optional[np.ndarray] = None
    ) -> float:
        """
        Calculate product visibility score

        Considers:
        - Is product fully visible (not cut off)
        - Is product ratio appropriate
        - Is product centered

        Args:
            img: Image
            mask: Product mask

        Returns:
            Visibility score from 0-1
        """
        width, height = img.size

        if mask is not None:
            # Calculate product area using mask
            product_pixels = np.sum(mask > 127)
            total_pixels = mask.shape[0] * mask.shape[1]
            product_ratio = product_pixels / total_pixels

            # Calculate product center of mass
            coords = np.column_stack(np.where(mask > 127))
            if len(coords) > 0:
                center_y, center_x = coords.mean(axis=0)
                center_x_ratio = center_x / mask.shape[1]
                center_y_ratio = center_y / mask.shape[0]
            else:
                center_x_ratio, center_y_ratio = 0.5, 0.5
        else:
            # No mask, use simplified calculation
            product_ratio = 0.5
            center_x_ratio, center_y_ratio = 0.5, 0.5

        # Scoring factors

        # 1. Product ratio score (Ideal 30%-70%)
        if 0.3 <= product_ratio <= 0.7:
            ratio_score = 1.0
        elif product_ratio < 0.3:
            ratio_score = product_ratio / 0.3
        else:
            ratio_score = max(0, 1 - (product_ratio - 0.7) / 0.3)

        # 2. Centering score (Closer to center is better)
        center_distance = np.sqrt(
            (center_x_ratio - 0.5) ** 2 + (center_y_ratio - 0.5) ** 2
        )
        # Maximum distance is 0.707 (Diagonal to center)
        center_score = 1 - min(center_distance / 0.5, 1)

        # 3. Completeness score (check if the edge is being cut)
        completeness_score = self._check_completeness(img, mask)

        # Composite score
        visibility = (
            ratio_score * 0.4 +
            center_score * 0.3 +
            completeness_score * 0.3
        )

        return visibility

    def _check_completeness(
        self,
        img: Image.Image,
        mask: Optional[np.ndarray] = None
    ) -> float:
        """
        Check if product is complete (not cut off at edges)

        Returns:
            Completeness score (0 - 1)
        """
        if mask is None:
            return 0.8  # No mask, assume mostly complete

        height, width = mask.shape
        edge_width = 5  # Edge check width

        # Check four edges
        edges = [
            mask[:edge_width, :],           # Top
            mask[-edge_width:, :],          # Bottom
            mask[:, :edge_width],           # Left
            mask[:, -edge_width:],          # Right
        ]

        # Calculate product pixels on each edge
        edge_product_ratios = []
        for edge in edges:
            product_pixels = np.sum(edge > 127)
            total_pixels = edge.size
            edge_product_ratios.append(product_pixels / total_pixels)

        # If any edge has >20% product, may be cut off
        max_edge_ratio = max(edge_product_ratios)

        if max_edge_ratio < 0.1:
            return 1.0  # Almost no product on edges, complete
        elif max_edge_ratio < 0.3:
            return 0.7  # Slightly cut off
        else:
            return 0.4  # Obviously cut off

    def calculate_composition_score(self, img: Image.Image) -> float:
        """
        Calculate composition score

            Factors to consider:
            - Is it a frontal view?
            - Is the background clean?
            - Overall aesthetics

        Returns:
            Composition score ranged 0 - 1
        """
        cv2_img = np.array(img)

        # Simplified implementation: evaluate based on color distribution

        # 1. Background cleanliness (low color variance = clean background)
        #   Assume background is near white or gray
        gray = np.mean(cv2_img, axis=2)

        # Find likely background areas (edges)
        edge_size = 20
        edges = np.concatenate([
            gray[:edge_size, :].flatten(),
            gray[-edge_size:, :].flatten(),
            gray[:, :edge_size].flatten(),
            gray[:, -edge_size:].flatten(),
        ])

        # Low background color variation = clean
        bg_std = np.std(edges)
        bg_cleanliness = 1 - min(bg_std / 50, 1)

        # 2. Symmetry assessment
        left_half = cv2_img[:, :cv2_img.shape[1]//2]
        right_half = cv2_img[:, cv2_img.shape[1]//2:]
        right_half_flipped = np.flip(right_half, axis=1)

        # Adjust size to ensure same
        min_width = min(left_half.shape[1], right_half_flipped.shape[1])
        left_half = left_half[:, :min_width]
        right_half_flipped = right_half_flipped[:, :min_width]

        # Calculate symmetry
        diff = np.abs(left_half.astype(float) - right_half_flipped.astype(float))
        symmetry = 1 - min(np.mean(diff) / 100, 1)

        # Composition score * weighting
        composition = bg_cleanliness * 0.6 + symmetry * 0.4

        return composition

    def evaluate_image(
        self,
        img: Image.Image,
        preprocess: bool = True
    ) -> Tuple[Image.Image, float, dict]:
        """
        evaluate_image

        Args:
            img: Image
            preprocess: Whether to preprocess

        Returns:
            (processed Image, Final score, Detailed score)
        """
        # Preprocessing
        if preprocess:
            result = self.preprocessor.standardize(img)
            processed_img = result.image
            mask = result.mask
        else:
            processed_img = img
            mask = None

        # Quality score
        quality = self.scorer.score(processed_img)

        # Visibility scoring
        visibility = self.calculate_visibility_score(processed_img, mask)

        # Composition scoring
        composition = self.calculate_composition_score(processed_img)

        # Calculate Final score
        final_score = (
            quality.overall * self.weights['quality'] +
            visibility * self.weights['visibility'] +
            composition * self.weights['composition']
        )

        details = {
            'quality': quality.overall,
            'quality_detail': quality,
            'visibility': visibility,
            'composition': composition,
        }

        return processed_img, final_score, details

    def select_best(
        self,
        images: List[Image.Image],
        return_all_scores: bool = False
    ) -> Tuple[int, Image.Image, Optional[List[ImageCandidate]]]:
        """
        Select the best main image from multiple images

        Args:
            images: List of images
            return_all_scores: Whether to return scores for all images

        Returns:
            (Best index, Processed best image, All candidate scores)
        """
        if not images:
            raise ValueError("No images provided")

        if len(images) == 1:
            processed, score, _ = self.evaluate_image(images[0])
            if return_all_scores:
                candidate = ImageCandidate(
                    index=0,
                    original_image=images[0],
                    processed_image=processed,
                    quality_score=self.scorer.score(processed),
                    visibility_score=self.calculate_visibility_score(processed),
                    final_score=score,
                )
                return 0, processed, [candidate]
            return 0, processed, None

        candidates = []

        for i, img in enumerate(images):
            try:
                # Preprocessing
                result = self.preprocessor.standardize(img)
                processed_img = result.image

                # Quality score
                quality = self.scorer.score(processed_img)

                # Filter out low-quality images
                if quality.overall < self.min_quality_threshold:
                    logger.info(f"Image {i} below quality threshold: {quality.overall:.2f}")
                    continue

                # Visibility scoring
                visibility = self.calculate_visibility_score(processed_img, result.mask)

                # Composition scoring
                composition = self.calculate_composition_score(processed_img)

                # Final score
                final_score = (
                    quality.overall * self.weights['quality'] +
                    visibility * self.weights['visibility'] +
                    composition * self.weights['composition']
                )

                candidate = ImageCandidate(
                    index=i,
                    original_image=img,
                    processed_image=processed_img,
                    quality_score=quality,
                    visibility_score=visibility,
                    final_score=final_score,
                )
                candidates.append(candidate)

            except Exception as e:
                logger.error(f"Failed to evaluate image {i}: {e}")
                continue

        if not candidates:
            # All images are unqualified, return the first image
            logger.warning("No acceptable images found, returning first image")
            processed, _, _ = self.evaluate_image(images[0], preprocess=True)
            return 0, processed, None

        # Sort by score
        candidates.sort(key=lambda x: x.final_score, reverse=True)

        best = candidates[0]
        logger.info(f"Selected image {best.index} with score {best.final_score:.3f}")

        if return_all_scores:
            return best.index, best.processed_image, candidates

        return best.index, best.processed_image, None

    def rank_images(
        self,
        images: List[Image.Image]
    ) -> List[ImageCandidate]:
        """
        Rank all images

        Args:
            images: List of images

        Returns:
            List of candidates sorted by score descending
        """
        _, _, candidates = self.select_best(images, return_all_scores=True)
        return candidates or []


# Usage
