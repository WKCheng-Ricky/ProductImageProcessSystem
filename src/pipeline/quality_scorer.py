"""
Image Quality Scoring Module (Facade)

Combines BasicQualityScorer and 
 for comprehensive image quality assessment.
Maintains backward compatibility with existing code.

Author: W Cheng
Co-Author: Claude (Anthropic AI Assistant)
Created: 2026-01-07
Updated: 2026-01-15 - Split into basic and aesthetic scorers
Project:  Product Image Processing System
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
from PIL import Image

from .basic_quality_scorer import BasicQualityScorer
from .aesthetic_scorer import AestheticScorer


@dataclass
class QualityScore:
    """Combined image quality scoring result (backward compatible)"""
    sharpness: float          # Sharpness (normalized 0-1)
    brightness: float         # Brightness (normalized 0-1)
    contrast: float           # Contrast (normalized 0-1)
    resolution_score: float   # Resolution score (0-1)
    product_ratio: float      # Product area ratio in image (0-1)
    has_watermark: bool       # Has watermark
    has_text_overlay: bool    # Has text overlay
    aesthetic: float          # Aesthetic score (normalized 0-1)
    overall: float            # Overall score (0-1)

    # Extended aesthetic fields
    clarity_score: float = 0.0       # CLIP clarity score
    clarity_score_est: float = 0.0       # CLIP clarity score
    composition_score: float = 0.0   # CLIP composition score
    composition_score_est: float = 0.0   # CLIP composition score
    defect_score: float = 0.0        # CLIP defect score
    defect_score_est: float = 0.0        # CLIP defect score
    aesthetic_overall: float = 0.0   # Raw aesthetic score (1-10)

    def is_acceptable(self, min_score: float = 0.3) -> bool:
        """Determine if image meets minimum quality requirements"""
        return self.overall >= min_score

class ImageQualityScorer:
    """Combined image quality scorer (backward compatible facade)"""

    def __init__(
        self,
        min_resolution: Tuple[int, int] = (200, 200),
        ideal_brightness_range: Tuple[float, float] = (80, 200),
        sharpness_weight: float = 0.20,
        brightness_weight: float = 0.10,
        contrast_weight: float = 0.10,
        resolution_weight: float = 0.15,
        product_ratio_weight: float = 0.15,
        aesthetic_weight: float = 0.30,
        watermark_penalty: float = 0.10,
        enable_aesthetic: bool = True,
        device: str = "auto",
    ):
        self.enable_aesthetic = enable_aesthetic
        self.aesthetic_weight = aesthetic_weight if enable_aesthetic else 0

        # Calculate basic weights (normalized when aesthetic is disabled)
        basic_total = sharpness_weight + brightness_weight + contrast_weight + resolution_weight + product_ratio_weight
        if not enable_aesthetic:
            scale = 1.0 / basic_total
            sharpness_weight *= scale
            brightness_weight *= scale
            contrast_weight *= scale
            resolution_weight *= scale
            product_ratio_weight *= scale

        self._basic_scorer = BasicQualityScorer(
            min_resolution=min_resolution,
            ideal_brightness_range=ideal_brightness_range,
            sharpness_weight=sharpness_weight,
            brightness_weight=brightness_weight,
            contrast_weight=contrast_weight,
            resolution_weight=resolution_weight,
            product_ratio_weight=product_ratio_weight,
            watermark_penalty=watermark_penalty,
        )

        self._aesthetic_scorer = AestheticScorer() if enable_aesthetic else None
        self.watermark_penalty = watermark_penalty

        self.weights = {
            'sharpness': sharpness_weight,
            'brightness': brightness_weight,
            'contrast': contrast_weight,
            'resolution': resolution_weight,
            'product_ratio': product_ratio_weight,
            'aesthetic': self.aesthetic_weight,
        }

    def score(
        self,
        img: Image.Image | np.ndarray,
        product_mask: Optional[np.ndarray] = None
    ) -> QualityScore:
        """
        Calculate comprehensive image quality score

        Args:
            img: Input image (PIL Image or numpy array)
            product_mask: Product mask (optional)

        Returns:
            QualityScore result
        """
        basic = self._basic_scorer.score(img, product_mask)

        aesthetic_detail = None
        aesthetic_normalized = 0.0
        if self._aesthetic_scorer:
            aesthetic_detail = self._aesthetic_scorer.score(img)
            aesthetic_normalized = aesthetic_detail.normalized_overall()

        # Calculate combined overall score
        normalized_scores = {
            'sharpness': basic.sharpness,
            'brightness': basic.brightness,
            'contrast': basic.contrast,
            'resolution': basic.resolution_score,
            'product_ratio': basic.product_ratio,
            'aesthetic': aesthetic_normalized,
        }

        overall = sum(
            normalized_scores[key] * self.weights[key]
            for key in self.weights
        )

        if basic.has_watermark:
            overall -= self.watermark_penalty
        if basic.has_text_overlay:
            overall -= self.watermark_penalty * 0.5

        overall = max(0, min(1, overall))

        return QualityScore(
            sharpness=basic.sharpness,
            brightness=basic.brightness,
            contrast=basic.contrast,
            resolution_score=basic.resolution_score,
            product_ratio=basic.product_ratio,
            has_watermark=basic.has_watermark,
            has_text_overlay=basic.has_text_overlay,
            aesthetic=aesthetic_normalized,
            overall=overall,
            clarity_score=aesthetic_detail.clarity_score if aesthetic_detail else 0.0,
            clarity_score_est=aesthetic_detail.clarity_score_est if aesthetic_detail else 0.0,
            composition_score=aesthetic_detail.composition_score if aesthetic_detail else 0.0,
            composition_score_est=aesthetic_detail.composition_score_est if aesthetic_detail else 0.0,
            defect_score=aesthetic_detail.defect_score if aesthetic_detail else 0.0,
            defect_score_est=aesthetic_detail.defect_score_est if aesthetic_detail else 0.0,
            aesthetic_overall=aesthetic_detail.aesthetic_overall if aesthetic_detail else 0.0,
        )

