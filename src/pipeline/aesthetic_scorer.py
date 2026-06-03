"""
Aesthetic Image Quality Scoring Module

Evaluates image aesthetic quality using CLIP-based models.
Provides detailed breakdown: clarity, composition, defects, and overall aesthetic score.

Author: W Cheng
Co-Author: Claude (Anthropic AI Assistant)
Created: 2026-01-15
Project:  Product Image Processing System
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional
from PIL import Image
from .clip_aesthetic_scorer import CLIPQualityScorer


@dataclass
class AestheticScore:
    """Aesthetic image quality scoring result"""
    clarity_score: float       # Product clarity/sharpness (CLIP-based)
    clarity_score_est: float       # Product clarity/sharpness (CLIP-based)
    composition_score: float   # Composition quality (CLIP-based)
    composition_score_est: float   # Composition quality (CLIP-based)
    defect_score: float        # Defect detection (watermarks, obstructions)
    defect_score_est: float        # Defect detection (watermarks, obstructions)
    aesthetic_overall: float   # Overall aesthetic score (1-10)

    def is_acceptable(self, min_score: float = 5.0) -> bool:
        """Determine if image meets minimum aesthetic requirements (1-10 scale)"""
        return self.aesthetic_overall >= min_score

    def normalized_overall(self) -> float:
        """Return aesthetic_overall normalized to 0-1 scale"""
        return max(0, min(1, (self.aesthetic_overall - 1) / 9))


class AestheticScorer:
    """Aesthetic image scorer using CLIP model"""

    def __init__(self, model_name: Optional[str] = None, pretrained: Optional[str] = None):
        """
        Initialize aesthetic scorer

        Args:
            model_name: CLIP model name (default: uses env or fallback)
            pretrained: Pretrained weights identifier
        """
        self._scorer = CLIPQualityScorer(model_name=model_name, pretrained=pretrained)

    def score(self, img: Image.Image | np.ndarray) -> AestheticScore:
        """
        Calculate aesthetic image quality score

        Args:
            img: Input image (PIL Image or numpy array)

        Returns:
            AestheticScore result with clarity, composition, defect, and overall scores
        """
        result = self._scorer.score_detailed(img)

        return AestheticScore(
            clarity_score=result['clarity_score'],
            clarity_score_est=result['clarity_score_est'],
            composition_score=result['composition_score'],
            composition_score_est=result['composition_score_est'],
            defect_score=result['defect_score'],
            defect_score_est=result['defect_score_est'],
            aesthetic_overall=result['aesthetic_overall'],
        )

    def score_simple(self, img: Image.Image | np.ndarray) -> float:
        """
        Get just the overall aesthetic score (1-10)

        Args:
            img: Input image (PIL Image or numpy array)

        Returns:
            Overall aesthetic score (1-10)
        """
        return self._scorer.score(img)


if __name__ == "__main__":
    scorer = AestheticScorer()
    print("AestheticScorer module loaded successfully.")
