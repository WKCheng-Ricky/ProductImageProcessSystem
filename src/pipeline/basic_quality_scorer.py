"""
Basic Image Quality Scoring Module

Evaluates basic image quality using OpenCV-based metrics (no ML models required).
Fast scoring for filtering low-quality images.

Author: W Cheng
Co-Author: Claude (Anthropic AI Assistant)
Created: 2026-01-15
Project:  Product Image Processing System
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple
from PIL import Image


@dataclass
class BasicQualityScore:
    """Basic image quality scoring result"""
    sharpness: float          # Sharpness (normalized 0-1)
    brightness: float         # Brightness (normalized 0-1)
    contrast: float           # Contrast (normalized 0-1)
    resolution_score: float   # Resolution score (0-1)
    product_ratio: float      # Product area ratio in image (0-1)
    has_watermark: bool       # Has watermark
    has_text_overlay: bool    # Has text overlay
    overall: float            # Overall score (0-1)

    def is_acceptable(self, min_score: float = 0.3) -> bool:
        """Determine if image meets minimum quality requirements"""
        return self.overall >= min_score


class BasicQualityScorer:
    """Basic image quality scorer using OpenCV (no ML models)"""

    def __init__(
        self,
        min_resolution: Tuple[int, int] = (200, 200),
        ideal_brightness_range: Tuple[float, float] = (80, 200),
        sharpness_weight: float = 0.30,
        brightness_weight: float = 0.15,
        contrast_weight: float = 0.15,
        resolution_weight: float = 0.20,
        product_ratio_weight: float = 0.20,
        watermark_penalty: float = 0.10,
    ):
        self.min_resolution = min_resolution
        self.ideal_brightness_range = ideal_brightness_range
        self.weights = {
            'sharpness': sharpness_weight,
            'brightness': brightness_weight,
            'contrast': contrast_weight,
            'resolution': resolution_weight,
            'product_ratio': product_ratio_weight,
        }
        self.watermark_penalty = watermark_penalty

    def _to_cv2(self, img: Image.Image | np.ndarray) -> np.ndarray:
        if isinstance(img, Image.Image):
            return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        return img

    def calculate_sharpness(self, img: np.ndarray) -> float:
        """
        Calculate sharpness using Laplacian

        Returns:
            Laplacian variance, higher values indicate sharper images
            Generally < 100 is blurry, > 300 is sharp
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        return float(laplacian.var())

    def calculate_brightness(self, img: np.ndarray) -> float:
        """
        Calculate average brightness

        Returns:
            Brightness value 0-255, ideal range is 80-200
        """
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        return float(hsv[:, :, 2].mean())

    def calculate_contrast(self, img: np.ndarray) -> float:
        """
        Calculate contrast (standard deviation)

        Returns:
            Contrast value, higher values indicate more contrast
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return float(gray.std())

    def calculate_resolution_score(self, img: np.ndarray) -> float:
        """
        Calculate resolution score

        Returns:
            0-1 score, based on pixel count
        """
        height, width = img.shape[:2]
        pixels = height * width

        min_pixels = self.min_resolution[0] * self.min_resolution[1]
        if pixels < min_pixels:
            return 0.0

        ideal_pixels = 1_000_000

        if pixels >= ideal_pixels:
            return 1.0

        return (pixels - min_pixels) / (ideal_pixels - min_pixels)

    def calculate_product_ratio(
        self,
        img: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> float:
        """
        Calculate product area ratio in the image

        Args:
            img: Input image
            mask: Product mask (if available)

        Returns:
            0-1 ratio value
        """
        if mask is not None:
            product_pixels = np.sum(mask > 0)
            total_pixels = mask.shape[0] * mask.shape[1]
            return product_pixels / total_pixels

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        foreground_pixels = np.sum(binary > 0)
        total_pixels = binary.shape[0] * binary.shape[1]

        ratio = foreground_pixels / total_pixels

        if ratio > 0.95 or ratio < 0.05:
            return 0.5

        return ratio

    def detect_watermark(self, img: np.ndarray) -> bool:
        """
        Detect watermark

        Returns:
            Whether watermark is detected
        """
        height, width = img.shape[:2]
        corner_size = min(height, width) // 8

        corners = [
            img[0:corner_size, 0:corner_size],
            img[0:corner_size, width-corner_size:width],
            img[height-corner_size:height, 0:corner_size],
            img[height-corner_size:height, width-corner_size:width],
        ]

        for corner in corners:
            gray = cv2.cvtColor(corner, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)
            edge_ratio = np.sum(edges > 0) / edges.size

            if edge_ratio > 0.15:
                return True

        return False

    def detect_text_overlay(self, img: np.ndarray) -> bool:
        """
        Detect text overlay (promotional text, etc.)

        Returns:
            Whether text overlay is detected
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        mser = cv2.MSER_create()
        regions, _ = mser.detectRegions(gray)

        if len(regions) > 50:
            return True

        total_area = 0
        for region in regions:
            hull = cv2.convexHull(region.reshape(-1, 1, 2))
            total_area += cv2.contourArea(hull)

        image_area = gray.shape[0] * gray.shape[1]
        text_ratio = total_area / image_area

        return text_ratio > 0.1

    def _normalize_sharpness(self, sharpness: float) -> float:
        """Normalize sharpness score to 0-1"""
        return min(max(sharpness / 500, 0), 1)

    def _normalize_brightness(self, brightness: float) -> float:
        """Normalize brightness score to 0-1 (closer to ideal range is higher)"""
        low, high = self.ideal_brightness_range
        mid = (low + high) / 2

        if low <= brightness <= high:
            distance_from_mid = abs(brightness - mid)
            max_distance = (high - low) / 2
            return 1 - (distance_from_mid / max_distance) * 0.2
        else:
            if brightness < low:
                return max(brightness / low * 0.8, 0)
            else:
                return max((255 - brightness) / (255 - high) * 0.8, 0)

    def _normalize_contrast(self, contrast: float) -> float:
        """Normalize contrast score to 0-1"""
        if contrast < 20:
            return contrast / 20 * 0.5
        elif contrast > 100:
            return max(1 - (contrast - 100) / 100, 0.5)
        else:
            return min(contrast / 60, 1)

    def score(
        self,
        img: Image.Image | np.ndarray,
        product_mask: Optional[np.ndarray] = None
    ) -> BasicQualityScore:
        """
        Calculate basic image quality score

        Args:
            img: Input image (PIL Image or numpy array)
            product_mask: Product mask (optional)

        Returns:
            BasicQualityScore result
        """
        cv2_img = self._to_cv2(img)

        sharpness = self.calculate_sharpness(cv2_img)
        brightness = self.calculate_brightness(cv2_img)
        contrast = self.calculate_contrast(cv2_img)
        resolution_score = self.calculate_resolution_score(cv2_img)
        product_ratio = self.calculate_product_ratio(cv2_img, product_mask)
        has_watermark = self.detect_watermark(cv2_img)
        has_text_overlay = self.detect_text_overlay(cv2_img)

        normalized_scores = {
            'sharpness': self._normalize_sharpness(sharpness),
            'brightness': self._normalize_brightness(brightness),
            'contrast': self._normalize_contrast(contrast),
            'resolution': float(resolution_score),
            'product_ratio': float(product_ratio),
        }

        overall = sum(
            normalized_scores[key] * self.weights[key]
            for key in self.weights
        )

        if has_watermark:
            overall -= self.watermark_penalty
        if has_text_overlay:
            overall -= self.watermark_penalty * 0.5

        overall = max(0, min(1, overall))

        return BasicQualityScore(
            sharpness=normalized_scores['sharpness'],
            brightness=normalized_scores['brightness'],
            contrast=normalized_scores['contrast'],
            resolution_score=normalized_scores['resolution'],
            product_ratio=normalized_scores['product_ratio'],
            has_watermark=has_watermark,
            has_text_overlay=has_text_overlay,
            overall=overall,
        )

    def filter_acceptable(
        self,
        images: list[Image.Image | np.ndarray],
        min_score: float = 0.3
    ) -> list[tuple[int, BasicQualityScore]]:
        """
        Filter images that meet quality standards

        Args:
            images: Image list
            min_score: Minimum score threshold

        Returns:
            List of (index, score) tuples, sorted by score descending
        """
        results = []

        for i, img in enumerate(images):
            score = self.score(img)
            if score.is_acceptable(min_score):
                results.append((i, score))

        results.sort(key=lambda x: x[1].overall, reverse=True)

        return results


if __name__ == "__main__":
    scorer = BasicQualityScorer()
    print("BasicQualityScorer module loaded successfully.")
