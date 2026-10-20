import logging
from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

try:
    import easyocr
except ImportError:
    easyocr = None

logger = logging.getLogger(__name__)


@dataclass
class WatermarkRegion:
    """Watermark region information"""
    bbox: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    text: Optional[str]               # Detected text content
    confidence: float                 # Detection confidence
    type: str                         # 'text', 'logo', 'sticker'


@dataclass
class WatermarkRemovalResult:
    """Watermark removal result"""
    image: Image.Image               # Processed image
    original_image: Image.Image      # Original image
    watermarks_detected: List[WatermarkRegion]  # Detected watermarks
    watermarks_removed: int          # Number of watermarks removed
    success: bool                    # Whether successful


class WatermarkRemover:
    def __init__(
        self,
        use_ocr: bool = False,
        min_text_confidence: float = 0.5,
        inpaint_method: str = 'telea',  # 'telea' or 'ns'
        padding: int = 5,
    ):
        """
        Initialize watermark remover

        Args:
            use_ocr: Whether to use OCR (Requires easyocr or paddleocr)
            min_text_confidence: Minimum text detection confidence
            inpaint_method: Inpainting method ('telea' or 'ns')
            padding: Watermark region padding
        """
        self.use_ocr = use_ocr
        self.min_confidence = min_text_confidence
        self.inpaint_method = inpaint_method
        self.padding = padding

        # OCR reader (lazy load)
        self._ocr_reader = None

    def _get_ocr_reader(self):
        """Lazy load OCR reader"""
        if self._ocr_reader is None and self.use_ocr:
            if easyocr is None:
                logger.warning("EasyOCR not installed. Falling back to simple text detection.")
                self.use_ocr = False
            else:
                logger.info("Loading EasyOCR reader...")
                self._ocr_reader = easyocr.Reader(['en', 'ch_tra'], gpu=False)
                logger.info("EasyOCR loaded successfully")
        return self._ocr_reader

    def _to_cv2(self, img: Image.Image) -> np.ndarray:
        """Convert to OpenCV format"""
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    def _to_pil(self, img: np.ndarray) -> Image.Image:
        """Convert to PIL format"""
        return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    def detect_text_regions_simple(
        self,
        img: np.ndarray
    ) -> List[WatermarkRegion]:
        """
        Simple text region detection (without using OCR)

        Detection strategy:
        1. High contrast small areas (usually text)
        2. Rectangle regions with dense edges
        3. Located at image edges or corners (common watermark positions)

        Args:
            img: Input image (BGR)

        Returns:
            List of detected text regions
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape

        # 1. Edge detection
        edges = cv2.Canny(gray, 50, 150)

        # 2. Dilate to connect text
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=2)

        # 3. Find contours
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        regions = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)

            # Filter conditions
            area = w * h
            aspect_ratio = w / h if h > 0 else 0

            # Text characteristics:
            # - Small area (< 20% of image area)
            # - Aspect ratio in reasonable range (0.1 ~ 10)
            # - Located at edge region or < 10% of image
            total_area = width * height
            is_small = area < total_area * 0.2
            is_reasonable_ratio = 0.1 < aspect_ratio < 10
            is_edge_region = (
                x < width * 0.15 or x + w > width * 0.85 or
                y < height * 0.15 or y + h > height * 0.85
            )

            # Calculate edge density (text usually has dense edges)
            roi_edges = edges[y:y+h, x:x+w]
            edge_density = np.sum(roi_edges > 0) / area if area > 0 else 0
            has_dense_edges = edge_density > 0.1

            if is_small and is_reasonable_ratio and (is_edge_region or has_dense_edges):
                # Calculate confidence
                confidence = 0.5
                if is_edge_region:
                    confidence += 0.2
                if has_dense_edges:
                    confidence += 0.2
                if area < total_area * 0.05:  # Very small area
                    confidence += 0.1

                regions.append(WatermarkRegion(
                    bbox=(x, y, x + w, y + h),
                    text=None,
                    confidence=min(confidence, 1.0),
                    type='text'
                ))

        logger.info(f"Detected {len(regions)} potential text regions (simple method)")
        return regions

    def detect_text_regions_ocr(
        self,
        img: np.ndarray
    ) -> List[WatermarkRegion]:
        """
        Detect text regions using OCR

        Args:
            img: Input image (BGR)

        Returns:
            List of detected text regions
        """
        reader = self._get_ocr_reader()
        if reader is None:
            logger.warning("OCR reader not available, using simple detection")
            return self.detect_text_regions_simple(img)

        # EasyOCR requires RGB
        rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        try:
            # OCR detection
            results = reader.readtext(rgb_img)

            regions = []
            for (bbox, text, conf) in results:
                if conf < self.min_confidence:
                    continue

                # bbox format: [[x1,y1], [x2,y1], [x2,y2], [x1,y2]]
                x_coords = [point[0] for point in bbox]
                y_coords = [point[1] for point in bbox]

                x1, y1 = int(min(x_coords)), int(min(y_coords))
                x2, y2 = int(max(x_coords)), int(max(y_coords))

                regions.append(WatermarkRegion(
                    bbox=(x1, y1, x2, y2),
                    text=text,
                    confidence=conf,
                    type='text'
                ))

            logger.info(f"OCR detected {len(regions)} text regions")
            return regions

        except Exception as e:
            logger.error(f"OCR detection failed: {e}")
            return self.detect_text_regions_simple(img)

    def detect_watermarks(
        self,
        img: Image.Image
    ) -> List[WatermarkRegion]:
        """
        Detect watermarks in image

        Args:
            img: Input image

        Returns:
            List of detected watermark regions
        """
        cv2_img = self._to_cv2(img)

        if self.use_ocr:
            return self.detect_text_regions_ocr(cv2_img)
        else:
            return self.detect_text_regions_simple(cv2_img)

    def remove_watermark(
        self,
        img: Image.Image,
        watermarks: Optional[List[WatermarkRegion]] = None,
    ) -> WatermarkRemovalResult:
        """
        Remove watermarks from image

        Args:
            img: Input image
            watermarks: Watermark regions (if None, will auto-detect)

        Returns:
            WatermarkRemovalResult containing removal results
        """
        original_img = img.copy()
        cv2_img = self._to_cv2(img)
        height, width = cv2_img.shape[:2]

        # Detect watermarks
        if watermarks is None:
            watermarks = self.detect_watermarks(img)

        if not watermarks:
            logger.info("No watermarks detected")
            return WatermarkRemovalResult(
                image=img,
                original_image=original_img,
                watermarks_detected=[],
                watermarks_removed=0,
                success=True
            )

        # Create inpainting mask
        mask = np.zeros((height, width), dtype=np.uint8)

        removed_count = 0
        for wm in watermarks:
            x1, y1, x2, y2 = wm.bbox

            # Add padding
            x1 = max(0, x1 - self.padding)
            y1 = max(0, y1 - self.padding)
            x2 = min(width, x2 + self.padding)
            y2 = min(height, y2 + self.padding)

            # Mark region to inpaint
            mask[y1:y2, x1:x2] = 255
            removed_count += 1

            logger.debug(f"Marking watermark region: ({x1},{y1}) to ({x2},{y2})")

        # Inpainting
        try:
            if self.inpaint_method == 'telea':
                inpainted = cv2.inpaint(cv2_img, mask, 3, cv2.INPAINT_TELEA)
            else:  # 'ns'
                inpainted = cv2.inpaint(cv2_img, mask, 3, cv2.INPAINT_NS)

            result_img = self._to_pil(inpainted)

            logger.info(f"Successfully removed {removed_count} watermarks")

            return WatermarkRemovalResult(
                image=result_img,
                original_image=original_img,
                watermarks_detected=watermarks,
                watermarks_removed=removed_count,
                success=True
            )

        except Exception as e:
            logger.error(f"Inpainting failed: {e}")
            return WatermarkRemovalResult(
                image=img,
                original_image=original_img,
                watermarks_detected=watermarks,
                watermarks_removed=0,
                success=False
            )

    def remove_watermark_auto(
        self,
        img: Image.Image
    ) -> Image.Image:
        """
        Automatically detect and remove watermarks (simplified version)

        Args:
            img: Input image

        Returns:
            Processed image
        """
        result = self.remove_watermark(img)
        return result.image


# Usage example
