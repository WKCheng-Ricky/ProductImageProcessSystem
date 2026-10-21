"""
Image Preprocessing Module

Processing flow: Background removal → Crop to subject → Remove watermark → Remove text → Standardize size

Author: W Cheng
Co-Author: Claude (Anthropic AI Assistant)
Created: 2026-01-07
Project:  Product Image Processing System
"""

import cv2
import numpy as np
from PIL import Image
from typing import Optional, Tuple
from dataclasses import dataclass
import logging

from .model_cache import get_model_cache

logger = logging.getLogger(__name__)


@dataclass
class PreprocessingResult:
    """Preprocessing result"""
    image: Image.Image           # Processed image
    original_size: Tuple[int, int]  # Original size
    product_bbox: Optional[Tuple[int, int, int, int]]  # Product bounding box (x1, y1, x2, y2)
    mask: Optional[np.ndarray]   # Product mask
    background_removed: bool     # Background removed
    watermark_removed: bool      # Watermark removed
    text_removed: bool           # Text removed


class ImagePreprocessor:
    """Image preprocessor"""

    def __init__(
        self,
        target_size: Tuple[int, int] = (512, 512),
        background_color: Tuple[int, int, int] = (255, 255, 255),
        padding_ratio: float = 0.1,
        use_gpu: bool = False,
    ):
        """
        Initialize preprocessor

        Args:
            target_size: Target output size
            background_color: Background color (RGB)
            padding_ratio: Padding ratio when cropping
            use_gpu: Whether to use GPU acceleration
        """
        self.target_size = target_size
        self.background_color = background_color
        self.padding_ratio = padding_ratio
        self.use_gpu = use_gpu

        # Lazy load models
        self._rembg_session = None

    @property
    def rembg_session(self):
        """Lazy load rembg using shared cache"""
        if self._rembg_session is None:
            try:
                # Configure ONNX Runtime providers
                providers = []

                if self.use_gpu:
                    try:
                        import onnxruntime as ort
                        available_providers = ort.get_available_providers()

                        if 'ROCMExecutionProvider' in available_providers:
                            providers.append('ROCMExecutionProvider')
                            logger.info("Using ROCm provider for AMD GPU")
                        elif 'CUDAExecutionProvider' in available_providers:
                            providers.append('CUDAExecutionProvider')
                            logger.info("Using CUDA provider for NVIDIA GPU")
                        else:
                            logger.warning("GPU requested but no GPU provider available, falling back to CPU")
                    except ImportError:
                        logger.warning("onnxruntime not available for provider detection, using CPU")

                providers.append('CPUExecutionProvider')

                # Use shared model cache
                cache = get_model_cache()
                self._rembg_session = cache.get_rembg_session("u2net", providers=tuple(providers))
                logger.info(f"Loaded rembg session with providers: {providers}")
            except ImportError:
                logger.warning("rembg not installed. Background removal disabled.")
        return self._rembg_session

    def remove_background(
        self,
        img: Image.Image,
        return_mask: bool = False
    ) -> Tuple[Image.Image, Optional[np.ndarray]]:
        """
        Remove background, convert to specified background color

        Args:
            img: Input image
            return_mask: Whether to return mask

        Returns:
            (Processed image, mask) or (Processed image, None)
        """
        try:
            from rembg import remove

            # Remove background, get RGBA image
            result = remove(
                img,
                session=self.rembg_session,
                bgcolor=None,  # Transparent background
            )

            # Extract alpha channel as mask
            mask = None
            if return_mask and result.mode == 'RGBA':
                mask = np.array(result.split()[3])

            # Convert to specified background color
            if result.mode == 'RGBA':
                background = Image.new('RGBA', result.size, (*self.background_color, 255))
                background.paste(result, mask=result.split()[3])
                result = background.convert('RGB')

            return result, mask

        except ImportError:
            logger.warning("rembg not available, returning original image")
            return img, None
        except Exception as e:
            logger.error(f"Background removal failed: {e}")
            return img, None

    def find_product_bbox(
        self,
        img: Image.Image,
        mask: Optional[np.ndarray] = None
    ) -> Optional[Tuple[int, int, int, int]]:
        """
        Find product bounding box

        Args:
            img: Input image
            mask: Product mask (if available)

        Returns:
            (x1, y1, x2, y2) bounding box or None
        """
        if mask is not None:
            # Use mask to find boundary
            coords = np.column_stack(np.where(mask > 127))
            if len(coords) == 0:
                return None
            y1, x1 = coords.min(axis=0)
            y2, x2 = coords.max(axis=0)
            return (int(x1), int(y1), int(x2), int(y2))

        # No mask, use edge detection
        cv2_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)

        # Edge detection
        edges = cv2.Canny(gray, 50, 150)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return None

        # Merge bounding boxes of all contours
        x_min, y_min = float('inf'), float('inf')
        x_max, y_max = 0, 0

        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            x_min = min(x_min, x)
            y_min = min(y_min, y)
            x_max = max(x_max, x + w)
            y_max = max(y_max, y + h)

        if x_min == float('inf'):
            return None

        return (int(x_min), int(y_min), int(x_max), int(y_max))

    def crop_to_product(
        self,
        img: Image.Image,
        bbox: Optional[Tuple[int, int, int, int]] = None,
        mask: Optional[np.ndarray] = None,
        padding: Optional[float] = None
    ) -> Image.Image:
        """
        Crop to product subject

        Args:
            img: Input image
            bbox: Bounding box (if known)
            mask: Product mask (if known)
            padding: Padding ratio (overrides default)

        Returns:
            Cropped image
        """
        if bbox is None:
            bbox = self.find_product_bbox(img, mask)

        if bbox is None:
            logger.warning("Could not find product bbox, returning original")
            return img

        x1, y1, x2, y2 = bbox
        width, height = img.size

        # Calculate padding
        pad = padding if padding is not None else self.padding_ratio
        pad_w = int((x2 - x1) * pad)
        pad_h = int((y2 - y1) * pad)

        # Expand bounding box
        x1 = max(0, x1 - pad_w)
        y1 = max(0, y1 - pad_h)
        x2 = min(width, x2 + pad_w)
        y2 = min(height, y2 + pad_h)

        return img.crop((x1, y1, x2, y2))

    def remove_watermark(
        self,
        img: Image.Image,
        watermark_mask: Optional[np.ndarray] = None
    ) -> Image.Image:
        """
        Remove watermark (using inpainting)

        Args:
            img: Input image
            watermark_mask: Watermark region mask

        Returns:
            Image after watermark removal
        """
        if watermark_mask is None:
            # Auto-detect watermark region
            watermark_mask = self._detect_watermark_region(img)

        if watermark_mask is None or watermark_mask.sum() == 0:
            return img

        try:
            # Use OpenCV inpainting
            cv2_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

            # Ensure mask is in correct format
            if len(watermark_mask.shape) == 3:
                watermark_mask = cv2.cvtColor(watermark_mask, cv2.COLOR_BGR2GRAY)
            watermark_mask = watermark_mask.astype(np.uint8)

            # Inpainting
            result = cv2.inpaint(cv2_img, watermark_mask, 3, cv2.INPAINT_TELEA)

            return Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))

        except Exception as e:
            logger.error(f"Watermark removal failed: {e}")
            return img

    def _detect_watermark_region(self, img: Image.Image) -> Optional[np.ndarray]:
        """
        Auto-detect watermark region

        Simple implementation: Detect semi-transparent text/logo in corners
        Advanced implementation: Can use trained watermark detection model
        """
        cv2_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        height, width = cv2_img.shape[:2]

        mask = np.zeros((height, width), dtype=np.uint8)
        corner_size = min(height, width) // 6

        corners = [
            (0, 0, corner_size, corner_size),                              # Top-left
            (width - corner_size, 0, width, corner_size),                  # Top-right
            (0, height - corner_size, corner_size, height),                # Bottom-left
            (width - corner_size, height - corner_size, width, height),    # Bottom-right
        ]

        for x1, y1, x2, y2 in corners:
            corner = cv2_img[y1:y2, x1:x2]
            gray = cv2.cvtColor(corner, cv2.COLOR_BGR2GRAY)

            # Detect high contrast regions (likely watermark)
            edges = cv2.Canny(gray, 50, 150)
            edge_ratio = edges.sum() / edges.size / 255

            if edge_ratio > 0.1:  # Corner has many edges
                # Mark this corner region
                mask[y1:y2, x1:x2] = edges

        return mask if mask.sum() > 0 else None

    def remove_text_overlay(
        self,
        img: Image.Image,
        text_mask: Optional[np.ndarray] = None
    ) -> Image.Image:
        """
        Remove promotional text overlay

        Args:
            img: Input image
            text_mask: Text region mask

        Returns:
            Image after text removal
        """
        if text_mask is None:
            text_mask = self._detect_text_region(img)

        if text_mask is None or text_mask.sum() == 0:
            return img

        # Use inpainting to remove text
        try:
            cv2_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

            # Dilate mask to ensure complete text coverage
            kernel = np.ones((3, 3), np.uint8)
            text_mask = cv2.dilate(text_mask, kernel, iterations=2)

            result = cv2.inpaint(cv2_img, text_mask, 3, cv2.INPAINT_TELEA)

            return Image.fromarray(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))

        except Exception as e:
            logger.error(f"Text removal failed: {e}")
            return img

    def _detect_text_region(self, img: Image.Image) -> Optional[np.ndarray]:
        """
        Detect text region

        Uses MSER detection, recommend PaddleOCR for production use
        """
        cv2_img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2GRAY)

        height, width = gray.shape
        mask = np.zeros((height, width), dtype=np.uint8)

        # MSER to detect text regions
        mser = cv2.MSER_create()
        regions, _ = mser.detectRegions(gray)

        # Filter regions that are too small or too large
        for region in regions:
            hull = cv2.convexHull(region.reshape(-1, 1, 2))
            area = cv2.contourArea(hull)

            # Filter: too small might be noise, too large might be product itself
            if 100 < area < (height * width * 0.1):
                cv2.fillPoly(mask, [hull], 255)

        return mask if mask.sum() > 0 else None

    def resize_and_pad(
        self,
        img: Image.Image,
        target_size: Optional[Tuple[int, int]] = None
    ) -> Image.Image:
        """
        Resize and maintain aspect ratio, fill with background color

        Args:
            img: Input image
            target_size: Target size (overrides default)

        Returns:
            Resized image
        """
        target = target_size or self.target_size

        # Calculate scaling ratio
        width, height = img.size
        ratio = min(target[0] / width, target[1] / height)

        new_width = int(width * ratio)
        new_height = int(height * ratio)

        # Resize
        resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

        # Create background and paste centered
        result = Image.new('RGB', target, self.background_color)
        paste_x = (target[0] - new_width) // 2
        paste_y = (target[1] - new_height) // 2
        result.paste(resized, (paste_x, paste_y))

        return result

    def standardize(
        self,
        img: Image.Image,
        remove_bg: bool = True,
        crop: bool = True,
        remove_watermark: bool = True,
        remove_text: bool = False,  # Default off, may damage product text
        resize: bool = True,
    ) -> PreprocessingResult:
        """
        Complete standardization processing flow

        Args:
            img: Input image
            remove_bg: Whether to remove background
            crop: Whether to crop to product
            remove_watermark: Whether to remove watermark
            remove_text: Whether to remove text overlay
            resize: Whether to resize

        Returns:
            PreprocessingResult processing result
        """
        original_size = img.size
        mask = None
        bbox = None
        bg_removed = False
        wm_removed = False
        text_removed = False

        # 1. Remove background
        if remove_bg:
            img, mask = self.remove_background(img, return_mask=True)
            bg_removed = True

        # 2. Find bounding box and crop
        if crop:
            bbox = self.find_product_bbox(img, mask)
            if bbox:
                img = self.crop_to_product(img, bbox, mask)

        # 3. Remove watermark
        if remove_watermark:
            img = self.remove_watermark(img)
            wm_removed = True

        # 4. Remove text overlay
        if remove_text:
            img = self.remove_text_overlay(img)
            text_removed = True

        # 5. Resize
        if resize:
            img = self.resize_and_pad(img)

        return PreprocessingResult(
            image=img,
            original_size=original_size,
            product_bbox=bbox,
            mask=mask,
            background_removed=bg_removed,
            watermark_removed=wm_removed,
            text_removed=text_removed,
        )

    def process_batch(
        self,
        images: list[Image.Image],
        **kwargs
    ) -> list[PreprocessingResult]:
        """
        Batch process images

        Args:
            images: Image list
            **kwargs: Parameters passed to standardize

        Returns:
            Processing result list
        """
        results = []
        for img in images:
            try:
                result = self.standardize(img, **kwargs)
                results.append(result)
            except Exception as e:
                logger.error(f"Failed to process image: {e}")
                # Return original image as fallback
                results.append(PreprocessingResult(
                    image=img,
                    original_size=img.size,
                    product_bbox=None,
                    mask=None,
                    background_removed=False,
                    watermark_removed=False,
                    text_removed=False,
                ))

        return results


# Usage example
