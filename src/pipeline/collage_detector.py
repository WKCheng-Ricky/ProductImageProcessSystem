"""
Collage Detection and Splitting Module

Process merchant-uploaded collages, auto-split into single product images.

Remark: Not producing great result. Could drop.
"""

import cv2
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class CollageInfo:
    """Collage information"""
    is_collage: bool            
    num_images: int             
    layout: str                 # Layout ('grid', 'horizontal', 'vertical', 'irregular')
    grid_size: Optional[Tuple[int, int]]  # Grid size (rows, cols)
    confidence: float           # Detection confidence


@dataclass
class SplitResult:
    images: List[Image.Image]   
    bboxes: List[Tuple[int, int, int, int]]  # Bounding boxes (x1, y1, x2, y2)
    collage_info: CollageInfo   # Collage information


# Not Good
class CollageDetector:
    """Collage detector"""

    def __init__(
        self,
        min_split_size: int = 100,
        line_detection_threshold: float = 0.3,
        min_product_area_ratio: float = 0.05,
    ):
        """
        Initialize detector

        Args:
            min_split_size: Minimum split size
            line_detection_threshold: Separator line detection threshold
            min_product_area_ratio: Minimum product area ratio
        """
        self.min_split_size = min_split_size
        self.line_threshold = line_detection_threshold
        self.min_area_ratio = min_product_area_ratio

    def _to_cv2(self, img: Image.Image) -> np.ndarray:
        return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    def _to_pil(self, img: np.ndarray) -> Image.Image:
        return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    def detect_grid_lines(
        self,
        img: np.ndarray
    ) -> Tuple[List[int], List[int]]:
        """
        Detect grid separator lines

        Args:
            img: Image (BGR)

        Returns:
            (List of horizontal line positions, List of vertical line positions)
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape

        # Find peaks (Low variance regions may be separator lines)
        # or Separator line edges

        # Find uniform colored horizontal/vertical bands
        h_lines = self._find_uniform_lines(gray, axis=0)  # Horizontal lines
        v_lines = self._find_uniform_lines(gray, axis=1)  # Vertical lines

        return h_lines, v_lines

    def _find_uniform_lines(
        self,
        gray: np.ndarray,
        axis: int,
        min_line_width: int = 2,
    ) -> List[int]:
        """
        Find uniform color lines

        Args:
            gray: Grayscale image
            axis: 0 = Horizontal lines, 1 = Vertical lines
            min_line_width: Minimum line width

        Returns:
            List of line positions
        """
        if axis == 0:
            # Horizontal lines: standard deviation
            line_std = gray.std(axis=1)
            size = gray.shape[0]
        else:
            # Vertical lines: standard deviation
            line_std = gray.std(axis=0)
            size = gray.shape[1]

        # Find areas with low standard deviation (uniform color)
        threshold = line_std.mean() * 0.3
        uniform_mask = line_std < threshold

        # Find consecutive uniform areas
        lines = []
        in_line = False
        line_start = 0

        for i in range(size):
            if uniform_mask[i] and not in_line:
                in_line = True
                line_start = i
            elif not uniform_mask[i] and in_line:
                in_line = False
                line_width = i - line_start
                if line_width >= min_line_width:
                    # Take center point
                    lines.append(line_start + line_width // 2)

        # Filter lines too close to edges
        margin = size * 0.1
        lines = [l for l in lines if margin < l < size - margin]

        return lines

    def detect_objects(
        self,
        img: np.ndarray
    ) -> List[Tuple[int, int, int, int]]:
        """
        Detect objects in image using edge detection

        Args:
            img: Input image

        Returns:
            List of bounding boxes [(x1, y1, x2, y2), ...]
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Edge detection
        edges = cv2.Canny(gray, 50, 150)

        # Dilate to connect nearby edges
        kernel = np.ones((5, 5), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=2)

        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Convert to bounding boxes
        height, width = img.shape[:2]
        min_area = height * width * self.min_area_ratio

        bboxes = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            area = w * h

            if area >= min_area and w >= self.min_split_size and h >= self.min_split_size:
                bboxes.append((x, y, x + w, y + h))

        return bboxes

    def is_collage(self, img: Image.Image) -> CollageInfo:
        cv2_img = self._to_cv2(img)
        height, width = cv2_img.shape[:2]

        # Approach 1: detect_grid_lines
        h_lines, v_lines = self.detect_grid_lines(cv2_img)

        # Approach 2: detect_objects
        objects = self.detect_objects(cv2_img)

        num_h_lines = len(h_lines)
        num_v_lines = len(v_lines)
        num_objects = len(objects)

        # Determine based on separator lines
        if num_h_lines > 0 or num_v_lines > 0:
            rows = num_h_lines + 1
            cols = num_v_lines + 1
            num_images = rows * cols

            if num_images > 1:
                return CollageInfo(
                    is_collage=True,
                    num_images=num_images,
                    layout='grid',
                    grid_size=(rows, cols),
                    confidence=0.8,
                )

        # Determine based on object count
        if num_objects > 1:
            # Check if objects do not overlap
            non_overlapping = self._check_non_overlapping(objects)

            if non_overlapping:
                # Determine layout
                layout = self._detect_layout(objects, width, height)

                return CollageInfo(
                    is_collage=True,
                    num_images=num_objects,
                    layout=layout,
                    grid_size=None,
                    confidence=0.6,
                )

        # Not a collage
        return CollageInfo(
            is_collage=False,
            num_images=1,
            layout='single',
            grid_size=None,
            confidence=0.9,
        )

    def _check_non_overlapping(
        self,
        bboxes: List[Tuple[int, int, int, int]],
        overlap_threshold: float = 0.3,
    ) -> bool:
        """Check if bounding boxes do not overlap"""
        for i, box1 in enumerate(bboxes):
            for j, box2 in enumerate(bboxes):
                if i >= j:
                    continue

                # Calculate overlap
                x1 = max(box1[0], box2[0])
                y1 = max(box1[1], box2[1])
                x2 = min(box1[2], box2[2])
                y2 = min(box1[3], box2[3])

                if x1 < x2 and y1 < y2:
                    intersection = (x2 - x1) * (y2 - y1)
                    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
                    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
                    min_area = min(area1, area2)

                    if intersection / min_area > overlap_threshold:
                        return False

        return True

    def _detect_layout(
        self,
        bboxes: List[Tuple[int, int, int, int]],
        img_width: int,
        img_height: int,
    ) -> str:
        """Detect layout type"""
        if len(bboxes) <= 1:
            return 'single'

        # Calculate center points
        centers = [(
            (box[0] + box[2]) / 2,
            (box[1] + box[3]) / 2
        ) for box in bboxes]

        # Check if mainly horizontal arrangement
        y_coords = [c[1] for c in centers]
        y_std = np.std(y_coords)

        # Check if mainly vertical arrangement
        x_coords = [c[0] for c in centers]
        x_std = np.std(x_coords)

        relative_y_std = y_std / img_height
        relative_x_std = x_std / img_width

        if relative_y_std < 0.1:
            return 'horizontal'
        elif relative_x_std < 0.1:
            return 'vertical'
        else:
            return 'irregular'

    def split_collage(
        self,
        img: Image.Image,
        collage_info: Optional[CollageInfo] = None,
    ) -> SplitResult:
        """
        Split collage into individual images

        Args:
            img: Input image
            collage_info: Collage information (if None, will auto-detect)

        Returns:
            SplitResult Split result
        """
        if collage_info is None:
            collage_info = self.is_collage(img)

        if not collage_info.is_collage:
            return SplitResult(
                images=[img],
                bboxes=[(0, 0, img.width, img.height)],
                collage_info=collage_info,
            )

        cv2_img = self._to_cv2(img)
        width, height = img.size

        if collage_info.layout == 'grid' and collage_info.grid_size:
            # Grid splitting
            bboxes = self._split_grid(width, height, collage_info.grid_size)
        else:
            # Object detection splitting
            bboxes = self.detect_objects(cv2_img)

            if not bboxes:
                # Fallback: return original image
                return SplitResult(
                    images=[img],
                    bboxes=[(0, 0, width, height)],
                    collage_info=collage_info,
                )

        # Split image
        images = []
        valid_bboxes = []

        for bbox in bboxes:
            x1, y1, x2, y2 = bbox

            # Ensure valid boundaries
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(width, x2)
            y2 = min(height, y2)

            if x2 - x1 >= self.min_split_size and y2 - y1 >= self.min_split_size:
                cropped = img.crop((x1, y1, x2, y2))
                images.append(cropped)
                valid_bboxes.append((x1, y1, x2, y2))

        if not images:
            # Fallback: return original image
            return SplitResult(
                images=[img],
                bboxes=[(0, 0, width, height)],
                collage_info=collage_info,
            )

        return SplitResult(
            images=images,
            bboxes=valid_bboxes,
            collage_info=collage_info,
        )

    def _split_grid(
        self,
        width: int,
        height: int,
        grid_size: Tuple[int, int]
    ) -> List[Tuple[int, int, int, int]]:
        """
        Grid splitting

        Args:
            width: Image width
            height: Image height
            grid_size: (rows, cols)

        Returns:
            List of bounding boxes
        """
        rows, cols = grid_size
        cell_width = width // cols
        cell_height = height // rows

        bboxes = []
        for row in range(rows):
            for col in range(cols):
                x1 = col * cell_width
                y1 = row * cell_height
                x2 = x1 + cell_width
                y2 = y1 + cell_height

                # Last column/row extends to boundary
                if col == cols - 1:
                    x2 = width
                if row == rows - 1:
                    y2 = height

                bboxes.append((x1, y1, x2, y2))

        return bboxes

    def smart_split(
        self,
        img: Image.Image,
        remove_duplicates: bool = True,
    ) -> SplitResult:
        """
        Intelligent splitting (combining multiple methods)

        Args:
            img: Input image
            remove_duplicates: Whether to remove duplicate/similar split results

        Returns:
            SplitResult Split result
        """
        # First detect
        collage_info = self.is_collage(img)

        if not collage_info.is_collage:
            return SplitResult(
                images=[img],
                bboxes=[(0, 0, img.width, img.height)],
                collage_info=collage_info,
            )

        # Try multiple splitting methods
        results = []

        # Method 1: Grid splitting (if grid information available)
        if collage_info.grid_size:
            grid_result = self.split_collage(img, collage_info)
            results.append(grid_result)

        # Method 2: Object detection splitting
        cv2_img = self._to_cv2(img)
        objects = self.detect_objects(cv2_img)

        if objects:
            obj_images = []
            obj_bboxes = []

            for bbox in objects:
                x1, y1, x2, y2 = bbox
                if x2 - x1 >= self.min_split_size and y2 - y1 >= self.min_split_size:
                    cropped = img.crop((x1, y1, x2, y2))
                    obj_images.append(cropped)
                    obj_bboxes.append(bbox)

            if obj_images:
                results.append(SplitResult(
                    images=obj_images,
                    bboxes=obj_bboxes,
                    collage_info=collage_info,
                ))

        # Select best result (method that produces most valid images)
        if not results:
            return SplitResult(
                images=[img],
                bboxes=[(0, 0, img.width, img.height)],
                collage_info=collage_info,
            )

        best_result = max(results, key=lambda r: len(r.images))

        # Remove duplicates
        if remove_duplicates and len(best_result.images) > 1:
            best_result = self._remove_duplicate_splits(best_result)

        return best_result

    def _remove_duplicate_splits(self, result: SplitResult) -> SplitResult:
        """Remove duplicate split results"""
        # Simple implementation: based on image size and color histogram comparison
        unique_images = []
        unique_bboxes = []

        for i, img in enumerate(result.images):
            is_duplicate = False

            for existing_img in unique_images:
                # Compare size
                if abs(img.size[0] - existing_img.size[0]) < 10 and \
                   abs(img.size[1] - existing_img.size[1]) < 10:
                    # Compare histogram
                    hist1 = img.histogram()
                    hist2 = existing_img.histogram()

                    diff = sum(abs(a - b) for a, b in zip(hist1, hist2))
                    if diff < len(hist1) * 10:  # Similarity threshold
                        is_duplicate = True
                        break

            if not is_duplicate:
                unique_images.append(img)
                unique_bboxes.append(result.bboxes[i])

        return SplitResult(
            images=unique_images,
            bboxes=unique_bboxes,
            collage_info=result.collage_info,
        )


# Usage example
