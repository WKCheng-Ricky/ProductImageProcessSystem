from pydantic import BaseModel, Field
from typing import Optional, List, Tuple, Dict, Any

from src.pipeline.identifiers import PlainId


class CollageInfoResponse(BaseModel):
    is_collage: bool
    num_images: int
    layout: str
    grid_size: Optional[Tuple[int, int]] = None
    confidence: float


class UploadResponse(BaseModel):
    session_id: str
    collage_info: CollageInfoResponse
    original_image_url: str
    watermark_skipped: bool = False
    collage_skipped: bool = False


class WatermarkRegionInfo(BaseModel):
    bbox: Tuple[int, int, int, int]
    text: Optional[str] = None
    confidence: float
    type: str


class WatermarkRemovalResponse(BaseModel):
    watermarks_detected: int
    watermarks_removed: int
    success: bool
    original_image_url: str
    watermark_removed_image_url: Optional[str] = None
    regions: List[WatermarkRegionInfo] = []


class CollageSplitResultResponse(BaseModel):
    num_images: int
    split_image_urls: List[str]
    bboxes: List[Tuple[int, int, int, int]]


class PreprocessingResultResponse(BaseModel):
    processed_image_url: str
    original_size: Tuple[int, int]
    background_removed: bool
    watermark_removed: bool
    text_removed: bool
    product_bbox: Optional[Tuple[int, int, int, int]] = None


class QualityScoreResponse(BaseModel):
    overall: float
    sharpness: float
    brightness: float
    contrast: float
    resolution_score: float
    product_ratio: float
    has_watermark: bool
    has_text_overlay: bool
    aesthetic: float = 0.0
    # Extended aesthetic fields
    clarity_score: float = 0.0
    clarity_score_est: float = 0.0
    composition_score: float = 0.0
    composition_score_est: float = 0.0
    defect_score: float = 0.0
    defect_score_est: float = 0.0
    aesthetic_overall: float = 0.0
    original_image_url: Optional[str] = None
    file_extension: Optional[str] = None
    resolution: Optional[Tuple[int, int]] = None

    @property
    def quality_level(self) -> str:
        """Get quality level classification"""
        if self.overall >= 0.7:
            return "excellent"
        elif self.overall >= 0.4:
            return "acceptable"
        else:
            return "poor"

    @property
    def overall_percentage(self) -> int:
        """Get overall score as percentage"""
        return int(self.overall * 100)


class EmbeddingResultResponse(BaseModel):
    model_name: str
    dimension: int
    normalized: bool
    generation_time: float
    id: Optional[PlainId] = None
    saved_to_db: bool = False
    is_duplicate: bool = False
    duplicate_of: Optional[PlainId] = None
    duplicate_score: Optional[float] = None


class SearchResultItem(BaseModel):
    """Single search result item"""
    id: PlainId
    score: float
    image_url: Optional[str] = None
    original_image_url: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    @property
    def confidence_level(self) -> str:
        """Get confidence level classification"""
        if self.score >= 0.9:
            return "high"
        elif self.score >= 0.7:
            return "moderate"
        else:
            return "low"

    @property
    def score_percentage(self) -> int:
        """Get score as percentage"""
        return int(self.score * 100)


class SearchResultsResponse(BaseModel):
    """Similar product search results"""
    results: List[SearchResultItem]
    total_count: int
    query_image_url: str


class BatchImageResult(BaseModel):
    """Result for a single image in batch processing"""
    filename: str
    id: PlainId
    success: bool
    error: Optional[str] = None
    overall_score: Optional[float] = None
    aesthetic_score: Optional[float] = None
    saved_to_db: bool = False
    is_duplicate: bool = False
    duplicate_of: Optional[PlainId] = None
    duplicate_score: Optional[float] = None
    image_url: Optional[str] = None
    original_image_url: Optional[str] = None


class BatchUploadResponse(BaseModel):
    """Response for batch upload initiation"""
    batch_id: str
    total_images: int
    message: str


class BatchProcessResponse(BaseModel):
    """Response for batch processing results"""
    batch_id: str
    total_images: int
    successful: int
    failed: int
    skipped_duplicates: int = 0
    results: List[BatchImageResult]


class GroupMemberItem(BaseModel):
    """Single product in a group"""
    id: PlainId
    image_url: Optional[str] = None
    original_image_url: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None


class ProductGroupResult(BaseModel):
    """Product group with members"""
    group_id: PlainId
    member_count: int
    representative_id: Optional[PlainId] = None
    members: List[GroupMemberItem] = []


class GroupingStatsResponse(BaseModel):
    """Statistics about product grouping"""
    total_products_in_db: int
    total_groups: int
    total_products_grouped: int
    groups_with_multiple_members: int
    avg_group_size: float
    max_group_size: int
    min_group_size: int


class GroupingResultsResponse(BaseModel):
    """All product groups response"""
    total_groups: int
    total_products: int
    groups_with_multiple_members: int
    groups: List[ProductGroupResult]


class TextSearchRequest(BaseModel):
    """Text search request"""
    query: str = Field(..., min_length=1, max_length=500, description="Text description to search for")
    top_k: int = Field(default=10, ge=1, le=100, description="Number of results to return")
    threshold: float = Field(default=0.20, ge=0.0, le=1.0, description="Minimum similarity threshold")


class TextSearchResultsResponse(BaseModel):
    """Text search results response"""
    results: List[SearchResultItem]
    total_count: int
    query_text: str


class BrowseImageItem(BaseModel):
    """A stored image offered on the Browse & Search page"""
    id: PlainId
    image_url: str
    original_image_url: Optional[str] = None


class RandomImagesResponse(BaseModel):
    """Images sampled for the Browse & Search page"""
    images: List[BrowseImageItem]
    mode: str
    total_in_db: int
    ungrouped_count: Optional[int] = None
