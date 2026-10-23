"""
Browse Service - Chooses the stored images offered on the Browse & Search page

Author: W Cheng
Co-Author: Claude (Anthropic AI Assistant)
Created: 2026-10-21
Project: Product Image Processing System
"""

import logging
import random
from dataclasses import dataclass
from typing import Callable, List, Optional

import numpy as np

from src.pipeline.config import CONFIG, GROUPING_SIMILARITY_THRESHOLD
from src.pipeline.milvus_db import MilvusVectorLiteDB, ProductGroupsDB
from src.pipeline.product_grouper import ProductGroupUUID, VectorDatabase
from src.pipeline.identifiers import PlainId

logger = logging.getLogger(__name__)

CANDIDATE_MULTIPLIER = 4


@dataclass
class BrowseImage:
    id: str
    image_url: str
    original_image_url: Optional[str]


@dataclass
class BrowseSample:
    images: List[BrowseImage]
    mode: str
    total_in_db: int
    ungrouped_count: Optional[int]


class BrowseService:
    def __init__(
        self,
        vector_db: VectorDatabase,
        list_groups: Callable[[], List[ProductGroupUUID]],
        rng: Optional[random.Random] = None,
    ):
        self._vector_db = vector_db
        self._list_groups = list_groups
        self._rng = rng or random.Random()

    def sample(self, count: int = 10) -> BrowseSample:
        all_ids = self._vector_db.get_all_ids()
        total_in_db = self._vector_db.size()
        live_groups = self._live_groups(set(all_ids))
        ungrouped_count = None
        if live_groups:
            covered_ids = {member_id for group in live_groups for member_id in group.member_ids}
            ungrouped_count = len(set(all_ids) - covered_ids)
            if ungrouped_count == 0:
                return BrowseSample(
                    images=self._best_image_per_group(live_groups, count),
                    mode="grouped",
                    total_in_db=total_in_db,
                    ungrouped_count=0,
                )
        return BrowseSample(
            images=self._random_images(all_ids, count),
            mode="random",
            total_in_db=total_in_db,
            ungrouped_count=ungrouped_count,
        )

    def _live_groups(self, live_ids: set) -> List[ProductGroupUUID]:
        try:
            groups = self._list_groups()
        except Exception as e:
            logger.warning(f"Could not read product groups, sampling at random: {e}")
            return []
        live_groups = []
        for group in groups:
            live_members = [member_id for member_id in group.member_ids if member_id in live_ids]
            if live_members:
                live_groups.append(ProductGroupUUID(
                    group_id=group.group_id,
                    member_ids=live_members,
                    representative_id=group.representative_id,
                ))
        return live_groups

    def _best_image_per_group(self, groups: List[ProductGroupUUID], count: int) -> List[BrowseImage]:
        shuffled_groups = self._rng.sample(groups, len(groups))
        images: List[BrowseImage] = []
        for group in shuffled_groups:
            image = self._best_image(group)
            if image is not None:
                images.append(image)
            if len(images) == count:
                break
        return images

    def _best_image(self, group: ProductGroupUUID) -> Optional[BrowseImage]:
        records = [record for record in map(self._vector_db.get, group.member_ids) if record is not None]
        scored = sorted((r for r in records if self._quality_score(r) is not None), key=self._quality_score, reverse=True)
        unscored = [r for r in records if self._quality_score(r) is None]
        unscored.sort(key=lambda r: r.id != group.representative_id)
        for record in scored + unscored:
            image = self._to_image(record)
            if image is not None:
                return image
        return None

    @staticmethod
    def _quality_score(record) -> Optional[float]:
        metadata = record.metadata or {}
        overall = metadata.get("overall")
        return overall if overall is not None else metadata.get("quality_score")

    def _random_images(self, all_ids: List[PlainId], count: int) -> List[BrowseImage]:
        candidate_ids = self._rng.sample(all_ids, min(len(all_ids), CANDIDATE_MULTIPLIER * count))
        kept_images: List[BrowseImage] = []
        kept_vectors: List[np.ndarray] = []
        for candidate_id in candidate_ids:
            record = self._vector_db.get(candidate_id)
            image = self._to_image(record)
            if image is None:
                continue
            vector = self._unit_vector(record.embedding)
            if any(float(np.dot(vector, other)) >= GROUPING_SIMILARITY_THRESHOLD for other in kept_vectors):
                continue
            kept_images.append(image)
            kept_vectors.append(vector)
            if len(kept_images) == count:
                break
        return kept_images

    @staticmethod
    def _to_image(record) -> Optional[BrowseImage]:
        metadata = (record.metadata if record else None) or {}
        image_url = metadata.get("image_url")
        if not image_url:
            return None
        return BrowseImage(
            id=record.id,
            image_url=image_url,
            original_image_url=metadata.get("original_image_url"),
        )

    @staticmethod
    def _unit_vector(embedding: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(embedding)
        return embedding / norm if norm > 0 else embedding


def build_browse_service(pipeline) -> BrowseService:
    vector_db = pipeline.vector_db
    if isinstance(vector_db, MilvusVectorLiteDB):
        groups_db = ProductGroupsDB(db_path=CONFIG.get("milvus_lite_path", "./data/milvus.db"))
        return BrowseService(vector_db, list_groups=groups_db.get_all_groups_if_present)
    return BrowseService(vector_db, list_groups=list)
