"""
Product Grouping Module

For grouping similar SKUs and searching for similar products.

Author: W Cheng
Co-Author: Claude (Anthropic AI Assistant)
Created: 2026-01-07
Project: Product Image Processing System
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from .clustering import dbscan_cluster
from .identifiers import PlainId

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Search result"""
    id: PlainId                 # ID
    score: float                # Similarity score
    embedding: np.ndarray       # Embedding vector
    metadata: Optional[dict]    # Other metadata

    def __post_init__(self):
        self.id = PlainId(self.id)


@dataclass
class ProductGroup:
    """Product group (legacy, uses int group_id)"""
    group_id: int               # Group ID
    ids: List[PlainId]          # IDs in group
    representative_id: PlainId  # Representative ID
    size: int                   # Number in group


@dataclass
class ProductGroupUUID:
    """Product group with UUID identifier"""
    group_id: PlainId                       # UUID string
    member_ids: List[PlainId] = field(default_factory=list)  # Product IDs in group
    representative_id: Optional[PlainId] = None  # Representative product ID
    created_at: Optional[str] = None        # Creation timestamp

    def __post_init__(self):
        self.group_id = PlainId(self.group_id)
        self.member_ids = [PlainId(member_id) for member_id in self.member_ids]
        if self.representative_id is not None:
            self.representative_id = PlainId(self.representative_id)

    @property
    def size(self) -> int:
        return len(self.member_ids)

    def add_member(self, product_id: PlainId):
        if product_id not in self.member_ids:
            self.member_ids.append(PlainId(product_id))

    def to_dict(self) -> dict:
        return {
            "group_id": self.group_id.plainId,
            "member_ids": [member_id.plainId for member_id in self.member_ids],
            "representative_id": self.representative_id.plainId if self.representative_id is not None else None,
            "created_at": self.created_at,
            "size": self.size,
        }

class VectorDatabase:
    """Vector database interface (Abstract class)"""

    def insert(self, id: PlainId, embedding: np.ndarray, metadata: Optional[dict] = None):
        raise NotImplementedError

    def search(self, embedding: np.ndarray, top_k: int = 10) -> List[SearchResult]:
        raise NotImplementedError

    def delete(self, id: PlainId):
        raise NotImplementedError

    def delete_all(self):
        raise NotImplementedError

    def get(self, id: PlainId) -> Optional[SearchResult]:
        raise NotImplementedError

    def get_all_ids(self) -> List[PlainId]:
        raise NotImplementedError


class InMemoryVectorDB(VectorDatabase):
    """In-memory vector database (for testing and small-scale applications)"""

    def __init__(self):
        self.embeddings: Dict[str, np.ndarray] = {}
        self.metadata: Dict[str, dict] = {}

    def insert(self, id: PlainId, embedding: np.ndarray, metadata: Optional[dict] = None):
        embedding = embedding / np.linalg.norm(embedding)
        self.embeddings[id.plainId] = embedding
        self.metadata[id.plainId] = metadata or {}
        logger.info(f"InMemory: Inserted ID {id}, dimension: {len(embedding)}, total: {self.size()}")

    def search(self, embedding: np.ndarray, top_k: int = 10, threshold: float = 0.0) -> List[SearchResult]:
        if not self.embeddings:
            return []

        embedding = embedding / np.linalg.norm(embedding)
        ids = list(self.embeddings.keys())
        vectors = np.array(list(self.embeddings.values()))
        similarities = np.dot(vectors, embedding)
        indices = np.argsort(similarities)[::-1]

        results = []
        for idx in indices[:top_k]:
            score = float(similarities[idx])
            if score < threshold:
                break
            id = ids[idx]
            results.append(SearchResult(
                id=id,
                score=score,
                embedding=self.embeddings[id],
                metadata=self.metadata.get(id),
            ))

        logger.info(f"InMemory search: found {len(results)} results above threshold {threshold}")
        return results

    def delete(self, id: PlainId):
        self.embeddings.pop(id.plainId, None)
        self.metadata.pop(id.plainId, None)

    def delete_all(self):
        self.embeddings.clear()
        self.metadata.clear()
        logger.info("InMemory: Deleted all vectors")

    def get(self, id: PlainId) -> Optional[SearchResult]:
        if id.plainId not in self.embeddings:
            return None
        return SearchResult(
            id=id,
            score=1.0,
            embedding=self.embeddings[id.plainId],
            metadata=self.metadata.get(id.plainId),
        )

    def get_all_ids(self) -> List[PlainId]:
        return [PlainId(stored_id) for stored_id in self.embeddings]

    def size(self) -> int:
        return len(self.embeddings)

    def get_all_embeddings(self) -> Tuple[List[PlainId], np.ndarray]:
        ids = [PlainId(stored_id) for stored_id in self.embeddings]
        vectors = np.array(list(self.embeddings.values()))
        return ids, vectors


class ProductGrouper:
    def __init__(self, vector_db: VectorDatabase, similarity_threshold: float = 0.85):
        self.db = vector_db
        self.threshold = similarity_threshold

    def find_similar_products(
        self,
        embedding: np.ndarray,
        top_k: int = 20,
        threshold: Optional[float] = None,
    ) -> List[SearchResult]:
        th = threshold if threshold is not None else self.threshold
        return self.db.search(embedding, top_k=top_k, threshold=th)

    def is_duplicate(self, embedding: np.ndarray, threshold: float = 0.95) -> Tuple[bool, Optional[str]]:
        results = self.db.search(embedding, top_k=1, threshold=threshold)
        if results and results[0].score >= threshold:
            return True, results[0].id
        return False, None

    def group_skus_clustering(
        self,
        embeddings: Optional[np.ndarray] = None,
        sku_ids: Optional[List[str]] = None,
        min_cluster_size: int = 2,
    ) -> List[ProductGroup]:
        ids = [PlainId(sku_id) for sku_id in sku_ids] if sku_ids is not None else None
        if embeddings is None or ids is None:
            if hasattr(self.db, 'get_all_embeddings'):
                ids, embeddings = self.db.get_all_embeddings()
            else:
                raise ValueError("Must provide embeddings and ids")

        if len(embeddings) == 0:
            return []

        labels = dbscan_cluster(embeddings, self.threshold, min_cluster_size)

        groups_dict: Dict[int, List[str]] = {}
        for id, label in zip(ids, labels):
            if label == -1:
                continue
            if label not in groups_dict:
                groups_dict[label] = []
            groups_dict[label].append(id)

        groups = []
        for group_id, group_ids in groups_dict.items():
            if len(group_ids) < min_cluster_size:
                continue

            group_embeddings = np.array([self.db.get(id).embedding for id in group_ids])
            centroid = group_embeddings.mean(axis=0)
            centroid = centroid / np.linalg.norm(centroid)
            similarities = np.dot(group_embeddings, centroid)
            representative_idx = np.argmax(similarities)

            groups.append(ProductGroup(
                group_id=int(group_id),
                ids=group_ids,
                representative_id=group_ids[representative_idx],
                size=len(group_ids),
            ))

        groups.sort(key=lambda x: x.size, reverse=True)
        logger.info(f"Found {len(groups)} product groups")
        return groups

# Re-export Milvus classes — import deferred to bottom to avoid circular dependency
# (milvus_db.py imports SearchResult/VectorDatabase/ProductGroupUUID from this module)
