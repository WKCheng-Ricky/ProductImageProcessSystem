"""
Product Grouping Module

For grouping similar SKUs and searching for similar products.

Author: W Cheng
Co-Author: Claude (Anthropic AI Assistant)
Created: 2026-01-07
Project: Product Image Processing System
"""

import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from .clustering import connected_components_cluster, dbscan_cluster

logger = logging.getLogger(__name__)


@dataclass
class SearchResult:
    """Search result"""
    id: str                     # ID
    score: float                # Similarity score
    embedding: np.ndarray       # Embedding vector
    metadata: Optional[dict]    # Other metadata


@dataclass
class ProductGroup:
    """Product group (legacy, uses int group_id)"""
    group_id: int               # Group ID
    ids: List[str]              # IDs in group
    representative_id: str      # Representative ID
    size: int                   # Number in group


@dataclass
class ProductGroupUUID:
    """Product group with UUID identifier"""
    group_id: str                           # UUID string
    member_ids: List[str] = field(default_factory=list)  # Product IDs in group
    representative_id: Optional[str] = None  # Representative product ID
    created_at: Optional[str] = None        # Creation timestamp

    @property
    def size(self) -> int:
        return len(self.member_ids)

    def add_member(self, product_id: str):
        if product_id not in self.member_ids:
            self.member_ids.append(product_id)

    def remove_member(self, product_id: str):
        if product_id in self.member_ids:
            self.member_ids.remove(product_id)

    def to_dict(self) -> dict:
        return {
            "group_id": self.group_id,
            "member_ids": self.member_ids,
            "representative_id": self.representative_id,
            "created_at": self.created_at,
            "size": self.size,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ProductGroupUUID":
        return cls(
            group_id=data["group_id"],
            member_ids=data.get("member_ids", []),
            representative_id=data.get("representative_id"),
            created_at=data.get("created_at"),
        )


class VectorDatabase:
    """Vector database interface (Abstract class)"""

    def insert(self, id: str, embedding: np.ndarray, metadata: Optional[dict] = None):
        raise NotImplementedError

    def search(self, embedding: np.ndarray, top_k: int = 10) -> List[SearchResult]:
        raise NotImplementedError

    def delete(self, id: str):
        raise NotImplementedError

    def delete_all(self):
        raise NotImplementedError

    def get(self, id: str) -> Optional[SearchResult]:
        raise NotImplementedError


class InMemoryVectorDB(VectorDatabase):
    """In-memory vector database (for testing and small-scale applications)"""

    def __init__(self):
        self.embeddings: Dict[str, np.ndarray] = {}
        self.metadata: Dict[str, dict] = {}

    def insert(self, id: str, embedding: np.ndarray, metadata: Optional[dict] = None):
        embedding = embedding / np.linalg.norm(embedding)
        self.embeddings[id] = embedding
        self.metadata[id] = metadata or {}
        logger.info(f"InMemory: Inserted ID {id}, dimension: {len(embedding)}, total: {self.size()}")

    def insert_batch(
        self,
        ids: List[str],
        embeddings: np.ndarray,
        metadata_list: Optional[List[dict]] = None,
    ):
        if metadata_list is None:
            metadata_list = [{}] * len(ids)
        for id, embedding, meta in zip(ids, embeddings, metadata_list):
            self.insert(id, embedding, meta)

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

    def delete(self, id: str):
        self.embeddings.pop(id, None)
        self.metadata.pop(id, None)

    def delete_all(self):
        self.embeddings.clear()
        self.metadata.clear()
        logger.info("InMemory: Deleted all vectors")

    def get(self, id: str) -> Optional[SearchResult]:
        if id not in self.embeddings:
            return None
        return SearchResult(
            id=id,
            score=1.0,
            embedding=self.embeddings[id],
            metadata=self.metadata.get(id),
        )

    def size(self) -> int:
        return len(self.embeddings)

    def get_all_embeddings(self) -> Tuple[List[str], np.ndarray]:
        ids = list(self.embeddings.keys())
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
        ids = sku_ids
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

    def group_skus_connected_components(
        self,
        embeddings: Optional[np.ndarray] = None,
        sku_ids: Optional[List[str]] = None,
    ) -> List[ProductGroup]:
        if embeddings is None or sku_ids is None:
            if hasattr(self.db, 'get_all_embeddings'):
                sku_ids, embeddings = self.db.get_all_embeddings()
            else:
                raise ValueError("Must provide embeddings and sku_ids")

        n = len(embeddings)
        if n == 0:
            return []

        _, labels = connected_components_cluster(embeddings, self.threshold)

        groups_dict: Dict[int, List[str]] = {}
        for id, label in zip(sku_ids, labels):
            if label not in groups_dict:
                groups_dict[label] = []
            groups_dict[label].append(id)

        groups = []
        for group_id, group_sku_ids in groups_dict.items():
            if len(group_sku_ids) < 2:
                continue
            groups.append(ProductGroup(
                group_id=int(group_id),
                ids=group_sku_ids,
                representative_id=group_sku_ids[0],
                size=len(group_sku_ids),
            ))

        groups.sort(key=lambda x: x.size, reverse=True)
        return groups

    def merge_groups(self, groups: List[ProductGroup], merge_threshold: float = 0.9) -> List[ProductGroup]:
        if len(groups) <= 1:
            return groups

        group_embeddings = []
        embedding_dim = None
        for group in groups:
            rep_result = self.db.get(group.representative_id)
            if rep_result:
                group_embeddings.append(rep_result.embedding)
                if embedding_dim is None:
                    embedding_dim = len(rep_result.embedding)
            else:
                dim = embedding_dim if embedding_dim is not None else 512
                group_embeddings.append(np.zeros(dim))

        group_embeddings = np.array(group_embeddings)
        merged = [False] * len(groups)
        new_groups = []

        for i, group_i in enumerate(groups):
            if merged[i]:
                continue

            similar_groups = [group_i]
            merged[i] = True

            for j in range(i + 1, len(groups)):
                if merged[j]:
                    continue
                sim = np.dot(group_embeddings[i], group_embeddings[j])
                if sim >= merge_threshold:
                    similar_groups.append(groups[j])
                    merged[j] = True

            if len(similar_groups) == 1:
                new_groups.append(group_i)
            else:
                merged_ids = []
                for g in similar_groups:
                    merged_ids.extend(g.ids)
                new_groups.append(ProductGroup(
                    group_id=len(new_groups),
                    ids=merged_ids,
                    representative_id=similar_groups[0].representative_id,
                    size=len(merged_ids),
                ))

        return new_groups


# Re-export Milvus classes — import deferred to bottom to avoid circular dependency
# (milvus_db.py imports SearchResult/VectorDatabase/ProductGroupUUID from this module)
from .milvus_db import MilvusVectorLiteDB, ProductGroupManager, ProductGroupsDB  # noqa: E402
