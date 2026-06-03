import json
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .product_grouper import ProductGroupUUID, SearchResult, VectorDatabase

try:
    from pymilvus import MilvusClient, DataType
except ImportError:
    MilvusClient = None
    DataType = None

logger = logging.getLogger(__name__)


class MilvusVectorLiteDB(VectorDatabase):
    def __init__(
        self,
        collection_name: str = "product_embeddings",
        db_path: str = "./data/milvus.db",
        dimension: int = 768,
        model_name: str = None,
    ):
        self.collection_name = collection_name
        self.db_path = db_path
        self.dimension = dimension
        self.model_name = model_name
        self._client = None

    def _connect(self):
        if self._client is not None:
            return

        if MilvusClient is None:
            raise ImportError("Please install pymilvus with milvus-lite: pip install pymilvus[milvus-lite]")

        dir_str = os.path.dirname(self.db_path)
        Path(dir_str).mkdir(exist_ok=True)

        self._client = MilvusClient(self.db_path)
        logger.info(f"Connected to Milvus Lite: {self.db_path}")

        collections = self._client.list_collections()
        if self.collection_name not in collections:
            self._create_collection()
        else:
            self._verify_collection_compatibility()

        self._client.load_collection(self.collection_name)

    def _verify_collection_compatibility(self):
        try:
            schema = self._client.describe_collection(self.collection_name)

            existing_dim = None
            for field in schema.get('fields', []):
                if field.get('type') == 101:
                    params = field.get('params', {})
                    existing_dim = params.get('dim')
                    break

            collection_desc = schema.get('description', '')
            stored_model = None
            if collection_desc.startswith('model:'):
                stored_model = collection_desc.replace('model:', '').strip()

            if existing_dim and existing_dim != self.dimension:
                error_msg = (
                    f"\n{'='*70}\n"
                    f"ERROR: Embedding Model/Dimension Mismatch Detected\n"
                    f"{'='*70}\n"
                    f"Existing collection: {self.collection_name}\n"
                    f"  - Dimension: {existing_dim}\n"
                    f"  - Model: {stored_model or 'unknown'}\n"
                    f"\n"
                    f"Current configuration:\n"
                    f"  - Dimension: {self.dimension}\n"
                    f"  - Model: {self.model_name or 'unknown'}\n"
                    f"\n"
                    f"REASON FOR ERROR:\n"
                    f"You cannot mix embeddings from different models in the same collection.\n"
                    f"Embeddings from different models live in incompatible semantic spaces.\n"
                    f"\n"
                    f"TO FIX THIS:\n"
                    f"Option 1: Revert to original model\n"
                    f"  - Change .env back to: {stored_model or 'original model'}\n"
                    f"  - Or set CLIP_MODEL and CLIP_PRETRAINED to match existing collection\n"
                    f"\n"
                    f"Option 2: Delete old embeddings and regenerate (DATA LOSS!)\n"
                    f"  - Delete: rm {self.db_path}\n"
                    f"  - Restart application\n"
                    f"  - Re-process all images with new model\n"
                    f"\n"
                    f"Option 3: Implement migration (requires custom code)\n"
                    f"  - See OVERVIEW.md 'Model Migration' section\n"
                    f"  - Requires re-embedding all existing images\n"
                    f"{'='*70}\n"
                )
                logger.error(error_msg)
                raise RuntimeError(error_msg)

            if stored_model and self.model_name and stored_model != self.model_name:
                logger.warning(
                    f"Model name changed: '{stored_model}' -> '{self.model_name}' "
                    f"(dimensions match: {self.dimension}). Proceeding with caution."
                )

            logger.info(f"Loaded existing collection: {self.collection_name} (model: {stored_model or 'unknown'}, dim: {existing_dim})")

        except RuntimeError:
            raise
        except Exception as e:
            logger.warning(f"Could not verify collection compatibility: {e}")
            logger.info(f"Loaded existing collection: {self.collection_name}")

    def _create_collection(self):
        description = f"model:{self.model_name}" if self.model_name else ""

        schema = self._client.create_schema(
            auto_id=False,
            enable_dynamic_field=False,
            description=description,
        )
        schema.add_field(field_name="id", datatype=DataType.VARCHAR, max_length=100, is_primary=True)
        schema.add_field(field_name="vector", datatype=DataType.FLOAT_VECTOR, dim=self.dimension)
        schema.add_field(field_name="metadata_json", datatype=DataType.VARCHAR, max_length=2000)

        index_params = self._client.prepare_index_params()
        index_params.add_index(
            field_name="vector",
            metric_type="IP",
            index_type="FLAT",
        )

        self._client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
            index_params=index_params,
        )
        logger.info(f"Created new collection: {self.collection_name} (model: {self.model_name}, dim: {self.dimension})")

    def insert(self, id: str, embedding: np.ndarray, metadata: Optional[dict] = None):
        self._connect()
        embedding = embedding / np.linalg.norm(embedding)
        metadata_json = json.dumps(metadata) if metadata else "{}"
        data = {
            "id": id,
            "vector": embedding.tolist(),
            "metadata_json": metadata_json,
        }
        self._client.insert(collection_name=self.collection_name, data=[data])
        logger.info(f"Milvus: Inserted ID {id}, dimension: {len(embedding)}")

    def search(self, embedding: np.ndarray, top_k: int = 10, threshold: float = 0.0) -> List[SearchResult]:
        self._connect()
        embedding = embedding / np.linalg.norm(embedding)
        results = self._client.search(
            collection_name=self.collection_name,
            data=[embedding.tolist()],
            limit=top_k,
            output_fields=["id", "metadata_json"],
        )

        raw_count = len(results[0]) if results else 0
        logger.info(f"Milvus search: found {raw_count} raw results")

        search_results = []
        for hits in results:
            for hit in hits:
                if hit['distance'] >= threshold:
                    metadata = None
                    metadata_json = hit.get('entity', {}).get('metadata_json')
                    if metadata_json:
                        try:
                            metadata = json.loads(metadata_json)
                        except json.JSONDecodeError:
                            pass
                    search_results.append(SearchResult(
                        id=hit['id'],
                        score=hit['distance'],
                        embedding=np.array([]),
                        metadata=metadata,
                    ))

        logger.info(f"Milvus search: {len(search_results)} results above threshold {threshold}")
        return search_results

    def delete(self, id: str):
        self._connect()
        self._client.delete(collection_name=self.collection_name, filter=f'id == "{id}"')

    def delete_all(self):
        self._connect()
        self._client.drop_collection(collection_name=self.collection_name)
        self._create_collection()
        logger.info(f"Milvus: Deleted all vectors from {self.collection_name}")

    def get(self, id: str) -> Optional[SearchResult]:
        self._connect()
        results = self._client.query(
            collection_name=self.collection_name,
            filter=f'id == "{id}"',
            output_fields=["id", "vector", "metadata_json"],
        )

        if not results:
            return None

        metadata = None
        metadata_json = results[0].get("metadata_json")
        if metadata_json:
            try:
                metadata = json.loads(metadata_json)
            except json.JSONDecodeError:
                pass

        return SearchResult(
            id=results[0]["id"],
            score=1.0,
            embedding=np.array(results[0]["vector"]),
            metadata=metadata,
        )

    def size(self) -> int:
        self._connect()
        stats = self._client.get_collection_stats(collection_name=self.collection_name)
        return stats['row_count']

    def get_all_ids(self) -> List[str]:
        self._connect()
        all_ids = []
        offset = 0
        batch_size = 16000

        while True:
            results = self._client.query(
                collection_name=self.collection_name,
                filter="",
                output_fields=["id"],
                limit=batch_size,
                offset=offset,
            )
            if not results:
                break
            all_ids.extend([r["id"] for r in results])
            if len(results) < batch_size:
                break
            offset += batch_size

        return all_ids

    def get_all_embeddings(self) -> Tuple[List[str], np.ndarray]:
        self._connect()
        all_ids = []
        all_vectors = []
        offset = 0
        batch_size = 16000

        while True:
            results = self._client.query(
                collection_name=self.collection_name,
                filter="",
                output_fields=["id", "vector"],
                limit=batch_size,
                offset=offset,
            )
            if not results:
                break
            all_ids.extend([r["id"] for r in results])
            all_vectors.extend([r["vector"] for r in results])
            if len(results) < batch_size:
                break
            offset += batch_size

        if not all_ids:
            return [], np.array([])
        return all_ids, np.array(all_vectors)


class ProductGroupsDB:
    """
    Database for storing product groups using Milvus Lite

    Stores groups as documents with UUID as primary key and member_ids as JSON array
    """

    def __init__(
        self,
        collection_name: str = "product_groups",
        db_path: str = "./data/milvus.db",
    ):
        self.collection_name = collection_name
        self.db_path = db_path
        self._client = None

    def _connect(self):
        if self._client is not None:
            return

        if MilvusClient is None:
            raise ImportError("Please install pymilvus: pip install pymilvus[milvus-lite]")

        Path(self.db_path).parent.mkdir(exist_ok=True)
        self._client = MilvusClient(self.db_path)
        logger.info(f"ProductGroupsDB connected to: {self.db_path}")

        collections = self._client.list_collections()
        if self.collection_name not in collections:
            self._create_collection()
        else:
            logger.info(f"Using existing groups collection: {self.collection_name}")

    def _create_collection(self):
        schema = self._client.create_schema(auto_id=False, enable_dynamic_field=False)
        schema.add_field(field_name="group_id", datatype=DataType.VARCHAR, max_length=50, is_primary=True)
        schema.add_field(field_name="member_ids_json", datatype=DataType.VARCHAR, max_length=65535)
        schema.add_field(field_name="representative_id", datatype=DataType.VARCHAR, max_length=100)
        schema.add_field(field_name="created_at", datatype=DataType.VARCHAR, max_length=30)

        self._client.create_collection(
            collection_name=self.collection_name,
            schema=schema,
        )
        logger.info(f"Created groups collection: {self.collection_name}")

    def insert_group(self, group: ProductGroupUUID):
        self._connect()

        if group.created_at is None:
            group.created_at = datetime.now().isoformat()

        data = {
            "group_id": group.group_id,
            "member_ids_json": json.dumps(group.member_ids),
            "representative_id": group.representative_id or "",
            "created_at": group.created_at,
        }
        self._client.insert(collection_name=self.collection_name, data=[data])
        logger.info(f"Inserted group {group.group_id} with {group.size} members")

    def get_group(self, group_id: str) -> Optional[ProductGroupUUID]:
        self._connect()
        results = self._client.query(
            collection_name=self.collection_name,
            filter=f'group_id == "{group_id}"',
            output_fields=["group_id", "member_ids_json", "representative_id", "created_at"],
        )
        if not results:
            return None

        r = results[0]
        return ProductGroupUUID(
            group_id=r["group_id"],
            member_ids=json.loads(r["member_ids_json"]),
            representative_id=r["representative_id"] if r["representative_id"] else None,
            created_at=r["created_at"],
        )

    def update_group(self, group: ProductGroupUUID):
        self._connect()
        self._client.delete(
            collection_name=self.collection_name,
            filter=f'group_id == "{group.group_id}"',
        )
        self.insert_group(group)
        logger.info(f"Updated group {group.group_id}")

    def delete_group(self, group_id: str):
        self._connect()
        self._client.delete(
            collection_name=self.collection_name,
            filter=f'group_id == "{group_id}"',
        )
        logger.info(f"Deleted group {group_id}")

    def get_all_groups(self) -> List[ProductGroupUUID]:
        self._connect()
        groups = []
        offset = 0
        batch_size = 16000

        while True:
            results = self._client.query(
                collection_name=self.collection_name,
                filter="",
                output_fields=["group_id", "member_ids_json", "representative_id", "created_at"],
                limit=batch_size,
                offset=offset,
            )
            if not results:
                break
            for r in results:
                groups.append(ProductGroupUUID(
                    group_id=r["group_id"],
                    member_ids=json.loads(r["member_ids_json"]),
                    representative_id=r["representative_id"] if r["representative_id"] else None,
                    created_at=r["created_at"],
                ))
            if len(results) < batch_size:
                break
            offset += batch_size

        return groups

    def find_group_by_product(self, product_id: str) -> Optional[ProductGroupUUID]:
        groups = self.get_all_groups()
        for group in groups:
            if product_id in group.member_ids:
                return group
        return None

    def size(self) -> int:
        self._connect()
        stats = self._client.get_collection_stats(collection_name=self.collection_name)
        return stats['row_count']

    def clear_all_groups(self):
        self._connect()
        self._client.drop_collection(collection_name=self.collection_name)
        self._create_collection()
        logger.info("Cleared all groups")


class ProductGroupManager:
    """
    Manages automatic grouping of products based on embedding similarity

    Features:
    - Groups products with >90% similarity (configurable)
    - Uses UUID for group identification
    - Supports group expansion and retrieval
    """

    def __init__(
        self,
        vector_db: VectorDatabase,
        groups_db: Optional[ProductGroupsDB] = None,
        similarity_threshold: float = 0.90,
        top_k: int = 10,
        db_path: str = "./data/milvus.db",
    ):
        self.vector_db = vector_db
        self.groups_db = groups_db or ProductGroupsDB(db_path=db_path)
        self.similarity_threshold = similarity_threshold
        self.top_k = top_k

    def group_all_products(self, clear_existing: bool = True) -> List[ProductGroupUUID]:
        """
        Loop through all products in database and create similarity groups

        Algorithm:
        1. Get all product IDs from vector DB
        2. For each unassigned product:
           - Search for similar products (>90% similarity, top-k=10)
           - Create a new group with UUID
           - Mark all similar products as assigned
        3. Store groups in groups DB
        """
        logger.info("Starting product grouping...")

        if clear_existing:
            self.groups_db.clear_all_groups()

        if not hasattr(self.vector_db, 'get_all_ids'):
            raise ValueError("Vector database must support get_all_ids() method")

        all_ids = self.vector_db.get_all_ids()
        if not all_ids:
            logger.info("No products in database to group")
            return []

        logger.info(f"Found {len(all_ids)} products to group")

        assigned: set = set()
        created_groups: List[ProductGroupUUID] = []

        for product_id in all_ids:
            if product_id in assigned:
                continue

            product_data = self.vector_db.get(product_id)
            if product_data is None:
                continue

            similar_results = self.vector_db.search(
                embedding=product_data.embedding,
                top_k=self.top_k,
                threshold=self.similarity_threshold,
            )

            group_members = []
            for result in similar_results:
                if result.id not in assigned:
                    group_members.append(result.id)
                    assigned.add(result.id)

            if not group_members:
                group_members = [product_id]
                assigned.add(product_id)

            if len(group_members) >= 1:
                group = ProductGroupUUID(
                    group_id=str(uuid.uuid4()),
                    member_ids=group_members,
                    representative_id=group_members[0],
                    created_at=datetime.now().isoformat(),
                )
                self.groups_db.insert_group(group)
                created_groups.append(group)

                if len(group_members) > 1:
                    logger.info(f"Created group {group.group_id[:8]}... with {len(group_members)} members")

        logger.info(f"Grouping complete: {len(created_groups)} groups created")
        return created_groups

    def expand_group(self, group_id: str) -> Optional[ProductGroupUUID]:
        group = self.groups_db.get_group(group_id)
        if group is None:
            logger.warning(f"Group {group_id} not found")
            return None

        if not group.representative_id:
            logger.warning(f"Group {group_id} has no representative")
            return group

        rep_data = self.vector_db.get(group.representative_id)
        if rep_data is None:
            logger.warning(f"Representative {group.representative_id} not found in vector DB")
            return group

        similar_results = self.vector_db.search(
            embedding=rep_data.embedding,
            top_k=self.top_k * 2,
            threshold=self.similarity_threshold,
        )

        new_members_added = 0
        for result in similar_results:
            if result.id not in group.member_ids:
                existing_group = self.groups_db.find_group_by_product(result.id)
                if existing_group is None:
                    group.add_member(result.id)
                    new_members_added += 1

        if new_members_added > 0:
            self.groups_db.update_group(group)
            logger.info(f"Expanded group {group_id[:8]}... with {new_members_added} new members")

        return group

    def get_group(self, group_id: str) -> Optional[ProductGroupUUID]:
        return self.groups_db.get_group(group_id)

    def get_all_groups(self) -> List[ProductGroupUUID]:
        return self.groups_db.get_all_groups()

    def get_group_for_product(self, product_id: str) -> Optional[ProductGroupUUID]:
        return self.groups_db.find_group_by_product(product_id)

    def get_group_statistics(self) -> Dict[str, Any]:
        groups = self.get_all_groups()
        if not groups:
            return {
                "total_groups": 0,
                "total_products_grouped": 0,
                "avg_group_size": 0,
                "max_group_size": 0,
                "min_group_size": 0,
                "groups_with_multiple_members": 0,
            }

        sizes = [g.size for g in groups]
        multi_member_groups = [g for g in groups if g.size > 1]

        return {
            "total_groups": len(groups),
            "total_products_grouped": sum(sizes),
            "avg_group_size": sum(sizes) / len(sizes),
            "max_group_size": max(sizes),
            "min_group_size": min(sizes),
            "groups_with_multiple_members": len(multi_member_groups),
        }
