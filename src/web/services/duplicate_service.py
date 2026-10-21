"""
Duplicate Service for Milvus Database

Handles duplication of existing records in Milvus database
with backup functionality.
"""

import logging
import os
import shutil
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

from src.pipeline.config import CONFIG
from src.pipeline.milvus_db import MilvusVectorLiteDB

from .pipeline_service import get_pipeline

logger = logging.getLogger(__name__)


def _path_size_bytes(path: str) -> int:
    if os.path.isfile(path):
        return os.path.getsize(path)
    return sum(
        os.path.getsize(os.path.join(root, name))
        for root, _, names in os.walk(path)
        for name in names
    )


def active_milvus_db() -> Optional[MilvusVectorLiteDB]:
    """The pipeline's own Milvus handle, so dimension and model checks match and the file is not opened twice."""
    vector_db = get_pipeline().vector_db
    return vector_db if isinstance(vector_db, MilvusVectorLiteDB) else None


def backup_database() -> str:
    """
    Backup the existing Milvus database file with timestamp.

    Returns:
        Path to the backup file
    """
    db_path = CONFIG.get("milvus_lite_path", "./data/milvus.db")

    if not os.path.exists(db_path):
        raise FileNotFoundError(f"Database file not found: {db_path}")

    timestamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    backup_path = db_path.replace(".db", f"_{timestamp}.bak")

    if os.path.isdir(db_path):
        shutil.copytree(db_path, backup_path)
    else:
        shutil.copy2(db_path, backup_path)
    logger.info(f"Database backed up to: {backup_path}")

    return backup_path


def duplicate_records(n: int) -> Dict[str, Any]:
    """
    Duplicate all existing records in Milvus database to n copies.

    Args:
        n: Number of times to duplicate records (total copies = n)

    Returns:
        Dictionary with duplication results
    """
    if n < 2:
        raise ValueError("n must be at least 2 (original + 1 copy)")

    vector_db = active_milvus_db()
    if vector_db is None:
        raise ValueError("Duplicating records needs the Milvus vector database (set USE_MILVUS=true)")

    original_size = vector_db.size()
    if original_size == 0:
        return {
            "success": True,
            "message": "No records to duplicate",
            "original_count": 0,
            "new_count": 0,
            "duplicates_added": 0,
        }

    logger.info(f"Found {original_size} records to duplicate")

    all_ids, all_vectors = vector_db.get_all_embeddings()

    all_metadata = []
    for id in all_ids:
        result = vector_db.get(id)
        all_metadata.append(result.metadata if result else {})

    duplicates_added = 0
    for copy_num in range(1, n):
        for i, (orig_id, vector, metadata) in enumerate(zip(all_ids, all_vectors, all_metadata)):
            new_id = f"{orig_id}_dup{copy_num}_{uuid.uuid4().hex[:8]}"

            new_metadata = metadata.copy() if metadata else {}
            new_metadata["original_id"] = orig_id
            new_metadata["duplicate_number"] = copy_num

            vector_db.insert(new_id, np.array(vector), new_metadata)
            duplicates_added += 1

        logger.info(f"Completed copy {copy_num}/{n-1}")

    new_size = vector_db.size()

    return {
        "success": True,
        "message": "Successfully duplicated records",
        "original_count": original_size,
        "new_count": new_size,
        "duplicates_added": duplicates_added,
        "multiplication_factor": n,
    }


def backup_and_duplicate(n: int) -> Dict[str, Any]:
    """
    Backup database and then duplicate records.

    Args:
        n: Number of times to duplicate records

    Returns:
        Dictionary with backup path and duplication results
    """
    if n < 2:
        raise ValueError("n must be at least 2 (original + 1 copy)")
    if active_milvus_db() is None:
        raise ValueError("Duplicating records needs the Milvus vector database (set USE_MILVUS=true)")

    backup_path = backup_database()

    result = duplicate_records(n)
    result["backup_path"] = backup_path

    return result


def get_database_stats() -> Dict[str, Any]:
    """Get current database statistics."""
    db_path = CONFIG.get("milvus_lite_path", "./data/milvus.db")
    collection_name = CONFIG.get("milvus_collection", "product_embeddings")

    vector_db = active_milvus_db()
    if vector_db is None or not os.path.exists(db_path):
        return {
            "exists": False,
            "record_count": 0,
            "db_path": db_path,
        }

    return {
        "exists": True,
        "record_count": vector_db.size(),
        "db_path": db_path,
        "collection_name": collection_name,
    }


def list_backups() -> List[Dict[str, Any]]:
    """List all backup files."""
    db_path = CONFIG.get("milvus_lite_path", "./data/milvus.db")
    db_dir = os.path.dirname(db_path)
    db_name = os.path.basename(db_path).replace(".db", "")

    backups = []
    if os.path.exists(db_dir):
        for file in os.listdir(db_dir):
            if file.startswith(db_name) and file.endswith(".bak"):
                file_path = os.path.join(db_dir, file)
                stat = os.stat(file_path)
                backups.append({
                    "filename": file,
                    "path": file_path,
                    "size_mb": round(_path_size_bytes(file_path) / (1024 * 1024), 2),
                    "created_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
                })

    backups.sort(key=lambda x: x["created_at"], reverse=True)
    return backups
