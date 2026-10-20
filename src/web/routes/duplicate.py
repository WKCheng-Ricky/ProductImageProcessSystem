"""
Duplicate API Routes

Handles database record duplication with backup.
"""

import logging
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from ..services.duplicate_service import (
    backup_and_duplicate,
    get_database_stats,
    list_backups,
)

logger = logging.getLogger(__name__)
router = APIRouter()


class DuplicateRequest(BaseModel):
    n: int = Field(..., ge=2, le=100, description="Number of total copies (2-100)")


class DuplicateResponse(BaseModel):
    success: bool
    message: str
    original_count: int
    new_count: int
    duplicates_added: int
    multiplication_factor: int
    backup_path: str


@router.get("/duplicate/stats")
def get_stats():
    """Get current database statistics."""
    try:
        stats = get_database_stats()
        return stats
    except Exception as e:
        logger.error(f"Error getting database stats: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.get("/duplicate/backups")
def get_backups():
    """List all backup files."""
    try:
        backups = list_backups()
        return {"backups": backups}
    except Exception as e:
        logger.error(f"Error listing backups: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")


@router.post("/duplicate", response_model=DuplicateResponse)
def duplicate_database(request: DuplicateRequest):
    """
    Backup and duplicate all records in the database.

    Args:
        n: Number of total copies (original counts as 1)
    """
    try:
        logger.info(f"Starting duplication with n={request.n}")
        result = backup_and_duplicate(request.n)
        return DuplicateResponse(**result)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error during duplication: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")
