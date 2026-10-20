"""
Grouping Route - Product Grouping by Similarity

Author: W Cheng
Co-Author: Claude (Anthropic AI Assistant)
Created: 2026-01-15
Project: Product Image Processing System
"""

from fastapi import APIRouter, HTTPException
import logging

from ..models.schemas import (
    GroupMemberItem,
    ProductGroupResult,
    GroupingStatsResponse,
    GroupingResultsResponse,
)
from ..services.pipeline_service import get_pipeline
from src.pipeline.milvus_db import ProductGroupManager, ProductGroupsDB
from src.pipeline.config import CONFIG
from src.pipeline.config import GROUPING_SIMILARITY_THRESHOLD, GROUPING_TOP_K

logger = logging.getLogger(__name__)
router = APIRouter()


def get_group_manager() -> ProductGroupManager:
    """Get or create the ProductGroupManager"""
    pipeline = get_pipeline()
    return ProductGroupManager(
        vector_db=pipeline.vector_db,
        similarity_threshold=GROUPING_SIMILARITY_THRESHOLD,
        top_k=GROUPING_TOP_K,
        db_path=CONFIG.get("milvus_lite_path", "./data/milvus.db"),
    )


@router.post("/grouping/run", response_model=GroupingResultsResponse)
def run_grouping(
    similarity_threshold: float = GROUPING_SIMILARITY_THRESHOLD,
    clear_existing: bool = True,
):
    """
    Run product grouping on all products in the database.

    This endpoint:
    - Loops through all products in the vector database
    - Groups products with similarity above threshold
    - Creates UUID-based groups stored in the groups database

    Args:
        similarity_threshold: Minimum similarity score to group products (0-1)
        clear_existing: If True, clears existing groups before creating new ones
    """
    if similarity_threshold < 0 or similarity_threshold > 1:
        raise HTTPException(status_code=400, detail="similarity_threshold must be between 0 and 1")

    try:
        pipeline = get_pipeline()
        db_size = pipeline.vector_db.size()

        if db_size == 0:
            return GroupingResultsResponse(
                total_groups=0,
                total_products=0,
                groups_with_multiple_members=0,
                groups=[],
            )

        logger.info(f"Starting grouping: {db_size} products, threshold={similarity_threshold}")

        group_manager = ProductGroupManager(
            vector_db=pipeline.vector_db,
            similarity_threshold=similarity_threshold,
            top_k=10,
            db_path=CONFIG.get("milvus_lite_path", "./data/milvus.db"),
        )

        groups = group_manager.group_all_products(clear_existing=clear_existing)

        logger.info(f"Grouping complete: {len(groups)} groups created")

        group_results = []
        for group in groups:
            members = []
            for member_id in group.member_ids:
                product_data = pipeline.vector_db.get(member_id)
                image_url = None
                original_image_url = None
                metadata = None
                if product_data and product_data.metadata:
                    image_url = product_data.metadata.get("image_url")
                    original_image_url = product_data.metadata.get("original_image_url")
                    metadata = product_data.metadata

                members.append(GroupMemberItem(
                    id=member_id,
                    image_url=image_url,
                    original_image_url=original_image_url,
                    metadata=metadata,
                ))

            group_results.append(ProductGroupResult(
                group_id=group.group_id,
                member_count=group.size,
                representative_id=group.representative_id,
                members=members,
            ))

        multi_member_count = sum(1 for g in group_results if g.member_count > 1)
        total_products = sum(g.member_count for g in group_results)

        return GroupingResultsResponse(
            total_groups=len(group_results),
            total_products=total_products,
            groups_with_multiple_members=multi_member_count,
            groups=group_results,
        )

    except Exception as e:
        logger.error(f"Error running grouping: {e}")
        raise HTTPException(status_code=500, detail="Error running grouping")


@router.get("/grouping/results", response_model=GroupingResultsResponse)
def get_grouping_results():
    """
    Get all existing product groups.

    Returns all groups that were previously created by the grouping algorithm.
    """
    try:
        pipeline = get_pipeline()

        groups_db = ProductGroupsDB(
            db_path=CONFIG.get("milvus_lite_path", "./data/milvus.db"),
        )

        groups = groups_db.get_all_groups()

        group_results = []
        for group in groups:
            members = []
            for member_id in group.member_ids:
                product_data = pipeline.vector_db.get(member_id)
                image_url = None
                original_image_url = None
                metadata = None
                if product_data and product_data.metadata:
                    image_url = product_data.metadata.get("image_url")
                    original_image_url = product_data.metadata.get("original_image_url")
                    metadata = product_data.metadata

                members.append(GroupMemberItem(
                    id=member_id,
                    image_url=image_url,
                    original_image_url=original_image_url,
                    metadata=metadata,
                ))

            group_results.append(ProductGroupResult(
                group_id=group.group_id,
                member_count=group.size,
                representative_id=group.representative_id,
                members=members,
            ))

        multi_member_count = sum(1 for g in group_results if g.member_count > 1)
        total_products = sum(g.member_count for g in group_results)

        return GroupingResultsResponse(
            total_groups=len(group_results),
            total_products=total_products,
            groups_with_multiple_members=multi_member_count,
            groups=group_results,
        )

    except Exception as e:
        logger.error(f"Error getting grouping results: {e}")
        raise HTTPException(status_code=500, detail="Error getting results")


@router.get("/grouping/stats", response_model=GroupingStatsResponse)
def get_grouping_stats():
    """
    Get statistics about product grouping.

    Returns summary statistics without the full group details.
    """
    try:
        pipeline = get_pipeline()
        db_size = pipeline.vector_db.size()

        groups_db = ProductGroupsDB(
            db_path=CONFIG.get("milvus_lite_path", "./data/milvus.db"),
        )

        groups = groups_db.get_all_groups()

        if not groups:
            return GroupingStatsResponse(
                total_products_in_db=db_size,
                total_groups=0,
                total_products_grouped=0,
                groups_with_multiple_members=0,
                avg_group_size=0.0,
                max_group_size=0,
                min_group_size=0,
            )

        sizes = [g.size for g in groups]
        multi_member_groups = [g for g in groups if g.size > 1]

        return GroupingStatsResponse(
            total_products_in_db=db_size,
            total_groups=len(groups),
            total_products_grouped=sum(sizes),
            groups_with_multiple_members=len(multi_member_groups),
            avg_group_size=sum(sizes) / len(sizes) if sizes else 0.0,
            max_group_size=max(sizes) if sizes else 0,
            min_group_size=min(sizes) if sizes else 0,
        )

    except Exception as e:
        logger.error(f"Error getting grouping stats: {e}")
        raise HTTPException(status_code=500, detail="Error getting stats")


@router.get("/grouping/group/{group_id}", response_model=ProductGroupResult)
def get_group_by_id(group_id: str):
    """
    Get a specific group by its UUID.
    """
    try:
        pipeline = get_pipeline()

        groups_db = ProductGroupsDB(
            db_path=CONFIG.get("milvus_lite_path", "./data/milvus.db"),
        )

        group = groups_db.get_group(group_id)
        if group is None:
            raise HTTPException(status_code=404, detail=f"Group {group_id} not found")

        members = []
        for member_id in group.member_ids:
            product_data = pipeline.vector_db.get(member_id)
            image_url = None
            original_image_url = None
            metadata = None
            if product_data and product_data.metadata:
                image_url = product_data.metadata.get("image_url")
                original_image_url = product_data.metadata.get("original_image_url")
                metadata = product_data.metadata

            members.append(GroupMemberItem(
                id=member_id,
                image_url=image_url,
                original_image_url=original_image_url,
                metadata=metadata,
            ))

        return ProductGroupResult(
            group_id=group.group_id,
            member_count=group.size,
            representative_id=group.representative_id,
            members=members,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting group {group_id}: {e}")
        raise HTTPException(status_code=500, detail="Error getting group")
