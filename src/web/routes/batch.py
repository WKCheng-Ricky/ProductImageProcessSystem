"""
Batch Processing Route - Upload and process multiple images

Author: W Cheng
Co-Author: Claude (Anthropic AI Assistant)
Created: 2026-01-14
Project: Product Image Processing System
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Request
from PIL import Image
import io
import logging
import uuid
import time
import gc
import asyncio
from pathlib import Path
from typing import List, Optional

from ..models.schemas import BatchUploadResponse, BatchProcessResponse, BatchImageResult
from ..services.session_manager import SessionManager
from ..services.pipeline_service import get_pipeline

logger = logging.getLogger(__name__)
router = APIRouter()

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_FILE_SIZE = 50 * 1024 * 1024


def get_session_manager(request: Request) -> SessionManager:
    return request.app.state.session_manager


@router.post("/batch/upload", response_model=BatchUploadResponse)
async def batch_upload(
    files: List[UploadFile] = File(...),
    skip_watermark_removal: bool = True,
    skip_collage_detection: bool = True,
    session_manager: SessionManager = Depends(get_session_manager)
):
    """
    Upload multiple images for batch processing

    Saves all uploaded images and creates a batch session.
    Returns batch_id for subsequent processing.
    """
    if not files:
        raise HTTPException(status_code=400, detail="No files provided")

    batch_id = str(uuid.uuid4())[:8]
    batch_path = Path("uploads/batches") / batch_id
    batch_path.mkdir(parents=True, exist_ok=True)

    valid_files = []

    for file in files:
        file_ext = Path(file.filename).suffix.lower()
        if file_ext not in ALLOWED_EXTENSIONS:
            logger.warning(f"Skipping invalid file type: {file.filename}")
            continue

        img = None
        try:
            content = await file.read()
            if len(content) > MAX_FILE_SIZE:
                logger.warning(f"Skipping file too large: {file.filename}")
                continue

            img = Image.open(io.BytesIO(content)).convert('RGB')

            file_id = str(uuid.uuid4())[:8]
            save_path = batch_path / f"{file_id}{file_ext}"
            img.save(save_path, quality=95)

            valid_files.append({
                "file_id": file_id,
                "filename": file.filename,
                "path": str(save_path),
                "extension": file_ext
            })

        except Exception as e:
            logger.warning(f"Error processing file {file.filename}: {e}")
            continue

        finally:
            # Close image to free memory
            if img is not None:
                img.close()

    if not valid_files:
        raise HTTPException(status_code=400, detail="No valid image files provided")

    batch_data = {
        "batch_id": batch_id,
        "files": valid_files,
        "skip_watermark_removal": skip_watermark_removal,
        "skip_collage_detection": skip_collage_detection,
        "status": "uploaded",
        "created_at": time.time()
    }

    session_manager.create_session(f"batch_{batch_id}", batch_data)

    return BatchUploadResponse(
        batch_id=batch_id,
        total_images=len(valid_files),
        message=f"Successfully uploaded {len(valid_files)} images"
    )


@router.post("/batch/{batch_id}/process", response_model=BatchProcessResponse)
async def process_batch(
    batch_id: str,
    session_manager: SessionManager = Depends(get_session_manager)
):
    """
    Process all images in a batch

    For each image:
    1. Get aesthetic scoring
    2. Preprocess (remove BG, crop, resize to 512x512)
    3. Generate embedding
    4. Save to vector database
    """
    try:
        batch_data = session_manager.get_session(f"batch_{batch_id}")
    except ValueError:
        raise HTTPException(status_code=404, detail=f"Batch {batch_id} not found")

    pipeline = get_pipeline()
    results = []
    successful = 0
    failed = 0
    skipped_duplicates = 0

    files = batch_data["files"]
    total = len(files)

    session_manager.update_session(f"batch_{batch_id}", {
        "status": "processing",
        "processed": 0,
        "total": total
    })

    for idx, file_info in enumerate(files):
        file_id = file_info["file_id"]
        filename = file_info["filename"]
        image_path = Path(file_info["path"])

        result = BatchImageResult(
            filename=filename,
            id=file_id,
            success=False
        )

        img = None
        preprocessed = None

        try:
            img = Image.open(image_path).convert('RGB')

            # Yield to event loop before CPU-intensive work
            await asyncio.sleep(0)

            # Step 1: Aesthetic scoring
            quality_score = pipeline.quality_scorer.score(img)
            result.overall_score = quality_score.overall
            result.aesthetic_score = quality_score.aesthetic

            # Yield to event loop
            await asyncio.sleep(0)

            # Step 2: Preprocessing (remove BG, crop, resize to 512x512)
            preprocessed = pipeline.preprocessor.standardize(
                img,
                remove_bg=True,
                crop=True,
                remove_watermark=True,
                remove_text=False,
                resize=True
            )

            processed_path = image_path.parent / f"{file_id}_processed.jpg"
            preprocessed.image.save(processed_path, quality=95)

            # Set image URLs in result
            result.image_url = f"/uploads/batches/{batch_id}/{file_id}_processed.jpg"
            result.original_image_url = f"/uploads/batches/{batch_id}/{file_id}{file_info['extension']}"

            # Yield to event loop
            await asyncio.sleep(0)

            # Step 3: Generate embedding
            embedding_result = pipeline.embedding_generator.generate(preprocessed.image)

            # Yield to event loop
            await asyncio.sleep(0)

            # Step 4: Check for duplicate before saving
            DUPLICATE_THRESHOLD = 0.999
            is_duplicate = False
            duplicate_of = None
            duplicate_score = None

            try:
                existing_results = pipeline.vector_db.search(
                    embedding=embedding_result.embedding,
                    top_k=1,
                    threshold=DUPLICATE_THRESHOLD
                )

                # Yield to event loop after search
                await asyncio.sleep(0)

                if existing_results and existing_results[0].score >= DUPLICATE_THRESHOLD:
                    is_duplicate = True
                    duplicate_of = existing_results[0].id
                    duplicate_score = existing_results[0].score
                    logger.info(f"Duplicate detected for {filename}: {duplicate_score:.4f} match with {duplicate_of}")
            except Exception as e:
                logger.warning(f"Failed to check for duplicates: {e}")

            result.is_duplicate = is_duplicate
            result.duplicate_of = duplicate_of
            result.duplicate_score = duplicate_score

            # Step 5: Save to vector database (skip if duplicate)
            if not is_duplicate:
                metadata = {
                    "batch_id": batch_id,
                    "filename": filename,
                    "aesthetic": quality_score.aesthetic,
                    "overall": quality_score.overall,
                    "image_url": f"/uploads/batches/{batch_id}/{file_id}_processed.jpg",
                    "original_image_url": f"/uploads/batches/{batch_id}/{file_id}{file_info['extension']}"
                }

                pipeline.vector_db.insert(
                    id=file_id,
                    embedding=embedding_result.embedding,
                    metadata=metadata
                )

                # Yield to event loop after insert
                await asyncio.sleep(0)

                result.saved_to_db = True
                successful += 1
                logger.info(f"Processed {filename}: aesthetic={quality_score.aesthetic:.2f}, saved to DB")
            else:
                skipped_duplicates += 1
                logger.info(f"Skipped {filename}: duplicate of {duplicate_of}")

            result.success = True

        except Exception as e:
            logger.error(f"Error processing {filename}: {e}")
            result.error = str(e)
            failed += 1

        finally:
            # Explicitly close PIL images to free memory
            if img is not None:
                img.close()
            if preprocessed is not None and hasattr(preprocessed, 'image'):
                preprocessed.image.close()

            # Force garbage collection every 5 images
            if (idx + 1) % 5 == 0:
                gc.collect()

        results.append(result)

        # Update session after each image so status endpoint can show progress
        session_manager.update_session(f"batch_{batch_id}", {
            "status": "processing",
            "processed": idx + 1,
            "total": total,
            "successful": successful,
            "failed": failed,
            "skipped_duplicates": skipped_duplicates
        })

        # Yield to event loop after each image to allow status requests
        await asyncio.sleep(0)

    session_manager.update_session(f"batch_{batch_id}", {
        "status": "completed",
        "results": [r.model_dump() for r in results],
        "successful": successful,
        "failed": failed,
        "skipped_duplicates": skipped_duplicates
    })

    # Final garbage collection to free all memory
    gc.collect()
    logger.info(f"Batch {batch_id} completed: {successful} successful, {failed} failed, {skipped_duplicates} duplicates")

    return BatchProcessResponse(
        batch_id=batch_id,
        total_images=len(files),
        successful=successful,
        failed=failed,
        skipped_duplicates=skipped_duplicates,
        results=results
    )


@router.get("/batch/{batch_id}/status")
async def get_batch_status(
    batch_id: str,
    session_manager: SessionManager = Depends(get_session_manager)
):
    """Get batch processing status"""
    try:
        batch_data = session_manager.get_session(f"batch_{batch_id}")
        return {
            "batch_id": batch_id,
            "status": batch_data.get("status", "unknown"),
            "total": batch_data.get("total", len(batch_data.get("files", []))),
            "processed": batch_data.get("processed", 0),
            "successful": batch_data.get("successful", 0),
            "failed": batch_data.get("failed", 0),
            "skipped_duplicates": batch_data.get("skipped_duplicates", 0),
            "results": batch_data.get("results", [])
        }
    except ValueError:
        raise HTTPException(status_code=404, detail=f"Batch {batch_id} not found")
