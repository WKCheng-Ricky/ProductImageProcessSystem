"""
FastAPI Web Application for Product Image Processing

7-page pipeline interface:
1. Upload & Collage Detection
2. Basic Quality Scoring (on raw image)
3. Aesthetic Scoring (on raw image)
4. Collage Split
5. Image Preprocessing
6. Embedding Generation
7. Similar Product Search

Author: W Cheng
Co-Author: Claude (Anthropic AI Assistant)
Created: 2026-01-09
Updated: 2026-01-15
Project: Product Image Processing System
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse

from src.logging_config import configure_logging
from src.pipeline.config import CONFIG
from .routes import upload, watermark, split, preprocessing, quality, embedding, search, batch, grouping, duplicate
from .services.pipeline_service import PipelineService, get_pipeline
from .services.session_manager import SessionManager

configure_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting up web application...")

    # Initialize session manager
    session_manager = SessionManager()
    app.state.session_manager = session_manager

    # Schedule periodic cleanup task
    async def cleanup_task():
        while True:
            try:
                await asyncio.sleep(3600)  # Every hour
                logger.info("Running scheduled session cleanup...")
                deleted = session_manager.cleanup_old_sessions(max_age_hours=24)
                logger.info(f"Cleanup complete: {deleted} sessions deleted")
            except Exception as e:
                logger.error(f"Error in cleanup task: {e}")

    cleanup_task_handle = asyncio.create_task(cleanup_task())

    logger.info("Web application started successfully")
    logger.info("Access the interface at http://localhost:8000")

    yield

    # Shutdown
    logger.info("Shutting down web application...")
    cleanup_task_handle.cancel()
    logger.info("Shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="Product Image Processing System",
    description="Web interface for processing product images with collage detection, preprocessing, quality scoring, and similarity search",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware (configure for production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Change to specific origins in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_dir = Path(__file__).parent / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Mount uploads directory
uploads_dir = Path("uploads")
uploads_dir.mkdir(exist_ok=True)
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# Configure templates
templates_dir = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(templates_dir))

# Include API routers
app.include_router(upload.router, prefix="/api", tags=["upload"])
app.include_router(watermark.router, prefix="/api", tags=["watermark"])
app.include_router(split.router, prefix="/api", tags=["split"])
app.include_router(preprocessing.router, prefix="/api", tags=["preprocessing"])
app.include_router(quality.router, prefix="/api", tags=["quality"])
app.include_router(embedding.router, prefix="/api", tags=["embedding"])
app.include_router(search.router, prefix="/api", tags=["search"])
app.include_router(batch.router, prefix="/api", tags=["batch"])
app.include_router(grouping.router, prefix="/api", tags=["grouping"])
app.include_router(duplicate.router, prefix="/api", tags=["duplicate"])


# Page Routes
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Home Page"""
    return templates.TemplateResponse(request, "index.html", {
        "current_step": 0
    })


@app.get("/upload", response_class=HTMLResponse)
async def upload_page(request: Request):
    """Page 1: Upload & Collage Detection"""
    return templates.TemplateResponse(request, "upload.html", {
        "current_step": 1
    })


@app.get("/watermark/{session_id}", response_class=HTMLResponse)
async def watermark_page(request: Request, session_id: str):
    """Page 2: Watermark Removal"""
    return templates.TemplateResponse(request, "watermark.html", {
        "session_id": session_id,
        "current_step": 2
    })


@app.get("/split/{session_id}", response_class=HTMLResponse)
async def split_page(request: Request, session_id: str):
    """Page 4: Collage Split"""
    return templates.TemplateResponse(request, "split.html", {
        "session_id": session_id,
        "current_step": 4
    })


@app.get("/preprocessing/{session_id}", response_class=HTMLResponse)
async def preprocessing_page(request: Request, session_id: str):
    """Page 5: Image Preprocessing"""
    return templates.TemplateResponse(request, "preprocessing.html", {
        "session_id": session_id,
        "current_step": 5
    })


@app.get("/quality_basic/{session_id}", response_class=HTMLResponse)
async def quality_basic_page(request: Request, session_id: str):
    """Page 2: Quality Scoring (on raw image)"""
    return templates.TemplateResponse(request, "quality_basic.html", {
        "session_id": session_id,
        "current_step": 2
    })


@app.get("/quality_aesthetic/{session_id}", response_class=HTMLResponse)
async def quality_aesthetic_page(request: Request, session_id: str):
    """Page 3: Aesthetic Scoring (on raw image)"""
    return templates.TemplateResponse(request, "quality_aesthetic.html", {
        "session_id": session_id,
        "current_step": 3
    })


@app.get("/embedding/{session_id}", response_class=HTMLResponse)
async def embedding_page(request: Request, session_id: str):
    """Page 6: Embedding Generation"""
    return templates.TemplateResponse(request, "embedding.html", {
        "session_id": session_id,
        "current_step": 6
    })


@app.get("/search/by-image/{image_id}", response_class=HTMLResponse)
async def search_by_image_page(request: Request, image_id: str):
    """Search similar products by image ID"""
    return templates.TemplateResponse(request, "search_by_image.html", {
        "image_id": image_id
    })


@app.get("/search/text", response_class=HTMLResponse)
async def search_text_page(request: Request):
    """Text Search Page"""
    return templates.TemplateResponse(request, "search_text.html", {
        "current_step": 0
    })


@app.get("/search/{session_id}", response_class=HTMLResponse)
async def search_page(request: Request, session_id: str):
    """Page 7: Similar Product Search"""
    return templates.TemplateResponse(request, "search.html", {
        "session_id": session_id,
        "current_step": 7
    })


@app.get("/batch/upload", response_class=HTMLResponse)
async def batch_upload_page(request: Request):
    """Batch Processing: Upload Page"""
    return templates.TemplateResponse(request, "batch_upload.html", {
        "current_step": 0
    })


@app.get("/grouping", response_class=HTMLResponse)
async def grouping_page(request: Request):
    """Product Grouping Page"""
    return templates.TemplateResponse(request, "grouping.html", {
        "current_step": 0
    })


@app.get("/duplicate", response_class=HTMLResponse)
async def duplicate_page(request: Request):
    """Database Duplication Page"""
    return templates.TemplateResponse(request, "duplicate.html", {
        "current_step": 0
    })


@app.get("/batch/complete/{batch_id}", response_class=HTMLResponse)
async def batch_complete_page(request: Request, batch_id: str):
    """Batch Processing: Completion Page"""
    return templates.TemplateResponse(request, "batch_complete.html", {
        "batch_id": batch_id,
        "current_step": 0
    })


# Utility endpoints
@app.get("/api/session/{session_id}/status")
async def get_session_status(session_id: str):
    """Get session status and progress"""
    session_manager = app.state.session_manager
    try:
        session = session_manager.get_session(session_id)
        original_path = session.get("original_path", "")
        original_filename = os.path.basename(original_path) if original_path else ""
        return {
            "session_id": session_id,
            "current_step": session.get("current_step", 0),
            "created_at": session.get("created_at"),
            "updated_at": session.get("updated_at"),
            "original_filename": original_filename,
            "steps_completed": [
                k for k in ["collage_info", "split_result", "preprocessing_result",
                           "quality_score", "embedding_result", "search_results"]
                if k in session
            ]
        }
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@app.delete("/api/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a session and cleanup files"""
    session_manager = app.state.session_manager
    try:
        session_manager.delete_session(session_id)
        return {"message": f"Session {session_id} deleted successfully"}
    except Exception as e:
        logger.error(f"Error deleting session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "product-image-processing", "version": app.version}


@app.get("/ready")
async def readiness_check():
    """Readiness: session storage writable and, if already loaded, vector DB reachable. Never loads models."""
    checks = {}
    try:
        probe = app.state.session_manager.storage_dir / ".ready_probe"
        probe.write_text("ok")
        probe.unlink()
        checks["session_storage"] = "ok"
    except Exception as e:
        checks["session_storage"] = f"error: {e}"

    pipeline = PipelineService._pipeline
    if pipeline is None:
        checks["vector_db"] = "not_initialised"
    else:
        try:
            pipeline.vector_db.size()
            checks["vector_db"] = "ok"
        except Exception as e:
            checks["vector_db"] = f"error: {e}"

    ready = not any(str(v).startswith("error") for v in checks.values())
    return JSONResponse({"status": "ready" if ready else "not_ready", "checks": checks},
                        status_code=200 if ready else 503)


@app.get("/api/vector-db/health")
async def vector_db_health_check():
    """Check vector database health and statistics"""
    try:
        pipeline = get_pipeline()
        db = pipeline.vector_db
        db_type = type(db).__name__

        health_info = {
            "database_type": db_type,
            "total_embeddings": db.size(),
            "status": "healthy"
        }

        if db_type == "MilvusVectorDB":
            health_info.update({
                "host": CONFIG["milvus_host"],
                "port": CONFIG["milvus_port"],
                "collection": CONFIG["milvus_collection"]
            })

        return health_info
    except Exception as e:
        logger.error(f"Error checking vector database health: {e}")
        return {
            "database_type": "unknown",
            "status": "error",
            "error": str(e)
        }


if __name__ == "__main__":
    logger.info("Starting development server...")
    uvicorn.run(
        "web.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
