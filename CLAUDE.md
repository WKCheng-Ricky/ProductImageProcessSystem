# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Important Rules

- DO NOT read folders/files in `.gitignore` (venv/, data/, uploads/, resource/, .env)
- Activate venv before running Python: `source venv/bin/activate`
- Prefer clear function names over comments

## Commands

```bash
# Setup
./scripts/setup.sh                        # Full setup (recommended)
source venv/bin/activate                  # Required before any python command

# Run pipeline
python main.py --test                     # Test pipeline + warm up / download models
python main.py --image path.jpg           # Process single image
python main.py --batch ./images/          # Batch process directory
python main.py --batch ./images/ --gpu    # With GPU acceleration

# Web interface
python main.py --serve --port 8000        # Start at http://localhost:8000
python main.py --serve --preload-model    # Pre-warm models before serving

# Search (requires embeddings already in DB)
python main.py --search-image query.jpg   # Image similarity search
python main.py --search-text "shampoo"    # Text-based search

# Docker (uses Milvus by default, exposes port 8001)
docker-compose up --build
```

## Architecture

### Key Design Decision: Preprocess BEFORE Embedding

```
Raw Image → Watermark Removal → Collage Split → Best Image Selection
         → Background Removal → Quality Score → CLIP Embedding → Vector DB
```

Backgrounds, watermarks, and text overlays distort embeddings. Cleaning first ensures vectors capture product features, not environmental noise.

### Pipeline (`src/pipeline/product_image_pipeline.py`)

`ProductImagePipeline` orchestrates all components with lazy-loaded ML models:

1. **WatermarkRemover** — simple heuristic by default (`use_ocr=False`); EasyOCR optional
2. **CollageDetector** — detects composite images, splits via `smart_split()`
3. **BestImageSelector** — picks highest quality from collage splits (uses scorer + preprocessor internally)
4. **ImagePreprocessor** — rembg/U2Net background removal, crops to product
5. **ImageQualityScorer** — facade combining `BasicQualityScorer` (cv2 metrics) + `AestheticScorer` (CLIP-based LAION predictor)
6. **TextImageEmbeddingGenerator** — open_clip ViT-L-14; supports both image and text queries
7. **ProductGrouper** — wraps `VectorDatabase`, runs DBSCAN clustering for grouping

Embedding generator, vector DB, and product grouper are lazy-loaded properties — they only initialize on first access to avoid loading CLIP/rembg at startup.

### Vector Database

Two backends, swapped via env var `USE_MILVUS`:
- `InMemoryVectorDB` — default dev mode, no persistence
- `MilvusVectorLiteDB` — file-based persistence (`./data/milvus.db`), used in Docker

The pipeline auto-detects embedding dimension from the loaded CLIP model before initializing Milvus.

### Web Interface (`src/web/`)

FastAPI app with session-based multi-page flow. Each uploaded image gets a `session_id` (UUID); session state is persisted as JSON in `uploads/sessions/<session_id>/session.json`.

Page flow (single image): upload → watermark → quality_basic → quality_aesthetic → split → preprocessing → embedding → search

Additional standalone pages: `/batch/upload`, `/grouping`, `/duplicate`

All ML operations are called via `src/web/services/pipeline_service.py` which holds a singleton `ProductImagePipeline`. Routes under `src/web/routes/` handle API endpoints (`/api/...`) that the JS frontend calls; page routes in `src/web/app.py` serve HTML templates.

### Model Caching

`src/pipeline/model_cache.py` provides a singleton `ModelCache` used by all components to share loaded CLIP and rembg sessions — prevents duplicate model loading when multiple pipeline components need the same model.

### Confidence-Based Similarity

| Similarity | Action |
|------------|--------|
| > 90% | Auto-match (same product) |
| 70–90% | Similar product (variant/related) |
| < 70% | Different product |

## Configuration

Via `.env` file or environment variables:

```bash
USE_MILVUS=true                      # Persistent vector DB (default: false)
MILVUS_LITE_PATH=./data/milvus.db    # Milvus Lite file path
MILVUS_COLLECTION=product_embeddings
CLIP_MODEL=ViT-L-14                  # CLIP model variant (default: ViT-L-14)
CLIP_PRETRAINED=openai               # Pretrained weights (default: openai)
```

Pipeline constructor parameters (override per-instance):
- `target_size=(512, 512)` — output image dimensions
- `similarity_threshold=0.85` — grouping/search threshold
- `quality_threshold=0.3` — minimum score to pass (0–1)

## Tech Stack

| Component | Technology |
|-----------|------------|
| Background Removal | rembg (U2Net) |
| Embedding Model | open_clip (ViT-L-14) |
| Quality Scoring | LAION aesthetic predictor (CLIP-based) |
| Vector DB | InMemoryVectorDB (dev) / Milvus Lite (prod) |
| Clustering | DBSCAN (scikit-learn) |
| OCR (watermark) | EasyOCR (optional) |
| Web | FastAPI + Jinja2 + uvicorn |
