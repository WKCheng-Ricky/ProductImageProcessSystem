#  Product Image Processing System

> **Author**: W Cheng
> **Co-Author**: Claude (Anthropic AI Assistant)
> **Created**: 2026-01-07

[![CI](https://github.com/WKCheng-Ricky/ProductImageProcessSystem/actions/workflows/ci.yml/badge.svg)](https://github.com/WKCheng-Ricky/ProductImageProcessSystem/actions/workflows/ci.yml)

A multimodal product-image pipeline (CLIP embeddings, vector search, clustering), built almost entirely with Claude Code from specs and prompts, then verified with tests and CI.

Product photos are cleaned (watermarks, collages, backgrounds), scored for quality, embedded with CLIP ViT-L-14 (text and images share one embedding space), stored in a vector database, and then used for similarity search, duplicate detection and DBSCAN product grouping. It is designed for an e-commerce catalog of 2M+ SKUs.

## Features

- **Watermark Removal**: Detect and remove watermarks/text overlays before processing
- **Collage Detection & Splitting**: Automatically detect and split composite images
- **Background Removal**: Remove backgrounds and standardize to white (rembg/U2Net)
- **Quality Scoring**: Evaluate sharpness, brightness using LAION aesthetic predictor
- **Best Image Selection**: Automatically select the best main image from multiple SKU photos
- **CLIP Embeddings**: Generate vector embeddings for image search (open_clip ViT-L-14)
- **Duplicate Detection**: Find duplicate SKUs using vector similarity
- **Product Grouping**: Cluster similar SKUs using DBSCAN
- **Web Interface**: Interactive multi-page UI for visualizing the processing pipeline

## Requirements

- Python 3.12+
- ~4GB disk space for models

## Quick Setup

```bash
chmod +x scripts/setup.sh
./scripts/setup.sh
```

Or manually:

```bash
# Install system dependencies (requires sudo)
sudo apt update
sudo apt install -y python3.12-venv python3-pip libgl1-mesa-glx libglib2.0-0

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

## Usage

### Activate Virtual Environment

```bash
source venv/bin/activate
```

### Web Interface

```bash
python main.py --serve --port 8000
# Open browser to http://localhost:8000
```

Web interface pages: upload, collage split, watermark removal, preprocessing, quality scoring, embedding, search, batch processing, product grouping, duplicate detection.

### Command Line

```bash
# Run pipeline tests + download models
python main.py --test

# Process single image
python main.py --image path/to/image.jpg

# Batch process directory
python main.py --batch path/to/images/

# With GPU
python main.py --batch images/ --gpu

# Image similarity search
python main.py --search-image query.jpg

# Text-based search
python main.py --search-text "shampoo bottle"

# Custom output directory
python main.py --image photo.jpg --output ./processed
```

## Project Structure

```
ProductImageProcessingSystem/
├── main.py                        # Main entry point (CLI + web server)
├── requirements.txt               # Python dependencies (input)
├── requirements.lock              # Pinned dependencies
├── pyproject.toml                 # pytest and ruff config
├── tests/                         # pytest suite
├── .github/workflows/ci.yml       # CI
├── docker-compose.yml             # Docker setup
├── Dockerfile
├── scripts/
│   ├── setup.sh                   # Full setup script
│   ├── start.sh                   # Start web server
│   └── start_detached.sh          # Start in background
├── docs/
│   └── OVERVIEW.md                # Architecture & tech stack overview
└── src/
    ├── pipeline/                  # Core ML pipeline
    │   ├── product_image_pipeline.py  # Main pipeline orchestrator
    │   ├── watermark_remover.py       # Watermark detection & removal
    │   ├── collage_detector.py        # Collage detection/splitting
    │   ├── image_preprocessor.py      # Background removal, cropping
    │   ├── quality_scorer.py          # LAION aesthetic quality scoring
    │   ├── best_image_selector.py     # Best image selection logic
    │   ├── embedding_generator.py     # CLIP embedding generation
    │   ├── product_grouper.py         # SKU grouping, vector DB, search
    │   ├── milvus_db.py               # Milvus Lite vector database
    │   ├── model_cache.py             # Shared model cache (CLIP, rembg)
    │   ├── config.py                  # Configuration from environment
    │   └── clustering.py              # DBSCAN clustering
    └── web/                       # Web interface (FastAPI)
        ├── app.py                     # FastAPI application
        ├── routes/                    # API endpoints per page
        │   ├── upload.py, split.py, watermark.py
        │   ├── preprocessing.py, quality.py
        │   ├── embedding.py, search.py
        │   ├── batch.py, grouping.py, duplicate.py
        ├── services/
        │   ├── session_manager.py     # Session state management
        │   ├── pipeline_service.py    # Pipeline wrapper for web
        │   └── duplicate_service.py  # Duplicate detection service
        ├── models/schemas.py          # Pydantic schemas
        ├── templates/                 # HTML templates (Jinja2)
        └── static/                   # CSS, JavaScript
```

## Pipeline Flow

Images are cleaned **before** embedding: backgrounds, watermarks and text overlays distort embeddings, so cleaning first keeps the vectors about the product itself.

```mermaid
flowchart TD
    A[Input image] --> B[Watermark removal]
    B --> C[Collage detection and split]
    C --> D[Best image selection]
    D --> E[Background removal and crop]
    E --> F[Quality scoring: basic + LAION aesthetic]
    F --> G[CLIP embedding ViT-L-14]
    G --> H[(Vector DB: in-memory or Milvus Lite)]
    H --> I[Search, duplicate detection, DBSCAN grouping]
```

## Configuration

Environment variables (via `.env` or shell):

```bash
USE_MILVUS=true                      # Enable persistent vector DB (default: false)
MILVUS_LITE_PATH=./data/milvus.db
MILVUS_COLLECTION=product_embeddings
CLIP_MODEL=ViT-L-14                  # CLIP model variant
CLIP_PRETRAINED=openai               # Pretrained weights
```

Pipeline parameters:

```python
pipeline = ProductImagePipeline(
    target_size=(512, 512),          # Output image dimensions
    similarity_threshold=0.85,       # Product grouping threshold
    quality_threshold=0.3,           # Minimum quality score (0-1)
    use_gpu=False,
    output_dir="./processed"
)
```

## Confidence-Based Workflow

| Similarity | Action |
|------------|--------|
| > 90% | Auto-match (same product) |
| 70–90% | Similar product (variant/related) |
| < 70% | Different product |

## Testing and CI

```bash
pip install -r requirements-dev.txt   # pytest, httpx, ruff
pytest                                 # fast suite, no model downloads
pytest -m model                        # opt-in tests that load real model weights
ruff check .
```

The default suite covers session handling, the in-memory vector DB, clustering, the OpenCV quality scorer, API schemas and the FastAPI routes. Heavy ML libraries are stubbed when they are not installed (`tests/conftest.py`), so it runs in seconds. GitHub Actions (`.github/workflows/ci.yml`) runs `ruff` and `pytest` on every push and pull request, and builds the Docker image without pushing it.

Health endpoints: `GET /health` (liveness) and `GET /ready` (session storage and vector DB; never loads models).

## Docker

```bash
docker compose up --build            # web UI on http://localhost:8001
```

The image is multi-stage, runs as a non-root user, and installs from the pinned `requirements.lock`. To change dependencies, edit `requirements.txt` and regenerate the lock:

```bash
uv pip compile requirements.txt -o requirements.lock --python-version 3.12 --index-strategy unsafe-best-match
```

## Built with spec-driven agentic development

Most of the code was written by Claude Code. The human role was to direct and review it:

- **Spec:** `CLAUDE.md` records the architecture, rules and thresholds the agent works to, including the key design decision to preprocess before embedding.
- **Agent:** Claude Code writes, runs and iterates on the code (see the `Co-Author` headers in the source files).
- **Guardrails:** tests and CI check the agent's output. Tests were added after the first version, so they pin current behaviour.

## API Reference

```python
from src.pipeline.product_image_pipeline import ProductImagePipeline

pipeline = ProductImagePipeline()

# Process single image
result = pipeline.process_image("image.jpg", id="SKU001")

# Batch process
results = pipeline.process_batch("./images/")

# Search by image
similar = pipeline.search_by_image("query.jpg", top_k=10)

# Search by text
similar = pipeline.search_by_text("shampoo bottle", top_k=10)

# Group products
groups = pipeline.group_products()

# Select best from multiple SKU photos
best_idx, best_img, scores = pipeline.select_best_from_sku(image_paths)
```

## Tech Stack

| Component | Technology |
|-----------|------------|
| Background Removal | rembg (U2Net) |
| Embedding Model | open_clip (ViT-L-14) |
| Quality Scoring | LAION aesthetic predictor |
| Vector DB | InMemoryVectorDB (dev) / Milvus Lite (prod) |
| Clustering | DBSCAN |
| OCR (watermark) | EasyOCR |
| Web | FastAPI + Jinja2 |

## Troubleshooting

### "ensurepip is not available"

```bash
sudo apt install python3.12-venv
```

### OpenCV import error

```bash
sudo apt install libgl1-mesa-glx libglib2.0-0
```

### CUDA out of memory

Use CPU mode (omit `--gpu` flag):

```bash
python main.py --image photo.jpg
```
