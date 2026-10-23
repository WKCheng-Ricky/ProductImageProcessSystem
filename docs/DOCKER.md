# Docker Deployment Guide

## Standard way to start and roll out (nerdctl compose)

This project is run with `nerdctl compose` against `docker-compose.yml`. The web UI is on http://localhost:8001 (host 8001 -> container 8000).

```bash
# Build the image and (re)create the container with the new code
nerdctl compose up -d --build --force-recreate

# Wait until it is healthy (models preload, about a minute)
curl -s localhost:8001/health

# Confirm the running container has your code (example)
nerdctl exec productimageprocesssystem-product-image-processor-1 ls /app/src/web/services

# Logs / stop
nerdctl compose logs -f
nerdctl compose down        # keeps the named volumes (app-data, app-uploads)
```

Things that bit us:

- **`--force-recreate` is required to roll out.** `nerdctl compose up -d --build` rebuilds the image but leaves a running container alone when the compose config is unchanged, so the old code keeps serving. A sign of this: new routes return 404, and a new page path such as `/search/browse` can even return 200 because an older catch-all route (`/search/{session_id}`) matches it.
- **Recreate can fail with `bind for :8001 failed: port is already allocated`.** The old container is stopped first, and its port is sometimes still held for a moment. Run the same command again.
- **Rolling out stops the service** for the duration of the recreate plus the model preload. Stored data survives because it lives in the named volumes.
- **Builds are fast after the first one.** The Dockerfile puts the dependency install and the model download before `COPY . .`, so a code-only change reuses those layers. `.dockerignore` keeps `.mcp.json`, `.playwright-mcp`, `.git` and `*.md` (except `CLAUDE.md`) out of the build context.

## Checking the UI with Playwright MCP

A local, gitignored `.mcp.json` declares a `playwright` MCP server. Create it with `claude mcp add --scope project playwright -- npx -y @playwright/mcp@0.0.83 --headless --isolated`. After a rollout, ask Claude Code to open `http://localhost:8001` with the Playwright tools. A session started before the server was added must be restarted to see it, and Claude Code asks you to approve project MCP servers the first time. Browsers are cached in `~/.cache/ms-playwright`; if one is missing, run `npx playwright install chromium`.

## Quick Start (docker-compose / docker CLI)

### Using Docker Compose (Recommended)

```bash
# Build and start the service
docker-compose up -d

# View logs
docker-compose logs -f

# Stop the service
docker-compose down
```

Access the web interface at: http://localhost:8000

### Using Docker CLI

```bash
# Build the image
docker build -t product-image-processor .

# Run the web server
docker run -d \
  -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/uploads:/app/uploads \
  -e USE_MILVUS=true \
  --name product-processor \
  product-image-processor

# Run with GPU support (if available)
docker run -d \
  --gpus all \
  -p 8000:8000 \
  -v $(pwd)/data:/app/data \
  -e USE_MILVUS=true \
  product-image-processor \
  python main.py --serve --port 8000 --gpu
```

## Different Modes

### Web Interface Mode (Default)
```bash
docker run -p 8000:8000 product-image-processor
```

### Test Mode
```bash
docker run product-image-processor python main.py --test
```

### Process Single Image
```bash
docker run -v $(pwd)/images:/app/images product-image-processor \
  python main.py --image /app/images/product.jpg
```

### Batch Processing
```bash
docker run -v $(pwd)/images:/app/images product-image-processor \
  python main.py --batch /app/images/
```

### Search Mode
```bash
# Image similarity search
docker run -v $(pwd)/images:/app/images product-image-processor \
  python main.py --search-image /app/images/query.jpg

# Text-based search
docker run product-image-processor \
  python main.py --search-text "shampoo"
```

## Environment Variables

Configure the container using environment variables:

| Variable | Default | Description |
|----------|---------|-------------|
| USE_MILVUS | false | Enable persistent vector DB |
| MILVUS_LITE_PATH | /app/data/milvus.db | Milvus database path |
| MILVUS_COLLECTION | product_embeddings | Collection name |

Example:
```bash
docker run -e USE_MILVUS=true -e MILVUS_COLLECTION=my_products product-image-processor
```

## Data Persistence

Mount volumes to persist data between container restarts:

```bash
docker run \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/uploads:/app/uploads \
  -v $(pwd)/resource:/app/resource \
  product-image-processor
```

## Development Mode

Run with code hot-reload:
```bash
docker run -v $(pwd):/app -p 8000:8000 product-image-processor
```

## Troubleshooting

### View container logs
```bash
docker logs -f product-processor
```

### Enter container shell
```bash
docker exec -it product-processor /bin/bash
```

### Check resource usage
```bash
docker stats product-processor
```

### Rebuild after code changes
```bash
docker-compose down
docker-compose build --no-cache
docker-compose up -d
```
