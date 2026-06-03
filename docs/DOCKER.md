# Docker Deployment Guide

## Quick Start

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
