# Multi-stage Dockerfile for Product Image Processing System
FROM python:3.11-slim AS base

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p data uploads resource

# Set environment variables
ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    # USE_MILVUS=false \ # false when using external MILVUS
    MILVUS_LITE_PATH=/app/data/milvus.db \
    MILVUS_COLLECTION=product_embeddings 
    # CLIP_MODEL=ViT-L-14 \
    # CLIP_PRETRAINED=openai
    # CLIP_MODEL=ViT-L-14 \
    # CLIP_PRETRAINED=laion2b_s32b_b82k


# Expose web server port
EXPOSE 8000

# Default command: run web server
CMD ["python", "main.py", "--preload-model", "--serve", "--port", "8000"]
