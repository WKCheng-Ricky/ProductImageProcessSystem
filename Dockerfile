FROM python:3.12-slim AS builder

WORKDIR /build
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements.lock .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.lock


FROM python:3.12-slim AS runtime

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 1000 app

COPY --from=builder /opt/venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH" \
    CLIP_MODEL=ViT-B-32 \
    CLIP_PRETRAINED=openai \
    HF_HOME=/opt/models/huggingface \
    U2NET_HOME=/opt/models/u2net

# Weights are public, so the anonymous download needs no token. Done before COPY . . so code changes keep this layer.
RUN python -c "import os, open_clip, rembg; open_clip.create_model_and_transforms(os.environ['CLIP_MODEL'], pretrained=os.environ['CLIP_PRETRAINED']); rembg.new_session('u2net', providers=['CPUExecutionProvider'])" && \
    chmod -R a+rX /opt/models

ENV HF_HUB_OFFLINE=1

WORKDIR /app
COPY --chown=app:app . .
RUN mkdir -p data uploads resource && chown -R app:app data uploads resource

ENV PYTHONPATH=/app \
    PYTHONUNBUFFERED=1 \
    MILVUS_LITE_PATH=/app/data/milvus.db \
    MILVUS_COLLECTION=product_embeddings

USER app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health', timeout=5)"

CMD ["python", "main.py", "--preload-model", "--serve", "--port", "8000"]
