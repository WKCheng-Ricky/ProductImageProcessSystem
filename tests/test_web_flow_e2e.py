"""Walks the multi-page web flow over HTTP against the real pipeline, with only the CLIP parts faked."""
import io

import pytest
from fastapi.testclient import TestClient

from src.web.app import app
from src.web.services.pipeline_service import PipelineService
from src.web.services.session_manager import SessionManager
from tests.fakes import build_fake_model_pipeline, product_photo


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)  # routes write under relative uploads/
    app.state.session_manager = SessionManager(storage_dir=str(tmp_path / "sessions"))
    pipeline = build_fake_model_pipeline(str(tmp_path / "processed"), monkeypatch)
    monkeypatch.setattr(PipelineService, "_pipeline", pipeline)
    return TestClient(app)


def _jpeg(color):
    buffer = io.BytesIO()
    product_photo(color).save(buffer, format="JPEG")
    return buffer.getvalue()


def _upload(client, color):
    response = client.post("/api/upload", files={"file": ("product.jpg", _jpeg(color), "image/jpeg")})
    assert response.status_code == 200, response.text
    return response.json()["session_id"]


def _run_pages(client, session_id, store_id):
    base = f"/api/session/{session_id}"
    assert client.post(f"{base}/watermark").status_code == 200
    assert client.post(f"{base}/quality").status_code == 200
    assert client.post(f"{base}/preprocess").status_code == 200
    embedding = client.post(f"{base}/embedding", params={"id": store_id})
    assert embedding.status_code == 200, embedding.text
    return embedding.json()


def test_full_page_flow_stores_embedding_and_finds_the_same_product(client):
    first = _upload(client, (200, 30, 30))
    _run_pages(client, first, store_id="red-original")

    second = _upload(client, (200, 30, 30))
    _run_pages(client, second, store_id="red-duplicate")
    search = client.post(f"/api/session/{second}/search", params={"threshold": 0.7})

    assert search.status_code == 200, search.text
    ids = [result["id"] for result in search.json()["results"]]
    assert "red-original" in ids


def test_search_before_embedding_is_rejected(client):
    session_id = _upload(client, (200, 30, 30))

    assert client.post(f"/api/session/{session_id}/search").status_code == 400


def test_batch_upload_and_process_saves_unique_images_and_skips_duplicates(client):
    files = [
        ("files", ("a.jpg", _jpeg((200, 30, 30)), "image/jpeg")),
        ("files", ("b.jpg", _jpeg((200, 30, 30)), "image/jpeg")),
        ("files", ("c.jpg", _jpeg((30, 40, 200)), "image/jpeg")),
    ]
    upload = client.post("/api/batch/upload", files=files)
    assert upload.status_code == 200, upload.text
    batch_id = upload.json()["batch_id"]

    processed = client.post(f"/api/batch/{batch_id}/process")

    assert processed.status_code == 200, processed.text
    body = processed.json()
    assert body["failed"] == 0
    assert body["skipped_duplicates"] == 1
    assert body["successful"] == 2
