import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src.web.app import app
from src.web.services.session_manager import SessionManager


@pytest.fixture
def client(tmp_path):
    app.state.session_manager = SessionManager(storage_dir=str(tmp_path / "sessions"))
    return TestClient(app)


def test_health_reports_healthy(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


SESSION_ID = "11111111-1111-4111-8111-111111111111"


def test_unknown_session_status_is_404(client):
    assert client.get("/api/session/does-not-exist/status").status_code == 404


def test_session_status_lists_completed_steps(client):
    manager = app.state.session_manager
    manager.create_session(SESSION_ID, {"current_step": 2, "collage_info": {}})

    body = client.get(f"/api/session/{SESSION_ID}/status").json()

    assert body["current_step"] == 2
    assert body["steps_completed"] == ["collage_info"]


def test_delete_session_removes_it(client):
    app.state.session_manager.create_session(SESSION_ID, {})

    assert client.delete(f"/api/session/{SESSION_ID}").status_code == 200
    assert client.get(f"/api/session/{SESSION_ID}/status").status_code == 404


def test_upload_rejects_disallowed_extension(client):
    response = client.post("/api/upload", files={"file": ("notes.txt", b"hello", "text/plain")})

    assert response.status_code == 400


def test_upload_rejects_corrupt_image(client):
    response = client.post("/api/upload", files={"file": ("bad.jpg", b"not an image", "image/jpeg")})

    assert response.status_code == 400


def test_health_includes_version(client):
    assert "version" in client.get("/health").json()


def test_ready_without_pipeline_does_not_load_models(client):
    response = client.get("/ready")

    assert response.status_code == 200
    assert response.json()["checks"] == {"session_storage": "ok", "vector_db": "not_initialised"}


def test_ready_returns_503_when_storage_unwritable(client, tmp_path, monkeypatch):
    monkeypatch.setattr(app.state.session_manager, "storage_dir", tmp_path / "missing" / "dir")

    response = client.get("/ready")

    assert response.status_code == 503
    assert response.json()["status"] == "not_ready"


def _png_bytes():
    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), "white").save(buffer, format="PNG")
    return buffer.getvalue()


def test_internal_errors_do_not_leak_exception_text(client, monkeypatch):
    class ExplodingCollageDetector:
        def is_collage(self, image):
            raise RuntimeError("secret internal detail /srv/models/key")

    class StubPipeline:
        collage_detector = ExplodingCollageDetector()

    monkeypatch.setattr("src.web.routes.upload.get_pipeline", lambda: StubPipeline())

    response = client.post(
        "/api/upload",
        params={"skip_watermark_removal": True},
        files={"file": ("a.png", _png_bytes(), "image/png")},
    )

    assert response.status_code == 500
    assert "secret" not in response.text
