"""HTML pages, session status/delete, readiness and the vector DB health endpoint."""
import pytest
from fastapi.testclient import TestClient

from src.web.app import app
from src.web.services.pipeline_service import PipelineService
from tests.fakes import build_fake_model_pipeline

SESSION_ID = "11111111-1111-4111-8111-111111111111"

PAGES = [
    "/", "/upload", f"/watermark/{SESSION_ID}", f"/split/{SESSION_ID}", f"/preprocessing/{SESSION_ID}",
    f"/quality_basic/{SESSION_ID}", f"/quality_aesthetic/{SESSION_ID}", f"/embedding/{SESSION_ID}",
    "/search/by-image/some-image", "/search/text", f"/search/{SESSION_ID}", "/batch/upload", "/grouping",
    "/duplicate", "/batch/complete/batch_0a1b2c3d",
]


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    pipeline = build_fake_model_pipeline(str(tmp_path / "processed"), monkeypatch)
    monkeypatch.setattr(PipelineService, "_pipeline", pipeline)
    with TestClient(app) as running:  # runs the lifespan, which creates the real session manager under tmp_path
        yield running


@pytest.mark.parametrize("path", PAGES)
def test_every_page_renders_html(client, path):
    response = client.get(path)

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_session_status_lists_completed_steps_then_the_session_can_be_deleted(client):
    manager = app.state.session_manager
    session_id = SESSION_ID
    manager.create_session(session_id, {"current_step": 2, "collage_info": {}, "original_path": "uploads/x/original.jpg"})

    status = client.get(f"/api/session/{session_id}/status").json()

    assert status["current_step"] == 2
    assert status["original_filename"] == "original.jpg"
    assert status["steps_completed"] == ["collage_info"]
    assert client.delete(f"/api/session/{session_id}").status_code == 200
    assert client.get(f"/api/session/{session_id}/status").status_code == 404


def test_readiness_reports_ready_with_the_fake_pipeline(client):
    response = client.get("/ready")

    assert response.status_code == 200
    assert response.json()["checks"] == {"session_storage": "ok", "vector_db": "ok"}


def test_readiness_is_503_when_the_vector_db_fails(client, monkeypatch):
    def broken_size():
        raise RuntimeError("db down")

    monkeypatch.setattr(PipelineService._pipeline.vector_db, "size", broken_size)

    response = client.get("/ready")

    assert response.status_code == 503
    assert response.json()["status"] == "not_ready"


def test_readiness_is_503_when_session_storage_is_not_writable(client, monkeypatch):
    monkeypatch.setattr(app.state.session_manager, "storage_dir", app.state.session_manager.storage_dir / "missing" / "dir")

    assert client.get("/ready").status_code == 503


def test_vector_db_health_reports_type_and_size(client):
    body = client.get("/api/vector-db/health").json()

    assert body == {"database_type": "InMemoryVectorDB", "total_embeddings": 0, "status": "healthy"}


def test_vector_db_health_reports_errors_instead_of_failing(client, monkeypatch):
    def broken_size():
        raise RuntimeError("db down")

    monkeypatch.setattr(PipelineService._pipeline.vector_db, "size", broken_size)

    assert client.get("/api/vector-db/health").json()["status"] == "error"
