"""Ids that reach the vector DB must be plain tokens; anything else is refused at the route."""
import io
from urllib.parse import quote

import pytest
from fastapi.testclient import TestClient

from src.web.app import app
from src.web.services.pipeline_service import PipelineService
from src.web.services.session_manager import SessionManager
from tests.fakes import build_fake_model_pipeline, product_photo

UNKNOWN_SESSION = "22222222-2222-4222-8222-222222222222"
BAD_IDS = [
    'x" or id != "', "tail\\", "it's", "a'] or id not in ['", "line\nbreak", "carriage\rreturn",
    "tab\there", "x" * 101, "semi;colon", "paren(1)",
]
GOOD_IDS = ["red", "with space", "IMG 001", "red-shade", "a.b_c", "6f1c2a9e-1b7d-4c1e-9d55-0a1b2c3d4e5f", "商品-01", "x" * 100]


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    app.state.session_manager = SessionManager(storage_dir=str(tmp_path / "sessions"))
    fake = build_fake_model_pipeline(str(tmp_path / "processed"), monkeypatch)
    monkeypatch.setattr(PipelineService, "_pipeline", fake)
    return fake


@pytest.fixture
def client(pipeline):
    return TestClient(app)


@pytest.fixture
def db_calls(pipeline, monkeypatch):
    calls = []
    real_get = pipeline.vector_db.get
    monkeypatch.setattr(pipeline.vector_db, "get", lambda id: calls.append(id) or real_get(id))
    return calls


def encoded(value):
    return quote(value, safe="")


def photo_upload():
    buffer = io.BytesIO()
    product_photo((200, 30, 30)).save(buffer, format="JPEG")
    return {"file": ("p.jpg", buffer.getvalue(), "image/jpeg")}


@pytest.mark.parametrize("bad_id", BAD_IDS)
def test_search_by_image_refuses_a_bad_id_before_the_database(client, db_calls, bad_id):
    assert client.get(f"/api/search/by-image/{encoded(bad_id)}").status_code == 422
    assert db_calls == []


@pytest.mark.parametrize("bad_id", BAD_IDS)
def test_group_lookup_refuses_a_bad_id(client, bad_id):
    assert client.get(f"/api/grouping/group/{encoded(bad_id)}").status_code == 422


@pytest.mark.parametrize("bad_id", BAD_IDS)
def test_search_page_refuses_a_bad_id(client, bad_id):
    assert client.get(f"/search/by-image/{encoded(bad_id)}").status_code == 422


@pytest.mark.parametrize("bad_id", BAD_IDS)
def test_upload_refuses_a_bad_id(client, bad_id):
    assert client.post("/api/upload", params={"id": bad_id}, files=photo_upload()).status_code == 422


@pytest.mark.parametrize("bad_id", BAD_IDS)
def test_embedding_refuses_a_bad_id(client, bad_id):
    response = client.post(f"/api/session/{UNKNOWN_SESSION}/embedding", params={"id": bad_id})

    assert response.status_code == 422


@pytest.mark.parametrize("good_id", GOOD_IDS)
def test_search_by_image_accepts_a_plain_id(client, good_id):
    assert client.get(f"/api/search/by-image/{encoded(good_id)}").status_code == 404


@pytest.mark.parametrize("good_id", GOOD_IDS)
def test_group_lookup_accepts_a_plain_id(client, good_id):
    assert client.get(f"/api/grouping/group/{encoded(good_id)}").status_code == 404


@pytest.mark.parametrize("good_id", GOOD_IDS)
def test_upload_accepts_a_plain_id(client, good_id):
    assert client.post("/api/upload", params={"id": good_id}, files=photo_upload()).status_code == 200


def test_upload_and_embedding_still_work_without_an_id(client):
    assert client.post("/api/upload", files=photo_upload()).status_code == 200
    assert client.post(f"/api/session/{UNKNOWN_SESSION}/embedding").status_code == 404
