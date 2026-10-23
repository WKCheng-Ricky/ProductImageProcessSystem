"""Page-by-page API behaviour: result endpoints, validation and not-found paths, and grouping on Milvus."""
import io

import pytest
from fastapi.testclient import TestClient

from src.pipeline.milvus_db import MilvusVectorLiteDB
from src.web.app import app
from src.web.services.browse_service import BrowseService
from src.web.services.pipeline_service import PipelineService
from src.web.services.session_manager import SessionManager
from tests.fakes import FAKE_DIMENSION, FakeEmbeddingGenerator, build_fake_model_pipeline, product_photo, two_by_two_collage

UNKNOWN_SESSION = "22222222-2222-4222-8222-222222222222"
RED = (200, 30, 30)
RED_SHADE = (170, 50, 50)  # cosine 0.99 to RED with the fake embedder: similar, but not a duplicate
BLUE = (30, 30, 200)
GREEN = (30, 200, 30)


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


def upload(client, color=RED):
    buffer = io.BytesIO()
    product_photo(color).save(buffer, format="JPEG")
    response = client.post("/api/upload", files={"file": ("p.jpg", buffer.getvalue(), "image/jpeg")})
    assert response.status_code == 200, response.text
    return response.json()["session_id"]


def run_through_embedding(client, color, product_id):
    session_id = upload(client, color)
    base = f"/api/session/{session_id}"
    for step in ("watermark", "quality", "preprocess"):
        assert client.post(f"{base}/{step}").status_code == 200
    assert client.post(f"{base}/embedding", params={"id": product_id}).status_code == 200
    return session_id


def test_each_stage_result_can_be_read_back_after_it_ran(client):
    session_id = run_through_embedding(client, RED, "red")
    base = f"/api/session/{session_id}"

    for step in ("watermark", "quality", "preprocess", "embedding", "collage"):
        assert client.get(f"{base}/{step}").status_code == 200, step


def test_split_runs_and_reads_back(client):
    session_id = upload(client)

    posted = client.post(f"/api/session/{session_id}/split")
    fetched = client.get(f"/api/session/{session_id}/split")

    assert posted.status_code == fetched.status_code == 200
    assert fetched.json()["num_images"] == 1


def test_preprocessing_can_be_bypassed(client):
    session_id = upload(client)

    response = client.post(f"/api/session/{session_id}/preprocess/bypass")

    assert response.status_code == 200
    assert response.json()["success"] is True


@pytest.mark.parametrize("method,path", [
    ("get", "collage"), ("post", "watermark"), ("get", "watermark"), ("post", "split"), ("get", "split"),
    ("post", "quality"), ("get", "quality"), ("post", "preprocess"), ("get", "preprocess"),
    ("post", "preprocess/bypass"), ("post", "embedding"), ("get", "embedding"), ("post", "search"), ("get", "search"),
])
def test_unknown_session_is_not_found_on_every_stage(client, method, path):
    response = getattr(client, method)(f"/api/session/{UNKNOWN_SESSION}/{path}")

    assert response.status_code == 404


def test_results_are_not_available_before_their_stage_ran(client):
    base = f"/api/session/{upload(client)}"

    for step in ("watermark", "quality", "preprocess", "embedding"):
        assert client.get(f"{base}/{step}").status_code in (400, 404), step


@pytest.mark.parametrize("params", [{"top_k": 0}, {"top_k": 101}, {"threshold": -0.1}, {"threshold": 1.1}])
def test_search_rejects_out_of_range_parameters(client, params):
    session_id = run_through_embedding(client, RED, "red")

    assert client.post(f"/api/session/{session_id}/search", params=params).status_code == 400


def test_search_ranks_the_matching_colour_first_and_hides_dissimilar_ones(client):
    run_through_embedding(client, RED, "red")
    run_through_embedding(client, BLUE, "blue")
    query = run_through_embedding(client, RED_SHADE, "red-shade")

    results = client.post(f"/api/session/{query}/search", params={"threshold": 0.9}).json()["results"]

    assert {r["id"] for r in results} >= {"red", "red-shade"}
    assert "blue" not in {r["id"] for r in results}
    assert client.get(f"/api/session/{query}/search").status_code == 200


def test_an_exact_duplicate_is_reported_and_not_stored_twice(client, pipeline):
    run_through_embedding(client, RED, "red")
    second = run_through_embedding(client, RED, "red-copy")

    stored = client.get(f"/api/session/{second}/embedding").json()

    assert stored["is_duplicate"] is True and stored["duplicate_of"] == "red"
    assert stored["saved_to_db"] is False
    assert pipeline.vector_db.size() == 1


def test_search_by_stored_image_id(client):
    run_through_embedding(client, RED, "red")
    run_through_embedding(client, RED_SHADE, "red-shade")

    response = client.get("/api/search/by-image/red", params={"threshold": 0.9})

    assert response.status_code == 200
    assert "red-shade" in {r["id"] for r in response.json()["results"]}


def test_search_by_unknown_image_id_is_not_found(client):
    assert client.get("/api/search/by-image/never-stored").status_code == 404


@pytest.fixture
def milvus_pipeline(pipeline, milvus_client):
    pipeline._vector_db = MilvusVectorLiteDB(db_path="unused.db", dimension=FAKE_DIMENSION, model_name="fake-thumbnail")
    return pipeline


def test_grouping_with_no_products_returns_an_empty_result(milvus_pipeline, client):
    response = client.post("/api/grouping/run")

    assert response.status_code == 200
    assert response.json()["total_groups"] == 0


def test_grouping_run_then_read_results_stats_and_one_group(milvus_pipeline, client):
    run_through_embedding(client, RED, "red")
    run_through_embedding(client, RED_SHADE, "red-shade")
    run_through_embedding(client, BLUE, "blue")

    run = client.post("/api/grouping/run", params={"similarity_threshold": 0.95}).json()
    results = client.get("/api/grouping/results").json()
    stats = client.get("/api/grouping/stats")
    one = client.get(f"/api/grouping/group/{run['groups'][0]['group_id']}")

    assert run["total_groups"] == 2 and run["groups_with_multiple_members"] == 1
    assert results["total_groups"] == 2
    assert stats.status_code == 200
    assert one.status_code == 200


def test_grouping_rejects_a_threshold_outside_zero_to_one(milvus_pipeline, client):
    assert client.post("/api/grouping/run", params={"similarity_threshold": 1.5}).status_code == 400


def test_unknown_group_is_not_found(milvus_pipeline, client):
    assert client.get("/api/grouping/group/not-a-group").status_code == 404


def upload_collage(client):
    buffer = io.BytesIO()
    two_by_two_collage().save(buffer, format="JPEG")
    response = client.post("/api/upload", files={"file": ("grid.jpg", buffer.getvalue(), "image/jpeg")})
    assert response.status_code == 200, response.text
    return response.json()["session_id"]


def test_collage_is_split_into_its_four_images(client):
    session_id = upload_collage(client)

    split = client.post(f"/api/session/{session_id}/split").json()

    assert split["num_images"] == 4 and len(split["split_image_urls"]) == 4
    assert client.get(f"/api/session/{session_id}/split").json()["num_images"] == 4


def test_split_can_keep_only_the_selected_images(client):
    session_id = upload_collage(client)

    split = client.post(f"/api/session/{session_id}/split", json=[0, 3]).json()

    assert split["num_images"] == 2


class FakeTextGenerator:
    def __init__(self, **options):
        pass

    def generate_text_embedding(self, text):
        return FakeEmbeddingGenerator().generate(product_photo(RED))


def test_text_search_returns_the_closest_stored_product_first(client, monkeypatch):
    monkeypatch.setattr("src.web.routes.search.TextImageEmbeddingGenerator", FakeTextGenerator)
    run_through_embedding(client, RED, "red")
    run_through_embedding(client, BLUE, "blue")

    body = client.post("/api/search/by-text", json={"query": "red bottle", "threshold": 0.0}).json()

    assert [r["id"] for r in body["results"]][0] == "red"
    assert len(body["results"]) == 2
    assert body["query_text"] == "red bottle"


def test_text_search_failure_is_a_generic_500(client, monkeypatch):
    class BrokenTextGenerator(FakeTextGenerator):
        def generate_text_embedding(self, text):
            raise RuntimeError("secret model path /opt/models")

    monkeypatch.setattr("src.web.routes.search.TextImageEmbeddingGenerator", BrokenTextGenerator)

    response = client.post("/api/search/by-text", json={"query": "anything"})

    assert response.status_code == 500
    assert "secret" not in response.text


def test_preprocessing_after_a_split_uses_the_selected_split_image(client):
    session_id = upload_collage(client)
    client.post(f"/api/session/{session_id}/split")

    response = client.post(f"/api/session/{session_id}/preprocess", params={"remove_bg": False, "resize": False})

    assert response.status_code == 200
    assert response.json()["original_size"] == [303, 303]  # one cell of the 606x606 grid, not the whole collage


def test_preprocessing_honours_its_switches(client):
    session_id = upload(client)

    response = client.post(
        f"/api/session/{session_id}/preprocess",
        params={"remove_bg": False, "crop": False, "remove_watermark": False, "remove_text": True, "resize": False},
    )

    body = response.json()
    assert (body["watermark_removed"], body["text_removed"]) == (False, True)


def test_preprocessing_failure_is_a_generic_500(client, pipeline, monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("internal detail")

    monkeypatch.setattr(pipeline.preprocessor, "standardize", broken)

    response = client.post(f"/api/session/{upload(client)}/preprocess")

    assert response.status_code == 500
    assert "internal detail" not in response.text


def test_random_images_returns_stored_images_with_urls(client):
    run_through_embedding(client, RED, "red")
    run_through_embedding(client, BLUE, "blue")

    response = client.get("/api/search/random")

    body = response.json()
    assert response.status_code == 200
    assert body["mode"] == "random"
    assert body["total_in_db"] == 2
    assert {image["id"] for image in body["images"]} == {"red", "blue"}
    assert all(image["image_url"].startswith("/uploads/sessions/") for image in body["images"])


def test_random_images_on_an_empty_db_is_an_empty_list(client):
    response = client.get("/api/search/random")

    assert response.status_code == 200
    assert response.json()["images"] == []
    assert response.json()["total_in_db"] == 0


@pytest.mark.parametrize("count", [0, 51])
def test_random_images_rejects_count_outside_one_to_fifty(client, count):
    assert client.get("/api/search/random", params={"count": count}).status_code == 400


def test_random_images_failure_is_a_generic_500(client, monkeypatch):
    def broken(self, count):
        raise RuntimeError("secret db path /opt/data")

    monkeypatch.setattr(BrowseService, "sample", broken)

    response = client.get("/api/search/random")

    assert response.status_code == 500
    assert response.json()["detail"] == "Error sampling images"
    assert "secret" not in response.text


def test_random_images_uses_groups_on_milvus(milvus_pipeline, client):
    run_through_embedding(client, RED, "red")
    run_through_embedding(client, RED_SHADE, "red-shade")
    run_through_embedding(client, BLUE, "blue")
    client.post("/api/grouping/run", params={"similarity_threshold": 0.95})

    body = client.get("/api/search/random").json()

    assert body["mode"] == "grouped"
    assert len(body["images"]) == 2
    assert body["ungrouped_count"] == 0


def test_in_memory_mode_never_opens_the_groups_db(client, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("opened the groups db")

    monkeypatch.setattr("src.web.services.browse_service.ProductGroupsDB", forbidden)
    run_through_embedding(client, RED, "red")

    response = client.get("/api/search/random")

    assert response.status_code == 200
    assert response.json()["mode"] == "random"


def test_browse_page_renders_and_is_not_taken_for_a_session(client):
    response = client.get("/search/browse")

    assert response.status_code == 200
    assert "/api/search/random" in response.text
    assert "in any group yet, so this is a random sample" in response.text


def test_home_page_links_to_browse(client):
    assert 'href="/search/browse"' in client.get("/").text


def test_by_image_page_links_back_to_browse(client):
    assert 'href="/search/browse"' in client.get("/search/by-image/red").text


def test_grouping_still_requires_the_milvus_vector_db(client, milvus_client):
    run_through_embedding(client, RED, "red")

    response = client.post("/api/grouping/run")

    assert response.status_code == 500


def test_browse_does_not_create_the_groups_collection(milvus_pipeline, client, milvus_client):
    run_through_embedding(client, RED, "red")

    body = client.get("/api/search/random").json()

    assert body["mode"] == "random" and len(body["images"]) == 1
    assert "product_groups" not in milvus_client.collections


def test_browse_stops_using_groups_once_a_new_image_is_added(milvus_pipeline, client):
    run_through_embedding(client, RED, "red")
    run_through_embedding(client, RED_SHADE, "red-shade")
    run_through_embedding(client, BLUE, "blue")
    client.post("/api/grouping/run", params={"similarity_threshold": 0.95})
    before = client.get("/api/search/random").json()

    run_through_embedding(client, GREEN, "green")
    after = client.get("/api/search/random").json()

    assert before["mode"] == "grouped" and before["ungrouped_count"] == 0
    assert after["mode"] == "random" and after["ungrouped_count"] == 1
    assert "green" in {image["id"] for image in after["images"]}
