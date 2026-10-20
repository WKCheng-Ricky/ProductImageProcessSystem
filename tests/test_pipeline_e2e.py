"""End-to-end runs of the real ProductImagePipeline with only the CLIP-backed parts faked."""
import numpy as np
import pytest

from tests.fakes import FAKE_DIMENSION, build_fake_model_pipeline, product_photo

RED = (200, 30, 30)
BLUE = (30, 40, 200)


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    return build_fake_model_pipeline(str(tmp_path / "processed"), monkeypatch)


@pytest.fixture
def photos(tmp_path):
    paths = {}
    for name, color in {"red": RED, "red_copy": RED, "blue": BLUE}.items():
        path = tmp_path / f"{name}.jpg"
        product_photo(color).save(path)
        paths[name] = str(path)
    return paths


def test_single_image_runs_every_stage_and_is_stored(pipeline, photos):
    result = pipeline.process_image(photos["red"], id="red")

    assert result.success, result.error_message
    assert result.is_collage is False and result.num_split_images == 1
    assert result.quality_score is not None and 0 <= result.quality_score <= 1
    assert result.embedding.shape == (FAKE_DIMENSION,)
    assert np.linalg.norm(result.embedding) == pytest.approx(1.0, abs=1e-5)
    assert pipeline.vector_db.size() == 1


def test_processed_image_is_saved_at_target_size(pipeline, photos):
    from PIL import Image

    result = pipeline.process_image(photos["red"], id="red")

    with Image.open(result.processed_image_path) as processed:
        assert processed.size == pipeline.target_size


def test_second_identical_upload_finds_the_first_as_a_near_perfect_match(pipeline, photos):
    pipeline.process_image(photos["red"], id="red")

    result = pipeline.process_image(photos["red_copy"], id="red_copy")

    assert result.similar_products[0]["id"] == "red"
    assert result.similar_products[0]["score"] > 0.99


def test_different_product_scores_below_identical_one(pipeline, photos):
    pipeline.process_image(photos["red"], id="red")
    pipeline.process_image(photos["blue"], id="blue")

    result = pipeline.process_image(photos["red_copy"], id="red_copy", search_similar=True)

    scores = {match["id"]: match["score"] for match in result.similar_products}
    assert scores["red"] > scores.get("blue", -1)


def test_search_by_image_returns_stored_product_first(pipeline, photos):
    pipeline.process_image(photos["red"], id="red")
    pipeline.process_image(photos["blue"], id="blue")

    matches = pipeline.search_by_image(photos["red_copy"], top_k=2)

    assert matches[0].id == "red"


def test_batch_processing_handles_a_directory_and_a_bad_file(pipeline, photos, tmp_path):
    (tmp_path / "broken.jpg").write_bytes(b"not an image")

    results = pipeline.process_batch(str(tmp_path), save_processed=False)

    by_ok = {r.image_path.rsplit("/", 1)[-1]: r.success for r in results}
    assert by_ok["red.jpg"] and by_ok["blue.jpg"] and by_ok["red_copy.jpg"]
    assert by_ok["broken.jpg"] is False


def test_missing_file_reports_failure_instead_of_raising(pipeline, tmp_path):
    result = pipeline.process_image(str(tmp_path / "nope.jpg"), id="x")

    assert result.success is False
    assert result.error_message
    assert pipeline.vector_db.size() == 0
