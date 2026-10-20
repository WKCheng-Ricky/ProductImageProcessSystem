"""Collage, watermark and preprocessing stages on small synthetic images (no models)."""
import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from src.pipeline.collage_detector import CollageDetector
from src.pipeline.image_preprocessor import ImagePreprocessor
from src.pipeline.watermark_remover import WatermarkRemover
from src.web.app import app
from src.web.routes import upload
from src.web.services.session_manager import SessionManager
from tests.fakes import product_photo

CELL_COLORS = [(200, 30, 30), (30, 40, 200), (30, 200, 40), (200, 200, 30)]


def two_by_two_collage() -> Image.Image:
    grid = Image.new("RGB", (606, 606), "white")
    for index, color in enumerate(CELL_COLORS):
        grid.paste(product_photo(color, (300, 300)), ((index % 2) * 303, (index // 2) * 303))
    return grid


def test_single_product_photo_is_not_a_collage():
    assert CollageDetector().is_collage(product_photo()).is_collage is False


def test_two_by_two_grid_is_detected_and_split_into_four():
    detector = CollageDetector()
    collage = two_by_two_collage()

    info = detector.is_collage(collage)
    split = detector.smart_split(collage, info)

    assert info.is_collage and info.grid_size == (2, 2)
    assert len(split.images) == 4


def test_clean_photo_has_no_watermark_removed():
    result = WatermarkRemover(use_ocr=False).remove_watermark(product_photo())

    assert result.watermarks_removed == 0


def test_standardize_resizes_to_target_size():
    result = ImagePreprocessor(target_size=(256, 256)).standardize(product_photo())

    assert result.image.size == (256, 256)


@pytest.fixture
def client(tmp_path):
    app.state.session_manager = SessionManager(storage_dir=str(tmp_path / "sessions"))
    return TestClient(app)


def test_upload_rejects_files_over_the_size_limit(client, monkeypatch):
    monkeypatch.setattr(upload, "MAX_FILE_SIZE", 1024)
    buffer = io.BytesIO()
    product_photo().save(buffer, format="PNG")

    response = client.post("/api/upload", files={"file": ("big.png", buffer.getvalue(), "image/png")})

    assert response.status_code == 400
    assert "too large" in response.json()["detail"].lower()
