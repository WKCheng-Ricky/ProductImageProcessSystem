"""Deterministic stand-ins for the model-backed components, so the real pipeline code can run in CI."""
import numpy as np
from PIL import Image

from src.pipeline.embedding_generator import EmbeddingResult
from src.pipeline.product_image_pipeline import ProductImagePipeline
from src.pipeline.product_grouper import InMemoryVectorDB
from src.pipeline.quality_scorer import ImageQualityScorer

FAKE_DIMENSION = 192


class FakeEmbeddingGenerator:
    """Embeds an image as its 8x8 RGB thumbnail: same picture -> same vector, different colours -> different vectors."""

    model_name = "fake-thumbnail"
    device = "cpu"
    dimension = FAKE_DIMENSION

    def generate(self, img: Image.Image) -> EmbeddingResult:
        thumbnail = np.asarray(img.convert("RGB").resize((8, 8)), dtype=np.float32).flatten()
        vector = thumbnail - thumbnail.mean()
        norm = np.linalg.norm(vector)
        vector = vector / norm if norm > 0 else vector
        return EmbeddingResult(embedding=vector, model_name=self.model_name, dimension=FAKE_DIMENSION, normalized=True)


def build_fake_model_pipeline(output_dir: str, monkeypatch) -> ProductImagePipeline:
    """Real ProductImagePipeline wiring with only the CLIP-backed pieces replaced."""
    monkeypatch.setattr(
        "src.pipeline.product_image_pipeline.ImageQualityScorer",
        lambda: ImageQualityScorer(enable_aesthetic=False),
    )
    pipeline = ProductImagePipeline(output_dir=output_dir)
    pipeline._embedding_generator = FakeEmbeddingGenerator()
    pipeline._vector_db = InMemoryVectorDB()
    return pipeline


def product_photo(color=(200, 30, 30), size=(400, 400)) -> Image.Image:
    """A textured coloured rectangle on a white background, standing in for a product shot."""
    rng = np.random.default_rng(7)
    canvas = np.full((size[1], size[0], 3), 255, dtype=np.uint8)
    top, bottom = size[1] // 5, size[1] * 4 // 5
    left, right = size[0] // 5, size[0] * 4 // 5
    noise = rng.integers(-15, 15, (bottom - top, right - left, 3))
    canvas[top:bottom, left:right] = np.clip(np.array(color) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(canvas)
