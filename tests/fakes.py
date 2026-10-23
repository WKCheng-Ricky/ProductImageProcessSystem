"""Deterministic stand-ins for the model-backed components, so the real pipeline code can run in CI."""
from types import SimpleNamespace

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


class InMemoryMilvusClient:
    """Stateful stand-in for pymilvus.MilvusClient covering the calls milvus_db.py makes."""

    VECTOR_FIELD_TYPE = 101

    def __init__(self):
        self.collections = {}
        self.index_requests = []
        self.collections_that_waited_for_index = []

    def list_collections(self):
        return list(self.collections)

    def create_schema(self, auto_id=False, enable_dynamic_field=False, description=""):
        return SimpleNamespace(description=description, fields=[], add_field=lambda **field: None)

    def prepare_index_params(self):
        return SimpleNamespace(add_index=lambda **options: None)

    def create_collection(self, collection_name, schema, index_params=None):
        if index_params is not None:
            self.collections_that_waited_for_index.append(collection_name)
        self.collections[collection_name] = {"rows": {}, "description": schema.description, "schema": schema}

    def create_index(self, collection_name, index_params, sync=True, **options):
        self.index_requests.append((collection_name, sync))

    def drop_collection(self, collection_name):
        self.collections.pop(collection_name)

    def load_collection(self, collection_name):
        pass

    def describe_collection(self, collection_name):
        collection = self.collections[collection_name]
        first_row = next(iter(collection["rows"].values()), None)
        dim = len(first_row["vector"]) if first_row and "vector" in first_row else collection.get("dim")
        return {"fields": [{"type": self.VECTOR_FIELD_TYPE, "params": {"dim": dim}}], "description": collection["description"]}

    def insert(self, collection_name, data):
        rows = self.collections[collection_name]["rows"]
        for row in data:
            rows[row.get("id", row.get("group_id"))] = dict(row)

    def delete(self, collection_name, filter):
        self.collections[collection_name]["rows"].pop(_quoted_value(filter), None)

    def query(self, collection_name, filter, output_fields, limit=16000, offset=0):
        rows = list(self.collections[collection_name]["rows"].values())
        if filter:
            wanted = _quoted_value(filter)
            rows = [row for row in rows if wanted in (row.get("id"), row.get("group_id"))]
        return [{field: row[field] for field in output_fields} for row in rows[offset:offset + limit]]

    def search(self, collection_name, data, limit, output_fields):
        query = np.asarray(data[0], dtype=np.float32)
        scored = [
            (float(np.dot(query, np.asarray(row["vector"], dtype=np.float32))), row)
            for row in self.collections[collection_name]["rows"].values()
        ]
        scored.sort(key=lambda pair: pair[0], reverse=True)
        return [[
            {"id": row["id"], "distance": score, "entity": {"metadata_json": row["metadata_json"]}}
            for score, row in scored[:limit]
        ]]

    def get_collection_stats(self, collection_name):
        return {"row_count": len(self.collections[collection_name]["rows"])}


def _quoted_value(filter_expression: str) -> str:
    return filter_expression.split('"')[1]



CELL_COLORS = [(200, 30, 30), (30, 40, 200), (30, 200, 40), (200, 200, 30)]


def two_by_two_collage() -> Image.Image:
    grid = Image.new("RGB", (606, 606), "white")
    for index, color in enumerate(CELL_COLORS):
        grid.paste(product_photo(color, (300, 300)), ((index % 2) * 303, (index // 2) * 303))
    return grid
