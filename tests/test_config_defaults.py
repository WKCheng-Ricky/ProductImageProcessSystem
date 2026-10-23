"""Pin the central defaults so refactors cannot drift them silently."""
import inspect

from src.pipeline import config
from src.pipeline.milvus_db import ProductGroupManager
from src.pipeline.product_image_pipeline import ProductImagePipeline


def test_central_defaults_match_historic_behaviour():
    assert config.DEFAULT_TARGET_SIZE == (512, 512)
    assert config.PIPELINE_SIMILARITY_THRESHOLD == 0.85
    assert config.GROUPING_SIMILARITY_THRESHOLD == 0.90
    assert config.DUPLICATE_SIMILARITY_THRESHOLD == 0.999
    assert config.QUALITY_THRESHOLD == 0.3
    assert (config.DEFAULT_CLIP_MODEL, config.DEFAULT_CLIP_PRETRAINED) == ("ViT-B-32", "openai")


def test_pipeline_constructor_uses_central_defaults():
    params = inspect.signature(ProductImagePipeline.__init__).parameters

    assert params["target_size"].default == config.DEFAULT_TARGET_SIZE
    assert params["similarity_threshold"].default == config.PIPELINE_SIMILARITY_THRESHOLD
    assert params["quality_threshold"].default == config.QUALITY_THRESHOLD


def test_group_manager_uses_grouping_threshold():
    params = inspect.signature(ProductGroupManager.__init__).parameters

    assert params["similarity_threshold"].default == config.GROUPING_SIMILARITY_THRESHOLD


def test_clip_env_readers(monkeypatch):
    monkeypatch.setenv("CLIP_MODEL", "ViT-L-14")
    monkeypatch.delenv("CLIP_PRETRAINED", raising=False)

    assert config.clip_model_from_env() == "ViT-L-14"
    assert config.clip_pretrained_from_env() is None
