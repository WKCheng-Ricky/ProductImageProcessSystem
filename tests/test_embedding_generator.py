"""EmbeddingGenerator and TextImageEmbeddingGenerator against a fake CLIP model that works on numpy arrays."""
import contextlib
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from src.pipeline import embedding_generator
from src.pipeline.embedding_generator import EmbeddingGenerator, EmbeddingResult, TextImageEmbeddingGenerator
from tests.fakes import product_photo

DIMENSION = 16


class Array:
    """The slice of the torch.Tensor API the generators use, backed by numpy."""

    def __init__(self, values):
        self.values = np.asarray(values, dtype=np.float32)

    def unsqueeze(self, axis):
        return Array(np.expand_dims(self.values, axis))

    def to(self, device):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.values


def preprocess(img):
    pixels = np.asarray(img.resize((4, 4)), dtype=np.float32).flatten()
    return Array(pixels[:DIMENSION])


class FakeClip:
    def encode_image(self, batch):
        return Array(batch.values + 1.0)

    def encode_text(self, tokens):
        return Array(np.tile(tokens.values, (1, 1))[:, :DIMENSION] + 1.0)


def tokenizer(texts):
    seed = sum(ord(c) for c in texts[0])
    return Array(np.random.default_rng(seed).random((1, DIMENSION)))


@pytest.fixture
def fake_clip(monkeypatch):
    requested = {}

    def get_clip_model(arch, pretrained, device):
        requested.update(arch=arch, pretrained=pretrained, device=device)
        return FakeClip(), preprocess, tokenizer, DIMENSION

    monkeypatch.setattr(embedding_generator, "get_model_cache", lambda: SimpleNamespace(get_clip_model=get_clip_model))
    monkeypatch.setattr(embedding_generator, "no_grad", contextlib.nullcontext)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(stack=lambda items: Array(np.stack([i.values for i in items]))))
    return requested


def generator(**options):
    return TextImageEmbeddingGenerator(model_name="clip-vit-base", device="cpu", **options)


def test_supported_alias_resolves_to_architecture_and_weights(fake_clip):
    assert generator().dimension == DIMENSION
    assert (fake_clip["arch"], fake_clip["pretrained"]) == ("ViT-B-32", "openai")


@pytest.mark.parametrize("name,expected", [
    ("ViT-B-32:laion2b_s34b_b79k", ("ViT-B-32", "laion2b_s34b_b79k")),
    ("hf-hub:laion/CLIP-ViT-H-14", ("hf-hub:laion/CLIP-ViT-H-14", None)),
    ("ViT-L-14", ("ViT-L-14", None)),
])
def test_other_model_name_formats_are_parsed(fake_clip, name, expected):
    EmbeddingGenerator(model_name=name, device="cpu").dimension

    assert (fake_clip["arch"], fake_clip["pretrained"]) == expected


def test_model_name_defaults_to_the_environment(fake_clip, monkeypatch):
    monkeypatch.setattr(embedding_generator, "clip_model_from_env", lambda: "ViT-L-14")
    monkeypatch.setattr(embedding_generator, "clip_pretrained_from_env", lambda: "openai")

    assert EmbeddingGenerator(device="cpu").model_name == "clip-vit-large"


def test_environment_model_outside_the_supported_list_uses_the_direct_format(monkeypatch):
    monkeypatch.setattr(embedding_generator, "clip_model_from_env", lambda: "RN50")
    monkeypatch.setattr(embedding_generator, "clip_pretrained_from_env", lambda: "yfcc15m")

    assert EmbeddingGenerator(device="cpu").model_name == "RN50:yfcc15m"


def test_model_name_without_environment_falls_back_to_the_base_clip(monkeypatch):
    monkeypatch.setattr(embedding_generator, "clip_model_from_env", lambda: None)
    monkeypatch.setattr(embedding_generator, "clip_pretrained_from_env", lambda: None)

    assert EmbeddingGenerator(device="cpu").model_name == "clip-vit-base"


def test_auto_device_uses_default_device(monkeypatch):
    monkeypatch.setattr(embedding_generator, "default_device", lambda: "cpu")

    assert EmbeddingGenerator(model_name="clip-vit-base").device == "cpu"


def test_generate_returns_a_unit_vector_and_converts_other_modes(fake_clip):
    result = generator().generate(product_photo().convert("L"))

    assert isinstance(result, EmbeddingResult)
    assert result.dimension == DIMENSION and result.normalized
    assert np.linalg.norm(result.embedding) == pytest.approx(1.0)


def test_generate_without_normalisation_keeps_the_raw_vector(fake_clip):
    result = generator(normalize=False).generate(product_photo())

    assert not result.normalized and np.linalg.norm(result.embedding) != pytest.approx(1.0)


def test_same_image_gives_similarity_one_and_different_images_less(fake_clip):
    gen = generator()
    red, blue = gen.generate(product_photo((200, 30, 30))), gen.generate(product_photo((30, 30, 200)))

    assert gen.similarity(red, red) == pytest.approx(1.0)
    assert gen.similarity(red, blue.embedding) < 1.0


def test_batch_generation_matches_single_generation_and_respects_batch_size(fake_clip):
    gen = generator()
    images = [product_photo((200, 30, 30)), product_photo((30, 200, 30)), product_photo((30, 30, 200))]

    batch = gen.generate_batch(images, batch_size=2)

    assert len(batch) == 3
    assert np.allclose(batch[1].embedding, gen.generate(images[1]).embedding, atol=1e-6)


def test_batch_generation_with_progress_bar_and_unnormalised_output(fake_clip):
    batch = generator(normalize=False).generate_batch([product_photo()], show_progress=True)

    assert len(batch) == 1 and not batch[0].normalized


def test_batch_similarity_scores_every_candidate(fake_clip):
    gen = generator()
    query = gen.generate(product_photo((200, 30, 30)))
    candidates = [gen.generate(product_photo((200, 30, 30))), gen.generate(product_photo((30, 30, 200)))]

    scores = gen.batch_similarity(query, candidates)

    assert scores[0] == pytest.approx(1.0) and scores[1] < scores[0]


def test_text_embedding_is_normalised_and_deterministic_per_text(fake_clip):
    gen = generator()

    first, again = gen.generate_text_embedding("red bottle"), gen.generate_text_embedding("red bottle")

    assert np.allclose(first.embedding, again.embedding)
    assert np.linalg.norm(first.embedding) == pytest.approx(1.0)
    assert not np.allclose(first.embedding, gen.generate_text_embedding("blue cup").embedding)


def test_search_by_text_returns_indices_ranked_by_similarity(fake_clip):
    gen = generator()
    images = [product_photo((200, 30, 30)), product_photo((30, 200, 30)), Image.new("RGB", (64, 64), (0, 0, 255))]

    ranked = gen.search_by_text("anything", images, top_k=2)

    assert len(ranked) == 2
    assert ranked[0][1] >= ranked[1][1]
    assert {index for index, _ in ranked} <= {0, 1, 2}


def test_missing_ml_packages_raise_an_actionable_import_error(monkeypatch):
    def not_installed():
        raise ImportError("torch")

    monkeypatch.setattr(embedding_generator, "get_model_cache", not_installed)

    with pytest.raises(ImportError, match="pip install open_clip_torch"):
        generator().dimension
