"""Opt-in checks against real CLIP weights. Run with `pytest -m model`.

Assert behavioural properties, not exact numbers, so they hold across model variants
(CLIP_MODEL / CLIP_PRETRAINED can select a smaller model such as ViT-B-32).
"""
import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageEnhance

pytestmark = pytest.mark.model

EXPECTED_DIMENSIONS = {"ViT-B-32": 512, "ViT-L-14": 768}


@pytest.fixture(scope="module")
def generator():
    pytest.importorskip("torch")
    pytest.importorskip("open_clip")
    from src.pipeline.embedding_generator import TextImageEmbeddingGenerator

    return TextImageEmbeddingGenerator(device="cpu")


def solid_image(color):
    return Image.new("RGB", (224, 224), color)


def checkerboard_image():
    image = Image.new("RGB", (224, 224), "white")
    draw = ImageDraw.Draw(image)
    for row in range(0, 224, 28):
        for col in range(0, 224, 28):
            if (row + col) // 28 % 2 == 0:
                draw.rectangle([col, row, col + 27, row + 27], fill="black")
    return image


def test_dimension_matches_model_architecture(generator):
    architecture = generator.model_name.split(":")[0]
    for key, (arch, _) in generator.SUPPORTED_MODELS.items():
        if key == generator.model_name:
            architecture = arch
    if architecture not in EXPECTED_DIMENSIONS:
        pytest.skip(f"no expected dimension recorded for {architecture}")
    assert generator.dimension == EXPECTED_DIMENSIONS[architecture]


def test_image_embedding_is_unit_norm(generator):
    result = generator.generate(solid_image("red"))
    assert result.dimension == generator.dimension
    assert result.normalized
    assert np.linalg.norm(result.embedding) == pytest.approx(1.0, abs=1e-4)


def test_same_image_has_cosine_one(generator):
    first = generator.generate(checkerboard_image())
    second = generator.generate(checkerboard_image())
    assert generator.similarity(first, second) == pytest.approx(1.0, abs=1e-4)


def test_lightly_augmented_copy_stays_similar(generator):
    original = checkerboard_image()
    brighter = ImageEnhance.Brightness(original).enhance(1.15)
    similarity = generator.similarity(generator.generate(original), generator.generate(brighter))
    assert similarity > 0.9


def test_unrelated_images_are_less_similar_than_a_copy(generator):
    checkerboard = generator.generate(checkerboard_image())
    red = generator.generate(solid_image("red"))
    copy = generator.generate(checkerboard_image())
    assert generator.similarity(checkerboard, red) < generator.similarity(checkerboard, copy)
    assert generator.similarity(checkerboard, red) < 0.9


def test_text_query_ranks_matching_image_first(generator):
    images = {"red": solid_image("red"), "blue": solid_image("blue"), "green": solid_image("green")}
    embeddings = {name: generator.generate(image) for name, image in images.items()}
    query = generator.generate_text_embedding("a solid blue color")
    ranking = sorted(embeddings, key=lambda name: generator.similarity(query, embeddings[name]), reverse=True)
    assert ranking[0] == "blue"
