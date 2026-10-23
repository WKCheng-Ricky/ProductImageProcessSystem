import numpy as np
import pytest
from PIL import Image, ImageFilter

from src.pipeline.basic_quality_scorer import BasicQualityScorer


@pytest.fixture(scope="module")
def scorer():
    return BasicQualityScorer()


def _checkerboard(size=512, brightness=1.0):
    tile = np.indices((size, size)).sum(axis=0) // 32 % 2
    pixels = (tile * 255 * brightness).astype(np.uint8)
    return Image.fromarray(pixels).convert("RGB")


def test_scores_stay_in_unit_range(scorer):
    result = scorer.score(_checkerboard())

    for value in (result.overall, result.sharpness, result.brightness, result.contrast):
        assert 0.0 <= value <= 1.0


def test_sharp_image_scores_higher_than_blurred(scorer):
    sharp = _checkerboard()
    blurred = sharp.filter(ImageFilter.GaussianBlur(radius=12))

    assert scorer.score(sharp).sharpness > scorer.score(blurred).sharpness


def test_very_dark_image_has_lower_brightness_score(scorer):
    normal = _checkerboard(brightness=1.0)
    dark = _checkerboard(brightness=0.03)

    assert scorer.score(dark).brightness < scorer.score(normal).brightness


def test_tiny_image_has_lower_resolution_score(scorer):
    assert (
        scorer.score(_checkerboard(size=64)).resolution_score
        < scorer.score(_checkerboard(size=512)).resolution_score
    )


def test_is_acceptable_follows_min_score(scorer):
    result = scorer.score(_checkerboard())

    assert result.is_acceptable(min_score=0.0)
    assert not result.is_acceptable(min_score=1.01)
