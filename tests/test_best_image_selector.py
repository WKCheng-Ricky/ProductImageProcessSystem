"""BestImageSelector picks the clearest, best-framed image and degrades safely."""
import numpy as np
import pytest
from PIL import Image, ImageFilter

from src.pipeline.best_image_selector import BestImageSelector
from src.pipeline.quality_scorer import ImageQualityScorer
from tests.fakes import product_photo


@pytest.fixture
def selector():
    return BestImageSelector(quality_scorer=ImageQualityScorer(enable_aesthetic=False))


def blurry(img):
    return img.filter(ImageFilter.GaussianBlur(12))


def centered_mask(size=100, fraction=0.5):
    mask = np.zeros((size, size), dtype=np.uint8)
    margin = int(size * (1 - fraction) / 2)
    mask[margin:size - margin, margin:size - margin] = 255
    return mask


def test_selects_the_sharp_photo_over_a_blurred_copy(selector):
    sharp = product_photo()

    index, best, candidates = selector.select_best([blurry(sharp), sharp], return_all_scores=True)

    assert index == 1
    assert candidates[0].final_score >= candidates[-1].final_score
    assert best.size == candidates[0].processed_image.size


def test_single_image_is_returned_with_its_score(selector):
    index, _, candidates = selector.select_best([product_photo()], return_all_scores=True)

    assert index == 0
    assert len(candidates) == 1


def test_single_image_without_scores(selector):
    assert selector.select_best([product_photo()])[2] is None


def test_no_images_is_an_error(selector):
    with pytest.raises(ValueError, match="No images"):
        selector.select_best([])


def test_when_every_image_fails_the_threshold_the_first_is_returned():
    selector = BestImageSelector(
        quality_scorer=ImageQualityScorer(enable_aesthetic=False), min_quality_threshold=1.1
    )

    index, _, candidates = selector.select_best([product_photo(), product_photo((30, 30, 200))])

    assert index == 0 and candidates is None


def test_rank_images_orders_by_score(selector):
    sharp = product_photo()

    ranked = selector.rank_images([blurry(sharp), sharp])

    assert [c.index for c in ranked][0] == 1
    assert [c.final_score for c in ranked] == sorted((c.final_score for c in ranked), reverse=True)


def test_evaluate_image_reports_processed_image_and_score(selector):
    processed, score, _ = selector.evaluate_image(product_photo())

    assert isinstance(processed, Image.Image)
    assert 0.0 <= score <= 1.0


def test_evaluate_image_can_skip_preprocessing(selector):
    original = product_photo()

    processed, _, _ = selector.evaluate_image(original, preprocess=False)

    assert processed.size == original.size


def test_visibility_prefers_a_centred_mid_sized_product_over_a_tiny_corner_one(selector):
    img = product_photo()
    corner = np.zeros((100, 100), dtype=np.uint8)
    corner[0:8, 0:8] = 255

    good = selector.calculate_visibility_score(img, centered_mask())
    poor = selector.calculate_visibility_score(img, corner)

    assert good > poor


def test_visibility_without_a_mask_is_a_neutral_score(selector):
    assert 0.0 < selector.calculate_visibility_score(product_photo()) <= 1.0


def test_completeness_drops_when_the_product_touches_the_edges(selector):
    img = product_photo()
    touching_edges = np.full((100, 100), 255, dtype=np.uint8)

    assert selector._check_completeness(img, centered_mask()) > selector._check_completeness(img, touching_edges)
    assert selector._check_completeness(img, None) == 0.8


def test_composition_score_is_bounded(selector):
    assert 0.0 <= selector.calculate_composition_score(product_photo()) <= 1.0
