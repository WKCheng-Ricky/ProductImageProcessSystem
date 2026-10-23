"""Aesthetic scoring paths with the CLIP-backed scorer replaced by canned detail scores."""
import pytest

from src.pipeline import aesthetic_scorer, quality_scorer
from src.pipeline.aesthetic_scorer import AestheticScore, AestheticScorer
from src.pipeline.quality_scorer import ImageQualityScorer
from tests.fakes import product_photo

DETAIL = {
    "clarity_score": 0.8, "clarity_score_est": 7.0,
    "composition_score": 0.7, "composition_score_est": 6.0,
    "defect_score": 0.1, "defect_score_est": 2.0,
    "aesthetic_overall": 10.0,
}


class CannedClipScorer:
    def __init__(self, model_name=None, pretrained=None):
        pass

    def score_detailed(self, img):
        return dict(DETAIL)


@pytest.fixture
def canned_clip(monkeypatch):
    monkeypatch.setattr(aesthetic_scorer, "CLIPQualityScorer", CannedClipScorer)


def test_aesthetic_scorer_maps_the_detail_dict(canned_clip):
    score = AestheticScorer().score(product_photo())

    assert (score.clarity_score, score.defect_score, score.aesthetic_overall) == (0.8, 0.1, 10.0)


@pytest.mark.parametrize("overall,normalized", [(1.0, 0.0), (10.0, 1.0), (5.5, 0.5), (0.0, 0.0), (14.0, 1.0)])
def test_aesthetic_overall_is_normalised_to_zero_one(overall, normalized):
    score = AestheticScore(0, 0, 0, 0, 0, 0, aesthetic_overall=overall)

    assert score.normalized_overall() == pytest.approx(normalized)


def test_aesthetic_acceptance_uses_the_one_to_ten_scale():
    assert AestheticScore(0, 0, 0, 0, 0, 0, 6.0).is_acceptable()
    assert not AestheticScore(0, 0, 0, 0, 0, 0, 4.0).is_acceptable()


def test_combined_score_includes_the_aesthetic_part_and_its_details(canned_clip):
    combined = ImageQualityScorer(enable_aesthetic=True).score(product_photo())
    basic_only = ImageQualityScorer(enable_aesthetic=False).score(product_photo())

    assert combined.aesthetic == 1.0
    assert (combined.clarity_score, combined.defect_score_est) == (0.8, 2.0)
    assert basic_only.aesthetic == 0.0 and basic_only.clarity_score == 0.0
    assert 0.0 <= combined.overall <= 1.0


def test_watermark_and_text_penalties_lower_the_overall_score(canned_clip, monkeypatch):
    scorer = ImageQualityScorer(enable_aesthetic=False)
    clean = scorer.score(product_photo())

    real_score = scorer._basic_scorer.score

    def flagged(img, mask=None):
        result = real_score(img, mask)
        result.has_watermark = True
        result.has_text_overlay = True
        return result

    monkeypatch.setattr(scorer._basic_scorer, "score", flagged)

    assert scorer.score(product_photo()).overall < clean.overall


def test_quality_score_acceptance_threshold():
    score = ImageQualityScorer(enable_aesthetic=False).score(product_photo())

    assert score.is_acceptable(0.0)
    assert not score.is_acceptable(1.01)
    assert quality_scorer.QualityScore is type(score)
