import pytest
from pydantic import ValidationError

from src.web.models.schemas import (
    EmbeddingResultResponse,
    QualityScoreResponse,
    SearchResultItem,
)


@pytest.mark.parametrize(
    "score, level",
    [(0.95, "high"), (0.9, "high"), (0.8, "moderate"), (0.7, "moderate"), (0.69, "low")],
)
def test_search_confidence_bands(score, level):
    assert SearchResultItem(id="a", score=score).confidence_level == level


def test_search_score_percentage_truncates():
    assert SearchResultItem(id="a", score=0.876).score_percentage == 87


def _quality(overall):
    return QualityScoreResponse(
        overall=overall, sharpness=0.5, brightness=0.5, contrast=0.5,
        resolution_score=0.5, product_ratio=0.5,
        has_watermark=False, has_text_overlay=False,
    )


@pytest.mark.parametrize(
    "overall, level", [(0.8, "excellent"), (0.5, "acceptable"), (0.1, "poor")]
)
def test_quality_level(overall, level):
    assert _quality(overall).quality_level == level


def test_required_fields_are_validated():
    with pytest.raises(ValidationError):
        EmbeddingResultResponse(model_name="m")
