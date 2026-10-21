"""WatermarkRemover: heuristic detection, inpainting, and the optional OCR path with a fake reader."""
import numpy as np
import pytest
from PIL import Image, ImageDraw

from src.pipeline import watermark_remover
from src.pipeline.watermark_remover import WatermarkRegion, WatermarkRemover


def image_with_corner_text():
    img = Image.new("RGB", (300, 300), (235, 235, 235))
    ImageDraw.Draw(img).text((215, 275), "SHOP.COM", fill=(0, 0, 0))
    return img


def plain_image():
    return Image.new("RGB", (200, 200), (180, 180, 180))


class FakeOcrReader:
    def __init__(self, results):
        self.results = results

    def readtext(self, image):
        return self.results


def ocr_remover(results):
    remover = WatermarkRemover(use_ocr=True, min_text_confidence=0.5)
    remover._ocr_reader = FakeOcrReader(results)
    return remover


def test_corner_text_is_detected_as_a_watermark_region():
    regions = WatermarkRemover().detect_watermarks(image_with_corner_text())

    assert regions
    assert all(region.type == "text" and 0.5 <= region.confidence <= 1.0 for region in regions)


def test_plain_image_has_nothing_to_remove():
    result = WatermarkRemover().remove_watermark(plain_image())

    assert (result.watermarks_detected, result.watermarks_removed, result.success) == ([], 0, True)


def test_removal_changes_pixels_inside_the_watermark_region_only():
    img = image_with_corner_text()

    result = WatermarkRemover().remove_watermark(img)

    before, after = np.asarray(img).astype(int), np.asarray(result.image).astype(int)
    assert result.watermarks_removed >= 1 and result.success
    assert np.abs(before - after)[:200, :200].sum() == 0
    assert np.abs(before - after).sum() > 0


@pytest.mark.parametrize("method", ["telea", "ns"])
def test_both_inpainting_methods_remove_explicit_regions(method):
    region = WatermarkRegion(bbox=(10, 10, 60, 40), text=None, confidence=0.9, type="text")

    result = WatermarkRemover(inpaint_method=method).remove_watermark(image_with_corner_text(), watermarks=[region])

    assert result.watermarks_removed == 1 and result.watermarks_detected == [region]


def test_inpainting_failure_returns_the_original_image_and_reports_failure(monkeypatch):
    def broken_inpaint(*args, **kwargs):
        raise RuntimeError("inpaint failed")

    monkeypatch.setattr(watermark_remover.cv2, "inpaint", broken_inpaint)
    img = image_with_corner_text()

    result = WatermarkRemover().remove_watermark(img)

    assert (result.success, result.watermarks_removed) == (False, 0)
    assert result.image is img


def test_auto_removal_returns_just_the_cleaned_image():
    assert isinstance(WatermarkRemover().remove_watermark_auto(image_with_corner_text()), Image.Image)


def test_ocr_regions_keep_text_and_drop_low_confidence_hits():
    box = [[10, 20], [90, 20], [90, 40], [10, 40]]
    remover = ocr_remover([(box, "SHOP.COM", 0.9), (box, "noise", 0.2)])

    regions = remover.detect_watermarks(plain_image())

    assert [(r.text, r.bbox) for r in regions] == [("SHOP.COM", (10, 20, 90, 40))]


def test_ocr_failure_falls_back_to_simple_detection():
    class BrokenReader:
        def readtext(self, image):
            raise RuntimeError("ocr crashed")

    remover = WatermarkRemover(use_ocr=True)
    remover._ocr_reader = BrokenReader()

    assert remover.detect_watermarks(image_with_corner_text())


def test_ocr_requested_but_easyocr_missing_falls_back_to_simple_detection(monkeypatch):
    monkeypatch.setattr(watermark_remover, "easyocr", None)
    remover = WatermarkRemover(use_ocr=True)

    regions = remover.detect_watermarks(image_with_corner_text())

    assert remover.use_ocr is False
    assert regions
