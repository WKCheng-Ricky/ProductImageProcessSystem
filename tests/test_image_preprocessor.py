"""ImagePreprocessor stages, with rembg replaced by a fake that cuts out a known rectangle."""
import sys
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from src.pipeline import image_preprocessor
from src.pipeline.image_preprocessor import ImagePreprocessor
from tests.fakes import product_photo

PRODUCT_BOX = (60, 60, 140, 140)


def cut_out_rectangle(img, session=None, bgcolor=None):
    rgba = img.convert("RGBA")
    alpha = Image.new("L", img.size, 0)
    ImageDraw.Draw(alpha).rectangle(PRODUCT_BOX, fill=255)
    rgba.putalpha(alpha)
    return rgba


@pytest.fixture
def with_fake_rembg(monkeypatch):
    monkeypatch.setitem(sys.modules, "rembg", SimpleNamespace(remove=cut_out_rectangle))


def preprocessor_with_session(**options):
    preprocessor = ImagePreprocessor(**options)
    preprocessor._rembg_session = object()
    return preprocessor


def test_background_removal_paints_the_background_and_returns_the_mask(with_fake_rembg):
    img = Image.new("RGB", (200, 200), (10, 120, 10))

    result, mask = preprocessor_with_session(background_color=(255, 255, 255)).remove_background(img, return_mask=True)

    assert result.getpixel((5, 5)) == (255, 255, 255)
    assert result.getpixel((100, 100)) == (10, 120, 10)
    assert mask[100, 100] == 255 and mask[5, 5] == 0


def test_background_removal_without_mask_request(with_fake_rembg):
    _, mask = preprocessor_with_session().remove_background(Image.new("RGB", (200, 200)))

    assert mask is None


def test_background_removal_failure_returns_the_original(monkeypatch):
    def broken(*args, **kwargs):
        raise RuntimeError("model crashed")

    monkeypatch.setitem(sys.modules, "rembg", SimpleNamespace(remove=broken))
    img = Image.new("RGB", (50, 50))

    result, mask = preprocessor_with_session().remove_background(img, return_mask=True)

    assert result is img and mask is None


def test_rembg_session_comes_from_the_shared_cache_with_cpu_provider(monkeypatch):
    asked = {}
    cache = SimpleNamespace(get_rembg_session=lambda name, providers: asked.update(name=name, providers=providers) or "session")
    monkeypatch.setattr(image_preprocessor, "get_model_cache", lambda: cache)

    session = ImagePreprocessor().rembg_session

    assert session == "session" and asked == {"name": "u2net", "providers": ("CPUExecutionProvider",)}


def test_gpu_request_uses_cuda_when_available(monkeypatch):
    asked = {}
    cache = SimpleNamespace(get_rembg_session=lambda name, providers: asked.update(providers=providers) or "session")
    monkeypatch.setattr(image_preprocessor, "get_model_cache", lambda: cache)
    monkeypatch.setitem(sys.modules, "onnxruntime", SimpleNamespace(get_available_providers=lambda: ["CUDAExecutionProvider"]))

    ImagePreprocessor(use_gpu=True).rembg_session

    assert asked["providers"] == ("CUDAExecutionProvider", "CPUExecutionProvider")


def test_gpu_request_falls_back_to_cpu_without_a_gpu_provider(monkeypatch):
    asked = {}
    cache = SimpleNamespace(get_rembg_session=lambda name, providers: asked.update(providers=providers) or "session")
    monkeypatch.setattr(image_preprocessor, "get_model_cache", lambda: cache)
    monkeypatch.setitem(sys.modules, "onnxruntime", SimpleNamespace(get_available_providers=lambda: []))

    ImagePreprocessor(use_gpu=True).rembg_session

    assert asked["providers"] == ("CPUExecutionProvider",)


def test_missing_rembg_leaves_the_session_unset(monkeypatch):
    def not_installed():
        raise ImportError("rembg")

    monkeypatch.setattr(image_preprocessor, "get_model_cache", not_installed)

    assert ImagePreprocessor().rembg_session is None


def test_product_bbox_from_mask_and_from_edges():
    preprocessor = ImagePreprocessor()
    mask = np.zeros((200, 200), dtype=np.uint8)
    mask[PRODUCT_BOX[1]:PRODUCT_BOX[3], PRODUCT_BOX[0]:PRODUCT_BOX[2]] = 255

    assert preprocessor.find_product_bbox(Image.new("RGB", (200, 200)), mask) == (60, 60, 139, 139)
    assert preprocessor.find_product_bbox(Image.new("RGB", (200, 200)), np.zeros((200, 200), np.uint8)) is None
    x1, y1, x2, y2 = preprocessor.find_product_bbox(product_photo(size=(200, 200)))
    assert x1 < x2 and y1 < y2


def test_bbox_of_a_flat_image_is_none():
    assert ImagePreprocessor().find_product_bbox(Image.new("RGB", (100, 100), (200, 200, 200))) is None


def test_resize_and_pad_keeps_aspect_ratio_on_a_background_canvas():
    wide = Image.new("RGB", (400, 100), (0, 0, 0))

    result = ImagePreprocessor(target_size=(200, 200), background_color=(255, 255, 255)).resize_and_pad(wide)

    assert result.size == (200, 200)
    assert result.getpixel((100, 100)) == (0, 0, 0)
    assert result.getpixel((100, 5)) == (255, 255, 255)


def test_watermark_removal_with_an_explicit_mask_changes_masked_pixels_only():
    img = product_photo(size=(200, 200))
    mask = np.zeros((200, 200), dtype=np.uint8)
    mask[10:30, 10:60] = 255

    result = ImagePreprocessor().remove_watermark(img, mask)

    assert np.abs(np.asarray(img).astype(int) - np.asarray(result).astype(int))[100:, 100:].sum() == 0


def test_watermark_removal_ignores_an_empty_mask():
    img = product_photo(size=(200, 200))

    assert ImagePreprocessor().remove_watermark(img, np.zeros((200, 200), dtype=np.uint8)) is img


def test_watermark_removal_detects_busy_corners_automatically():
    img = Image.new("RGB", (240, 240), (255, 255, 255))
    draw = ImageDraw.Draw(img)
    for offset in range(0, 40, 4):
        draw.line([(200 + offset, 200), (200 + offset, 239)], fill=(0, 0, 0), width=1)

    preprocessor = ImagePreprocessor()

    assert preprocessor._detect_watermark_region(img) is not None
    assert preprocessor.remove_watermark(img) is not img


def test_text_overlay_removal_covers_detected_text():
    img = Image.new("RGB", (300, 300), (240, 240, 240))
    ImageDraw.Draw(img).text((20, 120), "SALE 50% OFF", fill=(20, 20, 20), font=ImageFont.load_default(size=40))
    img = img.filter(ImageFilter.GaussianBlur(1.2))  # MSER needs soft edges, like a photographed overlay

    preprocessor = ImagePreprocessor()

    assert preprocessor._detect_text_region(img) is not None
    assert preprocessor.remove_text_overlay(img) is not img


def test_text_overlay_removal_leaves_a_blank_image_alone():
    blank = Image.new("RGB", (50, 50), (255, 255, 255))

    assert ImagePreprocessor().remove_text_overlay(blank) is blank


def test_standardize_with_background_removal_crops_to_the_product_and_reports_every_step(with_fake_rembg):
    img = Image.new("RGB", (200, 200), (10, 120, 10))

    result = preprocessor_with_session(target_size=(128, 128)).standardize(img, remove_text=True)

    assert result.image.size == (128, 128)
    assert result.original_size == (200, 200)
    assert result.product_bbox is not None and result.mask is not None
    assert (result.background_removed, result.watermark_removed, result.text_removed) == (True, True, True)


def test_standardize_can_skip_every_optional_step():
    img = Image.new("RGB", (80, 60))

    result = ImagePreprocessor().standardize(img, remove_bg=False, crop=False, remove_watermark=False, resize=False)

    assert result.image.size == (80, 60) and not result.background_removed


def test_process_batch_falls_back_to_the_original_when_one_image_fails(monkeypatch):
    preprocessor = ImagePreprocessor(target_size=(64, 64))
    good, bad = Image.new("RGB", (50, 50)), Image.new("RGB", (60, 60))
    real = preprocessor.standardize

    def standardize(img, **kwargs):
        if img is bad:
            raise RuntimeError("boom")
        return real(img, **kwargs)

    monkeypatch.setattr(preprocessor, "standardize", standardize)

    results = preprocessor.process_batch([good, bad], remove_bg=False)

    assert results[0].image.size == (64, 64)
    assert results[1].image is bad and results[1].background_removed is False
