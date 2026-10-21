"""ModelCache, clip_loader and rembg_loader with fake torch/open_clip/rembg modules, so no weights are needed."""
import contextlib
import sys
from types import SimpleNamespace

import pytest

from src.pipeline import clip_loader, model_cache, rembg_loader
from src.pipeline.model_cache import ModelCache, get_model_cache


@pytest.fixture
def cache(monkeypatch):
    monkeypatch.setattr(ModelCache, "_instance", None)
    monkeypatch.setattr(model_cache, "_cache", None)
    return get_model_cache()


@pytest.fixture
def loads(monkeypatch):
    calls = []

    def fake_load_clip(model_name, pretrained, device):
        calls.append(("clip", model_name, pretrained, device))
        return f"model-{model_name}", "preprocess", "tokenizer", 512

    def fake_load_rembg(model_name, providers):
        calls.append(("rembg", model_name, providers))
        return f"session-{model_name}"

    monkeypatch.setattr(model_cache, "load_clip_model", fake_load_clip)
    monkeypatch.setattr(model_cache, "load_rembg_session", fake_load_rembg)
    return calls


def test_cache_is_a_singleton(cache):
    assert ModelCache() is cache is get_model_cache()


def test_clip_model_loads_once_per_key(cache, loads):
    first = cache.get_clip_model("ViT-B-32", pretrained="openai", device="cpu")
    second = cache.get_clip_model("ViT-B-32", pretrained="openai", device="cpu")

    assert first is second and len(loads) == 1
    assert first == ("model-ViT-B-32", "preprocess", "tokenizer", 512)


def test_a_different_model_or_device_loads_separately(cache, loads):
    cache.get_clip_model("ViT-B-32", "openai", "cpu")
    cache.get_clip_model("ViT-L-14", "openai", "cpu")
    cache.get_clip_model("ViT-B-32", "openai", "cuda")

    assert len(loads) == 3


def test_auto_device_resolves_through_default_device(cache, loads, monkeypatch):
    monkeypatch.setattr(model_cache, "default_device", lambda: "cpu")

    cache.get_clip_model("ViT-B-32", "openai")

    assert loads == [("clip", "ViT-B-32", "openai", "cpu")]


def test_rembg_session_is_cached_and_receives_a_provider_list(cache, loads):
    first = cache.get_rembg_session("u2net", providers=("CPUExecutionProvider",))
    second = cache.get_rembg_session("u2net", providers=("CPUExecutionProvider",))

    assert first == second == "session-u2net"
    assert loads == [("rembg", "u2net", ["CPUExecutionProvider"])]


def test_cached_model_keys_are_listed_and_clear_empties_them(cache, loads):
    cache.get_clip_model("ViT-B-32", "openai", "cpu")
    cache.get_rembg_session("u2net")

    assert len(cache.get_cached_models()) == 2

    cache.clear()
    assert cache.get_cached_models() == []


class FakeTensor:
    shape = (1, 512)

    def unsqueeze(self, dim):
        return self

    def to(self, device):
        return self


class FakeClipModel:
    def __init__(self):
        self.device = None
        self.evaluating = False

    def to(self, device):
        self.device = device

    def eval(self):
        self.evaluating = True

    def encode_image(self, tensor):
        return FakeTensor()


@pytest.fixture
def fake_ml_modules(monkeypatch):
    model = FakeClipModel()
    torch = SimpleNamespace(
        cuda=SimpleNamespace(is_available=lambda: False),
        no_grad=lambda: contextlib.nullcontext(),
    )
    open_clip = SimpleNamespace(
        create_model_and_transforms=lambda name, pretrained: (model, None, lambda image: FakeTensor()),
        get_tokenizer=lambda name: "tokenizer",
    )
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "open_clip", open_clip)
    return model


def test_default_device_is_cpu_without_cuda(fake_ml_modules):
    assert clip_loader.default_device() == "cpu"


def test_default_device_is_cuda_when_available(fake_ml_modules, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: True)))

    assert clip_loader.default_device() == "cuda"


def test_load_clip_model_prepares_the_model_and_measures_its_dimension(fake_ml_modules):
    model, preprocess, tokenizer, dimension = clip_loader.load_clip_model("ViT-B-32", "openai", "cpu")

    assert (tokenizer, dimension) == ("tokenizer", 512)
    assert model is fake_ml_modules and model.device == "cpu" and model.evaluating


def test_rembg_session_loads_through_rembg(monkeypatch):
    seen = {}
    fake = SimpleNamespace(new_session=lambda name, providers: seen.update(name=name, providers=providers) or "session")
    monkeypatch.setattr(rembg_loader, "_REMBG_AVAILABLE", True)
    monkeypatch.setattr(rembg_loader, "rembg", fake, raising=False)

    assert rembg_loader.load_rembg_session("u2net", ["CPUExecutionProvider"]) == "session"
    assert seen == {"name": "u2net", "providers": ["CPUExecutionProvider"]}


def test_rembg_missing_raises_a_clear_import_error(monkeypatch):
    monkeypatch.setattr(rembg_loader, "_REMBG_AVAILABLE", False)

    with pytest.raises(ImportError, match="pip install rembg"):
        rembg_loader.load_rembg_session("u2net", None)
