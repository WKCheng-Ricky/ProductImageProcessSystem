import importlib.util
import sys
import types
from unittest.mock import MagicMock

import pytest

HEAVY_MODULES = ("torch", "open_clip")


class _Tensor:
    """Real class so scipy/numpy array-API probing of torch.Tensor keeps working."""


def _stub_module(name):
    module = types.ModuleType(name)
    module.Tensor = _Tensor
    module.IS_TEST_STUB = True
    module.__getattr__ = lambda attr: MagicMock(name=f"{name}.{attr}")
    return module


def _stub_missing_heavy_modules():
    """Let the web app import without ML libraries; real installs are never shadowed."""
    for name in HEAVY_MODULES:
        if importlib.util.find_spec(name) is None:
            sys.modules[name] = _stub_module(name)


_stub_missing_heavy_modules()


@pytest.fixture(autouse=True)
def in_memory_vector_db(monkeypatch):
    monkeypatch.setenv("USE_MILVUS", "false")
