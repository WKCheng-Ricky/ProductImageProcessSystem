import pytest


@pytest.fixture(autouse=True)
def in_memory_vector_db(monkeypatch):
    monkeypatch.setenv("USE_MILVUS", "false")
