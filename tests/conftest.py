import pytest


@pytest.fixture(autouse=True)
def in_memory_vector_db(monkeypatch):
    monkeypatch.setenv("USE_MILVUS", "false")


@pytest.fixture
def milvus_client(monkeypatch):
    """Patch milvus_db to talk to one shared in-memory client, whatever path it is asked to open."""
    from types import SimpleNamespace

    from src.pipeline import milvus_db
    from tests.fakes import InMemoryMilvusClient

    client = InMemoryMilvusClient()
    monkeypatch.setattr(milvus_db, "MilvusClient", lambda path: client)
    monkeypatch.setattr(milvus_db, "DataType", SimpleNamespace(VARCHAR="VARCHAR", FLOAT_VECTOR="FLOAT_VECTOR"))
    return client
