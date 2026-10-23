"""Regressions for the grouping and duplicate-record 500s caused by Milvus Lite 3.x."""
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

from src.pipeline import milvus_db
from src.pipeline.config import CONFIG
from src.pipeline.milvus_db import ProductGroupsDB
from src.pipeline.product_grouper import ProductGroupUUID, SearchResult
from src.web.app import app
from src.web.services import duplicate_service
from src.web.services.pipeline_service import PipelineService
from tests.fakes import build_fake_model_pipeline


class RecordingSchema:
    def __init__(self):
        self.fields = []

    def add_field(self, field_name, datatype, **options):
        self.fields.append((field_name, datatype))


class RecordingMilvusClient:
    def __init__(self):
        self.created_schema = None
        self.index_params = None
        self.inserted = []

    def create_schema(self, **options):
        return RecordingSchema()

    def prepare_index_params(self):
        return SimpleNamespace(add_index=lambda **options: None)

    def create_collection(self, collection_name, schema, index_params=None):
        self.created_schema = schema
        self.index_params = index_params

    def insert(self, collection_name, data):
        self.inserted.extend(data)


@pytest.fixture
def groups_db(monkeypatch):
    monkeypatch.setattr(milvus_db, "DataType", SimpleNamespace(VARCHAR="VARCHAR", FLOAT_VECTOR="FLOAT_VECTOR"))
    db = ProductGroupsDB(db_path="unused.db")
    db._client = RecordingMilvusClient()
    db._collection_ready = True
    return db


def test_groups_collection_has_a_vector_field_because_milvus_requires_one(groups_db):
    groups_db._create_collection()

    datatypes = [datatype for _, datatype in groups_db._client.created_schema.fields]
    assert "FLOAT_VECTOR" in datatypes
    assert groups_db._client.index_params is not None


def test_inserted_group_carries_a_value_for_the_placeholder_vector(groups_db):
    groups_db.insert_group(ProductGroupUUID(group_id="g1", member_ids=["a", "b"], representative_id="a"))

    row = groups_db._client.inserted[0]
    assert len(row["placeholder_vector"]) == milvus_db._PLACEHOLDER_VECTOR_DIM


def test_backup_copies_a_milvus_lite_directory(tmp_path, monkeypatch):
    database = tmp_path / "milvus.db"
    (database / "collections").mkdir(parents=True)
    (database / "collections" / "segment").write_text("data")
    monkeypatch.setitem(CONFIG, "milvus_lite_path", str(database))

    backup = duplicate_service.backup_database()

    assert (tmp_path / backup.rsplit("/", 1)[1] / "collections" / "segment").read_text() == "data"
    assert duplicate_service.list_backups()[0]["size_mb"] >= 0


class StubMilvusDB:
    def __init__(self):
        self.rows = {"a": np.array([1.0, 0.0]), "b": np.array([0.0, 1.0])}

    def size(self):
        return len(self.rows)

    def get_all_embeddings(self):
        return list(self.rows), list(self.rows.values())

    def get(self, id):
        return SearchResult(id=id, score=1.0, embedding=self.rows[id], metadata={"image_url": f"/{id}.png"})

    def insert(self, id, embedding, metadata=None):
        self.rows[id] = embedding


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(PipelineService, "_pipeline", build_fake_model_pipeline(str(tmp_path / "processed"), monkeypatch))
    return TestClient(app)


def test_duplicate_without_milvus_is_a_client_error_not_a_500(client):
    response = client.post("/api/duplicate", json={"n": 2})

    assert response.status_code == 400
    assert "USE_MILVUS" in response.json()["detail"]


def test_stats_without_milvus_report_no_database(client):
    assert client.get("/api/duplicate/stats").json()["exists"] is False


def test_duplicate_copies_every_record_and_reports_a_backup(client, tmp_path, monkeypatch):
    database = tmp_path / "milvus.db"
    database.mkdir()
    monkeypatch.setitem(CONFIG, "milvus_lite_path", str(database))
    stub = StubMilvusDB()
    monkeypatch.setattr(duplicate_service, "active_milvus_db", lambda: stub)

    response = client.post("/api/duplicate", json={"n": 3})

    body = response.json()
    assert response.status_code == 200, response.text
    assert (body["original_count"], body["new_count"], body["duplicates_added"]) == (2, 6, 4)
    assert (tmp_path / body["backup_path"].rsplit("/", 1)[1]).exists()
