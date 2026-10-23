import numpy as np
import pytest

from src.pipeline.product_grouper import InMemoryVectorDB, VectorDatabase
from src.pipeline.identifiers import PlainId


@pytest.fixture
def db():
    database = InMemoryVectorDB()
    database.insert(PlainId("x"), np.array([1.0, 0.0, 0.0]), {"name": "x"})
    database.insert(PlainId("xy"), np.array([1.0, 1.0, 0.0]))
    database.insert(PlainId("z"), np.array([0.0, 0.0, 1.0]))
    return database


def test_search_orders_by_similarity(db):
    results = db.search(np.array([1.0, 0.1, 0.0]), top_k=3)

    assert [r.id for r in results][:2] == ["x", "xy"]
    assert results[0].score >= results[1].score >= results[2].score


def test_search_respects_threshold(db):
    results = db.search(np.array([1.0, 0.0, 0.0]), threshold=0.9)

    assert [r.id for r in results] == ["x"]


def test_search_respects_top_k(db):
    assert len(db.search(np.array([1.0, 1.0, 1.0]), top_k=2)) == 2


def test_search_empty_db_returns_nothing():
    assert InMemoryVectorDB().search(np.array([1.0, 0.0])) == []


def test_inserted_vectors_are_unit_normalised(db):
    assert np.linalg.norm(db.get(PlainId("xy")).embedding) == pytest.approx(1.0)


def test_get_returns_metadata_and_none_for_unknown(db):
    assert db.get(PlainId("x")).metadata == {"name": "x"}
    assert db.get(PlainId("nope")) is None


def test_delete_and_delete_all(db):
    db.delete(PlainId("x"))
    assert db.size() == 2

    db.delete_all()
    assert db.size() == 0


def test_get_all_ids_lists_every_inserted_id(db):
    assert db.get_all_ids() == ["x", "xy", "z"]


def test_get_all_ids_drops_deleted_ids(db):
    db.delete(PlainId("xy"))

    assert db.get_all_ids() == ["x", "z"]


def test_base_class_get_all_ids_is_not_implemented():
    with pytest.raises(NotImplementedError):
        VectorDatabase().get_all_ids()
