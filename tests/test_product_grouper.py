"""ProductGrouper and ProductGroupUUID on the in-memory vector database."""
import json
import numpy as np
import pytest

from src.pipeline.product_grouper import InMemoryVectorDB, ProductGrouper, ProductGroupUUID, SearchResult
from src.pipeline.identifiers import PlainId

RED = np.array([1.0, 0.0, 0.0])
NEAR_RED = np.array([0.98, 0.1, 0.0])
BLUE = np.array([0.0, 0.0, 1.0])
NEAR_BLUE = np.array([0.0, 0.1, 0.98])


@pytest.fixture
def db():
    database = InMemoryVectorDB()
    for name, vector in [("red", RED), ("near-red", NEAR_RED), ("blue", BLUE), ("near-blue", NEAR_BLUE)]:
        database.insert(PlainId(name), vector, {"colour": name})
    return database


def test_find_similar_products_uses_the_default_and_an_override_threshold(db):
    grouper = ProductGrouper(db, similarity_threshold=0.9)

    default = {r.id for r in grouper.find_similar_products(RED)}
    strict = {r.id for r in grouper.find_similar_products(RED, threshold=0.9999)}

    assert default == {"red", "near-red"}
    assert strict == {"red"}


def test_is_duplicate_reports_the_matching_id(db):
    grouper = ProductGrouper(db)

    assert grouper.is_duplicate(RED, threshold=0.99) == (True, "red")
    assert grouper.is_duplicate(np.array([0.0, 1.0, 0.0]), threshold=0.99) == (False, None)


def test_clustering_groups_the_two_colour_families(db):
    groups = ProductGrouper(db, similarity_threshold=0.9).group_skus_clustering()

    assert sorted(sorted(g.ids) for g in groups) == [["blue", "near-blue"], ["near-red", "red"]]
    assert all(g.representative_id in g.ids and g.size == 2 for g in groups)


def test_clustering_accepts_explicit_embeddings_and_a_larger_minimum_cluster(db):
    ids, embeddings = db.get_all_embeddings()
    grouper = ProductGrouper(db, similarity_threshold=0.9)

    assert len(grouper.group_skus_clustering(embeddings=embeddings, sku_ids=ids)) == 2
    assert grouper.group_skus_clustering(embeddings=embeddings, sku_ids=ids, min_cluster_size=3) == []


def test_clustering_an_empty_database_returns_nothing():
    assert ProductGrouper(InMemoryVectorDB()).group_skus_clustering() == []


def test_in_memory_delete_and_clear():
    db = InMemoryVectorDB()
    db.insert(PlainId("a"), RED)
    db.insert(PlainId("b"), BLUE)
    assert db.size() == 2

    db.delete(PlainId("a"))
    assert db.get(PlainId("a")) is None

    ids, vectors = db.get_all_embeddings()
    assert ids == ["b"] and vectors.shape == (1, 3)

    db.delete_all()
    assert db.size() == 0


def test_group_uuid_add_member_ignores_duplicates_and_serialises():
    group = ProductGroupUUID(group_id="g", member_ids=["a"], representative_id="a")

    group.add_member(PlainId("b"))
    group.add_member(PlainId("b"))

    assert group.member_ids == ["a", "b"] and group.size == 2
    assert group.to_dict()["size"] == 2


def test_group_uuid_to_dict_holds_plain_text_that_json_can_write():
    group = ProductGroupUUID(group_id="g", member_ids=["a", "b"], representative_id="a")

    assert json.loads(json.dumps(group.to_dict())) == {
        "group_id": "g", "member_ids": ["a", "b"], "representative_id": "a", "created_at": None, "size": 2,
    }
    assert ProductGroupUUID(group_id="g").to_dict()["representative_id"] is None


def test_search_result_and_group_ids_are_made_plain_ids():
    result = SearchResult(id="a b", score=1.0, embedding=np.zeros(2), metadata=None)
    group = ProductGroupUUID(group_id="g", member_ids=["a", "b"], representative_id="a")

    assert type(result.id) is PlainId
    assert type(group.group_id) is PlainId and type(group.representative_id) is PlainId
    assert all(type(member) is PlainId for member in group.member_ids)


@pytest.mark.parametrize("bad_id", ['x" or 1', "a\nb", "", "x" * 101])
def test_search_result_and_group_refuse_a_bad_id(bad_id):
    with pytest.raises(ValueError):
        SearchResult(id=bad_id, score=1.0, embedding=np.zeros(2), metadata=None)
    with pytest.raises(ValueError):
        ProductGroupUUID(group_id=bad_id)
    with pytest.raises(ValueError):
        ProductGroupUUID(group_id="g", member_ids=[bad_id])
    with pytest.raises(ValueError):
        ProductGroupUUID(group_id="g", representative_id=bad_id)


def test_add_member_turns_text_into_a_plain_id_and_refuses_a_bad_one():
    group = ProductGroupUUID(group_id="g")

    group.add_member("a")
    group.add_member(PlainId("a"))

    assert group.member_ids == ["a"] and type(group.member_ids[0]) is PlainId
    with pytest.raises(ValueError):
        group.add_member('b" or 1')
    assert group.member_ids == ["a"]


def test_in_memory_database_refuses_raw_string_ids():
    database = InMemoryVectorDB()

    for call in (lambda: database.insert("a", np.ones(2)), lambda: database.get("a"), lambda: database.delete("a")):
        with pytest.raises(AttributeError, match="plainId"):
            call()
