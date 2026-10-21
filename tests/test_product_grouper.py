"""ProductGrouper and ProductGroupUUID on the in-memory vector database."""
import numpy as np
import pytest

from src.pipeline.product_grouper import InMemoryVectorDB, ProductGroup, ProductGrouper, ProductGroupUUID

RED = np.array([1.0, 0.0, 0.0])
NEAR_RED = np.array([0.98, 0.1, 0.0])
BLUE = np.array([0.0, 0.0, 1.0])
NEAR_BLUE = np.array([0.0, 0.1, 0.98])


@pytest.fixture
def db():
    database = InMemoryVectorDB()
    for name, vector in [("red", RED), ("near-red", NEAR_RED), ("blue", BLUE), ("near-blue", NEAR_BLUE)]:
        database.insert(name, vector, {"colour": name})
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


def test_connected_components_groups_the_colour_families(db):
    groups = ProductGrouper(db, similarity_threshold=0.9).group_skus_connected_components()

    assert sorted(sorted(g.ids) for g in groups) == [["blue", "near-blue"], ["near-red", "red"]]


def test_merge_groups_joins_groups_whose_representatives_match(db):
    groups = [
        ProductGroup(group_id=0, ids=["red"], representative_id="red", size=1),
        ProductGroup(group_id=1, ids=["near-red"], representative_id="near-red", size=1),
        ProductGroup(group_id=2, ids=["blue"], representative_id="blue", size=1),
    ]

    merged = ProductGrouper(db).merge_groups(groups, merge_threshold=0.9)

    assert sorted(sorted(g.ids) for g in merged) == [["blue"], ["near-red", "red"]]


def test_merge_groups_with_one_group_is_a_no_op(db):
    only = [ProductGroup(group_id=0, ids=["red"], representative_id="red", size=1)]

    assert ProductGrouper(db).merge_groups(only) is only


def test_in_memory_insert_batch_delete_and_clear():
    db = InMemoryVectorDB()

    db.insert_batch(["a", "b"], np.array([RED, BLUE]))
    assert db.size() == 2

    db.delete("a")
    assert db.get("a") is None

    ids, vectors = db.get_all_embeddings()
    assert ids == ["b"] and vectors.shape == (1, 3)

    db.delete_all()
    assert db.size() == 0


def test_group_uuid_member_management_and_round_trip():
    group = ProductGroupUUID(group_id="g", member_ids=["a"], representative_id="a")

    group.add_member("b")
    group.add_member("b")
    group.remove_member("a")
    restored = ProductGroupUUID.from_dict(group.to_dict())

    assert group.member_ids == ["b"] and group.size == 1
    assert (restored.group_id, restored.member_ids, restored.representative_id) == ("g", ["b"], "a")
