"""MilvusVectorLiteDB, ProductGroupsDB and ProductGroupManager against an in-memory Milvus client."""
import numpy as np
import pytest

from src.pipeline.identifiers import PlainId
from src.pipeline.milvus_db import MilvusVectorLiteDB, ProductGroupManager, ProductGroupsDB
from src.pipeline.product_grouper import ProductGroupUUID

RED = np.array([1.0, 0.0, 0.0])
NEAR_RED = np.array([0.99, 0.1, 0.0])
BLUE = np.array([0.0, 0.0, 1.0])


@pytest.fixture
def vector_db(milvus_client):
    return MilvusVectorLiteDB(db_path="unused.db", dimension=3, model_name="test-model")


@pytest.fixture
def groups_db(milvus_client):
    return ProductGroupsDB(db_path="unused.db")


def test_insert_then_get_returns_normalised_vector_and_metadata(vector_db):
    vector_db.insert(PlainId("red"), RED * 5, {"image_url": "/red.png"})

    found = vector_db.get(PlainId("red"))

    assert found.metadata == {"image_url": "/red.png"}
    assert np.linalg.norm(found.embedding) == pytest.approx(1.0)


def test_get_unknown_id_returns_none(vector_db):
    assert vector_db.get(PlainId("missing")) is None


def test_search_orders_by_similarity_and_applies_threshold(vector_db):
    vector_db.insert(PlainId("red"), RED)
    vector_db.insert(PlainId("near-red"), NEAR_RED)
    vector_db.insert(PlainId("blue"), BLUE)

    ids = [hit.id for hit in vector_db.search(RED, top_k=3, threshold=0.5)]

    assert ids == ["red", "near-red"]


def test_search_respects_top_k(vector_db):
    for name, vector in [("a", RED), ("b", NEAR_RED), ("c", BLUE)]:
        vector_db.insert(PlainId(name), vector)

    assert len(vector_db.search(RED, top_k=1)) == 1


def test_size_delete_and_delete_all(vector_db):
    vector_db.insert(PlainId("red"), RED)
    vector_db.insert(PlainId("blue"), BLUE)
    assert vector_db.size() == 2

    vector_db.delete(PlainId("red"))
    assert vector_db.get_all_ids() == ["blue"]

    vector_db.delete_all()
    assert vector_db.size() == 0


def test_get_all_embeddings_returns_ids_and_matrix(vector_db):
    vector_db.insert(PlainId("red"), RED)
    vector_db.insert(PlainId("blue"), BLUE)

    ids, vectors = vector_db.get_all_embeddings()

    assert ids == ["red", "blue"]
    assert vectors.shape == (2, 3)


def test_get_all_embeddings_on_empty_collection(vector_db):
    ids, vectors = vector_db.get_all_embeddings()

    assert ids == []
    assert len(vectors) == 0


def test_reopening_with_a_different_dimension_is_refused(milvus_client):
    first = MilvusVectorLiteDB(db_path="unused.db", dimension=3, model_name="test-model")
    first.insert(PlainId("red"), RED)

    with pytest.raises(RuntimeError, match="Mismatch"):
        MilvusVectorLiteDB(db_path="unused.db", dimension=5, model_name="other").size()


def group(group_id, members):
    return ProductGroupUUID(group_id=group_id, member_ids=members, representative_id=members[0])


def test_group_round_trip(groups_db):
    groups_db.insert_group(group("g1", ["a", "b"]))

    found = groups_db.get_group(PlainId("g1"))

    assert (found.member_ids, found.representative_id) == (["a", "b"], "a")
    assert groups_db.get_group(PlainId("missing")) is None


def test_update_group_replaces_members(groups_db):
    groups_db.insert_group(group("g1", ["a"]))

    groups_db.update_group(group("g1", ["a", "b", "c"]))

    assert groups_db.get_group(PlainId("g1")).member_ids == ["a", "b", "c"]
    assert groups_db.size() == 1


def test_clear_all_groups(groups_db):
    groups_db.insert_group(group("g1", ["a"]))
    groups_db.insert_group(group("g2", ["b"]))

    groups_db.clear_all_groups()
    assert groups_db.get_all_groups() == []


def test_find_group_by_product(groups_db):
    groups_db.insert_group(group("g1", ["a", "b"]))

    assert groups_db.find_group_by_product(PlainId("b")).group_id == "g1"
    assert groups_db.find_group_by_product(PlainId("zzz")) is None


def seeded_manager(vector_db, groups_db, **options):
    vector_db.insert(PlainId("red"), RED)
    vector_db.insert(PlainId("near-red"), NEAR_RED)
    vector_db.insert(PlainId("blue"), BLUE)
    return ProductGroupManager(vector_db=vector_db, groups_db=groups_db, similarity_threshold=0.9, **options)


def test_grouping_puts_similar_products_together_and_stores_the_groups(vector_db, groups_db):
    manager = seeded_manager(vector_db, groups_db)

    groups = manager.group_all_products()

    members = sorted(sorted(g.member_ids) for g in groups)
    assert members == [["blue"], ["near-red", "red"]]
    assert len(groups_db.get_all_groups()) == 2


def test_regrouping_replaces_previous_groups(vector_db, groups_db):
    manager = seeded_manager(vector_db, groups_db)
    manager.group_all_products()

    manager.group_all_products(clear_existing=True)

    assert len(groups_db.get_all_groups()) == 2


def test_grouping_an_empty_database_creates_nothing(vector_db, groups_db):
    manager = ProductGroupManager(vector_db=vector_db, groups_db=groups_db)

    assert manager.group_all_products() == []




def test_reading_groups_that_were_never_created_returns_nothing_and_creates_nothing(groups_db, milvus_client):
    assert groups_db.get_all_groups_if_present() == []
    assert "product_groups" not in milvus_client.collections


def test_reading_existing_groups_returns_them(groups_db):
    groups_db.insert_group(ProductGroupUUID(group_id="g1", member_ids=["a", "b"], representative_id="a"))

    assert [g.group_id for g in groups_db.get_all_groups_if_present()] == ["g1"]


def test_collections_are_created_without_waiting_for_the_index(vector_db, groups_db, milvus_client):
    vector_db.size()
    groups_db.get_all_groups()

    assert milvus_client.collections_that_waited_for_index == []
    assert sorted(milvus_client.index_requests) == sorted([
        (vector_db.collection_name, False),
        (groups_db.collection_name, False),
    ])


def test_clearing_groups_recreates_the_collection_without_waiting_for_the_index(groups_db, milvus_client):
    groups_db.clear_all_groups()

    assert milvus_client.collections_that_waited_for_index == []
    assert milvus_client.index_requests == [(groups_db.collection_name, False)] * 2


RAW_IDS = ["a", 'x" or id != "', "it's", "a'] or id not in ['", "line\nbreak"]
UNUSUAL_IDS = ["IMG 001", "商品-01", "a.b_c", "6f1c2a9e-1b7d-4c1e-9d55-0a1b2c3d4e5f"]


@pytest.mark.parametrize("raw_id", RAW_IDS)
def test_get_refuses_a_raw_string(vector_db, raw_id):
    vector_db.insert(PlainId("a"), RED)

    with pytest.raises(AttributeError, match="plainId"):
        vector_db.get(raw_id)


@pytest.mark.parametrize("raw_id", RAW_IDS)
def test_delete_refuses_a_raw_string_and_keeps_every_row(vector_db, raw_id):
    vector_db.insert(PlainId("a"), RED)
    vector_db.insert(PlainId("b"), BLUE)

    with pytest.raises(AttributeError, match="plainId"):
        vector_db.delete(raw_id)

    assert sorted(vector_db.get_all_ids()) == ["a", "b"]


@pytest.mark.parametrize("raw_id", RAW_IDS)
def test_insert_refuses_a_raw_string(vector_db, raw_id):
    with pytest.raises(AttributeError, match="plainId"):
        vector_db.insert(raw_id, RED)

    assert vector_db.get_all_ids() == []


@pytest.mark.parametrize("raw_id", RAW_IDS)
def test_group_lookup_and_update_refuse_a_raw_string(groups_db, raw_id):
    groups_db.insert_group(group("g1", ["a"]))
    swapped = group("g1", ["a"])
    swapped.group_id = raw_id

    with pytest.raises(AttributeError, match="plainId"):
        groups_db.get_group(raw_id)
    with pytest.raises(AttributeError, match="plainId"):
        groups_db.update_group(swapped)
    with pytest.raises(AttributeError, match="plainId"):
        groups_db.insert_group(swapped)

    assert groups_db.get_group(PlainId("g1")).member_ids == ["a"]
    assert groups_db.size() == 1


def test_inserting_a_group_with_a_raw_member_is_refused(groups_db):
    group_with_raw_member = group("g1", ["a"])
    group_with_raw_member.member_ids.append("b")

    with pytest.raises(AttributeError, match="plainId"):
        groups_db.insert_group(group_with_raw_member)

    assert groups_db.size() == 0


@pytest.mark.parametrize("unusual_id", UNUSUAL_IDS)
def test_unusual_but_valid_ids_round_trip(vector_db, groups_db, unusual_id):
    vector_db.insert(PlainId("a"), RED)
    vector_db.insert(PlainId(unusual_id), BLUE)
    groups_db.insert_group(group(unusual_id, [unusual_id]))

    assert vector_db.get(PlainId(unusual_id)).id == unusual_id
    assert groups_db.get_group(PlainId(unusual_id)).member_ids == [unusual_id]

    vector_db.delete(PlainId(unusual_id))
    groups_db.update_group(group(unusual_id, ["a"]))

    assert vector_db.get_all_ids() == ["a"]
    assert groups_db.get_group(PlainId(unusual_id)).member_ids == ["a"]


def test_ids_read_back_from_the_database_are_plain_ids(vector_db, groups_db):
    vector_db.insert(PlainId("a"), RED)
    groups_db.insert_group(group("g1", ["a"]))

    assert all(type(stored) is PlainId for stored in vector_db.get_all_ids())
    assert type(vector_db.get(PlainId("a")).id) is PlainId
    assert type(vector_db.search(RED)[0].id) is PlainId
    read_group = groups_db.get_group(PlainId("g1"))
    assert type(read_group.group_id) is PlainId
    assert all(type(member) is PlainId for member in read_group.member_ids)
