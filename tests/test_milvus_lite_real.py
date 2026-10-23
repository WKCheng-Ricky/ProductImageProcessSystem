"""The ids= lookups against the real Milvus Lite library, which the in-memory fake client cannot model."""
import numpy as np
import pytest

pytest.importorskip("pymilvus")
pytest.importorskip("milvus_lite")

from pymilvus import MilvusClient  # noqa: E402

from src.pipeline.identifiers import PlainId  # noqa: E402
from src.pipeline.milvus_db import MilvusVectorLiteDB, ProductGroupsDB  # noqa: E402
from src.pipeline.product_grouper import ProductGroupUUID  # noqa: E402

RED = np.array([1.0, 0.0, 0.0])
BLUE = np.array([0.0, 0.0, 1.0])
GROUP_ID_MAX_LENGTH = 50  # the group_id column is shorter than the product id column; group ids are uuids
ALLOWED_IDS = ["plain", "IMG 001", "商品-01", "a.b_c", "6f1c2a9e-1b7d-4c1e-9d55-0a1b2c3d4e5f", "x" * 100]
ALLOWED_GROUP_IDS = [group_id for group_id in ALLOWED_IDS if len(group_id) <= GROUP_ID_MAX_LENGTH]


@pytest.fixture
def database_path(tmp_path):
    return str(tmp_path / "milvus.db")


@pytest.fixture
def vector_db(database_path):
    return MilvusVectorLiteDB(db_path=database_path, dimension=3, model_name="test-model")


@pytest.mark.parametrize("allowed_id", ALLOWED_IDS)
def test_get_and_delete_by_ids_touch_only_the_requested_row(vector_db, allowed_id):
    for other in ("other-1", "other-2"):
        vector_db.insert(PlainId(other), BLUE)
    vector_db.insert(PlainId(allowed_id), RED)

    assert vector_db.get(PlainId(allowed_id)).id == allowed_id

    vector_db.delete(PlainId(allowed_id))

    assert vector_db.get(PlainId(allowed_id)) is None
    assert sorted(vector_db.get_all_ids()) == ["other-1", "other-2"]


@pytest.mark.parametrize("allowed_id", ALLOWED_GROUP_IDS)
def test_groups_are_read_and_replaced_by_ids(database_path, allowed_id):
    groups_db = ProductGroupsDB(db_path=database_path)
    groups_db.insert_group(ProductGroupUUID(group_id="other", member_ids=["a"], representative_id="a"))
    groups_db.insert_group(ProductGroupUUID(group_id=allowed_id, member_ids=["a"], representative_id="a"))

    groups_db.update_group(ProductGroupUUID(group_id=allowed_id, member_ids=["a", "b"], representative_id="a"))

    assert groups_db.get_group(PlainId(allowed_id)).member_ids == ["a", "b"]
    assert groups_db.get_group(PlainId("other")).member_ids == ["a"]
    assert groups_db.size() == 2


def test_the_library_pastes_the_id_unescaped_so_the_pattern_is_the_only_barrier(database_path):
    """Canary: if pymilvus starts escaping ids= values, this fails and the design notes need revisiting."""
    client = MilvusClient(database_path)
    client.create_collection("canary", dimension=2, id_type="string", max_length=100, auto_id=False)
    client.insert("canary", [{"id": name, "vector": [1.0, 0.0]} for name in ("a", "b", "c")])

    hostile = "a'] or id not in ['"
    client.delete("canary", ids=[hostile])

    assert client.query("canary", filter="", output_fields=["id"]) == []
