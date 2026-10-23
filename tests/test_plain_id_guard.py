"""Fails when a vector-DB method takes or returns an id as a plain `str`, so a new method cannot skip PlainId."""
import inspect
import typing

import pytest

from src.pipeline.identifiers import PlainId
from src.pipeline.milvus_db import MilvusVectorLiteDB, ProductGroupManager, ProductGroupsDB
from src.pipeline.product_grouper import InMemoryVectorDB, ProductGroup, ProductGroupUUID, SearchResult, VectorDatabase

DB_CLASSES = [VectorDatabase, InMemoryVectorDB, MilvusVectorLiteDB, ProductGroupsDB, ProductGroupManager]
ID_RETURNING_METHODS = {"get_all_ids", "get_all_embeddings"}


def is_id_parameter(name: str) -> bool:
    return name in {"id", "ids"} or name.endswith("_id") or name.endswith("_ids")


def leaf_types(hint) -> set:
    arguments = typing.get_args(hint)
    if not arguments:
        return {hint}
    return set().union(*(leaf_types(argument) for argument in arguments))


def public_methods(cls):
    return [(name, method) for name, method in inspect.getmembers(cls, inspect.isfunction) if not name.startswith("_")]


def id_parameters():
    for cls in DB_CLASSES:
        for name, method in public_methods(cls):
            hints = typing.get_type_hints(method)
            for parameter in inspect.signature(method).parameters:
                if is_id_parameter(parameter):
                    yield pytest.param(hints.get(parameter), id=f"{cls.__name__}.{name}({parameter})")


def id_returns():
    for cls in DB_CLASSES:
        for name, method in public_methods(cls):
            if name in ID_RETURNING_METHODS:
                yield pytest.param(typing.get_type_hints(method).get("return"), id=f"{cls.__name__}.{name}")


def test_the_guard_finds_the_id_parameters_it_is_meant_to_check():
    found = {parameter.id for parameter in id_parameters()}

    assert {
        "MilvusVectorLiteDB.insert(id)", "MilvusVectorLiteDB.get(id)", "MilvusVectorLiteDB.delete(id)",
        "ProductGroupsDB.get_group(group_id)", "ProductGroupsDB.find_group_by_product(product_id)",
        "ProductGroupManager.get_group(group_id)", "InMemoryVectorDB.get(id)", "VectorDatabase.delete(id)",
    } <= found


def test_the_guard_finds_the_methods_that_return_ids():
    found = {returned.id for returned in id_returns()}

    assert {
        "MilvusVectorLiteDB.get_all_ids", "MilvusVectorLiteDB.get_all_embeddings", "InMemoryVectorDB.get_all_ids",
    } <= found


@pytest.mark.parametrize("hint", list(id_parameters()))
def test_every_id_parameter_is_annotated_as_a_plain_id(hint):
    assert hint is not None, "id parameter has no annotation"
    leaves = leaf_types(hint)
    assert PlainId in leaves and str not in leaves


@pytest.mark.parametrize("hint", list(id_returns()))
def test_every_id_returned_is_annotated_as_a_plain_id(hint):
    leaves = leaf_types(hint)
    assert PlainId in leaves and str not in leaves


@pytest.mark.parametrize("cls, fields", [
    (SearchResult, ["id"]),
    (ProductGroup, ["ids", "representative_id"]),
    (ProductGroupUUID, ["group_id", "member_ids", "representative_id"]),
])
def test_id_fields_of_the_records_are_annotated_as_plain_ids(cls, fields):
    hints = typing.get_type_hints(cls)
    for name in fields:
        leaves = leaf_types(hints[name])
        assert PlainId in leaves and str not in leaves, name


def test_the_guard_would_catch_a_method_added_with_a_str_id():
    class Careless(VectorDatabase):
        def get(self, id: str):
            return None

    hint = typing.get_type_hints(Careless.get)["id"]

    assert str in leaf_types(hint)
