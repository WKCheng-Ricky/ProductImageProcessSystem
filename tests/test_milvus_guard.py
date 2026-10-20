"""The model-mismatch guard must stop two embedding spaces sharing one collection."""
import logging

import pytest

from src.pipeline.milvus_db import MilvusVectorLiteDB

VECTOR_FIELD_TYPE = 101  # pymilvus DataType.FLOAT_VECTOR


class FakeMilvusClient:
    def __init__(self, dimension, description):
        self._schema = {
            "fields": [{"type": VECTOR_FIELD_TYPE, "params": {"dim": dimension}}],
            "description": description,
        }

    def describe_collection(self, name):
        return self._schema


def database_with_existing_collection(existing_dim, existing_model, *, current_dim, current_model):
    db = MilvusVectorLiteDB(db_path="unused.db", dimension=current_dim, model_name=current_model)
    db._client = FakeMilvusClient(existing_dim, f"model:{existing_model}")
    return db


def test_dimension_mismatch_refuses_to_open_the_collection():
    db = database_with_existing_collection(768, "ViT-L-14", current_dim=512, current_model="ViT-B-32")

    with pytest.raises(RuntimeError, match="Mismatch"):
        db._verify_collection_compatibility()


def test_same_model_and_dimension_is_accepted():
    db = database_with_existing_collection(512, "ViT-B-32", current_dim=512, current_model="ViT-B-32")

    db._verify_collection_compatibility()


def test_same_dimension_but_different_model_only_warns(caplog):
    db = database_with_existing_collection(512, "other-model", current_dim=512, current_model="ViT-B-32")

    with caplog.at_level(logging.WARNING):
        db._verify_collection_compatibility()

    assert "Model name changed" in caplog.text
