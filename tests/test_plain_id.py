"""PlainId is the only way to make a value that may be sent to Milvus as an id."""
import dataclasses

import pytest
from pydantic import BaseModel, ValidationError

from src.pipeline.identifiers import ID_PATTERN, PlainId
import re

BAD_IDS = [
    'x" or id != "', "tail\\", "it's", "a'] or id not in ['", "line\nbreak", "carriage\rreturn", "tab\there",
    "semi;colon", "paren(1)", "", "x" * 101, "trailing\n",
]
GOOD_IDS = ["abc", "red-shade", "a.b_c", "with space", "6f1c2a9e-1b7d-4c1e-9d55-0a1b2c3d4e5f", "商品-01", "x" * 100]
CHARACTERS_THAT_BREAK_OUT_OF_A_MILVUS_STRING = ["'", '"', "\\", "\n", "\r", "\t", "\x00", "]", "[", ",", ";", "(", ")"]


@pytest.mark.parametrize("bad_id", BAD_IDS)
def test_a_bad_id_cannot_be_constructed(bad_id):
    with pytest.raises(ValueError, match="not a plain id"):
        PlainId(bad_id)


@pytest.mark.parametrize("good_id", GOOD_IDS)
def test_the_validated_text_is_stored_in_plain_id(good_id):
    assert PlainId(good_id).plainId == good_id


@pytest.mark.parametrize("character", CHARACTERS_THAT_BREAK_OUT_OF_A_MILVUS_STRING)
def test_the_pattern_allows_no_character_that_could_end_a_milvus_string(character):
    assert re.fullmatch(ID_PATTERN, f"a{character}b") is None


def test_a_non_string_is_refused():
    with pytest.raises(TypeError):
        PlainId(5)


class DisguisedStr(str):
    """A str whose formatting differs from the characters that were validated."""

    def __str__(self):
        return "a'] or id not in ['"

    def __format__(self, spec):
        return str(self)


def test_a_str_subclass_is_refused_even_when_its_characters_look_plain():
    with pytest.raises(TypeError):
        PlainId(DisguisedStr("ok"))


def test_constructing_from_another_plain_id_copies_its_text():
    assert PlainId(PlainId("red")).plainId == "red"


def test_a_plain_id_is_not_a_string_so_the_text_can_only_be_read_through_plain_id():
    plain = PlainId("red")

    assert not isinstance(plain, str)
    assert str(plain) == "red"
    assert f"{plain}-x" == "red-x"


def test_a_plain_id_is_immutable():
    plain = PlainId("red")

    with pytest.raises(dataclasses.FrozenInstanceError):
        plain.plainId = 'x" or 1'


def test_a_plain_id_equals_the_same_plain_id_and_the_same_text():
    assert PlainId("red") == PlainId("red")
    assert PlainId("red") == "red"
    assert PlainId("red") != PlainId("blue")
    assert PlainId("red") != "blue"


def test_a_plain_id_and_its_text_are_interchangeable_as_dict_keys_and_in_lists():
    plain = PlainId("red")

    assert {plain: 1}["red"] == 1
    assert {"red": 1}[plain] == 1
    assert plain in ["red"]
    assert "red" in [plain]


def test_plain_ids_sort_like_their_text():
    assert sorted([PlainId("b"), PlainId("a"), PlainId("c")]) == ["a", "b", "c"]


def test_a_plain_id_orders_against_text_in_both_directions():
    assert "a" < PlainId("b")
    assert PlainId("a") <= "a"
    assert PlainId("b") > "a"
    assert sorted([PlainId("b"), "a", PlainId("c")]) == ["a", "b", "c"]


def test_a_plain_id_does_not_match_or_order_against_other_types():
    assert PlainId("5") != 5
    with pytest.raises(TypeError):
        PlainId("a") < 5
    with pytest.raises(TypeError):
        5 < PlainId("a")


class Row(BaseModel):
    id: PlainId


def test_a_pydantic_field_accepts_text_and_a_plain_id():
    assert Row(id="red").id == PlainId("red")
    assert Row(id=PlainId("red")).id == PlainId("red")


@pytest.mark.parametrize("bad_id", BAD_IDS)
def test_a_pydantic_field_refuses_a_bad_id(bad_id):
    with pytest.raises(ValidationError):
        Row(id=bad_id)


def test_a_pydantic_field_serializes_to_plain_text():
    assert Row(id="red").model_dump() == {"id": "red"}
    assert Row(id="red").model_dump_json() == '{"id":"red"}'
