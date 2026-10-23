"""A PlainId is an id that has been checked against ID_PATTERN; Milvus only ever receives `.plainId`."""
import re
from dataclasses import dataclass
from functools import total_ordering

from pydantic_core import core_schema

# The length of the id column in the Milvus products collection.
ID_MAX_LENGTH = 100
# No character that can end a Milvus string (' " \ newline) is allowed: the ids= lookups paste the value between quotes.
ID_PATTERN = rf"^[\w.\- ]{{1,{ID_MAX_LENGTH}}}$"

_PLAIN_ID = re.compile(ID_PATTERN)


@total_ordering
@dataclass(frozen=True, eq=False)
class PlainId:
    """Identifies a product or a group. Built from text, it either holds a properly formatted id or does not exist."""

    plainId: str

    def __post_init__(self):
        if isinstance(self.plainId, PlainId):
            object.__setattr__(self, "plainId", self.plainId.plainId)
            return
        if type(self.plainId) is not str:
            raise TypeError(f"an id must be a string, not {type(self.plainId).__name__}")
        if _PLAIN_ID.fullmatch(self.plainId) is None:
            raise ValueError(f"not a plain id: {self.plainId!r}")

    def __str__(self) -> str:
        return self.plainId

    def __eq__(self, other) -> bool:
        if isinstance(other, PlainId):
            return self.plainId == other.plainId
        if isinstance(other, str):
            return self.plainId == other
        return NotImplemented

    def __lt__(self, other) -> bool:
        if isinstance(other, PlainId):
            return self.plainId < other.plainId
        if isinstance(other, str):
            return self.plainId < other
        return NotImplemented

    def __hash__(self) -> int:
        return hash(self.plainId)

    @classmethod
    def __get_pydantic_core_schema__(cls, source, handler):
        from_text = core_schema.no_info_after_validator_function(cls, core_schema.str_schema(pattern=ID_PATTERN))
        return core_schema.json_or_python_schema(
            json_schema=from_text,
            python_schema=core_schema.union_schema([core_schema.is_instance_schema(cls), from_text]),
            serialization=core_schema.plain_serializer_function_ser_schema(lambda value: value.plainId),
        )
