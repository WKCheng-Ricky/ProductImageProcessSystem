"""FastAPI only accepts scalar path and query parameters, so these take the text, check the pattern, and hand the route a PlainId."""
from typing import Annotated, Optional

from fastapi import Path, Query
from pydantic import AfterValidator

from src.pipeline.identifiers import ID_PATTERN, PlainId


def _plain_id_if_present(value: Optional[str]) -> Optional[PlainId]:
    return PlainId(value) if value is not None else None


# Static checkers see `str`; at runtime the route receives a PlainId.
PathPlainId = Annotated[str, Path(pattern=ID_PATTERN), AfterValidator(PlainId)]
QueryPlainId = Annotated[Optional[str], Query(pattern=ID_PATTERN), AfterValidator(_plain_id_if_present)]
