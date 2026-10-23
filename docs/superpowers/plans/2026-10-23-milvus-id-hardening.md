# Milvus Id Hardening

**Problem:** an id was pasted into a Milvus filter (`f'id == "{id}"'`). An id such as `x" or id != "` makes the filter match every row, so a lookup or delete can touch every product.

**Fix:** an id is validated once, when it becomes a `PlainId`. Only a `PlainId` can reach the vector database, and Milvus receives its text through `ids=`, never through a filter string.

**Tech Stack:** Python 3.12, FastAPI, pydantic v2, Milvus Lite (pymilvus), pytest.

## Facts the design rests on

Measured on Milvus Lite 3.2.2 with pymilvus 3.0.2.

- `ids=[x]` is not parameterized. pymilvus builds `id in ['<x>']` by pasting `x` between single quotes with no escaping. Only `'`, `\`, newline and carriage return break an `ids=` lookup (probed one character at a time). So `ids=` is safe only for a value that has none of them.
- `filter_params` (`{name}` templating) is documented, but Lite's parser rejects `{` with `unexpected character '{'`. It cannot be used here.
- Both `id` (products) and `group_id` (groups) are primary keys, so `ids=` works for both collections.
- The products `id` column is VARCHAR(100). The group column is 50, but group ids are uuids.
- Python does not enforce annotations at runtime. `def delete(self, id: PlainId)` accepts anything unless a static checker (pyright, mypy; not used here, only ruff) reads it.

## Design

**`ID_PATTERN = r"^[\w.\- ]{1,100}$"`** in `src/pipeline/identifiers.py`.
- Letters (any script), digits, `_`, `.`, `-` and a space; 1 to 100 characters.
- It allows none of `' " \`, newline, tab or control characters, which is what makes `ids=` safe.
- Matched with `re.fullmatch`, because `$` also accepts a trailing newline.
- It stays narrow on purpose. It is the only barrier between a request and the pasted `ids=` string. Ids are created by the web path, so characters such as `( ) [ ] , & @` give nothing and would depend on pymilvus quoting staying as it is.

**`PlainId`**, a frozen dataclass with one field, `plainId: str`.
- `__post_init__` validates: `TypeError` for a non-string, `ValueError("not a plain id: ...")` for a pattern mismatch. A `PlainId` passed in is unwrapped.
- It is not a `str`. The text comes out through `.plainId` (or `str()`), and a raw `str` has no `.plainId`. So `delete("abc")` fails with `AttributeError: 'str' object has no attribute 'plainId'`, with no `isinstance` helper. This is the enforcement, since annotations are not.
- `__str__` returns the text. `__eq__` also accepts the same `str`, `__hash__` is `hash(plainId)`, and `__lt__` compares the text so `sorted()` works. Comparison with any other type returns `NotImplemented`.
- A pydantic core schema accepts a `PlainId` or a string matching the pattern, and serializes to the string. Response models declare `PlainId` and return plain strings in JSON.

**Vector database** (`src/pipeline/milvus_db.py`, `product_grouper.py`)
- Every id parameter and return on `VectorDatabase`, `InMemoryVectorDB`, `MilvusVectorLiteDB`, `ProductGroupsDB`, `ProductGroupManager` and the record dataclasses is `PlainId`.
- `get`, `delete`, `get_group` and `update_group` query with `ids=[x.plainId]`. No filter string is built from a value anywhere.
- Writes use `.plainId` (`insert`, `insert_group`: `group_id`, the member ids JSON, `representative_id`).
- Ids read back from Milvus are wrapped in `PlainId(...)` and fail loudly if one does not match.
- Group ids are `PlainId(str(uuid.uuid4()))`.

**Web layer** (`src/web/identifiers.py`, routes, services, `schemas.py`)
- FastAPI path and query parameters must be scalar, so a dataclass cannot be the parameter type. `PathPlainId = Annotated[str, Path(pattern=ID_PATTERN), AfterValidator(PlainId)]` and `QueryPlainId` (optional) give 422 for a bad id, and the handler receives a `PlainId`. Static checkers see `str`.
- Routes, services and the pipeline hand a `PlainId` on. Wherever an id becomes text (session JSON, plain dicts, metadata, log lines), the code uses `.plainId`.
- `process_batch` turns a filename stem outside the pattern into a failed result for that file and continues.

## Tests that pin the design

- `tests/test_plain_id.py`: construction, `.plainId`, equality and hashing against `str`, ordering, immutability, pydantic accept, refuse and serialize. One test fails if the pattern ever allows `'`, `"`, `\`, newline, carriage return, tab, `;` or `(`. This is the only test that pins the pattern: the fake client's `ids=` is a dictionary lookup and cannot model pymilvus pasting.
- `tests/test_milvus_db.py`: a raw `str` handed to `get`, `delete`, `insert`, `get_group`, `update_group` and `insert_group` raises and changes no row. Ids with spaces, Chinese characters and uuids round-trip. Ids read back are `PlainId`.
- `tests/test_milvus_lite_real.py`: runs `ids=` lookups, deletes and group updates on the real Milvus Lite library (skipped if it is not installed), and a canary that fails if pymilvus starts escaping `ids=` values, which would mean this design should be revisited.
- `tests/test_plain_id_guard.py`: by annotation introspection, every id parameter or return on the classes above is `PlainId`, not `str`, so a new method with `id: str` fails.
- `tests/test_id_validation.py`: a bad id gets 422 at each web entry point, and a good id passes.

## Verification gates

- `ruff check .` and `pytest -q --cov=src --cov-fail-under=80` pass.
- Run pytest once with `-W "error:Pydantic serializer warnings:UserWarning"`; a model field typed `str` that receives a `PlainId` shows up as a warning.
- On a scratch Milvus Lite DB (never the live one): `ids=` lookups for plain, spaced, Chinese and uuid ids return only that row, grouping, `get_group` and `update_group` work, and a delete removes exactly one row.

## Known limits

- A row stored with an id outside the pattern raises `ValueError` when read (search, grouping, browse, duplicate check). The live DB has not been inspected. If such rows exist, either migrate them or make the read paths skip and log.
- The enforcement is the attribute access. Run a static type checker if the `PlainId` annotations should also be checked before runtime.
