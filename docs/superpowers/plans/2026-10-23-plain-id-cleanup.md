# PlainId Cleanup

**Goal:** close the small gaps an independent review found in the finished design (`2026-10-23-milvus-id-hardening.md`). None of them is an open injection path; they are loose ends.

**Out of scope:** the design itself (pattern, wrapper, `ids=`), and legacy ids already stored outside the pattern (decided: the web path controls ids, so no migration or skip logic).

**Rules:** one commit per task, TDD (watch each test fail first), `ruff check .` and `pytest -q` green before every commit. Commit only; do not push.

## Task 1: Reject a `str` subclass

**Problem.** `PlainId(Evil("ok"))` passes the pattern check on its real content, where `Evil` is a `str` subclass overriding `__str__` / `__format__`. Formatting it later (pymilvus builds `id in ['{value}']`) can yield different text from what was validated. It needs code running in the process, so the risk is low, but the check is one line.

**Change** (`src/pipeline/identifiers.py`, `__post_init__`): require `type(self.plainId) is str`, otherwise `TypeError` (the existing message already names the type).

**Test** (`tests/test_plain_id.py`): a `str` subclass with an overridden `__str__` and `__format__` is refused with `TypeError`; a plain `str` still passes.

**Decision for you.** Strict refusal means a `str` subclass such as `numpy.str_` is also refused. Ids read from Milvus are plain `str`, so nothing in the repo is affected, but a future caller might hit it. The alternative is to normalize (copy into an exact `str`) instead of refusing. I recommend refusing: it fails loudly and is simpler.

## Task 2: Comparisons return `NotImplemented` for other types

**Problem.** `"a" < PlainId("b")` raises `TypeError` while `PlainId("a") < "b"` works, and `PlainId("a") < 5` quietly compares against `"5"` (`__lt__` calls `str(other)`).

**Change** (`identifiers.py`):
- `__eq__` returns `NotImplemented` for anything that is neither a `PlainId` nor a `str`.
- `__lt__` accepts a `PlainId` or a `str` and returns `NotImplemented` otherwise.
- `@functools.total_ordering` supplies `>`, `<=`, `>=`.

**Tests:** `PlainId("a") < 5` and `PlainId("a") == 5` do not match (`==` is `False`, `<` raises `TypeError`); `"a" < PlainId("b")` and `PlainId("a") <= "a"` work; `sorted()` over a mix of `PlainId` and `str` works.

## Task 3: `ProductGroupUUID.to_dict()` and the legacy `ProductGroup` annotation

**Problem.** `to_dict()` returns `PlainId` objects, which `json.dumps` cannot handle. Its only caller is `tests/test_product_grouper.py:81` (it checks `["size"]`), so the bug is latent. Separately, `ProductGroup.ids` and `representative_id` are annotated `str` but are filled with `PlainId` values (`product_grouper.py`, `group_skus_clustering`).

**Change** (`src/pipeline/product_grouper.py`):
- `to_dict()` returns text: `group_id.plainId`, `[m.plainId for m in member_ids]`, `representative_id.plainId` or `None`.
- `ProductGroup.ids: List[PlainId]` and `representative_id: PlainId`.

**Tests:** `json.dumps(group.to_dict())` succeeds and contains plain strings; `tests/test_plain_id_guard.py` also covers `ProductGroup`.

## Task 4: The duplicate tool must not overflow the id limit

**Problem.** `duplicate_service.py` builds `PlainId(f"{orig_id}_dup{n}_{8 hex}")`, which adds about 14 characters. An original id of 87 or more characters raises `ValueError` in the middle of the loop, after some copies are already inserted.

**Change** (`src/web/services/duplicate_service.py`): cut the base so the copy id fits in the pattern's 100 characters (`ID_MAX_LENGTH` exported from `identifiers.py` next to `ID_PATTERN`, and the pattern built from it, so there is one number). The uuid part keeps copies unique.

**Test:** an original id of exactly 100 characters is duplicated without error, every copy id is valid, and the stored count matches `duplicates_added`.

**Decision for you.** Truncating loses the readable link between a copy and its original in the id, but `metadata["original_id"]` still holds the full original. The alternative is to fail before inserting anything. I recommend truncating, since this tool only multiplies test data.

## Task 5: Replace the tautological test, and pin `ids=` on real Milvus Lite

**Problem.**
- `test_a_raw_string_has_no_plain_id_to_read` only tests that Python's `str` has no `plainId` attribute.
- The fake client's `ids=` is a dictionary lookup. It cannot show what pymilvus does with the value, so nothing in the suite touches the real library.

**Change:**
- Delete the tautological test. The raw-string tests in `test_milvus_db.py` already prove what matters (`AttributeError` and no row changes).
- Add one test that uses a real `MilvusClient` on a `tmp_path` database. It shows that `ids=` lookups for an id with a space, a Chinese id and a uuid return only that row. It is skipped when `milvus_lite` is not importable, so the suite does not depend on it.

**Decision for you.** This adds the first test that needs the real library, and it slows the suite a little. It is the only guard against a pymilvus upgrade changing how `ids=` quotes the value. I recommend adding it; drop it if you want the suite to stay fake-only.

## Not doing

- **`embedding.py:55`**, where `PlainId(session.get("id"))` on a stored session with an odd id gives a bare 500. Sessions are written only by routes that already validated the id, so only a hand-edited session can trigger it.
- **A static type checker** (pyright or mypy) in the repo. Worth it only if you want annotations enforced before runtime; it is a separate decision.
- **The redundancy of `Path(pattern=...)` and `AfterValidator(PlainId)`.** The pattern feeds the OpenAPI schema; keep both.

## Verification

- `ruff check .` and `pytest -q --cov=src --cov-fail-under=80`.
- `pytest -q -W "error:Pydantic serializer warnings:UserWarning"`.
- Update the "Known limits" section of `2026-10-23-milvus-id-hardening.md`: remove the `str` subclass and duplicate-service items once fixed.
