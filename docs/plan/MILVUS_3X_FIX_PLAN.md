# Milvus Lite 3.x fix: Group Products and Duplicate Records 500s

Status: implemented in `58b26c2` on `refactor/code-quality`. Verified with a temporary real `pymilvus` install and with stub-based tests. Not yet verified against a real `data/milvus.db`.

## Symptom

In the web UI, "Group Products" (`POST /api/grouping/run`) and "Duplicate Records" (`POST /api/duplicate`) returned HTTP 500. The routes only logged `str(e)`, so the server log did not show the cause.

## Cause

Dependency drift. `requirements.lock` resolves `pymilvus` 3.0.2 and `milvus-lite` 3.2.2. The code was written for 2.x behaviour. Both failing paths run only against Milvus, which the fast test suite never exercised.

| Path | What failed | Why |
|---|---|---|
| Group Products | `create_collection` raised `schema has no vector field (FLOAT_VECTOR or SPARSE_FLOAT_VECTOR)` | `ProductGroupsDB` stores groups with only string fields. Milvus 3.x requires a vector field in every collection. |
| Duplicate Records | `shutil.copy2` raised `[Errno 21] Is a directory: '.../milvus.db'` | Milvus Lite 3.x stores the database as a directory, not a single file. |
| Duplicate Records (latent) | The model-mismatch guard would reject the second handle | `duplicate_service` opened its own `MilvusVectorLiteDB` with the default 512 dimension, and no model name. Any non-512 model (for example ViT-L-14) fails the guard. A second client on the same path is also unsafe. |

## Fix

1. **Groups collection** (`src/pipeline/milvus_db.py`)
   - Add a `placeholder_vector` field (`FLOAT_VECTOR`, dim `_PLACEHOLDER_VECTOR_DIM = 2`) and a FLAT/IP index when creating `product_groups`.
   - Include `[0.0, 0.0]` for it in `insert_group`.
   - Call `load_collection` after connecting and after `clear_all_groups`.
2. **Backup** (`src/web/services/duplicate_service.py`)
   - `backup_database` uses `copytree` when the path is a directory, `copy2` otherwise.
   - `list_backups` reports the summed size of a directory backup (`_path_size_bytes`).
3. **Duplicate service**
   - New `active_milvus_db()` returns the pipeline's own `MilvusVectorLiteDB`, or `None` if another backend is active.
   - `duplicate_records` and `backup_and_duplicate` raise `ValueError` (HTTP 400) when Milvus is not active, before any backup is made.
   - `get_database_stats` reports `exists: false` when Milvus is not active.
4. **Diagnosability**: the grouping and duplicate routes log with `exc_info=True`, so the next 500 shows a traceback.

## Tests

`tests/test_milvus_routes.py`, 6 tests, no `pymilvus` needed. Reverting the source changes makes 5 of the 6 fail.
- the groups collection schema includes a vector field and an index
- an inserted group carries the placeholder vector
- a directory backup is copied, and its size is reported
- duplicate without Milvus returns 400, not 500
- stats without Milvus report no database
- duplicate copies every record and returns an existing backup path (stubbed Milvus handle)

Also checked by hand with `pymilvus` 3.0.2 in `/tmp`, outside the repo and venv: grouping, duplicate, stats and backups all returned 200 against a real Milvus Lite directory.

## Known limits and follow-ups

- **Existing groups collections.** A `product_groups` collection created before this fix has no placeholder vector, so inserting groups into it fails. Drop that collection (or delete the db) once. There is no automatic migration.
- **Backup of a live database.** The backup copies the directory while the pipeline holds it open. The result may be inconsistent if writes are in flight. This was also true of the old file copy.
- **Unpinned range.** `requirements.txt` has `pymilvus[milvus-lite]>=2.3.0`. The lock is what ships, so the code must target 3.x. Consider pinning the range to `>=3,<4` so a future major version cannot break it silently.
- **Coverage gap.** `milvus_db.py` is at 20% coverage and was the source of this bug. The coverage work should include a fake client that exercises insert, search and group CRUD (see the test coverage section in `docs/MODERNISATION_PLAN.md`).
- **Not verified:** a run against your real `data/milvus.db`, and a Docker build with the fix.
