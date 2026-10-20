# Refactor plan: code-quality follow-up to Tier 1

Status: implemented on branch `refactor/code-quality` (see the table below for scope notes), nothing pushed. Sits between Tier 1 and Tier 2 of `docs/MODERNISATION_PLAN.md`.

## Context

An independent review of the repository after Tier 1 rated the code quality **mixed**. The code is readable, `src/pipeline` never imports `src/web`, and CI is green (`ruff` clean, 39 tests passing). The review was targeted (grep plus reading key files), not a line-by-line read of all code. It found robustness gaps, scattered configuration and thin test coverage.

Findings behind this plan:

| # | Finding | Where |
|---|---|---|
| 1 | `session_id` and `batch_id` are used as path segments without validation, so path traversal is possible; `delete_session` calls `rmtree` on that path | `src/web/services/session_manager.py` |
| 2 | All routes are `async def` but call CLIP, rembg and the vector DB synchronously, with `await asyncio.sleep(0)` as a fake yield; the server is unresponsive during processing | `src/web/routes/batch.py:115-290` and other routes |
| 3 | About 40 blanket `except Exception` blocks; many return `str(e)` to clients; Milvus silently falls back to in-memory | `src/web/routes/*`, `product_image_pipeline.py:121` |
| 4 | Hidden global state: unlocked lazy-init singleton, `ModelCache` singleton, `CONFIG` built at import time | `pipeline_service.py`, `model_cache.py`, `config.py` |
| 5 | Scattered configuration and inconsistent defaults (similarity threshold 0.85 in some modules, 0.90 in others; `CLIP_MODEL` read in three places) | `main.py`, `embedding_generator.py`, `clip_aesthetic_scorer.py`, `grouping.py`, `milvus_db.py` |
| 6 | `torch` and `open_clip` imported at module level, which is why `tests/conftest.py` must stub them | `model_cache.py`, `clip_loader.py`, `embedding_generator.py`, `clip_aesthetic_scorer.py` |
| 7 | Leftover `print` calls and `if __name__` demo blocks in library modules | most `src/pipeline` modules |
| 8 | No tests for `ProductImagePipeline`, watermark, collage, preprocessor, embedding, grouper, or the upload, batch and search routes | `tests/` |

The green `ruff` result is weaker than it looks: `pyproject.toml` ignores F401 (unused imports), F841 (unused variables), F541, E501 and E741, which hides real dead code.

## Rules

- One commit per step, each independently reviewable, with `ruff` and `pytest` green after each.
- No rewrite of the `pipeline/` or `web/` layers, no package moves, no new dependency-injection framework. Restructuring `src/` stays in Tier 3.
- No threshold or default-model changes while refactoring configuration.
- Do not mix lint-ratchet commits with behaviour changes.
- Never push or open a PR without asking first.

## P0: correctness and safety

1. **Validate `session_id` and `batch_id`.** Accept UUIDs only in the `SessionManager` path helpers. Batch IDs use a `batch_<id>` prefix and must keep working.
   - Files: `src/web/services/session_manager.py`, `tests/test_session_manager.py`
   - Risk: low
   - Verify: new tests show traversal IDs such as `../x` are rejected (`ValueError` or 404); existing tests pass.
2. **Stop blocking the event loop.** Make routes that call the pipeline plain `def` (FastAPI runs them in its threadpool), or wrap the calls in `run_in_threadpool`. Remove the `await asyncio.sleep(0)` hacks in `batch.py`. Do `batch.py` first, then the other routes one commit each if wanted.
   - Files: `src/web/routes/batch.py`, `search.py`, `embedding.py`, `preprocessing.py`, `split.py`, `quality.py`, `watermark.py`, `grouping.py`
   - Risk: medium. The shared pipeline's lazy properties are not locked, so do step 3 straight after.
   - Verify: `tests/test_api.py` passes, and a manual check that `/health` responds during a batch run.
3. **Make `PipelineService` lazy init thread-safe** with a lock, and stop swallowing errors in the duplicate check in `batch.py`.
   - Files: `src/web/services/pipeline_service.py`, `src/web/routes/batch.py`
   - Risk: low
   - Verify: a unit test that initialises from several threads with a stub pipeline.

## P1: configuration, duplication and lint

4. **Central settings.** Move the `(512, 512)` size, the 0.85 / 0.90 / 0.999 thresholds, the `CLIP_MODEL` and `CLIP_PRETRAINED` reads, `use_gpu` and `output_dir` into `src/pipeline/config.py`. The grouping 0.90 against pipeline 0.85 difference must be decided deliberately, not by accident. This is the plain-constants version of Tier 2 "typed settings"; the `pydantic-settings` upgrade can follow later.
   - Risk: low to medium (default drift)
   - Verify: a test that the config defaults equal the current behaviour.
5. **One error-handling path.** Collapse the repeated `except Exception` -> `HTTPException(500, str(e))` into one helper or an app-level exception handler. Log the detail and stop returning `str(e)` to clients. Make the Milvus-to-in-memory fallback explicit and logged.
   - Files: `src/web/routes/*.py`, `src/web/app.py`, `src/pipeline/product_image_pipeline.py`
   - Verify: API tests for the error response shape.
6. **Remove debug leftovers.** Replace `print` in library modules with `logging`, delete `if __name__` demo blocks and commented-out debug code. Keep the `main.py` CLI output.
   - Risk: low
   - Verify: `ruff`, `pytest`.
7. **Lint ratchet, one rule per commit.** Remove F401, then F841, F541 and E741 from the `pyproject.toml` ignore list, fixing the hits each time (`ruff check --fix` handles F401 and F541). E501 last, or leave it with the 120-column line length.
   - Verify: `ruff` clean, `pytest`.

## P2: testability

8. **Tests with a fake embedder and fake vector DB** for `ProductImagePipeline` orchestration, the upload route (size and extension checks), the batch route, and watermark, collage and preprocessor on tiny synthetic images (no rembg). Include the `milvus_db.py` model-mismatch guard.
9. **Lazy-import `torch` and `open_clip`** inside functions in `model_cache.py`, `clip_loader.py` and the two scorers, so the `conftest.py` stubs can be dropped.
   - Risk: medium
   - Verify: `pytest -m model` (opt-in) and a Docker build.
10. **Split `ProductImagePipeline.process_image`** (about 130 lines) into named step methods. Split `milvus_db.py` (589 lines) only if it is touched anyway.

## Status

| Step | Result |
|---|---|
| 1 | Done. `SessionManager` accepts only UUIDs and `batch_<8 hex>`; traversal tests added. |
| 2 | Done. `batch.py` and all other pipeline-calling routes are plain `def`; fake yields removed. `upload_image` reads via `file.file`. |
| 3 | Done. Double-checked lock, failed builds not cached; a vector DB failure in the batch duplicate check now fails that image instead of being swallowed. |
| 4 | Done as plain constants in `config.py`; defaults unchanged and pinned by `tests/test_config_defaults.py`. The ProductGrouper 0.85 default and `main.py`'s 0.95 are left as they were. |
| 5 | Partly done. Clients no longer receive `str(e)` on 500s and the Milvus fallback logs an error. The repeated `try/except` blocks were not collapsed into one handler. |
| 6 | Done for `src/pipeline`; `pipeline_preloader.py` keeps its script entry point. |
| 7 | Done for F401, F541, F841 and E741. E501 stays ignored. |
| 8 | Done. `tests/test_pipeline_e2e.py` runs the real pipeline and `tests/test_web_flow_e2e.py` the real page flow with a fake thumbnail embedder; plus Milvus guard and stage tests. |
| 9 | Done, but not yet run against real weights (`pytest -m model` skips in this venv, which has no torch). Run it and a Docker build before merging. |
| 10 | `process_image` split. `milvus_db.py` was not split. |

## Order of work

Steps 1 to 3 first: step 1 is the only security issue and is small, and steps 2 and 3 fix the main runtime weakness. Then 4 to 7, then 8 to 10. Step 8 can move earlier if a behaviour change needs a safety net.
