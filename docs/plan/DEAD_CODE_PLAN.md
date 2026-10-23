# Dead-code review plan

Status: `pipeline_preloader.py` and the candidates below are deleted. The web UI was checked by hand afterwards with no errors. A second `vulture` pass lists follow-ups at the end.

## How it was found

`vulture src main.py --min-confidence 60` (run with `uvx vulture`, nothing installed in the repo). It reports about 150 items. Most are false positives: FastAPI route handlers registered by decorators, pydantic fields and properties, and dataclass fields. Each real candidate was then checked by counting its name across `src/`, `main.py`, `tests/` and the templates. A count of 1 means the definition is the only mention.

Static analysis cannot see callers outside the repo (scripts, notebooks, other services), so "no caller" means no caller in this repo.

## Done

- Deleted `src/pipeline/pipeline_preloader.py`. Nothing imported it. `main.preload_models` (used by `--preload-model`, the Dockerfile and `scripts/start.sh`) does the same job.

## Removed

Trial: each candidate was deleted on its own in a throwaway worktree. All 32 still compiled and imported. The only failing tests were ones written to cover the candidates, and five deletions left unused imports (`ruff --fix` removed them). All were then removed together, with those tests.

| File | Removed |
|---|---|
| `product_grouper.py` | `ProductGroupUUID.remove_member`, `ProductGroupUUID.from_dict`, `InMemoryVectorDB.insert_batch`, `ProductGrouper.group_skus_connected_components`, `ProductGrouper.merge_groups` |
| `milvus_db.py` | `ProductGroupsDB.delete_group`, `ProductGroupManager.expand_group`, `get_group_for_product`, `get_group_statistics` |
| `product_image_pipeline.py` | `group_products`, `select_best_from_sku` |
| `quality_scorer.py` | `score_basic`, `score_aesthetic`, `filter_acceptable`, properties `basic_score`, `aesthetic_score_detail` |
| `basic_quality_scorer.py` | `filter_acceptable` |
| `best_image_selector.py` | `rank_images` |
| `embedding_generator.py` | `generate_from_path`, `text_image_similarity` |
| `clip_aesthetic_scorer.py` | `score_image` |
| `aesthetic_scorer.py` | `score_simple` |
| `watermark_remover.py` | `remove_watermark_auto` |
| `schemas.py` | `SessionStatusResponse`, `ErrorResponse` |
| `collage_detector.py`, `image_preprocessor.py` | attributes `line_threshold`, `_inpainting_model`, `_ocr_model` |

Kept on purpose: `delete_all` on the vector DB classes (interface parity, used by a test), route handlers, pydantic fields and properties.

## Follow-ups from the second vulture pass

Exposed by the removals, not yet checked or removed: `clustering.connected_components_cluster`, `ProductGroupsDB.update_group` and `find_group_by_product`, `ProductGrouper.group_skus_clustering`, `EmbeddingGenerator.similarity`, the `is_acceptable` methods on the score classes, and the unused `line_detection_threshold` parameter in `collage_detector.py`.

## Method for each group

1. Confirm there is no caller: grep `src/`, `main.py`, `tests/`, templates and JS, and the docs.
2. Decide per group: delete, or keep and cover with a test. Keeping untested public API is what drags coverage down, so "keep" should come with a test.
3. One commit per group. Run `pytest -q` and `ruff check .` after each.
4. Re-run vulture and coverage at the end, and record the new baseline in `docs/MODERNISATION_PLAN.md`.

## Context: why this matters

On the `refactor/code-quality` branch, "Group Products" and "Duplicate Records" in the web UI returned 500. The cause was not dead code but dependency drift: `requirements.lock` resolves pymilvus 3.0.2 and milvus-lite 3.2.2, which require a vector field in every collection and store the database as a directory. Both paths ran only against Milvus, which the fast test suite never exercised, so the tests did not catch it. Fixed in `fix: repair group products and duplicate records 500s on Milvus Lite 3.x`, with regression tests in `tests/test_milvus_routes.py`.
