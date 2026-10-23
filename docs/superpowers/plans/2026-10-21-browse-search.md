# Browse & Search Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A home-page card opens `/search/browse`, which shows about ten varied stored images; clicking one opens the existing by-image similarity search.

**Architecture:** A new `BrowseService` chooses the images (best image per group when groups exist, otherwise random candidates filtered for near-duplicates). A thin `GET /api/search/random` route wraps it, and a new template calls that route. The click reuses `/search/by-image/{id}`. `get_all_ids()` is added to the `VectorDatabase` interface so both backends can be sampled.

**Tech Stack:** FastAPI, Jinja2 + Bootstrap templates, numpy, pytest (fast suite uses `tests/fakes.py`), ruff.

**Spec:** `docs/superpowers/specs/2026-10-21-browse-search-design.md`

## Global Constraints

- `count`: default 10, valid 1–50, otherwise HTTP 400 (manual `HTTPException`, like `top_k`).
- Random path draws at most `4 × count` candidates; near-duplicate cutoff is cosine similarity `>= GROUPING_SIMILARITY_THRESHOLD` (0.90, from `src/pipeline/config.py`) means "same product".
- Quality score is `metadata["overall"]`, else `metadata["quality_score"]`; thumbnails are `metadata["image_url"]`. Images without `image_url` are never returned.
- Failures return HTTP 500 with the exact detail `"Error sampling images"`; never send exception text to the client.
- Groups are read only when the vector DB is `MilvusVectorLiteDB` (otherwise `ProductGroupsDB` would create `./data/milvus.db`).
- The service must never call `get_all_embeddings`.
- `/search/browse` must be registered before `/search/{session_id}` in `src/web/app.py`.
- Run Python via `source venv/bin/activate`. Do not read `venv/`, `data/`, `uploads/`, `resource/`, `.env`.
- Commit messages end with `Co-Authored-By: Claude Sonnet 5.5 <noreply@anthropic.com>`. Branch: `feat/browse-search`.
- Code style: clear function names over comments; match surrounding module headers and docstring density.

## Review Focus

- A DB of 30 near-identical images shows 1 tile, not 10 (Task 2).
- Every stored record lacks `image_url` (CLI-ingested): API returns 200 with `images: []` and `total_in_db > 0`; the page says there are no thumbnails instead of "No images yet" (Tasks 2, 5).
- Groups exist but every member has been deleted: fall back to the random path rather than showing an empty "grouped" page (Task 3).
- A stored zero-norm embedding must not crash the near-duplicate filter with a divide-by-zero (Task 2).
- Image IDs containing spaces or `/` must still produce a working link to `/search/by-image/{id}` (Task 5).

---

## File Structure

| File | Change | Responsibility |
|------|--------|----------------|
| `src/pipeline/product_grouper.py` | modify | `get_all_ids()` on `VectorDatabase` and `InMemoryVectorDB` |
| `src/web/services/browse_service.py` | create | `BrowseService`, `BrowseSample`, `BrowseImage`, `build_browse_service` |
| `src/web/models/schemas.py` | modify | `BrowseImageItem`, `RandomImagesResponse` |
| `src/web/routes/search.py` | modify | `GET /search/random` |
| `src/web/app.py` | modify | `GET /search/browse` page route |
| `src/web/templates/browse.html` | create | Tile grid, Shuffle, hint banner, empty states |
| `src/web/templates/index.html` | modify | "Browse & Search" card |
| `src/web/templates/search_by_image.html` | modify | "Pick another" link |
| `tests/test_vector_db.py` | modify | `get_all_ids` tests |
| `tests/test_browse_service.py` | create | Service unit tests (no web, no fakes beyond `InMemoryVectorDB`) |
| `tests/test_web_routes.py` | modify | Route, page and Milvus-backed tests (reuses its fixtures) |
| `CLAUDE.md` | modify | Add `/search/browse` to the standalone pages line |

---

### Task 1: `get_all_ids()` on the vector DB interface

**Files:**
- Modify: `src/pipeline/product_grouper.py` (`VectorDatabase`, `InMemoryVectorDB`)
- Test: `tests/test_vector_db.py`

**Interfaces:**
- Produces: `VectorDatabase.get_all_ids(self) -> List[str]` (base raises `NotImplementedError`); `InMemoryVectorDB.get_all_ids` returns IDs in insertion order. `MilvusVectorLiteDB.get_all_ids` already exists with the same signature.

- [ ] **Step 1: Write the failing tests** in `tests/test_vector_db.py`, reusing the `db` fixture (inserts `x`, `xy`, `z`):
  - `test_get_all_ids_lists_every_inserted_id`: `assert db.get_all_ids() == ["x", "xy", "z"]`.
  - `test_get_all_ids_drops_deleted_ids`: after `db.delete("xy")`, `assert db.get_all_ids() == ["x", "z"]`.
  - `test_base_class_get_all_ids_is_not_implemented`: `with pytest.raises(NotImplementedError): VectorDatabase().get_all_ids()` (import `VectorDatabase`).
- [ ] **Step 2: Run** `source venv/bin/activate && pytest tests/test_vector_db.py -v`. Expected: the three new tests FAIL with `AttributeError`.
- [ ] **Step 3: Implement** `get_all_ids` on `VectorDatabase` (raise `NotImplementedError`, like its siblings) and on `InMemoryVectorDB` (`list(self.embeddings)`).
- [ ] **Step 4: Run** the same command. Expected: all PASS.
- [ ] **Step 5: Commit**
  ```bash
  git add src/pipeline/product_grouper.py tests/test_vector_db.py
  git commit -m "feat: add get_all_ids to the VectorDatabase interface"
  ```

---

### Task 2: `BrowseService` — random path

**Files:**
- Create: `src/web/services/browse_service.py`
- Test: `tests/test_browse_service.py`

**Interfaces:**
- Consumes: `VectorDatabase.get_all_ids/get/size` (Task 1); `ProductGroupUUID` from `src.pipeline.product_grouper`; `GROUPING_SIMILARITY_THRESHOLD` from `src.pipeline.config`.
- Produces (used by Tasks 3–4):
  ```python
  @dataclass
  class BrowseImage:
      id: str
      image_url: str
      original_image_url: Optional[str]

  @dataclass
  class BrowseSample:
      images: List[BrowseImage]
      mode: str                      # "random" | "grouped"
      total_in_db: int
      ungrouped_count: Optional[int]  # None in random mode

  class BrowseService:
      def __init__(self, vector_db: VectorDatabase,
                   list_groups: Callable[[], List[ProductGroupUUID]],
                   rng: Optional[random.Random] = None): ...
      def sample(self, count: int = 10) -> BrowseSample: ...
  ```

Tests build an `InMemoryVectorDB` directly with small hand-made vectors (e.g. one-hot vectors are mutually dissimilar; the same vector repeated is a near-duplicate), `list_groups=lambda: []`, and `rng=random.Random(0)`. A helper `add(db, id, vector, **metadata)` inserts with `image_url=f"/img/{id}.jpg"` unless overridden.

- [ ] **Step 1: Write the failing tests** (`tests/test_browse_service.py`):
  - `test_random_mode_returns_count_distinct_images`: 15 one-hot vectors (dimension 16) → `sample(10)`: `mode == "random"`, 10 unique ids, `total_in_db == 15`, `ungrouped_count is None`.
  - `test_fewer_images_than_count_returns_all`: 3 one-hot images → 3 returned.
  - `test_empty_db_returns_no_images`: `images == []`, `total_in_db == 0`.
  - `test_near_identical_images_collapse_to_one_tile`: 30 inserts of the same vector → exactly 1 image.
  - `test_images_without_image_url_are_never_returned`: 5 with URL, 5 with `image_url=None` (pass no URL) → returned ids ⊆ the 5; and a DB where *all* lack URLs gives `images == []` with `total_in_db == 5`.
  - `test_zero_norm_embedding_does_not_crash`: insert one image whose vector bypasses normalisation (set `db.embeddings["z"] = np.zeros(16)` and `db.metadata["z"] = {"image_url": "/img/z.jpg"}`) alongside a normal one → `sample(10)` returns without error.
  - `test_sampling_never_loads_all_embeddings`: set `db.get_all_embeddings = lambda: pytest.fail("loaded every vector")`, call `sample(10)`.
  - `test_same_seed_gives_same_sample_and_different_seeds_can_differ`: two services with `Random(1)` equal; with 40 one-hot images (dimension 40) and `count=5`, `Random(1)` vs `Random(2)` differ.
- [ ] **Step 2: Run** `pytest tests/test_browse_service.py -v`. Expected: collection FAILS (`ModuleNotFoundError: browse_service`).
- [ ] **Step 3: Implement** the dataclasses and `BrowseService.sample` in `src/web/services/browse_service.py`, random path only for now (a private `_sample_random(count, all_ids)`; `mode="random"`). Algorithm: `rng.sample` of `min(len(ids), 4*count)` ids; `get()` each; skip `None` and records without `image_url`; normalise each embedding to unit length, treating zero norm as "keep, distinct from everything"; keep a candidate only if its dot product with every kept vector is `< GROUPING_SIMILARITY_THRESHOLD`; stop at `count`. `total_in_db = vector_db.size()`. Store `list_groups` and `rng` (default `random.Random()`); `list_groups` is unused until Task 3.
- [ ] **Step 4: Run** the same command. Expected: all PASS.
- [ ] **Step 5: Commit**
  ```bash
  git add src/web/services/browse_service.py tests/test_browse_service.py
  git commit -m "feat: add BrowseService random sampling with near-duplicate filter"
  ```

---

### Task 3: `BrowseService` — grouped path

**Files:**
- Modify: `src/web/services/browse_service.py`
- Test: `tests/test_browse_service.py`

**Interfaces:**
- Consumes: Task 2's `BrowseService`, `BrowseSample`.
- Produces: `sample()` now returns `mode="grouped"` with `ungrouped_count: int` when at least one group has a live member; otherwise behaves as Task 2.

Tests use `ProductGroupUUID(group_id=..., member_ids=[...], representative_id=...)` and `list_groups=lambda: groups`.

- [ ] **Step 1: Write the failing tests:**
  - `test_grouped_mode_returns_one_image_per_group`: 3 groups of 2 members each (6 one-hot images) → 3 images, one from each group, `mode == "grouped"`.
  - `test_best_member_is_the_highest_overall_score`: one group, members with `overall` 0.2 / 0.9 / 0.5 → returned id is the 0.9 one. A second case where only `quality_score` is set (0.3 vs 0.8) → the 0.8 one.
  - `test_unscored_group_uses_representative_id`: no scores, `representative_id="b"` of `["a","b"]` → `"b"`.
  - `test_dead_representative_falls_back_to_a_live_member`: representative not in DB → returns a live member.
  - `test_dead_members_and_emptied_groups_are_skipped`: group A all deleted, group B live → only B's image returned.
  - `test_best_member_without_url_falls_back_to_next_member`: best-scored member has no URL → the next-best member with a URL; a group where no member has a URL is dropped.
  - `test_fewer_groups_than_count_returns_every_group`.
  - `test_ungrouped_count_is_images_not_in_any_live_group`: 5 images, groups cover 3 → `ungrouped_count == 2`.
  - `test_all_groups_dead_falls_back_to_random_mode`: groups only reference deleted ids → `mode == "random"`, `ungrouped_count is None`.
  - `test_failing_list_groups_falls_back_to_random_mode`: `list_groups` raises `RuntimeError` → `mode == "random"`, no exception.
- [ ] **Step 2: Run** `pytest tests/test_browse_service.py -v`. Expected: the new tests FAIL; Task 2 tests still PASS.
- [ ] **Step 3: Implement** in `BrowseService.sample`: call `list_groups()` inside `try/except Exception` (log `logger.warning`, treat as no groups). Keep only live members (`id in set(all_ids)`), drop empty groups; if none remain use the random path. Otherwise shuffle groups with `self._rng` and walk them until `count` images are collected. For each group, rank members as: scored members by score descending, then `representative_id` (if live), then the remaining members; choose the first whose record has an `image_url`; skip the group if none does. `ungrouped_count = total_in_db − number of distinct live member ids across all live groups`.
- [ ] **Step 4: Run** the same command. Expected: all PASS.
- [ ] **Step 5: Commit**
  ```bash
  git add src/web/services/browse_service.py tests/test_browse_service.py
  git commit -m "feat: pick the best image per group in BrowseService"
  ```

---

### Task 4: `GET /api/search/random`

**Files:**
- Modify: `src/web/models/schemas.py`, `src/web/routes/search.py`, `src/web/services/browse_service.py`
- Test: `tests/test_web_routes.py`

**Interfaces:**
- Consumes: `BrowseService`, `BrowseSample` (Tasks 2–3).
- Produces:
  ```python
  # schemas.py
  class BrowseImageItem(BaseModel): id: str; image_url: str; original_image_url: Optional[str] = None
  class RandomImagesResponse(BaseModel):
      images: List[BrowseImageItem]; mode: str; total_in_db: int; ungrouped_count: Optional[int] = None
  # browse_service.py
  def build_browse_service(pipeline) -> BrowseService
  ```
  `build_browse_service` wires `list_groups` to `ProductGroupsDB(db_path=CONFIG.get("milvus_lite_path", "./data/milvus.db")).get_all_groups` only when `isinstance(pipeline.vector_db, MilvusVectorLiteDB)`, else to `lambda: []`. Route: `GET /api/search/random` with query `count: int = 10`.

Route tests live in `tests/test_web_routes.py` and use its `client`, `pipeline`, `milvus_pipeline` fixtures and `run_through_embedding(client, color, product_id)` helper (colours `RED`, `RED_SHADE`, `BLUE`).

- [ ] **Step 1: Write the failing tests:**
  - `test_random_images_returns_stored_images_with_urls`: store red and blue → `GET /api/search/random` is 200, `mode == "random"`, `total_in_db == 2`, ids `{"red","blue"}`, each `image_url` starts with `/uploads/sessions/`.
  - `test_random_images_on_an_empty_db_is_an_empty_list`: 200, `images == []`, `total_in_db == 0`.
  - `test_random_images_rejects_count_outside_one_to_fifty` parametrised over `0` and `51` → 400.
  - `test_random_images_failure_is_a_generic_500`: `monkeypatch.setattr(BrowseService, "sample", boom)` → 500 and `detail == "Error sampling images"` (assert the exception text is absent).
  - `test_random_images_uses_groups_on_milvus` (uses `milvus_pipeline`): store red, red-shade, blue; `POST /api/grouping/run?similarity_threshold=0.95`; then the endpoint returns `mode == "grouped"`, 2 images (one per group), `ungrouped_count == 0`.
  - `test_in_memory_mode_never_opens_the_groups_db`: with the default `pipeline` fixture, `monkeypatch.setattr("src.web.services.browse_service.ProductGroupsDB", boom)` and the endpoint still returns 200 in `random` mode.
- [ ] **Step 2: Run** `pytest tests/test_web_routes.py -k random_images -v`. Expected: FAIL (404 for the route / import errors).
- [ ] **Step 3: Implement** the schemas, `build_browse_service`, and the route in `routes/search.py` (validate `count`; `try: sample = build_browse_service(get_pipeline()).sample(count)`; `except Exception`: log, raise `HTTPException(500, "Error sampling images")`; map to `RandomImagesResponse`). Use the same `def` (sync) style as the other routes so it runs in the threadpool.
- [ ] **Step 4: Run** `pytest tests/test_web_routes.py tests/test_browse_service.py -v`. Expected: all PASS.
- [ ] **Step 5: Commit**
  ```bash
  git add src/web/models/schemas.py src/web/routes/search.py src/web/services/browse_service.py tests/test_web_routes.py
  git commit -m "feat: add /api/search/random endpoint backed by BrowseService"
  ```

---

### Task 5: Browse page, home card and back link

**Files:**
- Create: `src/web/templates/browse.html`
- Modify: `src/web/app.py`, `src/web/templates/index.html`, `src/web/templates/search_by_image.html`, `CLAUDE.md`
- Test: `tests/test_web_routes.py`

**Interfaces:**
- Consumes: `GET /api/search/random` response shape (Task 4).
- Produces: `GET /search/browse` (HTML).

- [ ] **Step 1: Write the failing tests:**
  - `test_browse_page_renders_and_is_not_taken_for_a_session`: `GET /search/browse` is 200 and the HTML contains `/api/search/random`.
  - `test_home_page_links_to_browse`: `GET /` contains `href="/search/browse"`.
  - `test_by_image_page_links_back_to_browse`: `GET /search/by-image/red` contains `href="/search/browse"`.
- [ ] **Step 2: Run** `pytest tests/test_web_routes.py -k "browse_page or links" -v`. Expected: FAIL.
- [ ] **Step 3: Implement the page route** in `app.py`, placed next to `/search/text` and above `/search/{session_id}`, rendering `browse.html`.
- [ ] **Step 4: Implement `browse.html`** extending `base.html`, modelled on `search_text.html` (spinner, dismissible error, `d-none` toggling). It fetches `/api/search/random` on load and on the Shuffle button, and renders tiles as `<a href="/search/by-image/${encodeURIComponent(id)}">` wrapping `<img src=image_url>`, building DOM nodes (no unescaped `innerHTML` of response fields). States: empty DB (`total_in_db == 0`) → "No images yet" with a link to `/batch/upload`; `images` empty but `total_in_db > 0` → "No stored images have thumbnails to show"; hint banner with a button to `/grouping` when `mode == "random"` ("Showing a random sample. Group your products for a better selection.") or when `mode == "grouped"` and `ungrouped_count > 0` ("N images aren't grouped yet."). Include a Back to Home button like the other pages.
- [ ] **Step 5: Add the home card** to `index.html` after the Text Search card, same markup: title "Browse & Search", short description, button to `/search/browse`. **Add the back link** in `search_by_image.html`: a "Pick another" button to `/search/browse` beside the existing Back button. **Update `CLAUDE.md`**: add `/search/browse` to the "Additional standalone pages" line.
- [ ] **Step 6: Run** `pytest tests/test_web_routes.py -v`. Expected: all PASS.
- [ ] **Step 7: Check in the running app** (`source venv/bin/activate && python main.py --serve --port 8000`; use the `run` skill if helpful): with some stored images the grid appears; Shuffle changes it; clicking a tile lands on its results; "Pick another" returns; an empty DB shows the empty state; the banner links to `/grouping`. Report what was and was not observed.
- [ ] **Step 8: Commit**
  ```bash
  git add src/web/app.py src/web/templates/browse.html src/web/templates/index.html src/web/templates/search_by_image.html CLAUDE.md tests/test_web_routes.py
  git commit -m "feat: add Browse & Search page, home card and back link"
  ```

---

### Task 6: Full verification

**Files:** none (fix anything found in the files above).

- [ ] **Step 1: Run the CI commands.** `source venv/bin/activate && ruff check . && pytest -q --cov=src --cov-fail-under=80`. Expected: ruff clean; all tests pass; coverage ≥ 80%.
- [ ] **Step 2: Confirm the spec checklist.** Every test named in spec §7 has a counterpart in Tasks 1–5; list any gap and add its test.
- [ ] **Step 3: Commit** any fixes (`fix: …`). If nothing changed, no commit.

---

## Self-review

- **Spec coverage:** §4 components → Tasks 1, 2, 4, 5; §5 sampling (random, grouped, fallbacks, URL filter, no `get_all_embeddings`, injected RNG, groups only on Milvus) → Tasks 2–4; response fields → Task 4; hint/empty states → Task 5; §6 errors (empty, 400, groups failure, generic 500, deleted pick via existing 404 + "Pick another") → Tasks 3–5; §7 tests → each task; manual front-end check → Task 5 step 7.
- **Added beyond the spec:** (a) all-groups-dead falls back to the random path; (b) page message when images exist but none has a thumbnail; (c) zero-norm embedding guard. These are decisions the spec left open.
- **Types:** `BrowseSample` fields and `RandomImagesResponse` fields match one-to-one; `build_browse_service` is the only constructor used by the route.
