# Browse & Search — Design Spec

Date: 2026-10-21
Author: W Cheng
Status: Draft, awaiting review
Branch: `feat/browse-search`

## 1. Purpose

Text search needs the user to think of words before they see anything. Browse & Search
removes that step. From the home page, a user opens a page that shows about ten images
already stored in the vector DB, clicks one, and sees similar products.

The audience is **product discovery**: the ten tiles should be varied, not ten
near-duplicates of the same product.

## 2. Non-goals

- Uploading a new image (covered by `/upload`).
- Pagination, filtering or full browsing of the DB.
- A user-selectable sample size in the UI (fixed at 10; `count` exists on the API only).
- In-page search results (a click goes to the existing results page).
- Changing how grouping chooses `representative_id`. Grouping currently sets it to
  the first member (`group_members[0]` in `ProductGroupManager.group_all_products`).
  Storing a "best" image at grouping time is a separate change.
- Any change to embeddings, thresholds or the confidence bands.

## 3. Existing behaviour relied on

| Fact | Where |
|------|-------|
| `GET /api/search/by-image/{image_id}` searches from a stored image's embedding, excludes the query image, and returns 404 for an unknown ID | `src/web/routes/search.py` |
| `GET /search/by-image/{image_id}` renders those results. It is linked only from batch-complete and grouping pages | `src/web/app.py`, `batch_complete.html`, `grouping.html` |
| `MilvusVectorLiteDB.get_all_ids()` exists. `InMemoryVectorDB` and the `VectorDatabase` base class do not have it | `milvus_db.py`, `product_grouper.py` |
| `ProductGroupUUID` has `member_ids` and `representative_id`. Groups are stored by `ProductGroupsDB` (Milvus only) and are a snapshot taken when grouping last ran | `milvus_db.py`, `product_grouper.py` |
| `metadata["image_url"]` and `metadata["overall"]` (quality score 0–1) are written by the single-image embedding route and the batch route. The CLI path (`ProductImagePipeline._store_embedding`) writes `quality_score` and no `image_url`. Any of these may be absent | `routes/embedding.py`, `routes/batch.py`, `product_image_pipeline.py` |
| `ProductGroupsDB` opens the Milvus file at `CONFIG["milvus_lite_path"]` even when the vector DB is in-memory, so groups must only be read when the vector DB is `MilvusVectorLiteDB` | `routes/grouping.py`, `milvus_db.py` |
| Cosine similarity above 0.90 means "same product" | `CLAUDE.md` |

## 4. Components

| Unit | Responsibility | Depends on |
|------|----------------|------------|
| `VectorDatabase.get_all_ids()` | List all stored IDs. Base class raises `NotImplementedError`; `InMemoryVectorDB` implements it | none |
| `BrowseService` in `src/web/services/browse_service.py` | `sample(count) -> BrowseSample`: decides which images to show | vector DB, groups DB (optional) |
| `GET /api/search/random?count=10` in `routes/search.py` | Validates `count` (1–50), calls the service, maps failures to a generic 500 | `BrowseService` |
| `GET /search/browse` in `app.py` | Page with the tile grid, Shuffle button and grouping hint banner. Must be registered before `/search/{session_id}` | the API above |
| Home card in `index.html` | "Browse & Search" card linking to `/search/browse` | none |
| Tile click | Goes to the existing `/search/by-image/{id}`, which gains a "Pick another" link back to `/search/browse` | existing |

## 5. Sampling rules (`BrowseService.sample(count)`)

1. Read the groups through an injected `list_groups` callable. The route wires it to
   `ProductGroupsDB.get_all_groups_if_present` only when the vector DB is `MilvusVectorLiteDB`, and to
   a function returning `[]` otherwise. If there are no groups, or `list_groups` raises
   (logged as a warning), use the random path (step 3). Otherwise use the grouped path (step 2).
2. **Grouped path**
   - Remove member IDs that are no longer in the vector DB. Remove groups left empty.
   - Shuffle the groups and take `count` of them. If there are fewer, take all.
   - From each group choose the member with the highest quality score, read as
     `metadata["overall"]`, else `metadata["quality_score"]`. If no member has a score, use
     `representative_id`; if that is also dead, use any live member.
3. **Random path**
   - Draw up to `4 × count` random IDs and fetch only those records with `get()`.
   - Keep a candidate only if its cosine similarity to every image already kept is below
     0.90. Stop at `count`. If the DB has fewer distinct products, return fewer.
4. In both paths, skip any image without `metadata["image_url"]`. In the grouped path,
   a group whose chosen image has no URL falls back to its next-best member with a URL;
   if none has one, the group is dropped.
5. The service never calls `get_all_embeddings`.
6. Randomness comes from an injected `random.Random`, so tests are deterministic.

### Response (`BrowseSample` / `RandomImagesResponse`)

| Field | Meaning |
|-------|---------|
| `images` | list of `{id, image_url, original_image_url}` |
| `mode` | `"grouped"` or `"random"` |
| `total_in_db` | `vector_db.size()` |
| `ungrouped_count` | `total_in_db` minus the number of IDs covered by live groups. `null` in random mode |

### Grouping hint on the page

- `mode == "random"`: "Showing a random sample. Group your products for a better
  selection." plus a button to `/grouping`.
- `mode == "grouped"` and `ungrouped_count > 0`: "N images aren't grouped yet." plus
  the same button.
- Empty DB: no hint. Show "No images yet" with a link to `/batch/upload`.

## 6. Errors and edge cases

| Case | Behaviour |
|------|-----------|
| Empty DB | 200, `images: []`, `total_in_db: 0` |
| `count` outside 1–50 | 400, as `top_k` validation does |
| Groups DB fails to read | Log a warning, fall back to the random path |
| Any other failure | Generic 500 `"Error sampling images"`; no exception text sent to the client |
| Picked image deleted after the sample was drawn | Existing by-image 404 message, plus a "Pick another" link |
| Fewer images than `count` | Return what exists; the grid is not padded |

## 7. Testing

Test-driven; use `tests/fakes.py` and the `client` fixture in `tests/test_web_routes.py`.

**Service (seeded RNG)**
- Random path: 15 distinct images give 10. 30 near-identical images give 1. Fewer than
  `count` returns all. An empty DB returns an empty list.
- Grouped path: one image per group. Highest score wins (`overall`, else `quality_score`). Falls back to
  `representative_id` when no scores exist. Dead member IDs and emptied groups are
  skipped. Fewer groups than `count` returns all groups. A group with no URL on its best
  member falls back to a member that has one.
- Both paths: images without `image_url` are never returned.
- `mode`, `total_in_db` and `ungrouped_count` are correct. A failing groups DB falls back
  to random. `get_all_embeddings` is never called.

**DB interface:** `InMemoryVectorDB.get_all_ids()` returns inserted IDs; the base class
raises `NotImplementedError`.

**Routes**
- `/api/search/random`: 200 shape, 400 for `count=0` and `count=51`, generic 500 on forced
  failure.
- `/search/browse` renders and is not captured by `/search/{session_id}`.
- The home page links to `/search/browse`; the by-image page links back to it.

**Milvus:** one test for the grouped path using the existing `milvus_pipeline` fixture.

**Manual:** the grid, Shuffle button and banner are front-end behaviour and are checked by
running the app. The repo has no browser tests.

## 8. Decisions recorded

- Reuse `/search/by-image/{id}` instead of adding a search endpoint, so there is one code
  path for "search from a stored image".
- Logic lives in `BrowseService`, not in the route (`search.py` already duplicates result
  mapping) and not in `ProductGrouper` (it would pull web concerns into `src/pipeline`).
- "Best image in a group" means highest `overall` (else `quality_score`), with fallbacks,
  because grouping does not record a best image.
- A 0.90 similarity cutoff for the random path reuses the existing "same product" band.

## 9. Amendment (2026-10-23, found in human review)

Two problems were found after the feature was implemented. Both are rules the original spec should have stated.

- **A1. Existing features keep their behaviour.** Adding `get_all_ids()` to `InMemoryVectorDB` made the `hasattr(..., 'get_all_ids')` guard in `ProductGroupManager.group_all_products` pass for the in-memory DB. Before this branch, grouping with an in-memory vector DB and stored products failed (`POST /api/grouping/run` returned 500 `"Error running grouping"`); on this branch it succeeded and wrote groups into the Milvus file, leaving groups that point at vectors that vanish on restart. Grouping must keep requiring the persistent `MilvusVectorLiteDB`: the guard becomes an explicit `isinstance` check that raises `ValueError("Product grouping requires the Milvus vector database")`.
- **A2. Read paths must not write.** The first Browse request created an empty `product_groups` collection, because `ProductGroupsDB` creates its collection on connect. Browse now reads groups through `ProductGroupsDB.get_all_groups_if_present()`, which returns `[]` and creates nothing when grouping has never run. Other existing pages that create the collection on read (for example grouping stats) are unchanged.
- **Not changed:** the Milvus Lite `Method not implemented!` (`AllocTimestamp`) line logged at `ERROR` when any collection is created or dropped is server noise that predates this feature, and is out of scope.

Tests: in-memory grouping with stored products still returns 500; a first Browse request on Milvus creates no `product_groups` collection; `get_all_groups_if_present` returns `[]` without creating the collection and returns stored groups once they exist.
