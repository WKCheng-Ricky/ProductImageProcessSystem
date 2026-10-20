# Tier 1 implementation plan (docs/MODERNISATION_PLAN.md)

## Context
The repo has no tests, CI or pinned deps, and a Docker setup with known flaws. Tier 1 adds the guardrail half of the "spec-driven agentic development" story. Work happens on branch `feat/tier1-guardrails` off `main`, with **one commit per item** (each ending with the Co-Authored-By trailer). No `src/` restructuring.

Findings from exploration that change the plan's assumptions:
- `/health` **already exists** (`src/web/app.py:~290`, static JSON). Item 3 is therefore: make it meaningful, switch the healthcheck to it, and fix logging. It is not a new endpoint.
- Logging is already stdlib `logging` (`basicConfig` in `app.py`), plus about 107 `print`/logging sites. "Structured" means a small formatter configured in one place, not a rewrite of call sites.
- `SearchResultItem.confidence_level` (`schemas.py:~112`) uses 0.9 / **0.7** bands while CLAUDE.md says 0.9 / **0.8**. The code is authoritative (see Decisions); CLAUDE.md gets corrected.
- Docker healthcheck imports `requests`, which is not in `requirements.txt`, so it is broken.
- `requirements.txt` pins torch to `+cpu` via an extra index; the other packages use `>=`.
- Open risk: `src.web.app` imports routes, which import `product_image_pipeline` and possibly torch/open_clip at module level. This must be checked in step 1.

## Status (updated 2026-10-20, branch `feat/tier1-guardrails`, nothing pushed)

| Step | State | Commit | Notes |
|---|---|---|---|
| 0. Branch and plan docs | Done | `d97c7f3`, `6038402` | |
| 1. Tests | Done | `c0fbb07` | 39 tests pass (36 plus 3 `/ready` tests added in step 3). Fixed a circular import (`milvus_db` / `product_grouper`) found when importing `src.web.app`. `torch` and `open_clip` are stubbed in `tests/conftest.py` only when not installed. Not done: the fake-encoder contract tests and the model-mismatch guard test. |
| 2. CI | Done, not yet run on GitHub | `e85da78` | `test` and `docker` jobs. Ruff ignores F401, F541, F841, E501, E741 (41 existing violations). The `model-smoke` job is not written. |
| 3. `/health`, `/ready`, logging | Done | `373fbc1` | `configure_logging()` in `src/logging_config.py`, used by `main.py` and `app.py`. |
| 4. Dockerfile and compose | Done, verified with rootless nerdctl | `e09a7e2`, `533d001` | Build OK, `/health` and `/ready` return 200, runs as `uid=1000(app)`, healthcheck command exits 0. Bind mounts failed with `PermissionError` (root-owned in rootless), so `data` and `uploads` are now named volumes. nerdctl does not report a "healthy" status, so that check was done by running the command by hand. |
| 5. Pinning and `pyproject.toml` | Done | `e85da78`, `46fa102` | `requirements.lock` generated with `uv pip compile`; Docker installs from it. The build with numpy 2.5.4 and torch 2.5.1 works. `requirements.txt` stays the human-edited input (no `requirements.in`). CI still installs a light dependency list rather than the lock. |
| 6. README rewrite | Done | `416ec23` | CLAUDE.md, README and plan docs now say 70% as the lower similarity band. The CI badge points at `WKCheng-Ricky/ProductImageProcessSystem`. |
| 7. Faster deployment | Planned | | ViT-B-32 default, weights baked into the image. See step 7. |

**Remaining from this plan**
- Run `pytest -m model` once against real weights (written in `tests/test_model_smoke.py`, manual-only `model-smoke` job, but never run with real torch).
- Step 7 (faster deployment), planned below.
- Fake-encoder contract tests, including the model-mismatch guard in `milvus_db.py`.
- Push the branch (needs your permission) so CI runs for the first time, then check both jobs.
- Optional: install CI from `requirements.lock`.

**Environment notes**
- Claude Code's shell runs inside an Ubuntu distrobox. Rootless nerdctl needs the host: use `distrobox-host-exec nerdctl ...`, or the alias `[ -f /run/.containerenv ] && alias nerdctl='distrobox-host-exec nerdctl'` in `~/.bashrc` (interactive shells only).
- `venv/` was built with `uv` and light dependencies only (no torch), so `pytest -m model` cannot run there.

## Steps (in order)

### 0. Branch and plan doc
`git checkout -b feat/tier1-guardrails`. First commit: this plan saved in the repo as `docs/TIER1_IMPLEMENTATION_PLAN.md`.
**Git rule: never `git push` (or open a PR) without asking you first.** Local commits only. The untracked `docs/MODERNISATION_PLAN.md` and `.bak` are not committed in the first code commit. Commit the plan doc alone as `docs: add modernisation plan` and delete or ignore the `.bak`.

### 1. Tests (pytest), about 20 fast tests
Create `tests/conftest.py` plus:
- `tests/test_session_manager.py`: create/get/update/delete/cleanup using `tmp_path` (`SessionManager(storage_dir=...)`). Missing session raises `ValueError`.
- `tests/test_vector_db.py`: `InMemoryVectorDB` (`src/pipeline/product_grouper.py:101`) for insert, search ordering, threshold, delete, get, and normalisation.
- `tests/test_clustering.py`: `dbscan_cluster` and `connected_components_cluster` (`src/pipeline/clustering.py`) on synthetic unit vectors with two obvious clusters and an outlier.
- `tests/test_quality_scorer.py`: `BasicQualityScorer` (`basic_quality_scorer.py`) on synthetic PIL images. Sharp vs blurred, dark vs normal, tiny vs large; `is_acceptable`.
- `tests/test_schemas.py`: pydantic validation, `confidence_level` / `score_percentage` bands, `QualityScoreResponse` helpers.
- `tests/test_api.py`: FastAPI `TestClient` against `src.web.app.app`. `/health`, `/api/session/{id}/status` for a missing session (404), and the upload route with `get_pipeline` monkeypatched to a stub. Use `app.dependency_overrides[get_session_manager]` with a tmp-dir manager. Mocking `PipelineService._pipeline` avoids model loads.
- Heavy-import handling: if importing the app pulls in torch/open_clip, `conftest.py` registers a `heavy` marker and, for API tests, stubs the modules in `sys.modules` before import. Prefer CI installing real deps only if the stub approach gets ugly. Decide after a trial import.
- No PR-gating test may download or load a model. An autouse fixture sets `USE_MILVUS=false`.

### Model behaviour: how it is verified (answering the CI-weight concern)
Standard practice is a test pyramid. Model weights stay out of the PR gate:
1. **PR gate (fast, no weights):** unit tests plus contract tests using a tiny fake encoder (deterministic vectors) to check the pipeline wiring, shapes, normalisation, thresholds and the model-mismatch guard in `milvus_db.py`. This is what verifies *our* logic.
2. **Opt-in `@pytest.mark.model` smoke tests** (excluded by default via `-m "not model"`): load the real model and assert behavioural properties, not exact numbers: embedding dim matches the model (512 for ViT-B-32, 768 for ViT-L-14), a vector is unit-norm, the same image gives cosine about 1.0, a lightly augmented copy scores above 0.9, an unrelated image scores below 0.8, and a text query ranks the matching image first. Run locally (`pytest -m model`) and in the manual-only CI job, with `actions/cache` on the weights directory. ViT-B-32 is now the default model, so the job needs no override.
3. **Evaluation set (Tier 2 item 9):** a few labelled pairs measuring precision/recall of the 90%/70% thresholds. This is where real quality regressions get caught.

### 2. GitHub Actions CI (`.github/workflows/ci.yml`)
- Job `test`: Python 3.12, pip cache, install the deps needed by the tests, then `ruff check .` and `pytest -q`, on `push` and `pull_request`.
- Job `docker`: image build only (no push), with buildx and GHA cache. (CI runners have Docker; local verification uses nerdctl.)
- Optional job `model-smoke`: `workflow_dispatch` only (no schedule), never on PRs (see "Model behaviour" below).
- Ruff config goes in `pyproject.toml` (step 5). Start with a lenient rule set (`E`, `F`, `I` ignored if noisy) and fix or ignore existing violations so CI starts green. List what was ignored in the commit message.

### 3. `/health` and logging
- Keep `GET /health` cheap and dependency-free (liveness). Add `version` to the response; never load models in it.
- Add `GET /ready` (readiness): checks that the session storage dir is writable and that the vector DB is reachable if the pipeline is already initialised (reuse `db.size()` as in `/api/vector-db/health`). It must not trigger model loading. Returns 503 if a check fails. Small, so it goes in this commit.
- New `src/logging_config.py` with `configure_logging()`: one-line JSON or key=value formatter, level from `LOG_LEVEL`. Call it from `app.py` in place of the inline `basicConfig`, and from `main.py`.
- Docker healthcheck: `python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"` (stdlib only, no `requests`).
- Test: `/health` returns 200 and the JSON shape.

### 4. Dockerfile and compose
- Real multi-stage build: `builder` creates a venv with the pip install; the runtime stage copies the venv, apt runtime libs only, and the app code.
- Non-root `app` user; chown `data/ uploads/ resource/`. Check that the bind mounts in compose work for uid 1000.
- Remove the commented-out ENV lines, and the `version: '3.8'` key from compose. Move the healthcheck to the stdlib `/health` call.
- Verify locally with nerdctl (see Verification).

### 5. Pinning and `pyproject.toml`
- `pyproject.toml` with `[tool.ruff]` and `[tool.pytest.ini_options]` (testpaths, markers). It carries tooling config only, not packaging.
- Lock: `requirements.in` (the current loose list) compiled to `requirements.lock` using `pip-compile`/`uv pip compile`, keeping the torch CPU extra index. Dockerfile and CI install from the lock. `requirements.txt` stays as the human-edited entry point, or becomes `.in`. Small `requirements-dev.txt` (pytest, ruff, httpx).
- Needs `httpx` for `TestClient`.

### 6. README rewrite
Tagline first, then a one-paragraph result, an architecture diagram (mermaid, from the CLAUDE.md pipeline), a quickstart (venv and Docker), a test/CI badge, and a "Built with spec-driven agentic development" section. The workflow doc is Tier 2, so link to it only once it exists, or say "coming". Use the "almost entirely" wording and avoid the "Claims to avoid" list. Fix stale facts (README says Python 3.12 but the Dockerfile uses 3.11; align them to 3.12).

### 7. Faster deployment (model footprint)
Context in `docs/MODERNISATION_PLAN.md`, "Model footprint". Decisions: ViT-B-32 (`openai`) becomes the default, ViT-L-14 stays opt-in, CLIP and U2Net weights are baked into the image, and no Hugging Face token is ever used.
- **Defaults:** change `ViT-L-14` to `ViT-B-32` in `main.py` and `src/pipeline/embedding_generator.py` (the default `clip-vit-large` key becomes `clip-vit-base`). Check the hard-coded `768` fallbacks in `milvus_db.py` and `product_grouper.py`; the dimension must come from the loaded model.
- **Compose and docs:** `docker-compose.yml` env defaults, README, CLAUDE.md, `docs/OVERVIEW.md` (768 to 512 where it describes the default).
- **Dockerfile:** a build step runs the app's own loaders once to download CLIP and U2Net into a fixed cache (`HF_HOME`, `U2NET_HOME` under `/opt/models`), copied into the runtime stage and readable by uid 1000. Set `HF_HUB_OFFLINE=1` at runtime. Order layers so code changes do not re-download weights.
- **Hugging Face warning:** the "unauthenticated requests" notice comes from anonymous downloads. Baking the weights plus offline mode removes it at runtime. Do not pass a token as a build arg or secret; the public models need none.
- **Existing data:** a Milvus collection built with ViT-L-14 is refused by the model-mismatch guard. Recreate it (`docker volume rm`) or set `CLIP_MODEL=ViT-L-14` and `CLIP_PRETRAINED=openai`.
- **Verify:** `pytest -m model` against real weights, then `nerdctl build`, start with the network disabled to prove offline start-up, `/health` and `/ready` return 200, and compare image size and first-start time with before.

## Critical files
`tests/*` (new), `.github/workflows/ci.yml` (new), `pyproject.toml` (new), `requirements.lock` and `requirements-dev.txt` (new), `src/logging_config.py` (new), `CLAUDE.md` (confidence table), `src/web/app.py`, `main.py`, `Dockerfile`, `docker-compose.yml`, `README.md`.

## Verification
- `source venv/bin/activate && pip install -r requirements-dev.txt && pytest -q && ruff check .` all green locally.
- With nerdctl: `nerdctl build -t pips .`, `nerdctl compose up -d`, `curl -f localhost:8001/health` and `/ready`, `nerdctl inspect --format '{{.State.Health.Status}}' <container>` becomes `healthy`, and `nerdctl exec <container> id` shows non-root. Then `nerdctl compose down`.
- `pytest -m model` run once locally to confirm the smoke tests pass against real weights.
- CI confirmation requires pushing. I will ask before any `git push`; nothing is pushed otherwise.
- `git log --oneline` shows one commit per item.

## Decisions already made
- Confidence bands: the code (0.9 / 0.7) is correct and was tuned on purpose. CLAUDE.md, README and the plan docs were updated to say 70% to match; tests pin the code's behaviour. Check the other docs (README, MODERNISATION_PLAN) for the same 80% claim.
- `/ready` is included.
- Nothing is pushed without explicit permission.

## Cost (public repo)
GitHub Actions on standard Linux runners is free and unlimited for public repos, so a build-only Docker job costs nothing. No registry push means no storage or bandwidth cost either. Guardrails: build only (`push: false`), standard `ubuntu-latest` runner (no larger paid runners), cache within the free 10 GB Actions cache limit, and the model-smoke job stays manual-only. A torch-CPU image of about 3 to 4 GB fits the runner's disk. If the build is slow, trigger the docker job only on changes to `Dockerfile`, `requirements*`, or `docker-compose.yml` using `paths:` filters.
