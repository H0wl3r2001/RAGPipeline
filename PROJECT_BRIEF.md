# Project Brief: Local RAG Service

Hand this file to opencode inside an empty project folder as the starting instruction. It defines scope, architecture, folder layout, and a phased task list so the agent can implement end-to-end with minimal back-and-forth.

## 1. Goal

Build a small, self-contained **Retrieval-Augmented Generation (RAG) API** that answers questions over a local document set, using a **locally-hosted LLM** (no external API calls, no API keys required to run). Everything runs via `docker-compose`. Include a lightweight evaluation script that measures retrieval and answer quality on a fixed test set.

This is a portfolio project — prioritize a clean, working, well-documented core over feature breadth. Time budget: roughly 8-12 hours total.

## 2. Success Criteria (Definition of Done)

- `docker-compose up` starts the full stack (Qdrant, Ollama, app) with no manual steps beyond a documented one-time setup (e.g. pulling a model into the Ollama container).
- A `/query` endpoint accepts a question and returns an answer with cited source chunks.
- An `/ingest` endpoint or CLI script loads documents from a `data/` folder into the vector store.
- `eval/eval.py` runs a fixed set of 15-20 question/expected-answer pairs and prints retrieval hit-rate and a simple relevance score.
- `README.md` documents architecture, setup, and one specific technical decision made along the way (e.g. chunking strategy, embedding model choice).
- No hardcoded secrets; config via `.env`.

## 3. Tech Stack

**Hardware note:** target machine is Windows 11 + WSL2, AMD Ryzen 7 7800X3D (8-core), 32GB RAM, AMD Radeon RX 9060 XT (16GB VRAM). In principle Ollama's Vulkan backend (`OLLAMA_VULKAN=1`) accelerates RDNA4 cards well — but passing AMD GPU compute (Vulkan or ROCm) *into a Docker container on WSL2* is currently unreliable: missing device nodes (`/dev/dri`), driver mismatches, and `ERROR_INCOMPATIBLE_DRIVER` failures are common and well-documented as of mid-2026, even when the same GPU works fine natively on Windows. Multiple sources are explicit that AMD + WSL2 + Docker GPU passthrough isn't a solved combination the way NVIDIA/CUDA passthrough is.

Given the priority is a fully-containerized setup (consistent with your existing Docker-per-project convention) over inference speed, **run Ollama CPU-only, inside its own container**, alongside `qdrant` and `app`. The 7800X3D is a strong CPU and handles a 7-8B model at a workable pace for a 15-20 question eval set — slower than GPU, but reliable and fully isolated, which matters more here.

| Component | Choice | Why |
|---|---|---|
| LLM inference | Ollama, containerized, **CPU-only**, `qwen2.5:7b` or `llama3.1:8b` (drop to `qwen2.5:3b`/`llama3.2:3b` if generation feels too slow to demo comfortably) | Keeps the whole stack isolated in Docker as intended; AMD GPU passthrough into WSL2 containers is unreliable enough right now that it isn't worth trading away isolation for it |
| Vector store | Qdrant | Official lightweight Docker image, simple REST/gRPC client, good docs |
| Embeddings | `sentence-transformers` (e.g. `all-MiniLM-L6-v2`) run in-process, OR Ollama's embedding models | Keep it local, avoid a second heavy service if possible — prefer Ollama embeddings if quality is acceptable, fall back to sentence-transformers otherwise |
| API layer | FastAPI + Uvicorn | Fast to build, automatic OpenAPI docs, easy to demo |
| Orchestration | `docker-compose` | Matches existing project convention (Docker-per-project) |
| Doc parsing | `pypdf` for PDF text extraction; built-in file I/O for Markdown/txt | Keep scope to PDF + Markdown/txt only — `unstructured` pulls in a much heavier dependency tree (OCR, layout models, extra system packages) than a 3-file-type scope needs |

## 4. Architecture

```
┌─────────────┐      ┌──────────────┐      ┌─────────────┐
│   Client /   │─────▶│   FastAPI    │─────▶│   Qdrant    │
│  curl / UI   │◀─────│  (Docker)    │◀─────│  (Docker)   │
└─────────────┘      └──────┬───────┘      └─────────────┘
                             │
                             ▼
                      ┌──────────────┐
                      │    Ollama    │
                      │  (Docker,    │
                      │  CPU-only)   │
                      └──────────────┘
```

Flow for `/query`:
1. Embed the incoming question.
2. Retrieve top-k chunks from Qdrant.
3. Construct a prompt with retrieved context + question.
4. Call Ollama for generation.
5. Return answer + the source chunks used (with document name / page), so answers are traceable.

## 5. Folder Structure

```
RAGPipeline/
├── docker-compose.yml
├── .env.example
├── README.md
├── PROJECT_BRIEF.md          # this file
├── .gitignore                # see required entries below
├── data/                     # source documents go here (gitignored except .gitkeep)
│   └── .gitkeep
├── app/
│   ├── Dockerfile              # builds the FastAPI service — lives with the code it packages
│   ├── requirements.txt
│   ├── main.py                # FastAPI app, routes
│   ├── config.py               # env/config loading
│   ├── ingest.py                # document loading, chunking, embedding, upsert to Qdrant
│   ├── retrieve.py              # query embedding + Qdrant search
│   ├── generate.py              # prompt construction + Ollama call
│   └── models.py                # pydantic request/response schemas
├── eval/
│   ├── eval.py                  # runs test_set.json against the running API, scores it
│   ├── requirements.txt
│   └── test_set.json            # 15-20 Q/A pairs with expected source doc
└── tests/
    └── test_api.py              # a few basic smoke tests (health check, ingest, query)
```

Only `app/` gets a Dockerfile — `qdrant` and `ollama` run from official prebuilt images in `docker-compose.yml`, so they don't need one of their own. `eval/` and `tests/` do **not** get their own Dockerfile or compose service — both just make HTTP calls against the already-running stack. Do **not** use a local venv (`python -m venv`) for these: that requires `python3-venv`/`pip` to be installed on the WSL host outside any project, which conflicts with this environment's "don't install runtimes on the base system" rule. Instead, run them via an ephemeral container that installs nothing persistent:

```bash
docker run --rm \
  --network ragpipeline_default \
  -v "$(pwd)/eval:/eval" -w /eval \
  -v eval_pip_cache:/root/.cache/pip \
  python:3.12-slim \
  bash -c "pip install -q -r requirements.txt && python eval.py"
```

(Same pattern for `tests/test_api.py`, swapping the mounted directory and command.) The `eval_pip_cache` named volume just speeds up repeat runs — it's optional. Document this exact command in the README so it's copy-pasteable; nothing about it needs a Dockerfile or a compose entry, so it stays consistent with keeping `eval/`/`tests/` out of the main stack.

`.gitignore` must include, at minimum: `.env` (only `.env.example` is committed — never the real `.env`), `data/*` with a negation for `data/.gitkeep`, `eval/.venv/`, `__pycache__/`, and `*.pyc`. Create `.gitignore` as the very first file in Phase 1, before any other file that might contain a secret or generated artifact — do not defer it to Phase 5 polish.

Two named Docker volumes are also required in `docker-compose.yml` (not shown in the tree above since they're Docker-managed, not project files) — `ollama_data:/root/.ollama` and `qdrant_data:/qdrant/storage` — so pulled models and ingested vectors survive `docker-compose down` and container recreation instead of silently re-downloading/re-ingesting.

## 6. Implementation Phases

Work through these in order. Each phase should leave the system in a runnable state.

**Stop after each phase and wait for explicit confirmation before starting the next one.** Do not chain all five phases into one continuous run. At the end of each phase: summarize what was added/changed, state the exact `docker-compose` and `curl` commands to verify it, and wait. This is a deliberate checkpoint, not a formality — it's how build issues get caught per-phase instead of compounding.

### Phase 1 — Scaffolding
- Create folder structure above.
- `docker-compose.yml` with three services: `qdrant`, `ollama`, `app`. Use official images for the first two, **no GPU device reservation for `ollama`** (CPU-only — see hardware note in section 3). Directly above the `ollama` service definition, add a comment block pinning this as a deliberate decision, not an oversight, e.g.:
  ```yaml
  # CPU-only by design: AMD GPU passthrough (Vulkan/ROCm) into a WSL2 Docker
  # container is unreliable as of mid-2026 (missing /dev/dri, driver mismatches,
  # ERROR_INCOMPATIBLE_DRIVER). Do not add a `deploy.resources.reservations.devices`
  # GPU block here without re-validating passthrough works first — see
  # PROJECT_BRIEF.md section 3.
  ```
  This exists so a future edit (by you, or by an agent working on this repo later) doesn't silently "fix" what looks like a missing GPU reservation. Add named volumes `ollama_data:/root/.ollama` and `qdrant_data:/qdrant/storage` for persistence. `app` builds via `build: { context: ./app }`, using `app/Dockerfile`.
- `.env.example` with `OLLAMA_MODEL` (default `qwen2.5:7b`), `QDRANT_HOST`, `QDRANT_PORT`, `EMBED_MODEL`, `TOP_K` (default 4).
- Basic FastAPI app with a `/health` endpoint that also pings the `ollama` service to confirm connectivity. Confirm `docker-compose up` boots cleanly and the app can reach both `qdrant` and `ollama` before moving on. Also confirm Docker Desktop's WSL2 backend/integration is enabled — required for the compose networking to work correctly from WSL.

### Phase 2 — Ingestion
- `app/ingest.py`: load `.pdf`, `.md`, `.txt` files from `data/`.
- Chunk with a simple strategy first (fixed-size with overlap, e.g. 500 tokens / 50 overlap) — note in code comments this is the baseline, swappable later.
- Embed each chunk, upsert into Qdrant with metadata: `{source_file, chunk_index, text}`.
- Expose as both a CLI (`python -m app.ingest`) and a `/ingest` POST endpoint that triggers re-ingestion of `data/`.

### Phase 3 — Retrieval + Generation
- `app/retrieve.py`: embed query, search Qdrant, return top-k chunks with scores.
- `app/generate.py`: build a prompt template that instructs the model to answer only from provided context and say "I don't know" if the context doesn't cover it (this matters for eval quality). Call Ollama's generate endpoint.
- `/query` endpoint: accepts `{"question": str}`, returns `{"answer": str, "sources": [{"file": str, "chunk_index": int, "score": float}]}`.

### Phase 4 — Evaluation Harness
- `eval/test_set.json`: hand-write 15-20 question/expected-answer pairs based on whatever documents you actually put in `data/` (your own notes, FEUP course material, or public docs — pick something you know well enough to write correct expected answers).
- `eval/eval.py`:
  - For each question, call `/query`, check whether the expected source document appears in `sources` → **retrieval hit-rate**.
  - Default the answer-relevance check to simple keyword/substring matching — no extra dependency needed. If you want embedding-similarity instead, call whichever embedding backend `app/` already settled on over HTTP (e.g. Ollama's `/api/embed`) rather than installing `sentence-transformers` a second time in `eval/requirements.txt` — no need to duplicate a heavy dependency the app already carries in-process.
  - Print a summary table at the end.
- Run `eval/eval.py` (and `tests/test_api.py`) via the ephemeral `docker run` command in section 5 — do not create a local venv or install Python packages on the host.

### Phase 5 — Polish
- `README.md`: architecture summary (reuse section 4 diagram), setup steps, one paragraph on a real design decision made (e.g. "chose fixed-size chunking over semantic chunking for simplicity; noted as a possible improvement"), and sample `curl` commands.
- Basic error handling (empty `data/`, Qdrant unreachable, Ollama not yet pulled the model).
- Optional stretch (only if time remains): a single static HTML page with a text box calling `/query` — skip entirely if short on time, not required for the portfolio value of this project.
- Possible points of improvement, as well as a section on how this can escalate, even more so if the possibility of GPU acceleration exists (on my device it cannot happen due to hardware restrictions, but if those did not exist, how could this project be escalated)

## 7. Explicit Non-Goals (skip these — out of scope)

- No fine-tuning.
- No multi-user auth / production hardening.
- No support for arbitrary file types beyond PDF/Markdown/txt.
- No fancy chunking (semantic chunking, sliding-window optimization) — note it as a "future improvement" in the README instead of building it.
- No frontend framework — a single static HTML file at most, or skip UI entirely.
- No GPU passthrough for the Ollama container as part of the core build (see hardware note in section 3) — evidence strongly suggests AMD GPU passthrough into WSL2 Docker containers is currently unreliable, not just untested, so it's reasonable to skip it entirely rather than treat it as a stretch goal worth spending time on.
- No persistent Dockerfile or compose service for `eval/` or `tests/` — both are simple scripts that call the already-running stack over HTTP; an ephemeral `docker run` (section 5) gets the same host isolation as a Dockerfile would without the extra image or compose entry, and without installing Python on the WSL host the way a local venv would require.

## 8. Notes for the Agent

- Favor small, testable commits per phase over one large change.
- Keep dependencies minimal — every added package should be justified.
- After each phase, verify with `docker-compose up` and a manual `curl` call before proceeding, then stop and wait for confirmation per section 6.
- If a choice in section 3 turns out to be impractical (e.g. Ollama embeddings underperform), swap it and note the change plus reasoning in the README — that reasoning is itself useful portfolio content.
- Scope discipline: all file writes, builds, and container operations stay inside this project directory (`~/projects/rag-service`). Do not read, write, or mount anything outside it — no `~/.ssh`, `~/.aws`, `~/.azure`, other project folders, or Windows paths under `/mnt/c` other than this one if it happens to be cloned there.
- Do not run containers with `--privileged`, extra bind mounts to the host filesystem beyond the two named volumes and the project's own `data/`/`app/` directories, or host networking mode. The three services communicate over the default compose network.
- Never write a real secret or API key into any tracked file. `.env` (real values) is gitignored; `.env.example` (placeholder values only) is committed. If a task seems to require a credential beyond what `.env.example` already anticipates, stop and ask rather than inventing a workaround.
- Create `.gitignore` in Phase 1 before any other file, with the entries listed in section 5.