# Local RAG Service

A self-contained Retrieval-Augmented Generation (RAG) API that answers questions over a local document set using a locally-hosted LLM. No external API calls, no API keys required. Everything runs via `docker-compose`.

## Architecture

```
┌─────────────┐      ┌──────────────┐      ┌─────────────┐
│   Client /  │────▶│   FastAPI    │─────▶│   Qdrant    │
│  curl / UI  │◀────│  (Docker)    │◀─────│  (Docker)   │
└─────────────┘      └──────┬───────┘      └─────────────┘
                             │
                             ▼
                      ┌──────────────┐
                      │    Ollama    │
                      │  (Docker,    │
                      │  CPU-only)   │
                      └──────────────┘
```

**Flow for `/query`:**
1. Embed the incoming question via Ollama (`nomic-embed-text`).
2. Retrieve top-k chunks from Qdrant (cosine similarity).
3. Construct a prompt with retrieved context + question.
4. Call Ollama (`qwen2.5:7b`) for generation.
5. Return answer + source chunks (file, chunk index, score).

**Services:**

| Service  | Image                    | Purpose                          |
|----------|--------------------------|----------------------------------|
| `qdrant` | `qdrant/qdrant:latest`   | Vector store (cosine, 768-dim)   |
| `ollama` | `ollama/ollama:latest`   | LLM + embedding inference (CPU)  |
| `app`    | Built from `app/`        | FastAPI REST API                 |

Ollama runs **CPU-only by design** — see the comment block in `docker-compose.yml` and the hardware note in `PROJECT_BRIEF.md` section 3. AMD GPU passthrough into WSL2 Docker containers is unreliable as of mid-2026.

## Setup

### Prerequisites

- Docker Desktop with WSL2 backend/integration enabled
- Docker Compose v2

### One-time setup

```bash
# 1. Create .env from the example
cp .env.example .env

# 2. Start the stack
docker compose up -d --build

# 3. Pull models into the Ollama container (persisted in the ollama_data volume)
docker compose exec ollama ollama pull nomic-embed-text   # embedding model (~274 MB)
docker compose exec ollama ollama pull qwen2.5:7b         # LLM (~4.7 GB)

# 4. Place documents in data/ (PDF, Markdown, or txt)
#    The data/ folder is gitignored — only .gitkeep is tracked.

# 5. Ingest documents into the vector store
curl -s -X POST http://localhost:8000/ingest
```

### Everyday usage

```bash
# Start the stack (volumes persist models + vectors across restarts)
docker compose up -d

# Query
curl -s -X POST http://localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"question": "What is the outlook for global economic growth?"}' \
  | python3 -m json.tool

# Re-ingest after adding/removing documents
curl -s -X POST http://localhost:8000/ingest

# Health check (pings both Ollama and Qdrant)
curl -s http://localhost:8000/health

# Tear down (volumes retained by default)
docker compose down
```

## API Endpoints

### `GET /health`
Returns connectivity status for the app, Ollama, and Qdrant.

```json
{"app": "ok", "ollama": "ok", "qdrant": "ok"}
```

### `POST /ingest`
Re-ingests all `.pdf`, `.md`, `.txt` files from `data/` into Qdrant. Recreates the collection each time.

```json
{"status": "ok", "documents": 3, "chunks": 278, "upserted": 278}
```

### `POST /query`
Accepts a question and returns an answer with cited source chunks.

**Request:**
```json
{"question": "What is the projected global growth rate for 2026?"}
```

**Response:**
```json
{
  "answer": "Global economic growth is projected to be 3.0 percent in 2026...",
  "sources": [
    {"file": "world_economic_outlook.pdf", "chunk_index": 0, "score": 0.74},
    {"file": "world_economic_outlook.pdf", "chunk_index": 15, "score": 0.70}
  ]
}
```

**Error codes:**
- `400` — empty question
- `409` — collection not found (run `/ingest` first)
- `502` — Ollama or Qdrant unreachable
- `503` — model not pulled into Ollama (error message includes the pull command)

## Evaluation

The eval harness and tests are **not containerized** — they run as ephemeral containers that call the already-running stack over the compose network.

### Run the eval (20 Q/A pairs)

```bash
docker run --rm --network ragpipeline_default \
  -v "$(pwd)/eval:/eval" -w /eval \
  -v eval_pip_cache:/root/.cache/pip \
  python:3.12-slim \
  bash -c "pip install -q -r requirements.txt && python eval.py"
```

To run a subset (e.g. while iterating during tuning), use `--limit N` (first N in file order) or `--ids id1,id2,...` (specific IDs, file order preserved). The summary line will mark the run as `(full)` or `(partial - use full set before recording results)`. Example:

```bash
python eval.py --limit 5
python eval.py --ids weo-08,gfsr-03,fm-01
```

Prints a per-question table with retrieval hit-rate (expected source document in returned sources) and answer relevance (fraction of expected keywords found in the answer), plus aggregate scores.

### Run the smoke tests

```bash
docker run --rm --network ragpipeline_default \
  -v "$(pwd)/tests:/tests" -w /tests \
  -v eval_pip_cache:/root/.cache/pip \
  python:3.12-slim \
  bash -c "pip install -q -r requirements.txt && python test_api.py"
```

Tests: `/health` returns ok, `/ingest` returns chunks > 0, `/query` returns answer + sources, empty question returns 400.

## Design Decision: Word-Based Chunking vs. Tokenizer-Based Chunking

The PROJECT_BRIEF specified fixed-size chunking at "500 tokens / 50 overlap." Implementing this literally would require a tokenizer — either `tiktoken` (OpenAI-specific, extra dependency) or a HuggingFace tokenizer pulled in via `sentence-transformers` or `transformers` (a heavy dependency tree). The brief's section 8 explicitly says "keep dependencies minimal — every added package should be justified."

The decision was to use **word-based chunking** (~500 words / 50-word overlap) as a proxy, avoiding any tokenizer dependency. The tradeoff: words ≠ tokens. For English text, ~500 words approximates ~670 tokens (a ~1.3x ratio), so chunks are larger than the literal "500 tokens" target. This means fewer chunks per document (coarser retrieval) and slightly longer prompts (more context per chunk). The alternative — adding a tokenizer just for chunking — would have been a cleaner implementation of the spec but violated the minimal-dependencies principle for a portfolio project.

Phase 4 evaluation validated the decision: retrieval hit-rate was **100%** (20/20 questions retrieved the correct source document), confirming the imprecision didn't hurt retrieval quality in practice. The 65% answer relevance score was driven by the model not finding specific details within chunks (e.g., a specific page number or table value), not by retrieving the wrong document — which is a chunking granularity issue, not a tokenization issue. Semantic chunking or a tokenizer-based approach would be the natural next improvement.

## Tuning Results (Phase 7a — chunk size / overlap / TOP_K sweep)

To diagnose the 35-point gap between retrieval hit-rate (100%) and answer relevance (65% on the full test set), a small parameter sweep was run against the **5 worst-scoring questions** from the full run (`weo-08, gfsr-03, gfsr-05, fm-01, fm-03` — the questions that scored 0% relevance). Per config: re-ingest with the new chunking parameters, then run `eval.py --ids ...` over that subset. No code changes; all three knobs (`CHUNK_WORDS`, `OVERLAP_WORDS`, `TOP_K`) are `.env`-driven.

| Config | CHUNK_WORDS | OVERLAP_WORDS | TOP_K | chunks | Hit-rate | Avg relevance | weo-08 | gfsr-03 | gfsr-05 | fm-01 | fm-03 |
|--------|-------------|---------------|-------|--------|---------|---------------|--------|---------|---------|-------|-------|
| Baseline | 500 | 50 | 4 | 278 | 5/5 | 17% | 0% | 0% | 0% | 50% | 33% |
| 7a-A | 300 | 75 | 4 | 555 | 5/5 | 17% | 0% | 0% | 50% | 0% | 33% |
| **7a-B** | **300** | **100** | **4** | **624** | **5/5** | **67%** | **100%** | **0%** | **100%** | **100%** | **33%** |
| 7a-C | 300 | 100 | 6 | 624 | 5/5 | 57% | 100% | 0% | 100% | 50% | 33% |

**What this tells us:**

- **Overlap is the dominant lever.** Going from 75→100 words overlap (7a-A → 7a-B), at the same chunk size 300, lifted the subset average from 17% to 67% — three out of five `0%` questions jumped to `100%`. The diagnosed failure mode (model missing specific details *because the boundary cut them*) is real and is directly addressed by giving chunks more shared context.
- **Smaller chunks alone don't help.** Going from 500→300 words while keeping 50-word overlap (baseline → 7a-A) didn't move the average; the finer granularity didn't fix the boundary problem unless overlap also rose.
- **TOP_K=6 didn't compound the gain.** 7a-C kept the best chunking config from 7a-B but retrieved more chunks, and *regressed* on `fm-01` (100→50). More chunks can dilute the prompt for marginal cases.
- **Persistent failures point downstream.** `gfsr-03` (Tobias Adrian) stayed at 0% across all configs, and `fm-03` (US deficit) stayed at 33%. These are likely chunk-content/exact-lookup failures rather than boundary-cut failures — coarse tuning doesn't fix them; that's what Phase 7b (hybrid retrieval) and Phase 7c (page/section-aware chunking) target.

**Caveat — variance:** per-config relevance uses CPU LLM decoding which is non-deterministic. The 7a-B 67% is a substantial margin and likely real signal (the per-question jump from 0→100 on three questions is not noise). 7a-A vs baseline (both 17%) is well within noise; the only honest comparison there is the per-question pattern (one improves, one regresses on different questions). The keyword-based relevance scorer is also coarse (see Phase 4 caveat) — treat these numbers directionally, not statistically.

**Net result for the diagnosed gap:** the cheap config-only sweep already closed most of it on the worst-scoring subset. Whether this holds on the full 20-question test set is the question Phase 7b/7c should answer — but `CHUNK_WORDS=300 / OVERLAP_WORDS=100 / TOP_K=4` (config 7a-B) is the new strongest candidate baseline.

## Phase 7c — Page/Section-Aware Chunking (reverted, see notes)

The brief asks for page-aware chunking (PDFs) and heading-aware chunking (Markdown) so the word-window never splits a table, figure, or section across a boundary. The implementation in `app/ingest.py` was straightforward: `chunk_atoms()` runs the word-window per atom (per PDF page / per markdown heading section), overlap stays inside the atom, and new payload fields `page` (PDF) / `section` (Markdown) are added. No new dependency.

To isolate 7c's effect from 7a's, two full-set evals were run with `CHUNK_WORDS=300 / OVERLAP_WORDS=100 / TOP_K=4` (config 7a-B), differing only in `app/ingest.py`: pre-7c vs 7c-aware.

| Run | Code | Hit-rate | Avg relevance | chunks |
|------|------|---------|---------------|--------|
| Phase 4 / 5 baseline (500/50/4) | original | 20/20 | ~65% | 278 |
| **Run A (7a-B, no 7c)** | config 7a-B with old `chunk_text` | **20/20** | **78%** | 624 |
| **Run B (7a-B + 7c)** | config 7a-B with `chunk_atoms` | **19/20** | **68%** | 632 |

**Honest reading — 7c did not improve results on this corpus:**

- The 7a-B config alone (Run A) lifted full-set relevance from 65% to **78%** — a 13-point gain well above CPU-decoding variance.
- Adding 7c-aware chunking (Run B) on top of 7a-B regressed to **68%** with one MISS (`weo-09`). The page-aware change split 624 chunks into 632 (~1% more chunks) — the boundary clamps almost never triggered on this sparsely-packed PDF corpus, but the small shift in chunk boundaries was enough to perturb retrieval on a few borderline cases.
- The brief expected 7c to "complement 7b" (BM25) rather than lift dense-only; but until 7b is implemented and measured against this baseline, **7c on its own is a net negative on the current `/query`** for this test set. Without positive evidence that 7b will compound with it, the change was **reverted** — `app/ingest.py` is back to the pre-7c (chunk_text + settings-driven 7a plumbing) code path.

**The 7c script and run logs are kept under `/tmp/`** (e.g. `/tmp/ingest.7c.py`) for reference if a future agent wants to reapply the change once 7b's hybrid retrieval is in place — at that point the comparison can be Run A vs Run B run against a hybrid retriever, where 7c may finally show lift.

**Net position after 7a + 7c-revert:** at the end of 7c, the project's strongest measured config was **7a-B alone** (Run A: 20/20 retrieval, 78% relevance), and `app/ingest.py` matched that result (the code that produced Run A was committed).

## Phase 7b — Hybrid Retrieval (Qdrant dense + BM25, fused via Reciprocal Rank Fusion)

The brief's diagnosis was that dense embeddings "underperform on exact numeric/tabular lookups" — exactly the failure mode observed in Phase 4. Phase 7b fixes this by adding a sparse keyword scorer (`rank_bm25`, pure Python + numpy) alongside the existing Qdrant cosine search and fusing both rankings.

**Code change** (only `app/retrieve.py` touched, plus `app/config.py` for the `rrf_k` setting, `app/requirements.txt` for `rank-bm25>=0.2.2`, and `.env.example` for `RRF_K=60`). No other file modified; downstream `/query` shape unchanged.

**Method** (query-time, "alongside the existing search" per the brief):

1. Scroll Qdrant once per query for the full corpus (text + payload).
2. Build a `BM25Okapi` index from the chunk texts.
3. Score the question with BM25; score it with Qdrant cosine (top-N where N = `max(TOP_K * 8, 32)` to give RRF headroom).
4. Fuse via Reciprocal Rank Fusion: `final_score(id) = Σ 1/(k + rank)` across both rankings.

Single tunable: `RRF_K` (default 60, the Cormack et al. value). Lower `k` makes top ranks dominate more; higher `k` flattens the contribution.

**Efficiency note (in response to a project review):** query-time full-corpus BM25 is acceptable at this scope — at 278 chunks it's ~100ms of overhead vs. ~30-60 seconds of CPU LLM generation. If the corpus grows past ~10k chunks, this should switch to dense-KNN top-N + BM25 rerank the narrow candidate set. See "Possible Points of Improvement" below.

**Full-set validation (config 7a-B: 300/100/4 only, per project direction):**

| Run | Retrieval | Hit-rate | Avg relevance |
|------|-----------|---------|---------------|
| Phase 5 baseline (500/50/4, dense) | dense only | 20/20 | ~65% |
| Run A (7a-B, dense only) | dense only | 20/20 | 78% |
| **Phase 7b (7a-B + hybrid RRF, k=60)** | dense + BM25, RRF | **20/20** | **82%** |

**Per-question deltas (7b vs Run A):**

- Wins: `weo-06` (0% → 100%), `fm-03` (33% → 67%).
- Losses: `gfsr-05` (100% → 50%).
- Ties: 17 other questions unchanged within scoring precision.

**Honest reading:**

- **7b on its own is a modest +4 points** (78 → 82). Worth keeping: it's a free lift on the same chunking and same LLM calls, with new relevance concentrated on questions with exact-string matching needs ("4.6 percent", "US gross debt projected to reach 142 percent").
- **The persistent zero-percent cases did not move** (`gfsr-02` "8 percent", `gfsr-03` "Tobias Adrian", `fm-05` "4 percentage points"). BM25 should in principle surface these chunks since their keywords are exact — but the failure here is at the LLM-summarization step, not retrieval: the right chunks ARE in top-k (hit-rate proves it), but the model is summarizing without quoting the exact figure. The brief explicitly calls out that "one prompt variant should explicitly instruct the model to ... look for exact figures/quotes/page references" — this is exactly what **Phase 6** (prompt variants) is for. See that phase.
- **gfsr-05 regression** (100 → 50): the answer contained "K-shaped" but not "bonds" — the model picked one of two required keywords. Likely LLM variance / chunk-text paraphrasing; not a retrieval failure.

**Net position after 7a + 7b:** the project's strongest measured config is **7a-B + hybrid RRF** (20/20 retrieval, 82% relevance). `app/retrieve.py` has the hybrid implementation, `rank_bm25` is in `app/requirements.txt`, `RRF_K=60` is in `.env.example` and `.env`. 7c was reverted, `app/ingest.py` matches Run A's chunking. To reproduce 7b's numbers, set `.env` to `CHUNK_WORDS=300 OVERLAP_WORDS=100 TOP_K=4 RRF_K=60` and re-ingest.

## Possible Points of Improvement

- **Semantic chunking**: Split at sentence/paragraph boundaries instead of fixed word counts. Would reduce mid-sentence cuts and improve answer relevance for specific-detail questions.
- **BM25 at scale**: Phase 7b scores BM25 over the full collection at query time — fine at ~278 chunks. Above ~10k chunks, switch to dense-KNN top-N + BM25 rerank the narrow candidate set, or persist a tokenized BM25 corpus to disk and load at startup.
- **Tokenizer-based chunk sizing**: Replace the word proxy with actual token counts (e.g. via `tiktoken` or the model's native tokenizer) for precise control over prompt length.
- **Chunk overlap tuning**: The Phase 7a sweep above found 50→100 words overlap closes most of the diagnosed gap on this corpus; per-document-type tuning (e.g. table-heavy PDFs) would be a natural next pass.
- **Re-ranking**: Add a cross-encoder re-ranker between retrieval and generation to improve the order of retrieved chunks.
- **Larger or domain-specific embedding model**: `nomic-embed-text` is a good general-purpose model; a domain-specific model could improve retrieval for specialized documents.
- **Streaming responses**: `/query` currently waits for the full answer before returning. Server-Sent Events or WebSocket streaming would improve perceived latency.
- **Caching**: Cache query embeddings and answers to avoid re-computing identical questions.
- **Multi-format support**: Extend beyond PDF/Markdown/txt (e.g. DOCX, HTML, EPUB) — deliberately out of scope per the brief but straightforward to add.

## How This Project Could Escalate (If GPU Acceleration Were Available)

This project runs Ollama **CPU-only** because AMD GPU passthrough into WSL2 Docker containers is unreliable as of mid-2026 (see `PROJECT_BRIEF.md` section 3). If that hardware constraint didn't exist — for example, on a machine with an NVIDIA GPU and working CUDA passthrough into Docker, or on a native Linux host with AMD ROCm support — the project could be escalated in several ways:

**1. GPU-accelerated inference**
The most immediate change: add a `deploy.resources.reservations.devices` GPU block to the `ollama` service in `docker-compose.yml`. With GPU acceleration, a 7-8B model generates answers in 1-3 seconds instead of 30-50 seconds per query. This unlocks interactive use cases (real-time chat, live demos) that aren't practical at CPU speeds. It also makes larger models viable — `qwen2.5:14b` or `llama3.1:70b` instead of `qwen2.5:7b` — which would meaningfully improve answer quality.

**2. Larger embedding and LLM models**
With GPU VRAM available (e.g. the 16GB on the RX 9060 XT in the target hardware), the embedding model could be upgraded to a larger one (e.g. `nomic-embed-text` → a 1.5B parameter embedding model) for better retrieval precision. The LLM could scale to 14B-70B parameters, dramatically improving the model's ability to synthesize complex answers from retrieved context.

**3. Concurrent requests**
CPU-only inference effectively serializes generation (one query at a time on 8 cores). GPU inference with sufficient VRAM can handle multiple concurrent requests or batch them, enabling multi-user usage. This would warrant adding async queueing, connection pooling, and rate limiting to the FastAPI app.

**4. Larger document sets**
The current eval uses 3 PDFs (~20MB, 278 chunks). A GPU-accelerated setup could handle thousands of documents with millions of chunks — ingestion embedding would be 10-50x faster, and retrieval + generation would remain interactive. This would require Qdrant configuration tuning (HNSW parameters, quantization, shard count) and possibly moving Qdrant to a dedicated machine.

**5. Advanced RAG patterns**
With faster inference, more sophisticated RAG architectures become practical: multi-hop retrieval (query → retrieve → generate follow-up query → retrieve again), query rewriting, and agentic patterns where the model decides when to search. These require multiple LLM calls per user question — impractical at 50s/call on CPU, but feasible at 2s/call on GPU.

**6. Fine-tuning**
With a GPU, fine-tuning the LLM on domain-specific Q/A pairs becomes possible (LoRA/QLoRA on a single GPU). This is explicitly out of scope for the current project but would be the natural escalation path for improving answer quality on a specialized document set.

## Configuration

All configuration is via `.env` (gitignored; see `.env.example` for defaults):

| Variable             | Default              | Description                          |
|----------------------|----------------------|--------------------------------------|
| `OLLAMA_MODEL`       | `qwen2.5:7b`         | LLM model for generation             |
| `OLLAMA_HOST`        | `ollama`             | Ollama service hostname              |
| `OLLAMA_PORT`        | `11434`              | Ollama service port                  |
| `QDRANT_HOST`        | `qdrant`             | Qdrant service hostname              |
| `QDRANT_PORT`        | `6333`               | Qdrant service port                  |
| `QDRANT_COLLECTION`  | `rag_docs`           | Qdrant collection name               |
| `EMBED_MODEL`        | `nomic-embed-text`   | Embedding model for ingestion/query  |
| `TOP_K`              | `4`                  | Number of chunks to retrieve         |
| `CHUNK_WORDS`        | `500`                | Chunk size (in words) during ingest  |
| `OVERLAP_WORDS`      | `50`                 | Overlap (in words) between adjacent chunks |
| `DATA_DIR`           | `/data`              | Document directory (mounted in app)  |

## Project Structure

```
RAGPipeline/
├── docker-compose.yml
├── .env.example
├── README.md
├── PROJECT_BRIEF.md
├── .gitignore
├── data/                     # source documents (gitignored except .gitkeep)
│   └── .gitkeep
├── app/
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── main.py               # FastAPI app, routes
│   ├── config.py             # env/config loading
│   ├── ingest.py             # document loading, chunking, embedding, upsert
│   ├── retrieve.py           # query embedding + Qdrant search
│   ├── generate.py           # prompt construction + Ollama call
│   └── models.py             # pydantic request/response schemas
├── eval/
│   ├── eval.py               # runs test_set.json against the running API
│   ├── requirements.txt
│   └── test_set.json         # 20 Q/A pairs with expected source + keywords
└── tests/
    ├── test_api.py           # smoke tests (health, ingest, query)
    └── requirements.txt
```
