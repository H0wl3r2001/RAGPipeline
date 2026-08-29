# Local RAG Service

A self-contained Retrieval-Augmented Generation (RAG) API that answers questions over a local document set using a locally-hosted LLM. No external API calls, no API keys required. Everything runs via `docker-compose`.

## Architecture

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

## Possible Points of Improvement

- **Semantic chunking**: Split at sentence/paragraph boundaries instead of fixed word counts. Would reduce mid-sentence cuts and improve answer relevance for specific-detail questions.
- **Tokenizer-based chunk sizing**: Replace the word proxy with actual token counts (e.g. via `tiktoken` or the model's native tokenizer) for precise control over prompt length.
- **Chunk overlap tuning**: The current 50-word overlap (~10%) is a reasonable default but could be tuned per document type.
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
