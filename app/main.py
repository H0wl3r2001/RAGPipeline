import httpx
from fastapi import FastAPI, HTTPException

from app.config import settings
from app.generate import generate_answer
from app.ingest import run_ingest
from app.models import IngestResponse, QueryRequest, QueryResponse
from app.retrieve import retrieve

app = FastAPI(title="Local RAG Service", version="0.1.0")


@app.get("/health")
async def health():
    status = {"app": "ok", "ollama": "unreachable", "qdrant": "unreachable"}
    async with httpx.AsyncClient(timeout=5.0) as client:
        try:
            r = await client.get(f"{settings.ollama_url}/api/tags")
            status["ollama"] = "ok" if r.status_code == 200 else f"error:{r.status_code}"
        except httpx.HTTPError as e:
            status["ollama"] = f"unreachable: {e}"
        try:
            r = await client.get(f"{settings.qdrant_url}/")
            status["qdrant"] = "ok" if r.status_code == 200 else f"error:{r.status_code}"
        except httpx.HTTPError as e:
            status["qdrant"] = f"unreachable: {e}"
    return status


@app.post("/ingest", response_model=IngestResponse)
def ingest():
    """Re-ingest all documents from data/ into the vector store."""
    try:
        result = run_ingest()
    except (FileNotFoundError, ValueError) as e:
        raise HTTPException(status_code=400, detail=str(e))
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Embedding service error: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    return IngestResponse(status="ok", **result)


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    """Answer a question using retrieved context from the vector store."""
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="question must not be empty")

    try:
        chunks = retrieve(req.question)
    except ValueError as e:
        raise HTTPException(status_code=409, detail=str(e))
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Embedding service error: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    try:
        answer = generate_answer(req.question, chunks)
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"Generation service error: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return QueryResponse(
        answer=answer,
        sources=[
            {"file": c["file"], "chunk_index": c["chunk_index"], "score": c["score"]}
            for c in chunks
        ],
    )
