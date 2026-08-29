import httpx
from fastapi import FastAPI, HTTPException

from app.config import settings
from app.ingest import run_ingest
from app.models import IngestResponse

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
