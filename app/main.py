import httpx
from fastapi import FastAPI

from app.config import settings

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
