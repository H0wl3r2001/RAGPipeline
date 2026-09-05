"""Pydantic request/response schemas for the API."""

from pydantic import BaseModel


class IngestResponse(BaseModel):
    status: str
    documents: int
    chunks: int
    upserted: int


class QueryRequest(BaseModel):
    question: str
    prompt_variant: str | None = None


class Source(BaseModel):
    file: str
    chunk_index: int
    score: float


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]
