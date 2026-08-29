"""Pydantic request/response schemas for the API."""

from pydantic import BaseModel


class IngestResponse(BaseModel):
    status: str
    documents: int
    chunks: int
    upserted: int
