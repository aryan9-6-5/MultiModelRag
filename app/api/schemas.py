"""Pydantic request/response models for the FastAPI backend."""
from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query_text: Optional[str] = Field(default=None, description="Text query")
    top_k: int = Field(default=10, ge=1, le=100)
    rerank_top_k: int = Field(default=5, ge=1, le=50)
    modalities: list[str] = Field(default=["text", "image"])


class EvidenceItem(BaseModel):
    chunk_id: str
    modality: str
    source: str
    file_path: str
    page_number: Optional[int] = None
    text_preview: Optional[str] = None
    image_path: Optional[str] = None
    clip_score: float
    final_score: float
    score_breakdown: dict


class SearchResponse(BaseModel):
    results: list[EvidenceItem]
    latency_ms: float


class QueryRequest(BaseModel):
    query: str
    top_k: int = Field(default=10, ge=1, le=100)
    rerank_top_k: int = Field(default=5, ge=1, le=50)


class QueryResponse(BaseModel):
    answer: str
    success: bool
    error: Optional[str] = None
    model: str
    generation_latency_ms: float
    retrieval_latency_ms: float
    evidence: list[EvidenceItem]


class IngestResponse(BaseModel):
    document_id: str
    file_type: str
    n_chunks: int
    source: str


class StatsResponse(BaseModel):
    documents: int
    text_chunks: int
    image_chunks: int
    total_chunks: int
    faiss_text_vectors: int
    faiss_image_vectors: int


class HealthResponse(BaseModel):
    status: str
    ollama_reachable: bool
    embedding_device: str
