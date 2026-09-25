"""FastAPI backend. Usable standalone (independent of the Streamlit frontend) —
run with `python -m app.api` or `uvicorn app.api.main:app`."""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from app.api.schemas import (
    EvidenceItem, HealthResponse, IngestResponse, QueryRequest, QueryResponse,
    SearchRequest, SearchResponse, StatsResponse,
)
from app.api.state import get_database, get_image_index, get_retriever, get_text_index, save_indexes
from app.embeddings.clip_embedder import get_embedder
from app.generation.generator import generate_answer
from app.generation.llm_client import OllamaClient
from app.ingestion.pipeline import ingest_path

app = FastAPI(title="MultimodalRAG API", version="0.1.0")
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
)


def _to_evidence_item(r) -> EvidenceItem:
    return EvidenceItem(
        chunk_id=r.chunk_id, modality=r.modality, source=r.source, file_path=r.file_path,
        page_number=r.page_number, text_preview=(r.text[:280] if r.text else None),
        image_path=r.image_path, clip_score=r.clip_score, final_score=r.final_score,
        score_breakdown=r.score_breakdown,
    )


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    ollama_ok = True
    try:
        OllamaClient()._available_models()
    except Exception:
        ollama_ok = False
    return HealthResponse(status="ok", ollama_reachable=ollama_ok, embedding_device=get_embedder().device)


@app.get("/stats", response_model=StatsResponse)
def stats() -> StatsResponse:
    db_stats = get_database().stats()
    return StatsResponse(
        **db_stats,
        faiss_text_vectors=get_text_index().ntotal,
        faiss_image_vectors=get_image_index().ntotal,
    )


@app.post("/ingest", response_model=list[IngestResponse])
async def ingest(file: UploadFile = File(...)) -> list[IngestResponse]:
    # Use a tmpdir (not NamedTemporaryFile) so the file keeps its original name —
    # ingest_path/_store_raw_copy derive `source`/`file_path` from the path's name,
    # and losing the original filename here would make every ingested document
    # traceable only to a random temp name instead of what the user uploaded.
    tmp_dir = Path(tempfile.mkdtemp())
    tmp_path = tmp_dir / Path(file.filename).name  # basename only — filename is user-controlled
    with open(tmp_path, "wb") as tmp:
        shutil.copyfileobj(file.file, tmp)
    try:
        results = ingest_path(
            tmp_path, get_database(), get_text_index(), get_image_index(), get_embedder(),
        )
        save_indexes()
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)

    return [
        IngestResponse(document_id=r.document_id, file_type=r.file_type, n_chunks=r.n_chunks, source=file.filename)
        for r in results
    ]


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest) -> SearchResponse:
    if not req.query_text:
        raise HTTPException(status_code=400, detail="query_text is required")
    retriever = get_retriever()
    results, latency_ms = retriever.search_with_logging(
        query_text=req.query_text, top_k=req.top_k, rerank_top_k=req.rerank_top_k,
        modalities=tuple(req.modalities),
    )
    return SearchResponse(results=[_to_evidence_item(r) for r in results], latency_ms=latency_ms)


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> QueryResponse:
    retriever = get_retriever()
    evidence, retrieval_latency_ms = retriever.search_with_logging(
        query_text=req.query, top_k=req.top_k, rerank_top_k=req.rerank_top_k,
    )
    response = generate_answer(req.query, evidence, db=get_database())
    return QueryResponse(
        answer=response.answer, success=response.success, error=response.error, model=response.model,
        generation_latency_ms=response.latency_ms, retrieval_latency_ms=retrieval_latency_ms,
        evidence=[_to_evidence_item(r) for r in evidence],
    )
