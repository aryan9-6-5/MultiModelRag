"""Query -> embedding -> FAISS (both indexes) -> DuckDB metadata join -> reranked evidence.

Supports all four required query/result modality combinations because a query
embedding (text or image) is searched against BOTH the text and image FAISS indexes,
since they live in the same CLIP space:
  text query  -> text index  => text results
  text query  -> image index => image results
  image query -> image index => image results
  image query -> text index  => text results
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from app.config.settings import settings
from app.database.db import Database
from app.embeddings.clip_embedder import ClipEmbedder
from app.ranking.reranker import Candidate, RerankWeights, rerank
from app.retrieval.faiss_index import FaissIndex


@dataclass
class RetrievalResult:
    chunk_id: str
    modality: str
    source: str
    file_path: str
    page_number: Optional[int]
    text: Optional[str]
    image_path: Optional[str]
    clip_score: float
    final_score: float
    score_breakdown: dict


class Retriever:
    def __init__(self, db: Database, text_index: FaissIndex, image_index: FaissIndex, embedder: ClipEmbedder):
        self.db = db
        self.text_index = text_index
        self.image_index = image_index
        self.embedder = embedder

    def _search_index(self, index: FaissIndex, faiss_key: str, query_vector, top_k: int) -> list[Candidate]:
        if index.ntotal == 0:
            return []
        scores, ids = index.search(query_vector, top_k)
        scores, ids = scores[0], ids[0]
        valid = [(int(i), float(s)) for i, s in zip(ids, scores) if i != -1]
        if not valid:
            return []
        embedding_ids = [i for i, _ in valid]
        score_by_id = dict(valid)
        rows = self.db.get_chunks_by_embedding_ids(faiss_key, embedding_ids)
        candidates = []
        for row in rows:
            candidates.append(
                Candidate(
                    chunk_id=row["chunk_id"],
                    modality=row["modality"],
                    clip_score=score_by_id[int(row["embedding_id"])],
                    text=row["text"],
                    image_path=row["image_path"],
                    metadata={"source": row["source"], "file_path": row["file_path"], "page_number": row["page_number"]},
                )
            )
        return candidates

    def search(
        self,
        query_text: Optional[str] = None,
        query_image_path: Optional[str] = None,
        top_k: Optional[int] = None,
        rerank_top_k: Optional[int] = None,
        modalities: tuple[str, ...] = ("text", "image"),
        weights: RerankWeights = RerankWeights(),
    ) -> list[RetrievalResult]:
        if not query_text and not query_image_path:
            raise ValueError("Provide query_text and/or query_image_path")

        top_k = top_k or settings.default_top_k
        rerank_top_k = rerank_top_k or settings.rerank_top_k

        if query_image_path:
            query_vector = self.embedder.embed_image(query_image_path)
            query_modality = "image"
        else:
            query_vector = self.embedder.embed_text(query_text)
            query_modality = "text"

        candidates: list[Candidate] = []
        if "text" in modalities:
            candidates += self._search_index(self.text_index, "text", query_vector, top_k)
        if "image" in modalities:
            candidates += self._search_index(self.image_index, "image", query_vector, top_k)

        ranked = rerank(
            query_text=query_text, query_modality=query_modality,
            candidates=candidates, weights=weights, top_k=rerank_top_k,
        )

        return [
            RetrievalResult(
                chunk_id=c.chunk_id, modality=c.modality,
                source=c.metadata.get("source", ""), file_path=c.metadata.get("file_path", ""),
                page_number=c.metadata.get("page_number"), text=c.text, image_path=c.image_path,
                clip_score=c.clip_score, final_score=c.extra["final_score"],
                score_breakdown=c.extra["score_breakdown"],
            )
            for c in ranked
        ]

    def search_with_logging(self, **kwargs) -> tuple[list[RetrievalResult], float]:
        start = time.perf_counter()
        results = self.search(**kwargs)
        latency_ms = (time.perf_counter() - start) * 1000
        self.db.log_query(
            query_text=kwargs.get("query_text"),
            query_image=kwargs.get("query_image_path"),
            modality="image" if kwargs.get("query_image_path") else "text",
            top_k=kwargs.get("top_k") or settings.default_top_k,
            latency_ms=latency_ms,
            n_results=len(results),
        )
        return results, latency_ms
