"""FAISS index wrapper. One index per modality (text, image), both in the same
CLIP vector space so a query embedding of either modality can search either index.

Row position in the FAISS index IS the `embedding_id` stored in DuckDB (see
app.database.db) — that's the join key that makes every vector traceable to metadata.
We use IndexFlatIP (exact cosine similarity, since embeddings are L2-normalized) rather
than an approximate index: at the scale this project targets (thousands of chunks),
exact search is fast enough and removes a whole class of recall-vs-speed tuning that
isn't worth the complexity here.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import faiss
import numpy as np


class FaissIndex:
    def __init__(self, dim: int, path: Optional[Path] = None):
        self.dim = dim
        self.path = path
        self.index = faiss.IndexFlatIP(dim)

    def add(self, vectors: np.ndarray) -> None:
        if vectors.ndim == 1:
            vectors = vectors[None, :]
        self.index.add(np.ascontiguousarray(vectors.astype("float32")))

    def search(self, query: np.ndarray, top_k: int) -> tuple[np.ndarray, np.ndarray]:
        if query.ndim == 1:
            query = query[None, :]
        top_k = min(top_k, self.index.ntotal) or 1
        if self.index.ntotal == 0:
            return np.array([[]]), np.array([[]])
        scores, ids = self.index.search(np.ascontiguousarray(query.astype("float32")), top_k)
        return scores, ids

    @property
    def ntotal(self) -> int:
        return self.index.ntotal

    def save(self, path: Optional[Path] = None) -> None:
        target = path or self.path
        if target is None:
            raise ValueError("No path given to save the index to")
        target.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(target))

    @classmethod
    def load(cls, path: Path, dim: int) -> "FaissIndex":
        obj = cls(dim=dim, path=path)
        if path.exists():
            obj.index = faiss.read_index(str(path))
        return obj
