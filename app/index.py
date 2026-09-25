"""CLI: python -m app.index

Rebuilds both FAISS indexes from scratch by re-embedding every chunk already
recorded in DuckDB, in embedding_id order. This is the reproducibility guarantee
required by the project spec: FAISS holds no information that doesn't also live in
DuckDB, so a corrupted/deleted .index file is always recoverable from the DB + the
raw files under data/.
"""
from __future__ import annotations

from app.config.settings import settings
from app.database.db import Database
from app.embeddings.clip_embedder import get_embedder
from app.retrieval.faiss_index import FaissIndex


def rebuild() -> None:
    db = Database(settings.duckdb_path)
    embedder = get_embedder()

    text_index = FaissIndex(dim=embedder.embedding_dim)
    image_index = FaissIndex(dim=embedder.embedding_dim)

    text_ids = db.all_embedding_ids("text")
    text_rows = db.get_chunks_by_embedding_ids("text", text_ids)
    if text_rows:
        vectors = embedder.embed_text([r["text"] for r in text_rows])
        text_index.add(vectors)

    image_ids = db.all_embedding_ids("image")
    image_rows = db.get_chunks_by_embedding_ids("image", image_ids)
    for row in image_rows:
        vector = embedder.embed_image(row["image_path"])
        image_index.add(vector)

    text_index.save(settings.faiss_text_index_path)
    image_index.save(settings.faiss_image_index_path)
    print(f"Rebuilt: text_index={text_index.ntotal} vectors, image_index={image_index.ntotal} vectors")


if __name__ == "__main__":
    rebuild()
