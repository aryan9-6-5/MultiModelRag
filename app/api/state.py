"""Process-wide singletons for the API: one DB connection, one pair of FAISS
indexes (loaded from disk if present), one CLIP embedder. Loaded lazily so
`import app.api.main` (e.g. for OpenAPI generation) doesn't require a GPU."""
from __future__ import annotations

from functools import lru_cache

from app.config.settings import settings
from app.database.db import Database
from app.embeddings.clip_embedder import ClipEmbedder, get_embedder
from app.retrieval.faiss_index import FaissIndex
from app.retrieval.retriever import Retriever


@lru_cache(maxsize=1)
def get_database() -> Database:
    return Database(settings.duckdb_path)


@lru_cache(maxsize=1)
def get_text_index() -> FaissIndex:
    embedder = get_embedder()
    return FaissIndex.load(settings.faiss_text_index_path, dim=embedder.embedding_dim)


@lru_cache(maxsize=1)
def get_image_index() -> FaissIndex:
    embedder = get_embedder()
    return FaissIndex.load(settings.faiss_image_index_path, dim=embedder.embedding_dim)


def get_retriever() -> Retriever:
    return Retriever(
        db=get_database(), text_index=get_text_index(), image_index=get_image_index(),
        embedder=get_embedder(),
    )


def save_indexes() -> None:
    get_text_index().save(settings.faiss_text_index_path)
    get_image_index().save(settings.faiss_image_index_path)
