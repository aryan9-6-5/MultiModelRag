"""DuckDB-backed metadata store. Every FAISS vector is traceable to one row here."""
from __future__ import annotations

import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Optional

import duckdb

from app.config.settings import settings

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    document_id   VARCHAR PRIMARY KEY,
    source        VARCHAR NOT NULL,      -- original filename / URL
    file_path     VARCHAR NOT NULL,
    file_type     VARCHAR NOT NULL,      -- pdf|txt|md|image
    ingested_at   TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS chunks (
    chunk_id       VARCHAR PRIMARY KEY,
    document_id    VARCHAR NOT NULL,
    modality       VARCHAR NOT NULL,     -- 'text' | 'image'
    source         VARCHAR NOT NULL,
    file_path      VARCHAR NOT NULL,
    page_number    INTEGER,
    text           VARCHAR,              -- chunk text, or image caption/OCR text
    image_path     VARCHAR,
    embedding_id   INTEGER NOT NULL,     -- row index into the FAISS index for this modality
    faiss_index    VARCHAR NOT NULL,     -- 'text' | 'image' -- which index embedding_id refers to
    ingested_at    TIMESTAMP NOT NULL,
    metadata       VARCHAR               -- JSON blob for extensible metadata
);

CREATE SEQUENCE IF NOT EXISTS text_embedding_seq START 0 MINVALUE 0;
CREATE SEQUENCE IF NOT EXISTS image_embedding_seq START 0 MINVALUE 0;

CREATE TABLE IF NOT EXISTS query_log (
    query_id       VARCHAR PRIMARY KEY,
    query_text     VARCHAR,
    query_image    VARCHAR,
    modality       VARCHAR,
    top_k          INTEGER,
    latency_ms     DOUBLE,
    n_results      INTEGER,
    created_at     TIMESTAMP NOT NULL
);

CREATE TABLE IF NOT EXISTS generation_log (
    generation_id  VARCHAR PRIMARY KEY,
    query_id       VARCHAR,
    model          VARCHAR,
    latency_ms     DOUBLE,
    prompt_tokens  INTEGER,
    completion_tokens INTEGER,
    success        BOOLEAN,
    error          VARCHAR,
    created_at     TIMESTAMP NOT NULL
);
"""


class Database:
    def __init__(self, path: Optional[Path] = None):
        self.path = str(path or settings.duckdb_path)
        self._conn = duckdb.connect(self.path)
        self._conn.execute(SCHEMA)

    @property
    def conn(self) -> duckdb.DuckDBPyConnection:
        return self._conn

    def close(self) -> None:
        self._conn.close()

    # --- documents ---
    def insert_document(
        self, source: str, file_path: str, file_type: str, document_id: Optional[str] = None
    ) -> str:
        document_id = document_id or str(uuid.uuid4())
        self._conn.execute(
            "INSERT INTO documents VALUES (?, ?, ?, ?, ?)",
            [document_id, source, file_path, file_type, datetime.now(timezone.utc)],
        )
        return document_id

    # --- chunks ---
    def next_embedding_id(self, faiss_index: str) -> int:
        seq = "text_embedding_seq" if faiss_index == "text" else "image_embedding_seq"
        return self._conn.execute(f"SELECT nextval('{seq}')").fetchone()[0]

    def insert_chunk(
        self,
        *,
        document_id: str,
        modality: str,
        source: str,
        file_path: str,
        faiss_index: str,
        embedding_id: int,
        page_number: Optional[int] = None,
        text: Optional[str] = None,
        image_path: Optional[str] = None,
        metadata: Optional[str] = None,
        chunk_id: Optional[str] = None,
    ) -> str:
        chunk_id = chunk_id or str(uuid.uuid4())
        self._conn.execute(
            """INSERT INTO chunks
               (chunk_id, document_id, modality, source, file_path, page_number,
                text, image_path, embedding_id, faiss_index, ingested_at, metadata)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            [
                chunk_id, document_id, modality, source, file_path, page_number,
                text, image_path, embedding_id, faiss_index,
                datetime.now(timezone.utc), metadata,
            ],
        )
        return chunk_id

    def get_chunks_by_embedding_ids(self, faiss_index: str, embedding_ids: list[int]) -> list[dict]:
        if not embedding_ids:
            return []
        placeholders = ",".join("?" * len(embedding_ids))
        rows = self._conn.execute(
            f"""SELECT * FROM chunks
                WHERE faiss_index = ? AND embedding_id IN ({placeholders})""",
            [faiss_index, *embedding_ids],
        ).fetchdf()
        # preserve caller's ordering (FAISS returns ranked order)
        rows_by_id = {int(r["embedding_id"]): r for _, r in rows.iterrows()}
        return [dict(rows_by_id[i]) for i in embedding_ids if i in rows_by_id]

    def get_chunk(self, chunk_id: str) -> Optional[dict]:
        row = self._conn.execute("SELECT * FROM chunks WHERE chunk_id = ?", [chunk_id]).fetchdf()
        return None if row.empty else dict(row.iloc[0])

    def all_embedding_ids(self, faiss_index: str) -> list[int]:
        rows = self._conn.execute(
            "SELECT embedding_id FROM chunks WHERE faiss_index = ? ORDER BY embedding_id", [faiss_index]
        ).fetchall()
        return [r[0] for r in rows]

    # --- stats ---
    def stats(self) -> dict[str, Any]:
        n_docs = self._conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        n_text = self._conn.execute("SELECT COUNT(*) FROM chunks WHERE modality='text'").fetchone()[0]
        n_img = self._conn.execute("SELECT COUNT(*) FROM chunks WHERE modality='image'").fetchone()[0]
        return {
            "documents": n_docs,
            "text_chunks": n_text,
            "image_chunks": n_img,
            "total_chunks": n_text + n_img,
        }

    # --- logging ---
    def log_query(
        self, *, query_text: Optional[str], query_image: Optional[str], modality: str,
        top_k: int, latency_ms: float, n_results: int,
    ) -> str:
        query_id = str(uuid.uuid4())
        self._conn.execute(
            "INSERT INTO query_log VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [query_id, query_text, query_image, modality, top_k, latency_ms, n_results, datetime.now(timezone.utc)],
        )
        return query_id

    def log_generation(
        self, *, query_id: Optional[str], model: str, latency_ms: float,
        prompt_tokens: Optional[int], completion_tokens: Optional[int],
        success: bool, error: Optional[str] = None,
    ) -> str:
        generation_id = str(uuid.uuid4())
        self._conn.execute(
            "INSERT INTO generation_log VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [generation_id, query_id, model, latency_ms, prompt_tokens, completion_tokens,
             success, error, datetime.now(timezone.utc)],
        )
        return generation_id


@contextmanager
def get_db(path: Optional[Path] = None) -> Iterator[Database]:
    db = Database(path)
    try:
        yield db
    finally:
        db.close()
