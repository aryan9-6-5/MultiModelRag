"""End-to-end ingestion: file -> parsed pages / image -> chunks -> embeddings ->
FAISS + DuckDB. Both indexes and the DB are passed in so callers (CLI, API, tests)
control persistence explicitly instead of this module owning global state."""
from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from app.config.settings import settings
from app.database.db import Database
from app.embeddings.clip_embedder import ClipEmbedder
from app.ingestion.captioning import generate_caption, run_ocr
from app.ingestion.chunking import chunk_pages
from app.ingestion.parsers import detect_file_type, extract_pages
from app.retrieval.faiss_index import FaissIndex


@dataclass
class IngestResult:
    document_id: str
    file_type: str
    n_chunks: int


def _store_raw_copy(src: Path, dest_dir: Path) -> Path:
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.name
    if src.resolve() != dest.resolve():
        shutil.copy2(src, dest)
    return dest


def ingest_text_document(
    path: str | Path, db: Database, text_index: FaissIndex, embedder: ClipEmbedder,
) -> IngestResult:
    path = Path(path)
    file_type = detect_file_type(path)
    stored_path = _store_raw_copy(path, settings.data_dir / "raw")
    document_id = db.insert_document(source=path.name, file_path=str(stored_path), file_type=file_type)

    pages = extract_pages(stored_path)
    chunks = chunk_pages(pages)
    if not chunks:
        return IngestResult(document_id=document_id, file_type=file_type, n_chunks=0)

    embeddings = embedder.embed_text([c.text for c in chunks])
    if embeddings.ndim == 1:
        embeddings = embeddings[None, :]

    for chunk, vector in zip(chunks, embeddings):
        embedding_id = db.next_embedding_id("text")
        text_index.add(vector)
        db.insert_chunk(
            document_id=document_id, modality="text", source=path.name,
            file_path=str(stored_path), faiss_index="text", embedding_id=embedding_id,
            page_number=chunk.page_number, text=chunk.text,
        )
    return IngestResult(document_id=document_id, file_type=file_type, n_chunks=len(chunks))


def ingest_image(
    path: str | Path, db: Database, image_index: FaissIndex, embedder: ClipEmbedder,
) -> IngestResult:
    path = Path(path)
    stored_path = _store_raw_copy(path, settings.data_dir / "images")
    document_id = db.insert_document(source=path.name, file_path=str(stored_path), file_type="image")

    vector = embedder.embed_image(str(stored_path))
    embedding_id = db.next_embedding_id("image")
    image_index.add(vector)

    caption = generate_caption(stored_path)
    ocr_text = run_ocr(stored_path)
    combined_text = " ".join(t for t in (caption, ocr_text) if t) or None
    metadata = json.dumps({"caption": caption, "ocr_text": ocr_text})

    db.insert_chunk(
        document_id=document_id, modality="image", source=path.name,
        file_path=str(stored_path), faiss_index="image", embedding_id=embedding_id,
        text=combined_text, image_path=str(stored_path), metadata=metadata,
    )
    return IngestResult(document_id=document_id, file_type="image", n_chunks=1)


def ingest_path(
    path: str | Path, db: Database, text_index: FaissIndex, image_index: FaissIndex,
    embedder: ClipEmbedder,
) -> list[IngestResult]:
    """Ingest a single file or every supported file under a directory (recursively)."""
    path = Path(path)
    if path.is_dir():
        results = []
        for f in sorted(path.rglob("*")):
            if f.is_file():
                try:
                    detect_file_type(f)
                except ValueError:
                    continue
                results.append(ingest_path(f, db, text_index, image_index, embedder)[0])
        return results

    file_type = detect_file_type(path)
    if file_type == "image":
        return [ingest_image(path, db, image_index, embedder)]
    return [ingest_text_document(path, db, text_index, embedder)]
