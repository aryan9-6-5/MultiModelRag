"""CLI: python -m app.ingest --path data/raw

Ingests a file or directory (PDF/TXT/MD/JPG/PNG), embeds it, and persists
both the FAISS indexes and the DuckDB metadata to disk."""
from __future__ import annotations

import argparse

from app.api.state import get_database, get_image_index, get_text_index, save_indexes
from app.embeddings.clip_embedder import get_embedder
from app.ingestion.pipeline import ingest_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", required=True, help="File or directory to ingest")
    args = parser.parse_args()

    db = get_database()
    text_index = get_text_index()
    image_index = get_image_index()
    embedder = get_embedder()

    results = ingest_path(args.path, db, text_index, image_index, embedder)
    save_indexes()

    for r in results:
        print(f"[{r.file_type}] document={r.document_id} chunks={r.n_chunks}")
    print(f"\nTotal: {len(results)} document(s) ingested.")
    print(f"Index sizes: text={text_index.ntotal}, image={image_index.ntotal}")


if __name__ == "__main__":
    main()
