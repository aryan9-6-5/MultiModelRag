from app.ingestion.pipeline import ingest_text_document
from app.retrieval.faiss_index import FaissIndex


def test_ingest_text_document_creates_traceable_chunks(tmp_path, tmp_db, fake_embedder, monkeypatch):
    monkeypatch.setattr("app.ingestion.pipeline.settings.data_dir", tmp_path / "data")

    src = tmp_path / "doc.txt"
    src.write_text(" ".join(f"word{i}" for i in range(50)), encoding="utf-8")

    text_index = FaissIndex(dim=fake_embedder.embedding_dim)
    result = ingest_text_document(src, tmp_db, text_index, fake_embedder)

    assert result.n_chunks > 0
    assert text_index.ntotal == result.n_chunks

    # Every FAISS row must be traceable to a DuckDB chunk (no orphaned vectors).
    embedding_ids = tmp_db.all_embedding_ids("text")
    assert len(embedding_ids) == text_index.ntotal


def test_ingest_empty_text_document_creates_no_chunks(tmp_path, tmp_db, fake_embedder, monkeypatch):
    monkeypatch.setattr("app.ingestion.pipeline.settings.data_dir", tmp_path / "data")
    src = tmp_path / "empty.txt"
    src.write_text("", encoding="utf-8")

    text_index = FaissIndex(dim=fake_embedder.embedding_dim)
    result = ingest_text_document(src, tmp_db, text_index, fake_embedder)
    assert result.n_chunks == 0
    assert text_index.ntotal == 0
