from app.ingestion.pipeline import ingest_text_document
from app.retrieval.faiss_index import FaissIndex
from app.retrieval.retriever import Retriever


def test_retriever_finds_ingested_text_chunk(tmp_path, tmp_db, fake_embedder, monkeypatch):
    monkeypatch.setattr("app.ingestion.pipeline.settings.data_dir", tmp_path / "data")

    src = tmp_path / "doc.txt"
    src.write_text("the quick brown fox jumps over the lazy dog", encoding="utf-8")

    text_index = FaissIndex(dim=fake_embedder.embedding_dim)
    image_index = FaissIndex(dim=fake_embedder.embedding_dim)
    ingest_text_document(src, tmp_db, text_index, fake_embedder)

    retriever = Retriever(db=tmp_db, text_index=text_index, image_index=image_index, embedder=fake_embedder)
    results = retriever.search(query_text="the quick brown fox jumps over the lazy dog", top_k=5, rerank_top_k=5)

    assert len(results) >= 1
    assert results[0].modality == "text"
    assert "fox" in results[0].text


def test_retriever_requires_a_query(tmp_path, tmp_db, fake_embedder):
    text_index = FaissIndex(dim=fake_embedder.embedding_dim)
    image_index = FaissIndex(dim=fake_embedder.embedding_dim)
    retriever = Retriever(db=tmp_db, text_index=text_index, image_index=image_index, embedder=fake_embedder)
    try:
        retriever.search()
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_retriever_logs_query(tmp_path, tmp_db, fake_embedder, monkeypatch):
    monkeypatch.setattr("app.ingestion.pipeline.settings.data_dir", tmp_path / "data")
    src = tmp_path / "doc.txt"
    src.write_text("alpha beta gamma delta epsilon zeta", encoding="utf-8")

    text_index = FaissIndex(dim=fake_embedder.embedding_dim)
    image_index = FaissIndex(dim=fake_embedder.embedding_dim)
    ingest_text_document(src, tmp_db, text_index, fake_embedder)

    retriever = Retriever(db=tmp_db, text_index=text_index, image_index=image_index, embedder=fake_embedder)
    results, latency_ms = retriever.search_with_logging(query_text="alpha beta", top_k=5)
    assert latency_ms >= 0
    logged = tmp_db.conn.execute("SELECT COUNT(*) FROM query_log").fetchone()[0]
    assert logged == 1
