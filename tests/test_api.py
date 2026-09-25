import pytest
from fastapi.testclient import TestClient

from app.ingestion.pipeline import ingest_text_document
from app.retrieval.faiss_index import FaissIndex


@pytest.fixture
def api_client(tmp_path, tmp_db, fake_embedder, monkeypatch):
    monkeypatch.setattr("app.ingestion.pipeline.settings.data_dir", tmp_path / "data")

    text_index = FaissIndex(dim=fake_embedder.embedding_dim)
    image_index = FaissIndex(dim=fake_embedder.embedding_dim)

    src = tmp_path / "doc.txt"
    src.write_text("the quick brown fox jumps over the lazy dog", encoding="utf-8")
    ingest_text_document(src, tmp_db, text_index, fake_embedder)

    import app.api.main as main_module

    monkeypatch.setattr(main_module, "get_database", lambda: tmp_db)
    monkeypatch.setattr(main_module, "get_text_index", lambda: text_index)
    monkeypatch.setattr(main_module, "get_image_index", lambda: image_index)
    monkeypatch.setattr(main_module, "get_embedder", lambda: fake_embedder)
    monkeypatch.setattr(main_module, "save_indexes", lambda: None)

    from app.retrieval.retriever import Retriever

    monkeypatch.setattr(
        main_module, "get_retriever",
        lambda: Retriever(db=tmp_db, text_index=text_index, image_index=image_index, embedder=fake_embedder),
    )

    return TestClient(main_module.app)


def test_health_endpoint(api_client):
    resp = api_client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_stats_endpoint(api_client):
    resp = api_client.get("/stats")
    assert resp.status_code == 200
    data = resp.json()
    assert data["text_chunks"] == 1
    assert data["faiss_text_vectors"] == 1


def test_search_endpoint_returns_results(api_client):
    resp = api_client.post("/search", json={"query_text": "quick brown fox", "top_k": 5, "rerank_top_k": 5})
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["results"]) >= 1
    assert data["results"][0]["modality"] == "text"


def test_search_endpoint_requires_query_text(api_client):
    resp = api_client.post("/search", json={"top_k": 5})
    assert resp.status_code == 400


def test_query_endpoint_reports_generation_failure_gracefully(api_client, monkeypatch):
    import app.api.main as main_module
    from app.generation.generator import GenerationResponse

    def fake_generate_answer(query, evidence, db=None):
        return GenerationResponse(
            answer="", evidence=evidence, model="none", latency_ms=1.0,
            success=False, error="Ollama not reachable in test environment",
        )

    monkeypatch.setattr(main_module, "generate_answer", fake_generate_answer)
    resp = api_client.post("/query", json={"query": "quick brown fox", "top_k": 5})
    assert resp.status_code == 200
    data = resp.json()
    assert data["success"] is False
    assert data["error"]
