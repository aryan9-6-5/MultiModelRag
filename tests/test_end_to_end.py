"""Full pipeline test with the fake (hash-based) embedder for speed: ingest text +
image, retrieve across modalities, generate a grounded answer with a stub LLM
client, and confirm nothing is orphaned in DuckDB."""
from PIL import Image

from app.generation.generator import generate_answer
from app.ingestion.pipeline import ingest_image, ingest_text_document
from app.retrieval.faiss_index import FaissIndex
from app.retrieval.retriever import Retriever


class _StubClient:
    model = "stub-model"

    def generate(self, prompt, temperature=0.2):
        return {"text": "Based on [E1], the document discusses foxes.", "model": self.model,
                "prompt_tokens": 10, "completion_tokens": 8}


def test_full_pipeline_text_and_image(tmp_path, tmp_db, fake_embedder, monkeypatch):
    monkeypatch.setattr("app.ingestion.pipeline.settings.data_dir", tmp_path / "data")

    text_index = FaissIndex(dim=fake_embedder.embedding_dim)
    image_index = FaissIndex(dim=fake_embedder.embedding_dim)

    text_src = tmp_path / "doc.txt"
    text_src.write_text("the quick brown fox jumps over the lazy dog repeatedly", encoding="utf-8")
    text_result = ingest_text_document(text_src, tmp_db, text_index, fake_embedder)

    img_src = tmp_path / "photo.png"
    Image.new("RGB", (32, 32), color=(10, 200, 10)).save(img_src)
    monkeypatch.setattr("app.ingestion.captioning.settings.enable_captioning", False)
    monkeypatch.setattr("app.ingestion.captioning.settings.enable_ocr", False)
    image_result = ingest_image(img_src, tmp_db, image_index, fake_embedder)

    assert text_result.n_chunks >= 1
    assert image_result.n_chunks == 1

    db_stats = tmp_db.stats()
    assert db_stats["total_chunks"] == text_result.n_chunks + 1
    assert db_stats["text_chunks"] + db_stats["image_chunks"] == db_stats["total_chunks"]

    retriever = Retriever(db=tmp_db, text_index=text_index, image_index=image_index, embedder=fake_embedder)

    text_query_results = retriever.search(query_text="the quick brown fox", top_k=5, rerank_top_k=5)
    assert any(r.modality == "text" for r in text_query_results)

    image_query_results = retriever.search(query_image_path=str(img_src), top_k=5, rerank_top_k=5)
    assert len(image_query_results) >= 1

    generation = generate_answer("What does the document say about foxes?", text_query_results, db=tmp_db, client=_StubClient())
    assert generation.success is True
    assert "[E1]" in generation.answer

    n_generations = tmp_db.conn.execute("SELECT COUNT(*) FROM generation_log").fetchone()[0]
    assert n_generations == 1
