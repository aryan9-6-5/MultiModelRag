def test_insert_and_fetch_document(tmp_db):
    doc_id = tmp_db.insert_document(source="a.txt", file_path="/tmp/a.txt", file_type="txt")
    assert doc_id

    chunk_id = tmp_db.insert_chunk(
        document_id=doc_id, modality="text", source="a.txt", file_path="/tmp/a.txt",
        faiss_index="text", embedding_id=0, page_number=1, text="hello world",
    )
    chunk = tmp_db.get_chunk(chunk_id)
    assert chunk["text"] == "hello world"
    assert chunk["document_id"] == doc_id


def test_no_orphaned_vectors_every_chunk_has_embedding_id(tmp_db):
    doc_id = tmp_db.insert_document(source="a.txt", file_path="/tmp/a.txt", file_type="txt")
    for i in range(3):
        tmp_db.insert_chunk(
            document_id=doc_id, modality="text", source="a.txt", file_path="/tmp/a.txt",
            faiss_index="text", embedding_id=tmp_db.next_embedding_id("text"), text=f"chunk {i}",
        )
    ids = tmp_db.all_embedding_ids("text")
    assert ids == [0, 1, 2]


def test_get_chunks_by_embedding_ids_preserves_order(tmp_db):
    doc_id = tmp_db.insert_document(source="a.txt", file_path="/tmp/a.txt", file_type="txt")
    for i in range(5):
        tmp_db.insert_chunk(
            document_id=doc_id, modality="text", source="a.txt", file_path="/tmp/a.txt",
            faiss_index="text", embedding_id=i, text=f"chunk {i}",
        )
    rows = tmp_db.get_chunks_by_embedding_ids("text", [3, 1, 4])
    assert [r["embedding_id"] for r in rows] == [3, 1, 4]


def test_stats_counts_modalities_separately(tmp_db):
    doc_id = tmp_db.insert_document(source="a.txt", file_path="/tmp/a.txt", file_type="txt")
    tmp_db.insert_chunk(
        document_id=doc_id, modality="text", source="a.txt", file_path="/tmp/a.txt",
        faiss_index="text", embedding_id=0, text="hi",
    )
    tmp_db.insert_chunk(
        document_id=doc_id, modality="image", source="a.png", file_path="/tmp/a.png",
        faiss_index="image", embedding_id=0, image_path="/tmp/a.png",
    )
    stats = tmp_db.stats()
    assert stats["text_chunks"] == 1
    assert stats["image_chunks"] == 1
    assert stats["total_chunks"] == 2


def test_query_and_generation_logging(tmp_db):
    query_id = tmp_db.log_query(
        query_text="what is x", query_image=None, modality="text",
        top_k=10, latency_ms=12.3, n_results=5,
    )
    assert query_id
    gen_id = tmp_db.log_generation(
        query_id=query_id, model="llama3.2:3b", latency_ms=200.0,
        prompt_tokens=100, completion_tokens=50, success=True,
    )
    assert gen_id
