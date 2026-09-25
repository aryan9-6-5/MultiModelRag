from app.ingestion.chunking import chunk_pages, chunk_text


def test_chunk_text_respects_size_and_overlap():
    text = " ".join(f"word{i}" for i in range(100))
    chunks = chunk_text(text, chunk_size=30, overlap=5)
    assert len(chunks) > 1
    for c in chunks:
        assert len(c.text.split()) <= 30


def test_chunk_text_overlap_shares_tokens():
    text = " ".join(f"word{i}" for i in range(50))
    chunks = chunk_text(text, chunk_size=20, overlap=5)
    first_tail = chunks[0].text.split()[-5:]
    second_head = chunks[1].text.split()[:5]
    assert first_tail == second_head


def test_chunk_text_empty_input():
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_chunk_text_rejects_bad_overlap():
    import pytest

    with pytest.raises(ValueError):
        chunk_text("a b c", chunk_size=10, overlap=10)


def test_chunk_pages_preserves_page_numbers():
    pages = [(1, "a b c d e f g h"), (2, "i j k l m n o p")]
    chunks = chunk_pages(pages, chunk_size=4, overlap=1)
    assert all(c.page_number in (1, 2) for c in chunks)
    assert any(c.page_number == 1 for c in chunks)
    assert any(c.page_number == 2 for c in chunks)


def test_chunk_pages_never_spans_pages():
    pages = [(1, "a b c"), (2, "d e f")]
    chunks = chunk_pages(pages, chunk_size=10, overlap=0)
    page1_text = " ".join(c.text for c in chunks if c.page_number == 1)
    page2_text = " ".join(c.text for c in chunks if c.page_number == 2)
    assert "d" not in page1_text.split()
    assert "a" not in page2_text.split()
