import pytest

from app.ingestion.parsers import detect_file_type, extract_pages, parse_text_file


def test_detect_file_type():
    assert detect_file_type("doc.pdf") == "pdf"
    assert detect_file_type("notes.txt") == "txt"
    assert detect_file_type("readme.md") == "md"
    assert detect_file_type("photo.jpg") == "image"
    assert detect_file_type("photo.PNG") == "image"


def test_detect_file_type_unsupported():
    with pytest.raises(ValueError):
        detect_file_type("archive.zip")


def test_parse_text_file(tmp_path):
    f = tmp_path / "note.txt"
    f.write_text("hello world", encoding="utf-8")
    pages = parse_text_file(f)
    assert pages == [(1, "hello world")]


def test_parse_text_file_empty(tmp_path):
    f = tmp_path / "empty.txt"
    f.write_text("", encoding="utf-8")
    assert parse_text_file(f) == []


def test_extract_pages_dispatches_by_extension(tmp_path):
    f = tmp_path / "note.md"
    f.write_text("# Title\ncontent", encoding="utf-8")
    pages = extract_pages(f)
    assert len(pages) == 1
    assert "Title" in pages[0][1]
