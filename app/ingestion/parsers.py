"""Format-specific text/page extraction. Add a new format by adding a parse_* function
and registering it in EXTENSION_PARSERS."""
from __future__ import annotations

from pathlib import Path

import fitz  # PyMuPDF

TEXT_EXTENSIONS = {".txt", ".md"}
PDF_EXTENSIONS = {".pdf"}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}


def parse_pdf(path: str | Path) -> list[tuple[int, str]]:
    """Returns [(page_number, text), ...], 1-indexed pages. Skips blank pages."""
    doc = fitz.open(path)
    pages = []
    for i, page in enumerate(doc, start=1):
        text = page.get_text().strip()
        if text:
            pages.append((i, text))
    doc.close()
    return pages


def parse_text_file(path: str | Path) -> list[tuple[int, str]]:
    text = Path(path).read_text(encoding="utf-8", errors="replace").strip()
    return [(1, text)] if text else []


def detect_file_type(path: str | Path) -> str:
    ext = Path(path).suffix.lower()
    if ext in PDF_EXTENSIONS:
        return "pdf"
    if ext in TEXT_EXTENSIONS:
        return "txt" if ext == ".txt" else "md"
    if ext in IMAGE_EXTENSIONS:
        return "image"
    raise ValueError(f"Unsupported file extension: {ext}")


def extract_pages(path: str | Path) -> list[tuple[int, str]]:
    """Dispatch to the right parser based on extension. Returns [(page_number, text)]."""
    file_type = detect_file_type(path)
    if file_type == "pdf":
        return parse_pdf(path)
    if file_type in ("txt", "md"):
        return parse_text_file(path)
    raise ValueError(f"extract_pages does not handle file_type={file_type}")
