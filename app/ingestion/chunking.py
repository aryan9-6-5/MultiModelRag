"""Simple whitespace-token chunker with overlap. Keeps chunks aligned to page
boundaries when page-tagged text is supplied, so every chunk traces back to one page."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from app.config.settings import settings


@dataclass
class Chunk:
    text: str
    page_number: Optional[int] = None


def _split_tokens(text: str) -> list[str]:
    return text.split()


def chunk_text(
    text: str,
    page_number: Optional[int] = None,
    chunk_size: Optional[int] = None,
    overlap: Optional[int] = None,
) -> list[Chunk]:
    """Split text into overlapping token windows. Empty/whitespace-only input yields no chunks."""
    chunk_size = chunk_size or settings.chunk_size_tokens
    overlap = overlap if overlap is not None else settings.chunk_overlap_tokens
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")

    tokens = _split_tokens(text)
    if not tokens:
        return []

    chunks: list[Chunk] = []
    start = 0
    step = chunk_size - overlap
    while start < len(tokens):
        window = tokens[start : start + chunk_size]
        chunks.append(Chunk(text=" ".join(window), page_number=page_number))
        if start + chunk_size >= len(tokens):
            break
        start += step
    return chunks


def chunk_pages(pages: list[tuple[int, str]], chunk_size: Optional[int] = None, overlap: Optional[int] = None) -> list[Chunk]:
    """pages: list of (page_number, page_text). Chunks never span pages."""
    all_chunks: list[Chunk] = []
    for page_number, page_text in pages:
        all_chunks.extend(chunk_text(page_text, page_number=page_number, chunk_size=chunk_size, overlap=overlap))
    return all_chunks
