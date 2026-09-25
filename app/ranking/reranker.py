"""Cross-modal reranking layer.

A bare FAISS top-k is just cosine similarity in CLIP space — it doesn't know that a
near-duplicate should be suppressed, or that a marginally-lower-similarity image
carries an OCR/caption text hit that makes it more relevant to a text query than a
higher-similarity chunk with no lexical overlap at all. This module recombines
multiple weak signals into a single reranked ordering, and is deliberately kept
separate from FAISS retrieval so the two can be A/B'd (see evaluation/ ablation).

Score = w_clip * clip_similarity + w_lexical * lexical_overlap(query, candidate.text)
        + w_modality_bonus (query/candidate modality match bonus)
        - duplicate_penalty (near-duplicate suppression by text/image-path)

Weights are configurable; defaults were chosen by inspection (CLIP similarity should
dominate since it's the only signal grounded in the shared embedding space — lexical
overlap and modality bonus are tie-breakers), and are revisited in the ablation study
in evaluation/ which measures whether this actually beats raw FAISS ranking.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

_WORD_RE = re.compile(r"[a-z0-9]+")


def _tokenize(text: str) -> set[str]:
    return set(_WORD_RE.findall(text.lower()))


def lexical_overlap(query: str, candidate_text: Optional[str]) -> float:
    """Jaccard overlap between query tokens and candidate tokens. 0 if no candidate text."""
    if not candidate_text:
        return 0.0
    q_tokens = _tokenize(query)
    c_tokens = _tokenize(candidate_text)
    if not q_tokens or not c_tokens:
        return 0.0
    intersection = q_tokens & c_tokens
    union = q_tokens | c_tokens
    return len(intersection) / len(union)


@dataclass
class RerankWeights:
    clip: float = 0.65
    lexical: float = 0.25
    modality_bonus: float = 0.10
    duplicate_penalty: float = 0.5


@dataclass
class Candidate:
    chunk_id: str
    modality: str
    clip_score: float
    text: Optional[str] = None
    image_path: Optional[str] = None
    metadata: dict = field(default_factory=dict)
    extra: dict = field(default_factory=dict)


def rerank(
    query_text: Optional[str],
    query_modality: str,
    candidates: list[Candidate],
    weights: RerankWeights = RerankWeights(),
    top_k: Optional[int] = None,
) -> list[Candidate]:
    """Rerank candidates in place of raw FAISS ordering. Adds `final_score` and
    `score_breakdown` into candidate.extra. Deduplicates near-identical text chunks."""
    seen_texts: set[str] = set()
    scored: list[tuple[float, Candidate]] = []

    for c in candidates:
        lex = lexical_overlap(query_text, c.text) if query_text else 0.0
        modality_match = 1.0 if c.modality == query_modality else 0.0

        dup_key = (c.text or "").strip().lower()[:200]
        is_dup = bool(dup_key) and dup_key in seen_texts
        if dup_key:
            seen_texts.add(dup_key)

        score = (
            weights.clip * c.clip_score
            + weights.lexical * lex
            + weights.modality_bonus * modality_match
        )
        if is_dup:
            score -= weights.duplicate_penalty

        c.extra["final_score"] = score
        c.extra["score_breakdown"] = {
            "clip_score": c.clip_score,
            "lexical_overlap": lex,
            "modality_match": modality_match,
            "duplicate_penalty_applied": is_dup,
        }
        scored.append((score, c))

    scored.sort(key=lambda pair: pair[0], reverse=True)
    ranked = [c for _, c in scored]
    return ranked[:top_k] if top_k else ranked
