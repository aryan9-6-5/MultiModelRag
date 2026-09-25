"""Prompt construction for grounded generation. Keeps evidence formatting in one
place so the API, CLI, and evaluation harness all build identical prompts."""
from __future__ import annotations

from app.retrieval.retriever import RetrievalResult

SYSTEM_INSTRUCTIONS = (
    "You are a careful research assistant. You answer ONLY using the EVIDENCE "
    "provided below. Rules:\n"
    "1. Use the retrieved evidence to answer the question.\n"
    "2. Do NOT invent facts that are not supported by the evidence.\n"
    "3. If the evidence is insufficient or missing to answer confidently, say so "
    "explicitly instead of guessing.\n"
    "4. Clearly distinguish stated evidence from any inference/assumption you make.\n"
    "5. Cite evidence by its [E1], [E2]... tag when you use it.\n"
)


def format_evidence(results: list[RetrievalResult]) -> str:
    lines = []
    for i, r in enumerate(results, start=1):
        tag = f"[E{i}]"
        if r.modality == "text":
            loc = f"{r.source}" + (f", page {r.page_number}" if r.page_number else "")
            lines.append(f"{tag} (text, {loc}): {r.text}")
        else:
            loc = r.source
            desc = r.text or "(no caption/OCR text available)"
            lines.append(f"{tag} (image, {loc}): {desc}")
    return "\n".join(lines) if lines else "(no evidence retrieved)"


def build_prompt(query: str, results: list[RetrievalResult]) -> str:
    evidence_block = format_evidence(results)
    return (
        f"{SYSTEM_INSTRUCTIONS}\n"
        f"EVIDENCE:\n{evidence_block}\n\n"
        f"QUESTION: {query}\n\n"
        f"ANSWER (cite [Ex] tags, then be explicit about any evidence gaps):"
    )
