"""Ties retrieval evidence to prompt construction to the LLM call, and logs the
generation event to DuckDB (latency, tokens, success/failure) for monitoring."""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional

from app.database.db import Database
from app.generation.llm_client import GenerationError, OllamaClient
from app.generation.prompt import build_prompt
from app.retrieval.retriever import RetrievalResult


@dataclass
class GenerationResponse:
    answer: str
    evidence: list[RetrievalResult]
    model: str
    latency_ms: float
    success: bool
    error: Optional[str] = None


def generate_answer(
    query: str,
    evidence: list[RetrievalResult],
    db: Optional[Database] = None,
    query_id: Optional[str] = None,
    client: Optional[OllamaClient] = None,
) -> GenerationResponse:
    client = client or OllamaClient()
    prompt = build_prompt(query, evidence)

    start = time.perf_counter()
    try:
        result = client.generate(prompt)
        latency_ms = (time.perf_counter() - start) * 1000
        response = GenerationResponse(
            answer=result["text"], evidence=evidence, model=result["model"],
            latency_ms=latency_ms, success=True,
        )
        if db:
            db.log_generation(
                query_id=query_id, model=result["model"], latency_ms=latency_ms,
                prompt_tokens=result.get("prompt_tokens"), completion_tokens=result.get("completion_tokens"),
                success=True,
            )
        return response
    except GenerationError as exc:
        latency_ms = (time.perf_counter() - start) * 1000
        if db:
            db.log_generation(
                query_id=query_id, model=client.model, latency_ms=latency_ms,
                prompt_tokens=None, completion_tokens=None, success=False, error=str(exc),
            )
        return GenerationResponse(
            answer="", evidence=evidence, model=client.model, latency_ms=latency_ms,
            success=False, error=str(exc),
        )
