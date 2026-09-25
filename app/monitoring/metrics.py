"""Aggregate monitoring metrics read straight from DuckDB's query_log / generation_log
/ chunks tables — no separate metrics store, so the dashboard can never drift from
what actually happened."""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from app.database.db import Database


@dataclass
class RetrievalMetrics:
    query_count: int
    avg_latency_ms: float
    p50_latency_ms: float
    p95_latency_ms: float
    modality_distribution: dict[str, int]
    avg_n_results: float


@dataclass
class GenerationMetrics:
    generation_count: int
    success_count: int
    failure_count: int
    avg_latency_ms: float
    avg_prompt_tokens: float
    avg_completion_tokens: float


def retrieval_metrics(db: Database) -> RetrievalMetrics:
    df: pd.DataFrame = db.conn.execute("SELECT * FROM query_log").fetchdf()
    if df.empty:
        return RetrievalMetrics(0, 0.0, 0.0, 0.0, {}, 0.0)
    return RetrievalMetrics(
        query_count=len(df),
        avg_latency_ms=float(df["latency_ms"].mean()),
        p50_latency_ms=float(df["latency_ms"].quantile(0.5)),
        p95_latency_ms=float(df["latency_ms"].quantile(0.95)),
        modality_distribution=df["modality"].value_counts().to_dict(),
        avg_n_results=float(df["n_results"].mean()),
    )


def generation_metrics(db: Database) -> GenerationMetrics:
    df: pd.DataFrame = db.conn.execute("SELECT * FROM generation_log").fetchdf()
    if df.empty:
        return GenerationMetrics(0, 0, 0, 0.0, 0.0, 0.0)
    success = df[df["success"] == True]  # noqa: E712
    failure = df[df["success"] == False]  # noqa: E712
    return GenerationMetrics(
        generation_count=len(df),
        success_count=len(success),
        failure_count=len(failure),
        avg_latency_ms=float(df["latency_ms"].mean()),
        avg_prompt_tokens=float(success["prompt_tokens"].mean()) if not success.empty else 0.0,
        avg_completion_tokens=float(success["completion_tokens"].mean()) if not success.empty else 0.0,
    )
