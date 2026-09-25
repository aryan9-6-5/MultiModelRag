"""Ablation study: A) CLIP+FAISS  B) CLIP+FAISS+reranking  C) full pipeline
(adds grounded generation on top of B). Reuses evaluation.benchmarks.run_benchmark
for A/B retrieval metrics (that script already runs baseline vs reranked side by
side) and adds an end-to-end generation pass for C.

C does NOT get a retrieval-quality number of its own — its retrieval stage is
identical to B. What C adds and measures is: does the generation stage produce an
answer at all (success rate), how long does it take, and does it actually cite
retrieved evidence ([Ex] tags) rather than answering ungrounded. This is a modest,
honest proxy for "the full pipeline works end-to-end" — it is NOT a claim about
answer correctness or faithfulness, which would need either human judgment or a
separate LLM-judge pass (not implemented, see README limitations).
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np

from app.embeddings.clip_embedder import get_embedder
from app.generation.generator import generate_answer
from app.generation.llm_client import OllamaClient
from app.retrieval.faiss_index import FaissIndex
from app.retrieval.retriever import RetrievalResult
from evaluation.benchmarks.run_benchmark import build_image_index, load_manifest

RESULTS_DIR = Path(__file__).parent / "results"
CITATION_RE = re.compile(r"\[E\d+\]")


def run_generation_ablation(n_queries: int = 10) -> dict:
    manifest = load_manifest()[:n_queries]
    embedder = get_embedder()
    image_index, image_ids = build_image_index(manifest, embedder)
    client = OllamaClient()

    successes, cited, latencies = [], [], []
    for item in manifest:
        query_text = item["captions"][0]
        # Build a single-item evidence list directly from this benchmark image
        # (bypassing DuckDB, since this ablation only needs the generation stage).
        evidence = [
            RetrievalResult(
                chunk_id=str(item["image_id"]), modality="image", source=item["file_name"],
                file_path=item["image_path"], page_number=None, text=item["captions"][1],
                image_path=item["image_path"], clip_score=1.0, final_score=1.0, score_breakdown={},
            )
        ]
        response = generate_answer(f"What is shown in this image? Query: {query_text}", evidence, client=client)
        successes.append(response.success)
        latencies.append(response.latency_ms)
        cited.append(bool(CITATION_RE.search(response.answer)) if response.success else False)

    return {
        "n_queries": len(manifest),
        "success_rate": float(np.mean(successes)),
        "citation_rate_among_successes": float(np.mean(cited)) if any(successes) else 0.0,
        "avg_latency_ms": float(np.mean(latencies)),
    }


def run(n_generation_queries: int = 10) -> dict:
    from evaluation.benchmarks.run_benchmark import run as run_retrieval_benchmark

    retrieval_results = run_retrieval_benchmark()
    print("Running Experiment C (full pipeline: retrieval + grounded generation)...")
    try:
        generation_results = run_generation_ablation(n_generation_queries)
    except Exception as exc:
        generation_results = {"error": str(exc)}

    ablation = {
        "experiment_A_clip_faiss_baseline": retrieval_results["text_to_image"]["baseline_clip_faiss"],
        "experiment_B_clip_faiss_reranked": retrieval_results["text_to_image"]["reranked"],
        "experiment_C_full_pipeline_generation": generation_results,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "ablation_study.json"
    with open(out_path, "w") as f:
        json.dump(ablation, f, indent=2)
    print(f"Ablation results written to {out_path}")
    print(json.dumps(ablation, indent=2))
    return ablation


if __name__ == "__main__":
    run()
