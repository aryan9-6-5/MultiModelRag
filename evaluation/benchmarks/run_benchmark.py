"""Cross-modal retrieval benchmark on the COCO subset built by
evaluation/datasets/download_coco_subset.py.

Measures REAL text->image retrieval (find the image a caption describes) and
image->text retrieval (find a caption describing a query image), for two
configurations:
  BASELINE  = CLIP + FAISS only (raw cosine-similarity ranking)
  RERANKED  = CLIP + FAISS + cross-modal reranking (app.ranking.reranker)

All numbers are computed from an actual run against the downloaded subset —
nothing here is hardcoded. Results are written to evaluation/results/ as JSON.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from app.embeddings.clip_embedder import get_embedder
from app.ranking.reranker import Candidate, rerank
from app.retrieval.faiss_index import FaissIndex
from evaluation.metrics import ndcg_at_k, precision_at_k, reciprocal_rank, recall_at_k

MANIFEST_PATH = Path(__file__).parent.parent / "datasets" / "coco_subset" / "manifest.json"
RESULTS_DIR = Path(__file__).parent.parent / "results"


def load_manifest() -> list[dict]:
    if not MANIFEST_PATH.exists():
        raise FileNotFoundError(
            f"{MANIFEST_PATH} not found. Run: python -m evaluation.datasets.download_coco_subset first."
        )
    with open(MANIFEST_PATH) as f:
        return json.load(f)


def build_image_index(manifest: list[dict], embedder) -> tuple[FaissIndex, list[int]]:
    """Returns (index, image_ids) where row i of the index corresponds to image_ids[i]."""
    vectors = []
    image_ids = []
    for item in manifest:
        vec = embedder.embed_image(item["image_path"])
        vectors.append(vec)
        image_ids.append(item["image_id"])
    index = FaissIndex(dim=embedder.embedding_dim)
    index.add(np.stack(vectors))
    return index, image_ids


def build_caption_index(manifest: list[dict], embedder) -> tuple[FaissIndex, list[int], list[str]]:
    """One caption per image (first ground-truth caption). Returns (index, image_ids, caption_texts)."""
    texts = [item["captions"][0] for item in manifest]
    image_ids = [item["image_id"] for item in manifest]
    vectors = embedder.embed_text(texts)
    index = FaissIndex(dim=embedder.embedding_dim)
    index.add(vectors)
    return index, image_ids, texts


def evaluate_text_to_image(
    manifest: list[dict], image_index: FaissIndex, image_ids: list[int],
    embedder, k_values: tuple[int, ...], use_rerank: bool, indexed_text_by_image_id: dict[int, str],
) -> dict:
    """Query = a caption NOT used as the indexed text side (index is image-only here);
    ground truth = the image_id that caption belongs to.

    `indexed_text_by_image_id` stands in for the caption/OCR text a real ingestion run
    would attach to each image (see app.ingestion.captioning) — here it's each image's
    *first* ground-truth caption, deliberately distinct from the *query* caption
    (its second ground-truth caption) so the reranker's lexical-overlap signal isn't
    trivially comparing a caption against itself.
    """
    per_query_metrics = {k: {"recall": [], "precision": [], "ndcg": []} for k in k_values}
    mrr_values = []
    latencies_ms = []

    for item in manifest:
        query_text = item["captions"][min(1, len(item["captions"]) - 1)]  # 2nd caption if available
        true_image_id = item["image_id"]

        start = time.perf_counter()
        query_vec = embedder.embed_text(query_text)
        max_k = max(k_values)
        scores, ids = image_index.search(query_vec, max_k)
        latencies_ms.append((time.perf_counter() - start) * 1000)

        ranked_image_ids = [image_ids[i] for i in ids[0] if i != -1]

        if use_rerank:
            candidates = [
                Candidate(
                    chunk_id=str(image_ids[i]), modality="image", clip_score=float(s),
                    text=indexed_text_by_image_id.get(image_ids[i]),
                )
                for i, s in zip(ids[0], scores[0]) if i != -1
            ]
            ranked = rerank(query_text=query_text, query_modality="text", candidates=candidates, top_k=max_k)
            ranked_image_ids = [int(c.chunk_id) for c in ranked]

        relevant = [iid == true_image_id for iid in ranked_image_ids]
        total_relevant = 1  # exactly one correct image per caption query

        for k in k_values:
            per_query_metrics[k]["recall"].append(recall_at_k(relevant, k, total_relevant))
            per_query_metrics[k]["precision"].append(precision_at_k(relevant, k))
            per_query_metrics[k]["ndcg"].append(ndcg_at_k(relevant, k, total_relevant))
        mrr_values.append(reciprocal_rank(relevant))

    result = {"mrr": float(np.mean(mrr_values)), "avg_latency_ms": float(np.mean(latencies_ms))}
    for k in k_values:
        result[f"recall@{k}"] = float(np.mean(per_query_metrics[k]["recall"]))
        result[f"precision@{k}"] = float(np.mean(per_query_metrics[k]["precision"]))
        result[f"ndcg@{k}"] = float(np.mean(per_query_metrics[k]["ndcg"]))
    return result


def evaluate_image_to_text(
    manifest: list[dict], caption_index: FaissIndex, caption_image_ids: list[int],
    embedder, k_values: tuple[int, ...],
) -> dict:
    """Query = image; ground truth = any caption belonging to the same image_id."""
    per_query_metrics = {k: {"recall": [], "precision": [], "ndcg": []} for k in k_values}
    mrr_values = []
    latencies_ms = []

    for item in manifest:
        true_image_id = item["image_id"]

        start = time.perf_counter()
        query_vec = embedder.embed_image(item["image_path"])
        max_k = max(k_values)
        scores, ids = caption_index.search(query_vec, max_k)
        latencies_ms.append((time.perf_counter() - start) * 1000)

        ranked_image_ids = [caption_image_ids[i] for i in ids[0] if i != -1]
        relevant = [iid == true_image_id for iid in ranked_image_ids]
        total_relevant = 1

        for k in k_values:
            per_query_metrics[k]["recall"].append(recall_at_k(relevant, k, total_relevant))
            per_query_metrics[k]["precision"].append(precision_at_k(relevant, k))
            per_query_metrics[k]["ndcg"].append(ndcg_at_k(relevant, k, total_relevant))
        mrr_values.append(reciprocal_rank(relevant))

    result = {"mrr": float(np.mean(mrr_values)), "avg_latency_ms": float(np.mean(latencies_ms))}
    for k in k_values:
        result[f"recall@{k}"] = float(np.mean(per_query_metrics[k]["recall"]))
        result[f"precision@{k}"] = float(np.mean(per_query_metrics[k]["precision"]))
        result[f"ndcg@{k}"] = float(np.mean(per_query_metrics[k]["ndcg"]))
    return result


def run(k_values=(5, 10)) -> dict:
    manifest = load_manifest()
    embedder = get_embedder()

    print(f"Building indexes from {len(manifest)} images...")
    image_index, image_ids = build_image_index(manifest, embedder)
    caption_index, caption_image_ids, caption_texts = build_caption_index(manifest, embedder)

    indexed_text_by_image_id = {item["image_id"]: item["captions"][0] for item in manifest}

    print("Evaluating text->image (baseline)...")
    t2i_baseline = evaluate_text_to_image(manifest, image_index, image_ids, embedder, k_values, False, {})
    print("Evaluating text->image (reranked)...")
    t2i_reranked = evaluate_text_to_image(
        manifest, image_index, image_ids, embedder, k_values, True, indexed_text_by_image_id
    )

    print("Evaluating image->text...")
    i2t = evaluate_image_to_text(manifest, caption_index, caption_image_ids, embedder, k_values)

    results = {
        "n_images": len(manifest),
        "k_values": list(k_values),
        "text_to_image": {"baseline_clip_faiss": t2i_baseline, "reranked": t2i_reranked},
        "image_to_text": i2t,
        "device": embedder.device,
    }

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "retrieval_benchmark.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results written to {out_path}")
    print(json.dumps(results, indent=2))
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--k", type=int, nargs="+", default=[5, 10])
    args = parser.parse_args()
    run(tuple(args.k))
