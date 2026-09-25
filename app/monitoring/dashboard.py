"""Standalone Streamlit monitoring dashboard. Reads DuckDB and the FAISS indexes
directly (no API dependency) so it works even if the FastAPI server is down.
Run: streamlit run app/monitoring/dashboard.py
"""
from __future__ import annotations

import numpy as np
import streamlit as st

from app.config.settings import settings
from app.database.db import Database
from app.embeddings.clip_embedder import get_embedder
from app.monitoring.drift import compute_drift
from app.monitoring.metrics import generation_metrics, retrieval_metrics
from app.retrieval.faiss_index import FaissIndex

st.set_page_config(page_title="MultimodalRAG Monitoring", layout="wide")
st.title("MultimodalRAG — Monitoring & Drift")

db = Database(settings.duckdb_path)

st.header("System")
db_stats = db.stats()
cols = st.columns(4)
cols[0].metric("Documents", db_stats["documents"])
cols[1].metric("Text chunks", db_stats["text_chunks"])
cols[2].metric("Image chunks", db_stats["image_chunks"])
cols[3].metric("Total chunks", db_stats["total_chunks"])

try:
    embedder = get_embedder()
    text_index = FaissIndex.load(settings.faiss_text_index_path, dim=embedder.embedding_dim)
    image_index = FaissIndex.load(settings.faiss_image_index_path, dim=embedder.embedding_dim)
    cols2 = st.columns(2)
    cols2[0].metric("FAISS text vectors", text_index.ntotal)
    cols2[1].metric("FAISS image vectors", image_index.ntotal)
except Exception as exc:
    st.warning(f"Could not load FAISS indexes: {exc}")
    text_index = image_index = None

st.header("Retrieval")
rm = retrieval_metrics(db)
cols = st.columns(4)
cols[0].metric("Query count", rm.query_count)
cols[1].metric("Avg latency (ms)", f"{rm.avg_latency_ms:.1f}")
cols[2].metric("P50 latency (ms)", f"{rm.p50_latency_ms:.1f}")
cols[3].metric("P95 latency (ms)", f"{rm.p95_latency_ms:.1f}")
if rm.modality_distribution:
    st.bar_chart(rm.modality_distribution)

st.header("Generation")
gm = generation_metrics(db)
cols = st.columns(4)
cols[0].metric("Generations", gm.generation_count)
cols[1].metric("Successes", gm.success_count)
cols[2].metric("Failures", gm.failure_count)
cols[3].metric("Avg latency (ms)", f"{gm.avg_latency_ms:.1f}")

st.header("Embedding Drift (KS test)")
st.caption(
    "Reference = a random sample of TEXT chunk embeddings from the current index "
    "(stand-in for 'at index-build time'). Current = the most recently ingested "
    "text chunks. In a longer-running deployment, snapshot the reference sample "
    "once at build time and keep it fixed; here it's recomputed from the same "
    "index for demonstration since this is a single-session dev environment."
)

if text_index and text_index.ntotal >= 20:
    n = text_index.ntotal
    split = max(10, n // 2)
    all_vectors = text_index.index.reconstruct_n(0, n)
    reference = all_vectors[:split]
    current = all_vectors[split:] if split < n else all_vectors[-10:]
    if len(current) >= 2 and len(reference) >= 2:
        report = compute_drift(reference, current)
        cols = st.columns(4)
        cols[0].metric("Drifted dimensions", f"{report.n_dimensions_drifted}/{report.n_dimensions}")
        cols[1].metric("Drift fraction", f"{report.drift_fraction:.2%}")
        cols[2].metric("Mean KS statistic", f"{report.mean_ks_statistic:.4f}")
        cols[3].metric("Centroid cosine shift", f"{report.centroid_cosine_shift:.4f}")
        if report.alert:
            st.error("ALERT: drift_fraction exceeds threshold (10% of dimensions drifted)")
        else:
            st.success("No drift alert: distribution is within threshold")
    else:
        st.info("Not enough current-sample vectors to compute drift yet.")
else:
    st.info("Ingest at least 20 text chunks to compute a meaningful drift report.")
