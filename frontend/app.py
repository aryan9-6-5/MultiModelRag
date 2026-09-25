"""Streamlit frontend for the multimodal RAG system. Talks to the FastAPI backend
over HTTP so the two stay decoupled (the API is independently usable/testable)."""
from __future__ import annotations

import os

import requests
import streamlit as st

API_BASE = os.getenv("API_BASE", "http://localhost:8000")

st.set_page_config(page_title="MultimodalRAG", layout="wide")


def call_api(method: str, path: str, **kwargs):
    try:
        resp = requests.request(method, f"{API_BASE}{path}", timeout=120, **kwargs)
        resp.raise_for_status()
        return resp.json(), None
    except requests.exceptions.RequestException as exc:
        return None, str(exc)


tab_query, tab_ingest, tab_monitor = st.tabs(["Query", "Ingest", "Monitoring"])

with tab_query:
    st.header("Multimodal Retrieval + Grounded Generation")
    query_text = st.text_input("Ask a question")
    col_a, col_b = st.columns(2)
    top_k = col_a.slider("Retrieve top-K", 1, 30, 10)
    rerank_k = col_b.slider("Keep after reranking", 1, 15, 5)

    run_search = st.button("Search only (no generation)")
    run_query = st.button("Search + Generate answer", type="primary")

    if (run_search or run_query) and query_text:
        if run_query:
            data, err = call_api(
                "POST", "/query",
                json={"query": query_text, "top_k": top_k, "rerank_top_k": rerank_k},
            )
            if err:
                st.error(f"API error: {err}")
            else:
                st.subheader("Answer")
                if not data["success"]:
                    st.warning(f"Generation failed: {data['error']}")
                else:
                    st.write(data["answer"])
                    st.caption(
                        f"model={data['model']} | retrieval={data['retrieval_latency_ms']:.0f}ms "
                        f"| generation={data['generation_latency_ms']:.0f}ms"
                    )
                evidence = data["evidence"]
        else:
            data, err = call_api(
                "POST", "/search",
                json={"query_text": query_text, "top_k": top_k, "rerank_top_k": rerank_k},
            )
            if err:
                st.error(f"API error: {err}")
                evidence = []
            else:
                st.caption(f"latency={data['latency_ms']:.0f}ms")
                evidence = data["results"]

        if (run_search or run_query) and not err:
            st.subheader(f"Evidence ({len(evidence)})")
            for i, item in enumerate(evidence, start=1):
                with st.container(border=True):
                    header = f"[E{i}] {item['modality'].upper()} — {item['source']}"
                    if item.get("page_number"):
                        header += f" (page {item['page_number']})"
                    st.markdown(f"**{header}**")
                    st.caption(
                        f"clip_score={item['clip_score']:.3f}  final_score={item['final_score']:.3f}"
                    )
                    if item["modality"] == "image" and item.get("image_path"):
                        st.image(item["image_path"], width=240)
                    if item.get("text_preview"):
                        st.write(item["text_preview"])

with tab_ingest:
    st.header("Ingest documents / images")
    uploaded = st.file_uploader(
        "Upload a PDF, TXT, MD, JPG, or PNG", type=["pdf", "txt", "md", "jpg", "jpeg", "png"]
    )
    if uploaded and st.button("Ingest file"):
        files = {"file": (uploaded.name, uploaded.getvalue())}
        data, err = call_api("POST", "/ingest", files=files)
        if err:
            st.error(f"Ingest failed: {err}")
        else:
            for r in data:
                st.success(f"Ingested {r['source']} ({r['file_type']}): {r['n_chunks']} chunk(s)")

    st.divider()
    stats, err = call_api("GET", "/stats")
    if stats:
        cols = st.columns(4)
        cols[0].metric("Documents", stats["documents"])
        cols[1].metric("Text chunks", stats["text_chunks"])
        cols[2].metric("Image chunks", stats["image_chunks"])
        cols[3].metric("FAISS vectors (text+image)", stats["faiss_text_vectors"] + stats["faiss_image_vectors"])

with tab_monitor:
    st.header("Pipeline Monitoring")
    health, herr = call_api("GET", "/health")
    if health:
        st.write(f"API status: **{health['status']}** | Ollama reachable: **{health['ollama_reachable']}** "
                  f"| embedding device: **{health['embedding_device']}**")
    if herr:
        st.error(f"Cannot reach API at {API_BASE}: {herr}")
    st.info(
        "For query/generation latency, modality distribution, and drift metrics, run "
        "`python -m app.monitoring.dashboard` (a dedicated Streamlit dashboard reading "
        "directly from DuckDB — no API dependency), or see evaluation/results/ for the "
        "latest benchmark run."
    )
