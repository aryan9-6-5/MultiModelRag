---
title: MultimodalRAG
emoji: 🔎
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: false
---

Text-image retrieval-augmented generation. OpenCLIP embeddings in a shared text/image
space, FAISS vector search, DuckDB metadata, a cross-modal reranker, and grounded
answers from Llama 3.3 70B on Groq. FastAPI backend, Streamlit frontend.

The demo index holds 100 random MS-COCO val2017 images (captioned with BLIP). Try
queries such as "a person riding a bike" or "a plate of food". Uploads are disabled
and queries are rate-limited per client.

Source code: https://github.com/aryan9-6-5/MultiModelRag

Benchmark (200-image COCO subset, RTX 3050): MRR 0.849, Recall@10 1.0, 27 ms retrieval.
