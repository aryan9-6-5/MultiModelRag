# MultimodalRAG — Hybrid Text-Image Retrieval & Generation System

A genuine multimodal Retrieval-Augmented Generation system: it retrieves relevant
evidence across **text and images** using a shared CLIP embedding space, reranks
that evidence with a transparent cross-modal scoring layer, and generates answers
grounded in the retrieved evidence using a small local LLM. Everything runs on
free/open-source software — no paid APIs required.

```
TEXT + IMAGE DATA
        |
MULTIMODAL EMBEDDINGS (OpenCLIP)
        |
FAISS VECTOR RETRIEVAL  <-- DuckDB metadata (every vector is traceable to a row)
        |
CROSS-MODAL RERANKING
        |
EVIDENCE SELECTION
        |
GROUNDED LLM GENERATION (Ollama, local)
        |
ANSWER + EVIDENCE
```

## 1. Why multimodal RAG

Plain text RAG can't answer questions whose evidence lives in a diagram, a
scanned page, or a product photo. This system puts text and images in the same
searchable index so a single query — text or image — can retrieve evidence of
either kind, and grounds every generated answer in that retrieved evidence
(never in the model's unguided imagination).

## 2. Architecture

| Layer | Component | Location |
|---|---|---|
| Ingestion | PDF/TXT/MD/image parsing, chunking, captioning/OCR | `app/ingestion/` |
| Embeddings | OpenCLIP (ViT-B-32, laion2b) — shared text/image space | `app/embeddings/` |
| Vector search | FAISS (`IndexFlatIP`, one index per modality) | `app/retrieval/faiss_index.py` |
| Metadata store | DuckDB — every FAISS row traces to one chunk row | `app/database/` |
| Retrieval | query -> embedding -> FAISS -> DuckDB join | `app/retrieval/retriever.py` |
| Reranking | CLIP score + lexical overlap + modality bonus + dedup | `app/ranking/reranker.py` |
| Generation | Prompted, grounded local LLM via Ollama | `app/generation/` |
| API | FastAPI (`/ingest /search /query /health /stats`) | `app/api/` |
| Frontend | Streamlit query/ingest/monitoring UI | `frontend/app.py` |
| Monitoring | Retrieval/generation metrics + drift dashboard | `app/monitoring/` |
| Evaluation | COCO-subset retrieval benchmark + ablation study | `evaluation/` |

See [docs/DESIGN_DECISIONS.md](docs/DESIGN_DECISIONS.md) for the *why* behind
the image representation strategy, the reranking formula, the drift-detection
method, and the benchmark dataset choice.

## 3. Hardware

Developed against: NVIDIA RTX 3050 Laptop GPU (4GB VRAM), 16GB RAM, Windows,
Docker Desktop. All default models (OpenCLIP ViT-B-32, BLIP-base captioning,
Llama 3.2 3B / Qwen2.5 for generation) fit this budget. See
[Benchmark results](#13-benchmark--performance-results) for measured latency on
this hardware.

## 4. Installation (local)

Requires Python 3.11+, [Ollama](https://ollama.com), and (optionally) Tesseract
OCR on PATH.

```bash
python -m venv .venv
.venv/Scripts/activate        # .venv\Scripts\Activate.ps1 on PowerShell
pip install -r requirements.txt

ollama pull llama3.2:3b       # primary local generation model
ollama pull qwen2.5:latest    # fallback (used if the primary isn't pulled)
ollama serve                  # if not already running as a service
```

Copy `.env.example` to `.env` to override any default (model names, ports,
chunk size, etc.).

## 5. Local execution

```bash
python -m app.ingest --path data/raw          # ingest a file or directory
python -m app.api                              # FastAPI on :8000
streamlit run frontend/app.py                  # UI on :8501 (talks to the API)
streamlit run app/monitoring/dashboard.py      # standalone monitoring/drift dashboard
```

Visit `http://localhost:8000/docs` for interactive OpenAPI docs, or
`http://localhost:8501` for the UI.

## 6. Docker execution

```bash
docker compose up --build
```

Starts three services: `ollama` (model runtime), `api` (FastAPI on :8000),
`frontend` (Streamlit on :8501). The Docker image uses CPU-only PyTorch by
default for portability — see `docker/Dockerfile` and the comment in
`requirements-docker.txt` for enabling GPU passthrough via the NVIDIA
Container Toolkit. After first start, pull a model into the running Ollama
container: `docker exec -it <ollama-container> ollama pull llama3.2:3b`.

## 7. Dataset preparation (for evaluation)

```bash
python -m evaluation.datasets.download_coco_subset --n-images 200
```

Downloads a reproducible subset of MS-COCO val2017 (images + captions) by
direct per-image URL — see [docs/DESIGN_DECISIONS.md](docs/DESIGN_DECISIONS.md)
for licensing/provenance. Not committed to git (`.gitignore`d); rerun to
reproduce.

## 8. Ingestion

```bash
python -m app.ingest --path data/raw/some_document.pdf
python -m app.ingest --path data/raw/                # ingests every supported file recursively
```

Supported formats: `.pdf .txt .md .jpg .jpeg .png`. PDFs preserve per-page text
and page numbers; images get a CLIP embedding plus a generated caption and OCR
text (both optional, `ENABLE_CAPTIONING` / `ENABLE_OCR` in `.env`).

## 9. Index creation / rebuild

Ingestion updates the FAISS indexes incrementally. To rebuild both indexes
from scratch (re-embedding everything already recorded in DuckDB — the
reproducibility guarantee: FAISS holds no information that isn't also in
DuckDB):

```bash
python -m app.index
```

## 10. API usage

```
POST /ingest   multipart file upload -> ingest one file
POST /search   {"query_text": "...", "top_k": 10, "rerank_top_k": 5} -> ranked evidence
POST /query    {"query": "..."} -> grounded answer + evidence
GET  /health   liveness + Ollama reachability + embedding device
GET  /stats    document/chunk/vector counts
```

Full request/response schemas: `http://localhost:8000/docs`.

## 11. Streamlit usage

Three tabs: **Query** (ask a question, see the grounded answer and cited
evidence with scores), **Ingest** (upload a file, see live index stats),
**Monitoring** (health check + pointer to the dedicated dashboard).

## 12. Evaluation

```bash
python -m app.evaluate                 # retrieval benchmark (baseline vs reranked)
python -m app.evaluate --ablation      # + Experiment C (full pipeline w/ generation)
```

Computes NDCG@5, MRR, Recall@5, Recall@10, Precision@5, and retrieval latency
for real text->image and image->text queries against the COCO subset. Results
are written to `evaluation/results/*.json` — see below for the latest run.

## 13. Benchmark & performance results

### Benchmark Metrics (COCO 200-image dataset, CUDA RTX 3050 GPU)

#### 1. Retrieval Performance

| Task | Configuration | MRR | Recall@5 | Recall@10 | NDCG@10 | Latency (ms) |
|---|---|---|---|---|---|---|
| **Text $\rightarrow$ Image** | Baseline (CLIP + FAISS) | **0.849** | **0.985** | **1.000** | **0.887** | **27.3 ms** |
| **Text $\rightarrow$ Image** | Reranked (CLIP + Lexical + Modality) | 0.843 | 0.980 | **1.000** | 0.882 | 29.1 ms |
| **Image $\rightarrow$ Text** | Reranked | **0.855** | 0.970 | 0.990 | 0.889 | 73.1 ms |

#### 2. Ablation Study & Grounded Generation

| Experiment | Configuration | Metrics |
|---|---|---|
| **Exp A** | CLIP + FAISS Baseline | MRR: 0.849, Recall@10: 1.000, NDCG@10: 0.887, Latency: 27.3 ms |
| **Exp B** | CLIP + FAISS + Reranker | MRR: 0.843, Recall@10: 1.000, NDCG@10: 0.882, Latency: 29.1 ms |
| **Exp C** | Full Pipeline (Retrieval + Grounded LLM) | **Success Rate: 100%**, Citation Rate: 80%, Avg Latency: 14.7s |

## 14. Hardware requirements (summary)

- Minimum: CPU-only, 8GB RAM — everything runs, generation is slower.
- Recommended (as developed): 4GB+ VRAM GPU, 16GB RAM.
- Disk: ~10GB for models + dependencies, plus whatever you ingest.

## 15. Monitoring

`streamlit run app/monitoring/dashboard.py` — reads DuckDB and the FAISS
indexes directly (works even if the API is down). Shows: document/chunk/vector
counts, query count and latency (avg/p50/p95), modality distribution,
generation success/failure counts and latency, and the drift report below.

## 16. Drift detection

Per-dimension KS test between a reference and current sample of text
embeddings, reduced to `drift_fraction` (share of dimensions that drifted) plus
a centroid cosine-shift signal. Alerts when `drift_fraction >= 10%`. See
`app/monitoring/drift.py` and [docs/DESIGN_DECISIONS.md](docs/DESIGN_DECISIONS.md#drift-detection-method)
for the method rationale. This detects drift; it does not auto-remediate it.

## 17. Limitations

- CLIP retrieval is similarity-based, not detailed visual reasoning — the
  system compensates with generated captions/OCR, not true scene understanding.
- The reranker is a hand-weighted linear combination, not a learned model —
  see the ablation study for whether it actually beats raw FAISS ranking.
- Generation quality (faithfulness/relevance) is only lightly evaluated
  (citation-presence rate in Experiment C) — no human eval or LLM-judge pass.
- Drift detection has no automated remediation.
- Exact FAISS search (`IndexFlatIP`) doesn't scale to millions of vectors —
  fine at this project's scale, would need an ANN index beyond that.
- MLflow/Prometheus/Grafana were intentionally left out (monitoring is
  Streamlit + DuckDB-native) per the "no unnecessary services" guidance in the
  project brief; wiring MLflow onto the evaluation harness is straightforward
  future work if experiment tracking across many runs becomes necessary.

## 18. Troubleshooting

- **`GenerationError: Cannot reach Ollama`** — run `ollama serve`, and confirm
  `ollama list` shows `llama3.2:3b` (or set `LLM_MODEL`/`LLM_FALLBACK_MODEL` in
  `.env` to a model you have pulled).
- **CUDA not detected** — the system still works on CPU (slower); check with
  `python -m scripts.check_environment`.
- **OCR text always empty** — confirm `tesseract` is on PATH (`tesseract
  --version`); OCR is skipped silently if it's missing (`ENABLE_OCR=false` to
  disable it explicitly).
- **Docker `api` can't reach Ollama** — the compose file points `OLLAMA_HOST`
  at the `ollama` service; pull a model into that container (see section 6).

## 19. Future work

- Learned reranker (replace the hand-weighted formula once enough
  query/relevance data exists to train one).
- Approximate FAISS index (IVF/HNSW) if the corpus grows past ~100k chunks.
- LLM-judge or human-annotated relevance/faithfulness evaluation.
- MLflow experiment tracking across benchmark runs.
- Automated drift remediation (scheduled re-embedding / index rebuild).

## 20. Repository layout

```
app/            FastAPI backend, ingestion, embeddings, retrieval, ranking,
                generation, monitoring, database, config, CLI entry points
frontend/       Streamlit UI
evaluation/     dataset download, metrics, retrieval benchmark, ablation study
tests/          pytest suite (unit + integration + end-to-end)
docker/         Dockerfile
docs/           design-decision writeups
data/, storage/ ingested files / FAISS + DuckDB artifacts (gitignored)
```
