# Design decisions

Kept separate from the README so the README stays a usage guide. These are the
non-obvious calls made while building this system, and why.

## Image representation strategy (why CLIP alone isn't enough)

CLIP gives a shared embedding space for *similarity*, not scene understanding —
it can tell you an image is close to the phrase "a red car on a street" but the
local LLM used for generation is text-only and cannot look at pixels. So every
ingested image gets, in addition to its CLIP embedding:
- a short generated caption (BLIP, `Salesforce/blip-image-captioning-base`, ~990MB,
  runs on CPU or the 4GB GPU)
- OCR text via Tesseract, when the image contains legible text

Both are stored as the chunk's `text` field (see `app/ingestion/captioning.py`).
This makes images inspectable in the generation context even though the LLM
never sees pixels, and searchable by plain-text queries beyond pure visual
similarity (e.g. a scanned page or a screenshot with a chart title).

## Cross-modal reranking formula

`final_score = 0.65 * clip_similarity + 0.25 * lexical_overlap + 0.10 *
modality_match_bonus`, minus a duplicate penalty for near-identical text (see
`app/ranking/reranker.py`). CLIP dominates because it's the only signal actually
grounded in the shared embedding space; lexical overlap and modality bonus are
tie-breakers, not primary signals. This is deliberately simple and fully
transparent (no learned reranker) so it can be inspected and A/B'd — see
`evaluation/ablation.py`, which measures whether it actually beats raw FAISS
ranking rather than assuming it does.

## Drift detection method

Per-embedding-dimension two-sample Kolmogorov-Smirnov test, reduced to a single
`drift_fraction` (share of dimensions where the KS test rejects "same
distribution" at p<0.05), plus a cheaper reference/current centroid
cosine-shift as a secondary signal. KS was chosen over PSI or mean/variance
checks because CLIP embedding dimensions are continuous and roughly unimodal —
exactly what KS is built for — and it doesn't assume the shift shows up in the
mean (PSI/mean-variance can miss a shape-only shift). See
`app/monitoring/drift.py`. This module *detects* drift; it does not remediate
it — no automatic re-embedding or index rebuild is triggered.

## Benchmark dataset

MS-COCO val2017 captions + a downloaded image subset (default 200 images), via
`evaluation/datasets/download_coco_subset.py`. Chosen because: it's a standard
text<->image retrieval benchmark, captions are CC BY 4.0, images are
Flickr-sourced under mixed CC licenses (COCO's stated terms of use), and it's
downloadable without authentication straight from `cocodataset.org` by URL —
no Kaggle account, no gated dataset. Only the images actually used are
downloaded (via each image's `coco_url`), not the full ~1GB zip.

## Why IndexFlatIP instead of an approximate FAISS index

At the scale this project targets (thousands, not millions, of chunks), exact
cosine search (`IndexFlatIP` over L2-normalized vectors) is fast enough that
there is no recall-vs-speed tradeoff worth tuning. An IVF/HNSW index would add
training-set requirements and parameter tuning for no measurable benefit here.

## Why Ollama instead of a raw llama.cpp binding

Ollama wraps llama.cpp with model management (`ollama pull`), a stable local
HTTP API, and quantized GGUF models out of the box — all free/local, no API
key. `app/generation/llm_client.py` talks to it over HTTP so swapping to a
direct llama.cpp binding later doesn't touch any other module.
