"""Central configuration, overridable via environment variables / .env."""
from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

ROOT_DIR = Path(__file__).resolve().parents[2]


def _bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    return default if val is None else val.lower() in {"1", "true", "yes"}


class Settings:
    # Storage
    data_dir: Path = Path(os.getenv("DATA_DIR", ROOT_DIR / "data"))
    storage_dir: Path = Path(os.getenv("STORAGE_DIR", ROOT_DIR / "storage"))
    duckdb_path: Path = storage_dir / "metadata.duckdb"
    faiss_text_index_path: Path = storage_dir / "text.index"
    faiss_image_index_path: Path = storage_dir / "image.index"

    # Embeddings (OpenCLIP)
    clip_model_name: str = os.getenv("CLIP_MODEL_NAME", "ViT-B-32")
    clip_pretrained: str = os.getenv("CLIP_PRETRAINED", "laion2b_s34b_b79k")
    embedding_dim: int = int(os.getenv("EMBEDDING_DIM", "512"))
    device: str = os.getenv("EMBEDDING_DEVICE", "auto")  # auto|cuda|cpu

    # Chunking
    chunk_size_tokens: int = int(os.getenv("CHUNK_SIZE_TOKENS", "220"))
    chunk_overlap_tokens: int = int(os.getenv("CHUNK_OVERLAP_TOKENS", "40"))

    # Captioning / OCR
    enable_captioning: bool = _bool("ENABLE_CAPTIONING", True)
    enable_ocr: bool = _bool("ENABLE_OCR", True)
    caption_model_name: str = os.getenv(
        "CAPTION_MODEL_NAME", "Salesforce/blip-image-captioning-base"
    )

    # Retrieval
    default_top_k: int = int(os.getenv("DEFAULT_TOP_K", "10"))
    rerank_top_k: int = int(os.getenv("RERANK_TOP_K", "5"))

    # Generation (Ollama)
    ollama_host: str = os.getenv("OLLAMA_HOST", "http://localhost:11434")
    llm_model: str = os.getenv("LLM_MODEL", "llama3.2:3b")
    llm_fallback_model: str = os.getenv("LLM_FALLBACK_MODEL", "qwen2.5:latest")
    generation_timeout_s: float = float(os.getenv("GENERATION_TIMEOUT_S", "60"))

    # API
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", "8000"))


settings = Settings()
settings.data_dir.mkdir(parents=True, exist_ok=True)
settings.storage_dir.mkdir(parents=True, exist_ok=True)
