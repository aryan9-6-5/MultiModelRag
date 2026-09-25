"""Quick environment sanity check: python -m scripts.check_environment
Verifies GPU/CUDA, Ollama reachability + required models, and Tesseract."""
from __future__ import annotations

import shutil
import subprocess
import sys


def check_torch_cuda() -> None:
    try:
        import torch

        print(f"torch {torch.__version__} | CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("torch not installed")


def check_ollama() -> None:
    from app.config.settings import settings
    from app.generation.llm_client import OllamaClient

    try:
        client = OllamaClient()
        models = client._available_models()
        print(f"Ollama reachable at {settings.ollama_host}. Models: {sorted(models)}")
        for required in (settings.llm_model, settings.llm_fallback_model):
            status = "OK" if required in models else "MISSING (run: ollama pull " + required + ")"
            print(f"  {required}: {status}")
    except Exception as exc:
        print(f"Ollama NOT reachable: {exc}")
        print("  Start it with: ollama serve")


def check_tesseract() -> None:
    path = shutil.which("tesseract")
    print(f"tesseract binary: {path or 'NOT FOUND (OCR will be skipped)'}")


def check_docker() -> None:
    path = shutil.which("docker")
    print(f"docker binary: {path or 'NOT FOUND'}")


if __name__ == "__main__":
    print(f"Python: {sys.version}")
    check_torch_cuda()
    check_ollama()
    check_tesseract()
    check_docker()
