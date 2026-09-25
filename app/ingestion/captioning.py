"""Image captioning (BLIP) and OCR (Tesseract) — enrich images with searchable text.

CLIP embeddings capture visual *similarity*, not detailed scene understanding, and the
local LLM used for generation is text-only. So each ingested image also gets a short
generated caption (what's in the picture) and, when it contains legible text, OCR
output. Both are stored as the chunk's `text` field, which makes images: (a) reachable
by a pure text query via the shared CLIP space is still primary for retrieval, but
(b) inspectable/groundable in the generation context, where the LLM cannot look at
pixels directly. This is the "defensible image representation strategy" called for
in the project brief, chosen for being cheap enough to run on a 4GB-VRAM laptop.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Optional

from PIL import Image

from app.config.settings import settings


@lru_cache(maxsize=1)
def _get_caption_pipeline():
    from transformers import BlipForConditionalGeneration, BlipProcessor

    processor = BlipProcessor.from_pretrained(settings.caption_model_name)
    model = BlipForConditionalGeneration.from_pretrained(settings.caption_model_name)
    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    return processor, model, device


def generate_caption(image_path: str | Path) -> Optional[str]:
    if not settings.enable_captioning:
        return None
    try:
        processor, model, device = _get_caption_pipeline()
        import torch

        image = Image.open(image_path).convert("RGB")
        inputs = processor(image, return_tensors="pt").to(device)
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=30)
        return processor.decode(out[0], skip_special_tokens=True).strip()
    except Exception:
        return None


def run_ocr(image_path: str | Path) -> Optional[str]:
    if not settings.enable_ocr:
        return None
    try:
        import pytesseract

        text = pytesseract.image_to_string(Image.open(image_path)).strip()
        return text or None
    except Exception:
        return None
