"""OpenCLIP-based multimodal embedder. Text and images share one vector space."""
from __future__ import annotations

from functools import lru_cache
from typing import Union

import numpy as np
import open_clip
import torch
from PIL import Image

from app.config.settings import settings


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    return "cuda" if torch.cuda.is_available() else "cpu"


class ClipEmbedder:
    """Wraps an OpenCLIP model to embed text and images into one shared space."""

    def __init__(self, model_name: str | None = None, pretrained: str | None = None, device: str | None = None):
        self.model_name = model_name or settings.clip_model_name
        self.pretrained = pretrained or settings.clip_pretrained
        self.device = _resolve_device(device or settings.device)

        model, _, preprocess = open_clip.create_model_and_transforms(
            self.model_name, pretrained=self.pretrained, device=self.device
        )
        self.model = model.eval()
        self.preprocess = preprocess
        self.tokenizer = open_clip.get_tokenizer(self.model_name)
        with torch.no_grad():
            self.embedding_dim = self.model.encode_text(
                self.tokenizer(["dimension probe"]).to(self.device)
            ).shape[-1]

    @torch.no_grad()
    def embed_text(self, texts: Union[str, list[str]]) -> np.ndarray:
        single = isinstance(texts, str)
        texts_list = [texts] if single else list(texts)
        tokens = self.tokenizer(texts_list).to(self.device)
        features = self.model.encode_text(tokens)
        features = features / features.norm(dim=-1, keepdim=True)
        arr = features.cpu().numpy().astype("float32")
        return arr[0] if single else arr

    @torch.no_grad()
    def embed_image(self, images: Union[str, Image.Image, list]) -> np.ndarray:
        single = not isinstance(images, list)
        images_list = [images] if single else images
        pil_images = [
            Image.open(im).convert("RGB") if isinstance(im, str) else im.convert("RGB")
            for im in images_list
        ]
        batch = torch.stack([self.preprocess(im) for im in pil_images]).to(self.device)
        features = self.model.encode_image(batch)
        features = features / features.norm(dim=-1, keepdim=True)
        arr = features.cpu().numpy().astype("float32")
        return arr[0] if single else arr


@lru_cache(maxsize=1)
def get_embedder() -> ClipEmbedder:
    """Process-wide singleton so the model is loaded once."""
    return ClipEmbedder()
