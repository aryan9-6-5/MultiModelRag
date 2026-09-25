"""Sanity checks against the real OpenCLIP model — confirms text and image
embeddings actually land in a shared, meaningfully-similar space. Slower and
needs the model weights (auto-downloaded on first run), so marked accordingly."""
import numpy as np
import pytest
from PIL import Image

pytestmark = pytest.mark.requires_model


@pytest.fixture(scope="module")
def embedder():
    from app.embeddings.clip_embedder import ClipEmbedder

    return ClipEmbedder()


def test_embed_text_returns_unit_vector(embedder):
    vec = embedder.embed_text("a photo of a cat")
    assert vec.shape == (embedder.embedding_dim,)
    assert np.isclose(np.linalg.norm(vec), 1.0, atol=1e-3)


def test_embed_text_batch(embedder):
    vecs = embedder.embed_text(["a photo of a cat", "a photo of a dog"])
    assert vecs.shape == (2, embedder.embedding_dim)


def test_matching_image_and_text_are_more_similar_than_unrelated(embedder, tmp_path):
    red_image = Image.new("RGB", (224, 224), color=(220, 20, 20))
    red_path = tmp_path / "red.png"
    red_image.save(red_path)

    image_vec = embedder.embed_image(str(red_path))
    red_text_vec = embedder.embed_text("a solid red square")
    unrelated_text_vec = embedder.embed_text("a page of handwritten calculus notes")

    sim_related = float(np.dot(image_vec, red_text_vec))
    sim_unrelated = float(np.dot(image_vec, unrelated_text_vec))
    assert sim_related > sim_unrelated
