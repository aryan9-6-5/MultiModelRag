import os
import numpy as np
import pytest

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


@pytest.fixture
def tmp_db(tmp_path):
    from app.database.db import Database

    db = Database(tmp_path / "test.duckdb")
    yield db
    db.close()


@pytest.fixture
def fake_embedder():
    """A deterministic, hash-based stand-in for ClipEmbedder — no model download,
    no GPU. Real embedding-quality behavior is covered separately by tests marked
    'requires_model' that load the actual OpenCLIP model."""

    class FakeEmbedder:
        embedding_dim = 32
        device = "cpu"

        def _vec(self, s: str) -> np.ndarray:
            rng = np.random.default_rng(abs(hash(s)) % (2**32))
            v = rng.normal(size=self.embedding_dim).astype("float32")
            return v / np.linalg.norm(v)

        def embed_text(self, texts):
            single = isinstance(texts, str)
            texts_list = [texts] if single else list(texts)
            arr = np.stack([self._vec("text:" + t) for t in texts_list])
            return arr[0] if single else arr

        def embed_image(self, images):
            single = not isinstance(images, list)
            images_list = [images] if single else images
            arr = np.stack([self._vec("image:" + str(im)) for im in images_list])
            return arr[0] if single else arr

    return FakeEmbedder()
