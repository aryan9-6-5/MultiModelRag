import numpy as np

from app.retrieval.faiss_index import FaissIndex


def _random_unit_vectors(n, dim, seed=0):
    rng = np.random.default_rng(seed)
    vecs = rng.normal(size=(n, dim)).astype("float32")
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs


def test_add_and_search_returns_exact_match_first():
    index = FaissIndex(dim=16)
    vectors = _random_unit_vectors(10, 16)
    index.add(vectors)
    scores, ids = index.search(vectors[3], top_k=1)
    assert ids[0][0] == 3
    assert np.isclose(scores[0][0], 1.0, atol=1e-4)


def test_search_top_k_clamped_to_index_size():
    index = FaissIndex(dim=8)
    index.add(_random_unit_vectors(3, 8))
    scores, ids = index.search(_random_unit_vectors(1, 8)[0], top_k=100)
    assert ids.shape[1] == 3


def test_search_empty_index_returns_empty():
    index = FaissIndex(dim=8)
    scores, ids = index.search(_random_unit_vectors(1, 8)[0], top_k=5)
    assert ids.size == 0


def test_save_and_load_roundtrip(tmp_path):
    index = FaissIndex(dim=8)
    vectors = _random_unit_vectors(5, 8)
    index.add(vectors)
    path = tmp_path / "test.index"
    index.save(path)

    loaded = FaissIndex.load(path, dim=8)
    assert loaded.ntotal == 5
    scores, ids = loaded.search(vectors[2], top_k=1)
    assert ids[0][0] == 2
