import numpy as np
import pytest

from app.monitoring.drift import compute_drift


def _normalized(vecs: np.ndarray) -> np.ndarray:
    return vecs / np.linalg.norm(vecs, axis=1, keepdims=True)


def test_no_drift_when_same_distribution():
    rng = np.random.default_rng(0)
    reference = _normalized(rng.normal(size=(200, 16)).astype("float32"))
    current = _normalized(rng.normal(size=(200, 16)).astype("float32"))
    report = compute_drift(reference, current)
    assert report.alert is False
    assert report.drift_fraction < 0.3


def test_drift_detected_when_distribution_shifts():
    rng = np.random.default_rng(0)
    reference = _normalized(rng.normal(loc=0.0, size=(200, 16)).astype("float32"))
    current = _normalized(rng.normal(loc=5.0, size=(200, 16)).astype("float32"))
    report = compute_drift(reference, current)
    assert report.alert is True
    assert report.drift_fraction > 0.5
    assert report.centroid_cosine_shift > 0.0


def test_compute_drift_rejects_mismatched_dims():
    rng = np.random.default_rng(0)
    reference = rng.normal(size=(50, 16)).astype("float32")
    current = rng.normal(size=(50, 8)).astype("float32")
    with pytest.raises(ValueError):
        compute_drift(reference, current)


def test_compute_drift_rejects_too_few_samples():
    reference = np.zeros((1, 4), dtype="float32")
    current = np.zeros((1, 4), dtype="float32")
    with pytest.raises(ValueError):
        compute_drift(reference, current)
