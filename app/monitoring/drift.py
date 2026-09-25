"""Embedding-distribution drift detection.

Method: for each embedding dimension, run a two-sample Kolmogorov-Smirnov test
between a REFERENCE distribution (embeddings captured at index-build time) and a
CURRENT distribution (embeddings from recent queries/ingested chunks). KS was chosen
over PSI/mean-variance because CLIP embedding dimensions are continuous and roughly
unimodal per-dimension, which is exactly what KS is built for, and it makes no
assumption about the underlying distribution shape (unlike a mean/variance-only
check, which can miss a shape shift that leaves the mean unchanged). PSI is more
suited to binned/categorical features (e.g. score buckets) and is used here instead
for a coarser aggregate the summary drift score.

We reduce the per-dimension KS results to a single scalar CURRENT_DRIFT_SCORE (the
fraction of dimensions whose KS test rejects the null hypothesis of "same
distribution" at the configured p-value threshold), and separately report mean
cosine-similarity-to-centroid shift as a second, cheaper signal. An alert fires when
CURRENT_DRIFT_SCORE crosses `alert_fraction_threshold`.

This module detects drift; it does NOT remediate it (no automatic re-embedding or
index rebuild is triggered — that would need to be a human/ops decision).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import stats


@dataclass
class DriftReport:
    n_dimensions: int
    n_dimensions_drifted: int
    drift_fraction: float
    mean_ks_statistic: float
    centroid_cosine_shift: float
    alert: bool
    per_dimension_p_values: list[float] = field(default_factory=list)


def compute_drift(
    reference: np.ndarray,
    current: np.ndarray,
    p_value_threshold: float = 0.05,
    alert_fraction_threshold: float = 0.10,
) -> DriftReport:
    """reference, current: (n_samples, n_dims) float arrays of L2-normalized embeddings."""
    if reference.ndim != 2 or current.ndim != 2:
        raise ValueError("reference and current must be 2D arrays (n_samples, n_dims)")
    if reference.shape[1] != current.shape[1]:
        raise ValueError("reference and current must have the same embedding dimension")
    if len(reference) < 2 or len(current) < 2:
        raise ValueError("Need at least 2 samples in each distribution to run a KS test")

    n_dims = reference.shape[1]
    p_values = []
    ks_stats = []
    for d in range(n_dims):
        ks_stat, p_value = stats.ks_2samp(reference[:, d], current[:, d])
        ks_stats.append(ks_stat)
        p_values.append(p_value)

    p_values_arr = np.array(p_values)
    n_drifted = int(np.sum(p_values_arr < p_value_threshold))
    drift_fraction = n_drifted / n_dims

    ref_centroid = reference.mean(axis=0)
    cur_centroid = current.mean(axis=0)
    ref_centroid_n = ref_centroid / (np.linalg.norm(ref_centroid) + 1e-12)
    cur_centroid_n = cur_centroid / (np.linalg.norm(cur_centroid) + 1e-12)
    cosine_sim = float(np.dot(ref_centroid_n, cur_centroid_n))
    centroid_cosine_shift = 1.0 - cosine_sim

    return DriftReport(
        n_dimensions=n_dims,
        n_dimensions_drifted=n_drifted,
        drift_fraction=drift_fraction,
        mean_ks_statistic=float(np.mean(ks_stats)),
        centroid_cosine_shift=centroid_cosine_shift,
        alert=drift_fraction >= alert_fraction_threshold,
        per_dimension_p_values=p_values,
    )
