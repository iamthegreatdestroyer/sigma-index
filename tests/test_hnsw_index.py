"""HNSW index tests — Sprint 2."""

import numpy as np
import pytest

from sigma_index.index import HNSWIndex


def _rand_vecs(n: int, dim: int = 384, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, dim)).astype(np.float32)


def test_hnsw_add_search():
    idx = HNSWIndex(dim=384)
    vecs = _rand_vecs(50)
    for i, v in enumerate(vecs):
        idx.add(v, f"doc_{i}")
    results = idx.search(vecs[0], k=10)
    assert len(results) == 10
    distances, doc_ids = zip(*results)
    # All distances should be in [0, 2] for cosine space
    assert all(0.0 <= d <= 2.0 for d in distances)
    assert all(isinstance(d, str) for d in doc_ids)


def test_hnsw_knn_accuracy():
    """Identical vector query must return itself as nearest neighbor."""
    idx = HNSWIndex(dim=384)
    vecs = _rand_vecs(20)
    for i, v in enumerate(vecs):
        idx.add(v, f"doc_{i}")

    query = vecs[7].copy()
    results = idx.search(query, k=5)
    assert len(results) >= 1
    nearest_doc = results[0][1]
    nearest_dist = results[0][0]
    assert nearest_doc == "doc_7", f"Expected doc_7, got {nearest_doc}"
    assert nearest_dist < 1e-4, f"Distance to identical vector: {nearest_dist}"


def test_hnsw_fallback_no_ryzanstein(monkeypatch):
    """With hnswlib unavailable, hash fallback must still return results."""
    monkeypatch.setenv("RYZANSTEIN_URL", "")

    idx = HNSWIndex(dim=384)
    idx._hnswlib_available = False  # force fallback path

    vecs = _rand_vecs(10)
    for i, v in enumerate(vecs):
        idx.add(v, f"doc_{i}")

    query = vecs[3].copy()
    results = idx.search(query, k=5)
    assert len(results) >= 1
    # Nearest should be itself in cosine fallback
    assert results[0][1] == "doc_3"


def test_hnsw_empty_search():
    idx = HNSWIndex(dim=384)
    query = _rand_vecs(1)[0]
    results = idx.search(query, k=5)
    assert results == []


def test_hnsw_k_capped_at_n():
    idx = HNSWIndex(dim=8)
    vecs = _rand_vecs(3, dim=8)
    for i, v in enumerate(vecs):
        idx.add(v, f"doc_{i}")
    results = idx.search(vecs[0], k=10)
    assert len(results) <= 3
