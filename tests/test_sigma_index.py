"""SigmaIndex unified API tests — Sprint 3."""

import os
import pathlib
import tempfile

import numpy as np
import pytest

from sigma_index import SigmaIndex, SearchResult


def _write_file(directory: pathlib.Path, name: str, content: str) -> pathlib.Path:
    p = directory / name
    p.write_text(content, encoding="utf-8")
    return p


def test_index_file_and_exact_search(tmp_path):
    p = _write_file(tmp_path, "sample.py", "def hello_world():\n    return 42\n")
    idx = SigmaIndex()
    idx.index_file(str(p))
    results = idx.search_exact("hello_world")
    assert len(results) >= 1
    assert any("hello_world" in r.snippet for r in results)
    assert all(isinstance(r, SearchResult) for r in results)


def test_index_directory(tmp_path):
    _write_file(tmp_path, "a.py", "def foo(): pass\n")
    _write_file(tmp_path, "b.py", "def bar(): pass\n")
    sub = tmp_path / "sub"
    sub.mkdir()
    _write_file(sub, "c.py", "def baz(): pass\n")

    idx = SigmaIndex()
    count = idx.index_directory(str(tmp_path))
    assert count >= 2  # at least a.py and b.py


def test_search_exact_not_found(tmp_path):
    p = _write_file(tmp_path, "a.py", "def hello(): pass\n")
    idx = SigmaIndex()
    idx.index_file(str(p))
    results = idx.search_exact("zzz_nonexistent_zzz")
    assert results == []


def test_search_semantic(tmp_path):
    p = _write_file(tmp_path, "a.py", "def compress_data(buf):\n    return buf\n")
    idx = SigmaIndex()
    idx.index_file(str(p))
    results = idx.search_semantic("compress data", k=5)
    assert len(results) >= 1
    assert all(isinstance(r, SearchResult) for r in results)


def test_search_hybrid(tmp_path):
    p = _write_file(tmp_path, "a.py", "def search_index():\n    pass\n")
    idx = SigmaIndex()
    idx.index_file(str(p))
    results = idx.search("search_index", mode="hybrid")
    assert len(results) >= 1
    assert all(r.mode == "hybrid" for r in results)


def test_search_mode_exact(tmp_path):
    p = _write_file(tmp_path, "a.py", "def exact_match(): pass\n")
    idx = SigmaIndex()
    idx.index_file(str(p))
    results = idx.search("exact_match", mode="exact")
    assert len(results) >= 1
    assert all(r.mode == "exact" for r in results)


def test_search_mode_semantic(tmp_path):
    p = _write_file(tmp_path, "a.py", "def semantic_search(): pass\n")
    idx = SigmaIndex()
    idx.index_file(str(p))
    results = idx.search("semantic search", mode="semantic")
    assert len(results) >= 1
    assert all(r.mode == "semantic" for r in results)


def test_save_and_load(tmp_path):
    p = _write_file(tmp_path, "a.py", "def round_trip(): pass\n")
    idx = SigmaIndex()
    idx.index_file(str(p))

    save_path = str(tmp_path / "index.pkl")
    idx.save(save_path)

    idx2 = SigmaIndex()
    idx2.load(save_path)

    results = idx2.search_exact("round_trip")
    assert len(results) >= 1


def test_search_result_fields(tmp_path):
    p = _write_file(tmp_path, "fields.py", "def check_fields():\n    x = 1\n")
    idx = SigmaIndex()
    idx.index_file(str(p))
    results = idx.search_exact("check_fields")
    assert len(results) >= 1
    r = results[0]
    assert isinstance(r.file, str)
    assert isinstance(r.line, int)
    assert isinstance(r.snippet, str)
    assert isinstance(r.score, float)
    assert r.mode in ("exact", "semantic", "hybrid")


def test_hybrid_deduplication(tmp_path):
    """Hybrid results must not contain duplicate (file, line) pairs."""
    p = _write_file(tmp_path, "dup.py", "def deduplicate_me(): pass\n")
    idx = SigmaIndex()
    idx.index_file(str(p))
    results = idx.search("deduplicate_me", mode="hybrid")
    seen = set()
    for r in results:
        key = (r.file, r.line)
        assert key not in seen, f"Duplicate result: {key}"
        seen.add(key)
