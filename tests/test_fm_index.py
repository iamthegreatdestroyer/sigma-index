"""FM-index tests — Sprint 1."""

import random
import string
import time

import pytest

from sigma_index.index import FMIndex


def test_fm_exact_match():
    fm = FMIndex()
    fm.build("banana")
    positions = fm.search("ana")
    assert sorted(positions) == [1, 3]


def test_fm_count():
    fm = FMIndex()
    fm.build("banana")
    assert fm.count("na") == 2
    assert fm.count("a") == 3
    assert fm.count("b") == 1


def test_fm_not_found():
    fm = FMIndex()
    fm.build("banana")
    assert fm.search("xyz") == []
    assert fm.count("xyz") == 0


def test_fm_single_char():
    fm = FMIndex()
    fm.build("aababaa")
    positions = fm.search("a")
    assert len(positions) == 5


def test_fm_full_string():
    text = "hello"
    fm = FMIndex()
    fm.build(text)
    positions = fm.search("hello")
    assert 0 in positions


def test_fm_large_corpus():
    """100KB corpus — 10 pattern searches must all complete under 2s."""
    random.seed(42)
    # Build a realistic corpus with repeated patterns
    base = "".join(random.choices(string.ascii_lowercase + " \n", k=90_000))
    # Embed known patterns
    patterns = [
        "sigma", "index", "search", "def run", "class Foo",
        "return val", "import os", "for i in", "while true", "async def",
    ]
    parts = [base]
    for p in patterns:
        insert_at = random.randint(0, len(base) - 1)
        parts.append(p)
    corpus = "".join(parts)

    fm = FMIndex()
    t0 = time.time()
    fm.build(corpus)
    build_time = time.time() - t0
    assert build_time < 30, f"Build took {build_time:.1f}s"

    t1 = time.time()
    for p in patterns:
        results = fm.search(p)
        assert len(results) >= 1, f"Pattern '{p}' not found"
    search_time = time.time() - t1
    assert search_time < 2.0, f"10 searches took {search_time:.2f}s"


def test_fm_empty_pattern():
    fm = FMIndex()
    fm.build("hello")
    assert fm.search("") == []
    assert fm.count("") == 0


def test_fm_not_built():
    fm = FMIndex()
    assert fm.search("x") == []
    assert fm.count("x") == 0


def test_fm_sentinel_in_build():
    fm = FMIndex()
    fm.build("banana$")  # already has sentinel
    assert fm.count("na") == 2
