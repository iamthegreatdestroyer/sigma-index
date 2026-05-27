"""Core index implementations: FM-index (exact) + HNSW (semantic) + TF-IDF (legacy)."""

from __future__ import annotations

import logging
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Tokenizer (shared) ──────────────────────────────────────────────────────

_SPLIT_RE = re.compile(r"[a-zA-Z_]\w*|[0-9]+")
_CAMEL_RE = re.compile(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")

_STOP_WORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "do", "does", "did", "have", "has", "had", "in", "on", "at",
    "to", "for", "of", "with", "by", "from", "as", "it", "this",
    "that", "not", "and", "or", "if", "else", "then", "than",
    "self", "none", "true", "false", "null",
})


def tokenize(text: str) -> list[str]:
    """Tokenize text — splits camelCase and snake_case, removes stop words.

    >>> tokenize("getUserName")
    ['get', 'user', 'name']
    >>> tokenize("http_response_code")
    ['http', 'response', 'code']
    """
    tokens: list[str] = []
    for word in _SPLIT_RE.findall(text):
        parts = _CAMEL_RE.sub("_", word).split("_")
        for part in parts:
            lower = part.lower()
            if lower and lower not in _STOP_WORDS and len(lower) > 1:
                tokens.append(lower)
    return tokens


# ── FM-Index ────────────────────────────────────────────────────────────────

_SAMPLE_RATE = 32  # sample every 32 positions in suffix array


def _build_suffix_array(text: str) -> list[int]:
    """O(n log^2 n) suffix array construction."""
    n = len(text)
    sa = list(range(n))
    rank = [ord(c) for c in text]
    tmp = [0] * n
    k = 1
    while k < n:
        key_rank = rank[:]

        def sort_key(i: int) -> tuple[int, int]:
            r2 = rank[i + k] if i + k < n else -1
            return (key_rank[i], r2)

        sa.sort(key=sort_key)
        tmp[sa[0]] = 0
        for j in range(1, n):
            tmp[sa[j]] = tmp[sa[j - 1]]
            if sort_key(sa[j]) != sort_key(sa[j - 1]):
                tmp[sa[j]] += 1
        rank = tmp[:]
        if rank[sa[-1]] == n - 1:
            break
        k <<= 1
    return sa


class FMIndex:
    """FM-index for O(m) exact pattern matching via backward search.

    Build once, search many times. The text must be terminated with a
    sentinel character ('$' by default) that is lexicographically smaller
    than every character in the alphabet.
    """

    def __init__(self) -> None:
        self._bwt: str = ""
        self._sa_sample: dict[int, int] = {}   # bwt-row -> original position
        self._c_table: dict[str, int] = {}      # C[c]: # chars < c in text
        self._occ: dict[str, list[int]] = {}    # sampled occ[c][i//rate] = rank(c, i)
        self._n: int = 0
        self._built: bool = False

    def build(self, text: str) -> None:
        """Build FM-index from text. Appends '$' sentinel if not present."""
        if not text.endswith("$"):
            text = text + "$"
        n = len(text)
        self._n = n

        sa = _build_suffix_array(text)

        # BWT: character preceding each suffix
        bwt = "".join(text[(sa[i] - 1) % n] for i in range(n))
        self._bwt = bwt

        # Sampled suffix array: every _SAMPLE_RATE-th row
        self._sa_sample = {i: sa[i] for i in range(0, n, _SAMPLE_RATE)}

        # C table: for each char c, number of chars in text that are < c
        freq: dict[str, int] = defaultdict(int)
        for ch in text:
            freq[ch] += 1
        chars = sorted(freq)
        c_table: dict[str, int] = {}
        running = 0
        for ch in chars:
            c_table[ch] = running
            running += freq[ch]
        self._c_table = c_table

        # Occ table (sampled): occ[c][k] = rank of c in bwt[0 .. k*rate - 1]
        occ: dict[str, list[int]] = {ch: [] for ch in chars}
        counts: dict[str, int] = {ch: 0 for ch in chars}
        for i, ch in enumerate(bwt):
            if i % _SAMPLE_RATE == 0:
                for c in chars:
                    occ[c].append(counts[c])
            counts[ch] += 1
        # Append final counts
        for c in chars:
            occ[c].append(counts[c])
        self._occ = occ

        self._built = True

    def _occ_rank(self, c: str, i: int) -> int:
        """Count occurrences of c in bwt[0..i] (inclusive)."""
        if c not in self._occ:
            return 0
        occ_list = self._occ[c]
        block = i // _SAMPLE_RATE
        count = occ_list[block]
        # Walk forward from block*rate to i
        start = block * _SAMPLE_RATE
        for j in range(start, min(i + 1, self._n)):
            if self._bwt[j] == c:
                count += 1
        return count

    def _backward_search(self, pattern: str) -> tuple[int, int]:
        """Return (lo, hi) BWT row range for pattern. hi is exclusive."""
        lo, hi = 0, self._n
        for ch in reversed(pattern):
            if ch not in self._c_table:
                return 0, 0
            c = self._c_table[ch]
            lo = c + (self._occ_rank(ch, lo - 1) if lo > 0 else 0)
            hi = c + self._occ_rank(ch, hi - 1)
            if lo >= hi:
                return 0, 0
        return lo, hi

    def _resolve_position(self, row: int) -> int:
        """Walk BWT rows via LF-mapping until we hit a sampled row."""
        steps = 0
        r = row
        while r not in self._sa_sample:
            ch = self._bwt[r]
            r = self._c_table[ch] + (self._occ_rank(ch, r - 1) if r > 0 else 0)
            steps += 1
        return (self._sa_sample[r] + steps) % self._n

    def search(self, pattern: str) -> list[int]:
        """Return all starting positions of pattern in the original text."""
        if not self._built or not pattern:
            return []
        lo, hi = self._backward_search(pattern)
        return sorted(self._resolve_position(r) for r in range(lo, hi))

    def count(self, pattern: str) -> int:
        """Return count of pattern occurrences."""
        if not self._built or not pattern:
            return 0
        lo, hi = self._backward_search(pattern)
        return max(0, hi - lo)


# ── HNSW Index ──────────────────────────────────────────────────────────────

class HNSWIndex:
    """Approximate nearest-neighbor index using hnswlib (cosine space).

    Falls back to hash-based nearest neighbor when hnswlib is unavailable
    or when no vectors have been added yet. Never raises on search.
    """

    def __init__(self, dim: int = 384, M: int = 16, ef_construction: int = 200) -> None:
        self._dim = dim
        self._M = M
        self._ef_construction = ef_construction
        self._id_map: dict[int, str] = {}
        self._label_counter = 0
        self._initialized = False
        self._index = None
        self._fallback_store: list[tuple[np.ndarray, str]] = []  # (vec, doc_id)

        try:
            import hnswlib  # noqa: F401
            self._hnswlib_available = True
        except ImportError:
            logger.warning("hnswlib not available — using hash-based fallback")
            self._hnswlib_available = False

    def _init_index(self, max_elements: int = 10_000) -> None:
        import hnswlib
        self._index = hnswlib.Index(space="cosine", dim=self._dim)
        self._index.init_index(
            max_elements=max_elements,
            ef_construction=self._ef_construction,
            M=self._M,
        )
        self._initialized = True

    def add(self, vector: np.ndarray, doc_id: str) -> None:
        vec = np.array(vector, dtype=np.float32).reshape(1, -1)
        if not self._hnswlib_available:
            self._fallback_store.append((vec[0], doc_id))
            return

        if not self._initialized:
            self._init_index()

        # Grow index if needed
        current = self._index.get_current_count()
        max_el = self._index.get_max_elements()
        if current >= max_el:
            self._index.resize_index(max_el * 2)

        label = self._label_counter
        self._index.add_items(vec, [label])
        self._id_map[label] = doc_id
        self._label_counter += 1

    def search(self, query: np.ndarray, k: int = 10) -> list[tuple[float, str]]:
        """Return [(distance, doc_id)] sorted by distance ascending."""
        if not self._hnswlib_available or not self._initialized:
            return self._fallback_search(query, k)

        n = self._index.get_current_count()
        if n == 0:
            return []

        k_actual = min(k, n)
        q = np.array(query, dtype=np.float32).reshape(1, -1)
        try:
            labels, distances = self._index.knn_query(q, k=k_actual)
            return [(float(d), self._id_map[int(lbl)])
                    for lbl, d in zip(labels[0], distances[0])]
        except Exception as exc:
            logger.warning("HNSW search error: %s — falling back", exc)
            return self._fallback_search(query, k)

    def _fallback_search(self, query: np.ndarray, k: int) -> list[tuple[float, str]]:
        """Hash-based approximate NN — always returns best-effort results."""
        if not self._fallback_store:
            return []
        q = np.array(query, dtype=np.float32)
        norm_q = np.linalg.norm(q)
        if norm_q == 0:
            return [(float(abs(hash(doc_id) % 1000) / 1000.0), doc_id)
                    for _, doc_id in self._fallback_store[:k]]
        scored: list[tuple[float, str]] = []
        for vec, doc_id in self._fallback_store:
            norm_v = np.linalg.norm(vec)
            if norm_v == 0:
                dist = 1.0
            else:
                cosine_sim = float(np.dot(q, vec) / (norm_q * norm_v))
                dist = 1.0 - cosine_sim
            scored.append((dist, doc_id))
        scored.sort(key=lambda x: x[0])
        return scored[:k]


# ── TF-IDF Inverted Index (retained for compatibility) ──────────────────────

@dataclass
class SearchResult:
    """A single search result with TF-IDF score."""

    doc_id: str
    score: float
    snippet: str = ""
    positions: list[int] = field(default_factory=list)


@dataclass
class _Document:
    doc_id: str
    content: str
    tokens: list[str]
    token_count: int
    metadata: dict[str, str] = field(default_factory=dict)


class InvertedIndex:
    """TF-IDF inverted index for code and document search.

    Example:
        >>> idx = InvertedIndex()
        >>> idx.add_document("server.py", "def start_server(): ...")
        >>> idx.add_document("client.py", "def connect_client(): ...")
        >>> results = idx.search("server start", top_k=5)
        >>> results[0].doc_id
        'server.py'
    """

    def __init__(self) -> None:
        self._docs: dict[str, _Document] = {}
        self._posting: dict[str, dict[str, list[int]]] = defaultdict(dict)
        self._idf_cache: dict[str, float] = {}
        self._idf_dirty = True

    @property
    def doc_count(self) -> int:
        return len(self._docs)

    def add_document(
        self,
        doc_id: str,
        content: str,
        metadata: Optional[dict[str, str]] = None,
    ) -> None:
        if doc_id in self._docs:
            self.remove_document(doc_id)
        tokens = tokenize(content)
        doc = _Document(
            doc_id=doc_id,
            content=content,
            tokens=tokens,
            token_count=len(tokens),
            metadata=metadata or {},
        )
        self._docs[doc_id] = doc
        for pos, token in enumerate(tokens):
            if doc_id not in self._posting[token]:
                self._posting[token][doc_id] = []
            self._posting[token][doc_id].append(pos)
        self._idf_dirty = True

    def remove_document(self, doc_id: str) -> bool:
        doc = self._docs.pop(doc_id, None)
        if doc is None:
            return False
        for token in set(doc.tokens):
            if token in self._posting:
                self._posting[token].pop(doc_id, None)
                if not self._posting[token]:
                    del self._posting[token]
        self._idf_dirty = True
        return True

    def search(
        self,
        query: str,
        top_k: int = 10,
        doc_filter: Optional[set[str]] = None,
    ) -> list[SearchResult]:
        query_tokens = tokenize(query)
        if not query_tokens:
            return []
        self._ensure_idf()
        scores: dict[str, float] = defaultdict(float)
        hit_positions: dict[str, list[int]] = defaultdict(list)
        for token in query_tokens:
            if token not in self._posting:
                continue
            idf = self._idf_cache.get(token, 0.0)
            for doc_id, positions in self._posting[token].items():
                if doc_filter and doc_id not in doc_filter:
                    continue
                doc = self._docs[doc_id]
                tf = 1.0 + math.log(len(positions)) if positions else 0.0
                norm_tf = tf / (1.0 + math.log(doc.token_count)) if doc.token_count > 0 else 0.0
                scores[doc_id] += norm_tf * idf
                hit_positions[doc_id].extend(positions)
        results: list[SearchResult] = []
        for doc_id, score in scores.items():
            doc = self._docs[doc_id]
            snippet = _extract_snippet(doc.content, hit_positions.get(doc_id, []), doc.tokens)
            results.append(SearchResult(doc_id=doc_id, score=score, snippet=snippet,
                                        positions=sorted(set(hit_positions.get(doc_id, [])))))
        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    def _ensure_idf(self) -> None:
        if not self._idf_dirty:
            return
        n = len(self._docs)
        if n == 0:
            self._idf_cache = {}
            self._idf_dirty = False
            return
        self._idf_cache = {}
        for token, postings in self._posting.items():
            df = len(postings)
            self._idf_cache[token] = math.log((n + 1) / (df + 1)) + 1.0
        self._idf_dirty = False

    def get_document(self, doc_id: str) -> Optional[str]:
        doc = self._docs.get(doc_id)
        return doc.content if doc else None

    def get_stats(self) -> dict[str, int]:
        return {
            "documents": len(self._docs),
            "unique_tokens": len(self._posting),
            "total_tokens": sum(d.token_count for d in self._docs.values()),
        }


def _extract_snippet(
    content: str, positions: list[int], tokens: list[str], context_chars: int = 120
) -> str:
    if not positions or not content:
        return content[:context_chars] if content else ""
    lines = content.splitlines()
    if not lines:
        return ""
    first_pos = min(positions)
    token_count = 0
    for i, line in enumerate(lines):
        line_tokens = tokenize(line)
        token_count += len(line_tokens)
        if token_count > first_pos:
            start = max(0, i - 1)
            end = min(len(lines), i + 2)
            snippet = "\n".join(lines[start:end])
            if len(snippet) > context_chars:
                snippet = snippet[:context_chars] + "..."
            return snippet
    return content[:context_chars]
