"""Inverted index with TF-IDF scoring for code and document search."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class SearchResult:
    """A single search result with TF-IDF score."""

    doc_id: str
    score: float
    snippet: str = ""
    positions: list[int] = field(default_factory=list)


@dataclass
class _Document:
    """Internal representation of an indexed document."""

    doc_id: str
    content: str
    tokens: list[str]
    token_count: int
    metadata: dict[str, str] = field(default_factory=dict)


# ── Tokenizer ──

_SPLIT_RE = re.compile(r"[a-zA-Z_]\w*|[0-9]+")
_CAMEL_RE = re.compile(r"(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")

# Common stop words to skip in code search
_STOP_WORDS = frozenset({
    "the", "a", "an", "is", "are", "was", "were", "be", "been",
    "do", "does", "did", "have", "has", "had", "in", "on", "at",
    "to", "for", "of", "with", "by", "from", "as", "it", "this",
    "that", "not", "and", "or", "if", "else", "then", "than",
    "self", "none", "true", "false", "null",
})


def tokenize(text: str) -> list[str]:
    """Tokenize text for indexing — splits camelCase and snake_case.

    >>> tokenize("getUserName")
    ['get', 'user', 'name']
    >>> tokenize("http_response_code")
    ['http', 'response', 'code']
    """
    tokens: list[str] = []
    for word in _SPLIT_RE.findall(text):
        # Split camelCase
        parts = _CAMEL_RE.sub("_", word).split("_")
        for part in parts:
            lower = part.lower()
            if lower and lower not in _STOP_WORDS and len(lower) > 1:
                tokens.append(lower)
    return tokens


class InvertedIndex:
    """TF-IDF inverted index for code and document search.

    Supports incremental updates — add and remove documents without
    rebuilding the entire index.

    Example:
        >>> idx = InvertedIndex()
        >>> idx.add_document("server.py", "def start_server(): ...")
        >>> idx.add_document("client.py", "def connect_client(): ...")
        >>> results = idx.search("server start", top_k=5)
        >>> results[0].doc_id
        'server.py'
    """

    def __init__(self) -> None:
        # doc_id -> _Document
        self._docs: dict[str, _Document] = {}
        # token -> {doc_id -> [positions]}
        self._posting: dict[str, dict[str, list[int]]] = defaultdict(dict)
        # Cached IDF values (invalidated on add/remove)
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
        """Add or update a document in the index."""
        # Remove old version if exists
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

        # Build posting list with positions
        for pos, token in enumerate(tokens):
            if doc_id not in self._posting[token]:
                self._posting[token][doc_id] = []
            self._posting[token][doc_id].append(pos)

        self._idf_dirty = True

    def remove_document(self, doc_id: str) -> bool:
        """Remove a document from the index. Returns True if found."""
        doc = self._docs.pop(doc_id, None)
        if doc is None:
            return False

        # Clean posting lists
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
        """Search the index using TF-IDF scoring.

        Args:
            query: Search query text.
            top_k: Maximum results to return.
            doc_filter: Optional set of doc_ids to restrict search to.

        Returns:
            List of SearchResult sorted by score (descending).
        """
        query_tokens = tokenize(query)
        if not query_tokens:
            return []

        self._ensure_idf()

        # Accumulate scores per document
        scores: dict[str, float] = defaultdict(float)
        hit_positions: dict[str, list[int]] = defaultdict(list)

        for token in query_tokens:
            if token not in self._posting:
                continue

            idf = self._idf_cache.get(token, 0.0)
            postings = self._posting[token]

            for doc_id, positions in postings.items():
                if doc_filter and doc_id not in doc_filter:
                    continue

                doc = self._docs[doc_id]
                # TF: term frequency with log normalization
                tf = 1.0 + math.log(len(positions)) if positions else 0.0
                # Normalize by document length
                norm_tf = tf / (1.0 + math.log(doc.token_count)) if doc.token_count > 0 else 0.0

                scores[doc_id] += norm_tf * idf
                hit_positions[doc_id].extend(positions)

        # Build results
        results: list[SearchResult] = []
        for doc_id, score in scores.items():
            doc = self._docs[doc_id]
            snippet = _extract_snippet(doc.content, hit_positions.get(doc_id, []), doc.tokens)
            results.append(
                SearchResult(
                    doc_id=doc_id,
                    score=score,
                    snippet=snippet,
                    positions=sorted(set(hit_positions.get(doc_id, []))),
                )
            )

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    def _ensure_idf(self) -> None:
        """Recompute IDF cache if dirty."""
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
            # IDF with smoothing
            self._idf_cache[token] = math.log((n + 1) / (df + 1)) + 1.0

        self._idf_dirty = False

    def get_document(self, doc_id: str) -> Optional[str]:
        """Retrieve original document content."""
        doc = self._docs.get(doc_id)
        return doc.content if doc else None

    def get_stats(self) -> dict[str, int]:
        """Return index statistics."""
        return {
            "documents": len(self._docs),
            "unique_tokens": len(self._posting),
            "total_tokens": sum(d.token_count for d in self._docs.values()),
        }


def _extract_snippet(
    content: str, positions: list[int], tokens: list[str], context_chars: int = 120
) -> str:
    """Extract a relevant snippet around the first hit position."""
    if not positions or not content:
        return content[:context_chars] if content else ""

    # Find the character offset of the first hit token
    lines = content.splitlines()
    if not lines:
        return ""

    # Use the first token position to find the relevant line
    first_pos = min(positions)
    # Map token position to approximate line
    token_count = 0
    for i, line in enumerate(lines):
        line_tokens = tokenize(line)
        token_count += len(line_tokens)
        if token_count > first_pos:
            # Return this line and neighbors
            start = max(0, i - 1)
            end = min(len(lines), i + 2)
            snippet = "\n".join(lines[start:end])
            if len(snippet) > context_chars:
                snippet = snippet[:context_chars] + "..."
            return snippet

    return content[:context_chars]
