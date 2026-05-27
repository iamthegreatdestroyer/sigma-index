"""sigma-index: FM-index exact search + HNSW semantic search for codebases."""

from __future__ import annotations

import logging
import os
import pathlib
import pickle
import re
from dataclasses import dataclass
from typing import Optional

import numpy as np

from sigma_index.index import FMIndex, HNSWIndex, InvertedIndex, tokenize

logger = logging.getLogger(__name__)

__all__ = ["SigmaIndex", "SearchResult", "InvertedIndex"]

# ── Embedding helpers ────────────────────────────────────────────────────────

_VOCAB: dict[str, int] = {}   # built lazily on first embed call
_DIM = 384


def _tfidf_vector(text: str, dim: int = _DIM) -> np.ndarray:
    """Deterministic TF-IDF-style sparse projection to a dense vector."""
    global _VOCAB
    tokens = tokenize(text)
    vec = np.zeros(dim, dtype=np.float32)
    for tok in tokens:
        if tok not in _VOCAB:
            _VOCAB[tok] = len(_VOCAB) % dim
        idx = _VOCAB[tok]
        vec[idx] += 1.0
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


def _ryzanstein_embed(text: str) -> Optional[np.ndarray]:
    """Call Ryzanstein /v1/embeddings. Returns None if unavailable."""
    url = os.environ.get("RYZANSTEIN_URL", "").rstrip("/")
    if not url:
        return None
    try:
        import urllib.request, json
        payload = json.dumps({"input": text}).encode()
        req = urllib.request.Request(
            f"{url}/v1/embeddings",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read())
            return np.array(data["data"][0]["embedding"], dtype=np.float32)
    except Exception as exc:
        logger.debug("Ryzanstein embed failed: %s", exc)
        return None


def _embed(text: str) -> np.ndarray:
    vec = _ryzanstein_embed(text)
    return vec if vec is not None else _tfidf_vector(text)


# ── Search result ────────────────────────────────────────────────────────────

@dataclass
class SearchResult:
    file: str
    line: int
    snippet: str
    score: float
    mode: str  # "exact" | "semantic" | "hybrid"


# ── Code chunking ────────────────────────────────────────────────────────────

_CHUNK_RE = re.compile(r"^(?:def |class |func )", re.MULTILINE)


def _split_chunks(text: str) -> list[tuple[int, str]]:
    """Return [(start_line, chunk_text)] split on function/class boundaries."""
    lines = text.splitlines(keepends=True)
    boundaries = [0]
    for i, line in enumerate(lines):
        if _CHUNK_RE.match(line) and i > 0:
            boundaries.append(i)
    boundaries.append(len(lines))
    chunks = []
    for a, b in zip(boundaries, boundaries[1:]):
        chunk = "".join(lines[a:b])
        if chunk.strip():
            chunks.append((a, chunk))
    return chunks if chunks else [(0, text)]


# ── SigmaIndex ───────────────────────────────────────────────────────────────

class SigmaIndex:
    """Unified code search engine combining FM-index (exact) and HNSW (semantic).

    Usage::

        idx = SigmaIndex()
        idx.index_directory("./src")
        results = idx.search("def compress", mode="hybrid")
    """

    def __init__(self, config: Optional[dict] = None) -> None:
        self._fm = FMIndex()
        self._hnsw = HNSWIndex(dim=_DIM)
        # file_text maps file path -> full text (for snippet recovery)
        self._file_text: dict[str, str] = {}
        # fm_offset maps file path -> char offset in the concatenated FM corpus
        self._fm_offsets: list[tuple[int, str]] = []  # [(start_char, file_path)]
        self._fm_corpus: str = ""
        self._fm_dirty = True  # corpus needs rebuild

    # ── Indexing ──────────────────────────────────────────────────────────────

    def index_file(self, path: str) -> None:
        abs_path = str(pathlib.Path(path).resolve())
        text = pathlib.Path(abs_path).read_text(encoding="utf-8", errors="replace")
        self._file_text[abs_path] = text
        self._fm_dirty = True

        chunks = _split_chunks(text)
        for start_line, chunk in chunks:
            doc_id = f"{abs_path}:{start_line}"
            vec = _embed(chunk)
            self._hnsw.add(vec, doc_id)

    def index_directory(self, path: str, glob_pattern: str = "**/*.py") -> int:
        count = 0
        # rglob prepends its own "**/" so strip any leading "**/" or "**\" prefix
        rel = re.sub(r"^\*\*[/\\]", "", glob_pattern)
        for fp in pathlib.Path(path).rglob(rel):
            try:
                self.index_file(str(fp))
                count += 1
            except Exception as exc:
                logger.warning("Skipping %s: %s", fp, exc)
        return count

    def _ensure_fm(self) -> None:
        """Rebuild FM-index from all indexed files."""
        if not self._fm_dirty:
            return
        parts: list[str] = []
        offsets: list[tuple[int, str]] = []
        pos = 0
        for file_path, text in self._file_text.items():
            offsets.append((pos, file_path))
            # Separate files with a newline sentinel so patterns don't span files
            parts.append(text + "\n")
            pos += len(text) + 1
        corpus = "".join(parts)
        self._fm_corpus = corpus
        self._fm_offsets = offsets
        self._fm.build(corpus)
        self._fm_dirty = False

    # ── Search ────────────────────────────────────────────────────────────────

    def search_exact(self, pattern: str) -> list[SearchResult]:
        self._ensure_fm()
        positions = self._fm.search(pattern)
        results: list[SearchResult] = []
        seen: set[tuple[str, int]] = set()
        for char_pos in positions:
            file_path, line_no, snippet = self._resolve_char_pos(char_pos, pattern)
            if file_path and (file_path, line_no) not in seen:
                seen.add((file_path, line_no))
                results.append(SearchResult(
                    file=file_path,
                    line=line_no,
                    snippet=snippet,
                    score=1.0,
                    mode="exact",
                ))
        return results

    def search_semantic(self, query: str, k: int = 10) -> list[SearchResult]:
        vec = _embed(query)
        hits = self._hnsw.search(vec, k=k)
        results: list[SearchResult] = []
        for dist, doc_id in hits:
            file_path, start_line = self._parse_doc_id(doc_id)
            if file_path not in self._file_text:
                continue
            snippet = self._snippet_at_line(file_path, start_line)
            results.append(SearchResult(
                file=file_path,
                line=start_line,
                snippet=snippet,
                score=max(0.0, 1.0 - dist),
                mode="semantic",
            ))
        return results

    def search(self, query: str, mode: str = "hybrid") -> list[SearchResult]:
        if mode == "exact":
            return self.search_exact(query)
        if mode == "semantic":
            return self.search_semantic(query)
        # hybrid: both, deduplicated by (file, line)
        exact = self.search_exact(query)
        semantic = self.search_semantic(query)
        seen: set[tuple[str, int]] = set()
        merged: list[SearchResult] = []
        for r in exact:
            key = (r.file, r.line)
            if key not in seen:
                seen.add(key)
                merged.append(SearchResult(
                    file=r.file, line=r.line, snippet=r.snippet,
                    score=r.score, mode="hybrid",
                ))
        for r in semantic:
            key = (r.file, r.line)
            if key not in seen:
                seen.add(key)
                merged.append(SearchResult(
                    file=r.file, line=r.line, snippet=r.snippet,
                    score=r.score, mode="hybrid",
                ))
        return merged

    # ── Persistence ───────────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        self._ensure_fm()
        state = {
            "fm": self._fm,
            "hnsw": self._hnsw,
            "file_text": self._file_text,
            "fm_offsets": self._fm_offsets,
            "fm_corpus": self._fm_corpus,
        }
        with open(path, "wb") as fh:
            pickle.dump(state, fh, protocol=pickle.HIGHEST_PROTOCOL)

    def load(self, path: str) -> None:
        with open(path, "rb") as fh:
            state = pickle.load(fh)
        self._fm = state["fm"]
        self._hnsw = state["hnsw"]
        self._file_text = state["file_text"]
        self._fm_offsets = state["fm_offsets"]
        self._fm_corpus = state["fm_corpus"]
        self._fm_dirty = False

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _resolve_char_pos(self, char_pos: int, pattern: str) -> tuple[str, int, str]:
        """Map char position in FM corpus to (file_path, line_number, snippet)."""
        file_path = ""
        file_start = 0
        for start, fp in reversed(self._fm_offsets):
            if char_pos >= start:
                file_path = fp
                file_start = start
                break
        if not file_path:
            return "", 0, ""
        local_pos = char_pos - file_start
        text = self._file_text.get(file_path, "")
        if local_pos >= len(text):
            return file_path, 0, ""
        before = text[:local_pos]
        line_no = before.count("\n")
        lines = text.splitlines()
        start_line = line_no
        end_line = min(len(lines), line_no + 3)
        snippet = "\n".join(lines[start_line:end_line])[:120]
        return file_path, line_no, snippet

    def _parse_doc_id(self, doc_id: str) -> tuple[str, int]:
        """Parse doc_id formatted as 'file_path:line_no'."""
        if ":" in doc_id:
            # find last colon that is followed by digits
            m = re.match(r"^(.*):(\d+)$", doc_id)
            if m:
                return m.group(1), int(m.group(2))
        return doc_id, 0

    def _snippet_at_line(self, file_path: str, line_no: int, context: int = 3) -> str:
        text = self._file_text.get(file_path, "")
        lines = text.splitlines()
        start = max(0, line_no)
        end = min(len(lines), line_no + context)
        return "\n".join(lines[start:end])[:120]
