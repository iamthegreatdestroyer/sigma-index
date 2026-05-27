# sigma-index — Autonomous Completion Brief

## Project Identity
- **Repo:** `iamthegreatdestroyer/sigma-index`
- **Local path:** `S:\sigma-index`
- **Language:** Python (core) + Go (serving layer exists as reference)
- **Castle Layer:** Layer 4 — Storage & Inference (Code Search)
- **Current completion:** ~70%
- **Mission:** Succinct code search engine using FM-index for exact pattern matching + HNSW for approximate nearest-neighbor semantic search across large codebases

## Current State (verified 2026-05-25)
| Component | Status |
|-----------|--------|
| `src/sigma_index/__init__.py` | ✅ Exists |
| `src/sigma_index/index.py` — core index module | ✅ Skeleton exists |
| `src/distributed/` — distributed coordination | ✅ Done (from PHASE2_DEVELOPMENT) |
| `src/serving/` — API serving layer | ✅ Done (gRPC + REST) |
| `PHASE2_DEVELOPMENT/` — advanced cache/batch impl | ✅ Done |
| FM-index for exact pattern matching | ❌ Missing / incomplete |
| HNSW graph for semantic search | ❌ Missing |
| Query optimizer (batch search) | ❌ Missing |
| Index CLI | ❌ Missing |
| Integration with sigmalang corpus | ❌ Missing |

## Key File Map
```
sigma-index/
├── src/
│   ├── sigma_index/
│   │   ├── __init__.py       # Package init + public API
│   │   └── index.py          # Core index implementation (FM-index + HNSW)
│   ├── distributed/          # Distributed coordination layer
│   └── serving/              # gRPC + REST API server
├── PHASE2_DEVELOPMENT/
│   └── src/                  # Advanced cache/batching from Ryzanstein LLM ecosystem
├── tests/                    # Test suite
├── docs/                     # Architecture docs
└── configs/                  # Configuration files
```

## What Remains (Final 30%)

### Sprint 1 — FM-Index Implementation (Day 1)
**Goal:** FM-index for O(m log n) exact pattern search over tokenized code.

```
@APEX implement in src/sigma_index/index.py:
  class FMIndex:
    def __init__(self): ...
    def build(self, text: str) -> None:
      # 1. Compute suffix array (use pysuffixarray or implement SA-IS)
      # 2. Compute BWT (Burrows-Wheeler Transform)
      # 3. Build F column (sorted BWT)
      # 4. Build Occ table (rank/select structure using wavelet tree or sampled occ)
    def search(self, pattern: str) -> list[int]:
      # Backward search: O(m) using LF-mapping
      # Returns list of starting positions in original text
    def count(self, pattern: str) -> int:
      # O(m) count of occurrences

Optimize: store sampled suffix array (sample rate=32) to balance space/time.
Tests: test_fm_exact_match, test_fm_count, test_fm_not_found, test_fm_large_corpus.
Run: python -m pytest tests/test_fm_index.py -v
```

### Sprint 2 — HNSW Semantic Search (Day 1–2)
**Goal:** HNSW graph for approximate nearest-neighbor search using code embeddings.

```
@APEX implement in src/sigma_index/index.py:
  class HNSWIndex:
    def __init__(self, dim: int, M: int = 16, ef_construction: int = 200): ...
    def add(self, vector: np.ndarray, doc_id: str) -> None:
    def search(self, query: np.ndarray, k: int = 10) -> list[tuple[float, str]]:
      # Returns (distance, doc_id) pairs sorted by distance

Note: Use hnswlib (pip install hnswlib) rather than implementing from scratch.
  import hnswlib
  self._index = hnswlib.Index(space='cosine', dim=dim)

Wire: when Ryzanstein is available (RYZANSTEIN_URL env var), call /v1/embeddings
to get code block vectors. Fall back to hash-based nearest-neighbor if unavailable.

Tests: test_hnsw_add_search, test_hnsw_knn_accuracy, test_hnsw_fallback_no_ryzanstein.
```

### Sprint 3 — Query Optimizer + Unified Search (Day 2)
**Goal:** Combine FM-index (exact) + HNSW (semantic) into a unified SigmaIndex API.

```
@APEX create class SigmaIndex in src/sigma_index/__init__.py:
  class SigmaIndex:
    def __init__(self, config: dict = None): ...
    def index_file(self, path: str) -> None:
      # Tokenize file → add to FM-index (full text) + HNSW (per-function embeddings)
    def index_directory(self, path: str, glob: str = "**/*.py") -> int:
      # Returns count of indexed files
    def search_exact(self, pattern: str) -> list[SearchResult]:
      # FM-index exact match, returns file + line + context
    def search_semantic(self, query: str, k: int = 10) -> list[SearchResult]:
      # HNSW semantic search
    def search(self, query: str, mode: str = "hybrid") -> list[SearchResult]:
      # mode: "exact" | "semantic" | "hybrid" (both, deduplicated)
    def save(self, path: str) -> None:
    def load(self, path: str) -> None:

  @dataclass
  class SearchResult:
    file: str; line: int; snippet: str; score: float; mode: str

Add CLI: python -m sigma_index index ./src && python -m sigma_index search "def compress"
```

### Sprint 4 — Integration Test + Tag (Day 3)
```
@CORE run:
  pip install -e . --no-deps
  pip install pytest hnswlib numpy

  python -m pytest tests/ -v --timeout=60
  
  # Quick integration test:
  python -c "
  from sigma_index import SigmaIndex
  idx = SigmaIndex()
  idx.index_file('src/sigma_index/index.py')
  results = idx.search_exact('class')
  assert len(results) > 0, 'Expected results'
  print('Integration test passed:', len(results), 'results')
  "

Fix any failures. Then:
git tag v0.2.0 && git push origin v0.2.0
```

## Done Criteria (all must pass)
- [ ] `pytest tests/ -v` passes — zero failures
- [ ] FM-index: exact pattern search returns correct positions
- [ ] HNSW: semantic search returns k-nearest neighbors (falls back without Ryzanstein)
- [ ] `SigmaIndex.index_directory()` + `search()` hybrid mode works
- [ ] CLI: `python -m sigma_index index ./src` and `search "pattern"` work
- [ ] `pip install -e .` succeeds
- [ ] `v0.2.0` tag pushed

## Completion Signal
```bash
git tag v0.2.0 && git push origin v0.2.0
```

## Critical Rules
1. **Ryzanstein fallback** — semantic search must work without Ryzanstein (hash-based NN)
2. **FM-index must be lossless** — every occurrence of a pattern must be found; no false negatives
3. **Index must be serializable** — `save()` and `load()` must round-trip exactly
4. **Tests run offline** — no network calls required for test suite; mock Ryzanstein
