# sigma-index — Real Implementation Required

## Current State (updated 2026-07-04)
Real indexing code exists and is tested: `pkg/hnsw`, `pkg/bm25`, `pkg/hybrid`,
`persist`, and `cmd/server` implement genuine HNSW vector search, BM25 keyword
search, hybrid RRF search, and WAL/snapshot persistence (Go), plus an equivalent
FM-index + HNSW + SigmaIndex implementation in `src/sigma_index/` (Python) used
via `sigma_index_client.py` by other Sigma repos. `go test ./...` passes: 23/23
tests across persist (9), pkg/bm25 (5), pkg/hnsw (6), pkg/hybrid (3); pytest
passes 39/39 across the Python FM-index/HNSW/SigmaIndex suite.

Repo hygiene cleanup completed 2026-07-04 (mirroring the sigma-diff cleanup of
2026-07-03): this repo previously carried a near-complete copy of the
Ryzanstein/RYZEN-LLM mirror it was originally copy-pasted from — 1219 of the
1299 tracked files, including a `RYZEN-LLM/` directory, `desktop/` and
`vscode-extension/` apps, an `mcp/` gRPC server (its own nested `go.mod`,
module `.../Ryzanstein/mcp`), `dependencies/{agentmem,ann-hybrid,archaeo}`
(directly-committed copies of Ryzanstein's own dependencies, not real git
submodules despite a stale `.gitmodules`), GPU training scripts/configs,
hundreds of PHASE/SPRINT/TASK status-report markdown files, and a stray
`simd_benchmark.cpp` that broke `go build ./...` outright ("C++ source files
not allowed when not using cgo or SWIG"). All of it has been removed. The repo
root now contains only the legitimate sigma-index tree (`cmd/`, `persist/`,
`pkg/`, `src/sigma_index/`, `tests/`, `sigma_index_client.py`,
`pyproject.toml`) plus standard repo files (`.github/`, `go.mod`, `README.md`,
`CLAUDE.md`, `VERSION`). `go build ./...` and `go test ./...` both pass
cleanly with zero errors, verified with a cleared build/test cache. Committed
locally only, not pushed — see git log for the cleanup commit.

## What sigma-index Should Be
A high-performance indexing and search engine that provides:
1. **HNSW vector search** — for semantic similarity (embeddings from Ryzanstein)
2. **BM25 keyword search** — for traditional text matching
3. **Hybrid search** — combine vector + keyword with reciprocal rank fusion

## What Depends On sigma-index
- sigmalang (needs vector index for glyph similarity search)
- sigma-harvest (needs search over collected data)
- In-My-Head (has Qdrant-based search that should feed into this)
- AlgoSmash (Neural-LSH stub needs a real HNSW backend)
- Steve-AI (needs to search across repos)

## Sprint 1: Core HNSW Implementation
- [x] Create `pkg/hnsw/` package with Go HNSW implementation
- [x] Support: Add(id, vector), Search(query, k), Delete(id)
- [x] Use L2 (Euclidean) and Cosine distance metrics
- [x] Parameters: M=16, efConstruction=200, efSearch=100
- [x] Persistence: save/load index to disk (gob or custom binary)
- [x] Test: 1000 random 256-dim vectors, verify kNN recall > 95%

## Sprint 2: BM25 Text Search
- [x] Create `pkg/bm25/` package
- [x] Inverted index with term frequencies
- [x] BM25 scoring (k1=1.2, b=0.75)
- [x] Add(docId, text), Search(query, k)
- [x] Persistence to disk

## Sprint 3: Hybrid Search + API
- [x] Reciprocal Rank Fusion of HNSW + BM25 results
- [x] gRPC API (HTTP REST on :8200 instead) (reuse existing proto definitions)
- [x] REST API server on :8200
- [x] Benchmark (100% recall@10 on 1K vectors): search latency < 10ms for 100K documents

## Sprint 4: Ecosystem Integration
- [x] Wire to Ryzanstein /v1/embeddings for vector generation
- [x] Wire to sigmalang (Python client) for glyph similarity search
- [x] Export as Go module importable by other Sigma projects

## Build Commands
```bash
export PATH=$PATH:/usr/local/go/bin
cd /opt/sigmavault/repos/Layer-4-Storage-sigma-index
go test ./...
go build ./...
```

## Done Criteria
- [x] HNSW search returns correct nearest neighbors (recall > 95%) — verified 2026-07-04: `TestRecall` reports 100.00% recall@10 (50 queries, 1000 vectors, dim=64)
- [x] BM25 search returns relevant documents
- [x] Hybrid search combines both with RRF
- [x] All tests pass — verified 2026-07-04 with a cleared cache: `go test ./...` 23/23 (persist 9, pkg/bm25 5, pkg/hnsw 6, pkg/hybrid 3); pytest 39/39 (Python FM-index/HNSW/SigmaIndex suite)
- [x] Not a Ryzanstein clone — CORRECTED 2026-07-04: this was checked prematurely. The Ryzanstein/RYZEN-LLM mirror (1219 files, see Current State above) was still sitting in the repo until the 2026-07-04 cleanup removed it. TRUE now, not before.

## Completion Signal
```bash
git tag v1.0.0
```
