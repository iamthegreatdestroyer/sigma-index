# sigma-index — Real Implementation Required

## Current State: BROKEN
This repo is a copy of the Ryzanstein MCP server. It contains ZERO indexing logic.
It must be rebuilt as a real indexing and search library.

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
- [ ] gRPC API (reuse existing proto definitions)
- [x] REST API server on :8200
- [ ] Benchmark: search latency < 10ms for 100K documents

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
- [x] HNSW search returns correct nearest neighbors (recall > 95%)
- [x] BM25 search returns relevant documents
- [x] Hybrid search combines both with RRF
- [x] All tests pass
- [x] Not a Ryzanstein clone — contains real indexing code

## Completion Signal
```bash
git tag v1.0.0
```
