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
- [ ] Create `pkg/hnsw/` package with Go HNSW implementation
- [ ] Support: Add(id, vector), Search(query, k), Delete(id)
- [ ] Use L2 (Euclidean) and Cosine distance metrics
- [ ] Parameters: M=16, efConstruction=200, efSearch=100
- [ ] Persistence: save/load index to disk (gob or custom binary)
- [ ] Test: 1000 random 256-dim vectors, verify kNN recall > 95%

## Sprint 2: BM25 Text Search
- [ ] Create `pkg/bm25/` package
- [ ] Inverted index with term frequencies
- [ ] BM25 scoring (k1=1.2, b=0.75)
- [ ] Add(docId, text), Search(query, k)
- [ ] Persistence to disk

## Sprint 3: Hybrid Search + API
- [ ] Reciprocal Rank Fusion of HNSW + BM25 results
- [ ] gRPC API (reuse existing proto definitions)
- [ ] REST API wrapper
- [ ] Benchmark: search latency < 10ms for 100K documents

## Sprint 4: Ecosystem Integration
- [ ] Wire to Ryzanstein /v1/embeddings for vector generation
- [ ] Wire to sigmalang for glyph similarity search
- [ ] Export as Go module importable by other Sigma projects

## Build Commands
```bash
export PATH=$PATH:/usr/local/go/bin
cd /opt/sigmavault/repos/Layer-4-Storage-sigma-index
go test ./...
go build ./...
```

## Done Criteria
- [ ] HNSW search returns correct nearest neighbors (recall > 95%)
- [ ] BM25 search returns relevant documents
- [ ] Hybrid search combines both with RRF
- [ ] All tests pass
- [ ] Not a Ryzanstein clone — contains real indexing code

## Completion Signal
```bash
git tag v1.0.0
```
