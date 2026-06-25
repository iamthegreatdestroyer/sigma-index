package hybrid

import (
	"fmt"
	"math/rand"
	"testing"
)

func randVec(dim int) []float32 {
	v := make([]float32, dim)
	for i := range v {
		v[i] = rand.Float32()*2 - 1
	}
	return v
}

func TestHybridSearch(t *testing.T) {
	dim := 32
	idx := New(dim)

	idx.Add("crypto", randVec(dim), "AES-256-GCM encryption with Kyber-1024 post-quantum key encapsulation")
	idx.Add("network", randVec(dim), "WireGuard VPN tunnel with UDP transport and NAT traversal")
	idx.Add("search", randVec(dim), "HNSW approximate nearest neighbor search with cosine distance")
	idx.Add("compress", randVec(dim), "sigmalang semantic compression using 256 glyph primitives")
	idx.Add("diff", randVec(dim), "AST-level structural code diffing for Go and Python source files")

	// Text-only search
	textResults := idx.SearchText("encryption Kyber quantum", 3)
	if len(textResults) == 0 {
		t.Fatal("text search returned no results")
	}
	if textResults[0].ID != "crypto" {
		t.Errorf("expected crypto as top text result, got %s", textResults[0].ID)
	}

	// Hybrid search (vector + text)
	query := randVec(dim)
	hybridResults := idx.Search(query, "compression glyph semantic", 3)
	if len(hybridResults) == 0 {
		t.Fatal("hybrid search returned no results")
	}

	// All results should have RRF scores
	for _, r := range hybridResults {
		if r.Score <= 0 {
			t.Errorf("result %s has zero score", r.ID)
		}
	}

	t.Logf("Hybrid results:")
	for _, r := range hybridResults {
		t.Logf("  %s: score=%.4f vecDist=%.4f textRank=%d", r.ID, r.Score, r.VecDist, r.TextRank)
	}
}

func TestVectorOnlySearch(t *testing.T) {
	dim := 16
	idx := New(dim)

	target := randVec(dim)
	idx.Add("target", target, "target document")
	for i := 0; i < 20; i++ {
		idx.Add(fmt.Sprintf("other_%d", i), randVec(dim), "random document")
	}

	results := idx.SearchVector(target, 1)
	if len(results) == 0 {
		t.Fatal("vector search returned no results")
	}
	if results[0].ID != "target" {
		t.Errorf("expected target as nearest, got %s", results[0].ID)
	}
}

func TestAddDelete(t *testing.T) {
	idx := New(8)
	idx.Add("a", randVec(8), "hello world")
	idx.Add("b", randVec(8), "goodbye world")
	if idx.Len() != 2 {
		t.Fatalf("expected 2, got %d", idx.Len())
	}
	idx.Delete("a")
	if idx.Len() != 1 {
		t.Fatalf("expected 1 after delete, got %d", idx.Len())
	}
}
