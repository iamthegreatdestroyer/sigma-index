package bm25

import (
	"testing"
)

func TestAddAndSearch(t *testing.T) {
	idx := New()
	idx.Add("1", "the quick brown fox jumps over the lazy dog")
	idx.Add("2", "a fast brown fox leaps across a sleeping hound")
	idx.Add("3", "quantum computing breaks cryptographic protocols")
	idx.Add("4", "post-quantum cryptography uses lattice-based algorithms")

	results := idx.Search("brown fox", 3)
	if len(results) < 2 {
		t.Fatalf("expected at least 2 results for 'brown fox', got %d", len(results))
	}
	top := results[0]
	if top.ID != "1" && top.ID != "2" {
		t.Errorf("expected doc 1 or 2 as top result, got %s", top.ID)
	}
}

func TestCryptoSearch(t *testing.T) {
	idx := New()
	idx.Add("a", "AES-256-GCM encryption with Argon2id key derivation")
	idx.Add("b", "Kyber-1024 post-quantum key encapsulation mechanism")
	idx.Add("c", "Dilithium-3 digital signature scheme for authentication")
	idx.Add("d", "HTTP server with REST API and JSON responses")

	results := idx.Search("post-quantum cryptography", 3)
	if len(results) == 0 {
		t.Fatal("no results for 'post-quantum cryptography'")
	}
	if results[0].ID != "b" {
		t.Errorf("expected doc b (Kyber) as top result, got %s", results[0].ID)
	}
}

func TestDelete(t *testing.T) {
	idx := New()
	idx.Add("x", "hello world")
	idx.Add("y", "goodbye world")
	if idx.Len() != 2 {
		t.Fatalf("expected 2 docs, got %d", idx.Len())
	}
	idx.Delete("x")
	if idx.Len() != 1 {
		t.Fatalf("expected 1 doc after delete, got %d", idx.Len())
	}
	results := idx.Search("hello", 5)
	for _, r := range results {
		if r.ID == "x" {
			t.Error("deleted doc x still appears in results")
		}
	}
}

func TestEmptySearch(t *testing.T) {
	idx := New()
	results := idx.Search("anything", 5)
	if len(results) != 0 {
		t.Errorf("empty index should return no results, got %d", len(results))
	}
}

func TestDuplicateAdd(t *testing.T) {
	idx := New()
	idx.Add("dup", "original text content")
	idx.Add("dup", "updated text content replacement")
	if idx.Len() != 1 {
		t.Errorf("duplicate add should replace, got %d docs", idx.Len())
	}
	results := idx.Search("replacement", 1)
	if len(results) == 0 || results[0].ID != "dup" {
		t.Error("updated doc should be searchable by new content")
	}
}
