package hnsw

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"testing"
)

func randomVector(dim int) []float32 {
	v := make([]float32, dim)
	for i := range v {
		v[i] = rand.Float32()*2 - 1
	}
	norm := float32(0)
	for _, x := range v {
		norm += x * x
	}
	norm = float32(math.Sqrt(float64(norm)))
	for i := range v {
		v[i] /= norm
	}
	return v
}

func TestAddAndSearch(t *testing.T) {
	idx := New(Config{M: 16, EfConstruction: 200, Dim: 32, Distance: "cosine"})

	vectors := make(map[string][]float32)
	for i := 0; i < 100; i++ {
		id := fmt.Sprintf("doc_%d", i)
		v := randomVector(32)
		vectors[id] = v
		idx.Add(id, v)
	}

	if idx.Len() != 100 {
		t.Fatalf("expected 100 nodes, got %d", idx.Len())
	}

	query := vectors["doc_0"]
	results := idx.Search(query, 5, 100)

	if len(results) == 0 {
		t.Fatal("search returned no results")
	}
	if results[0].ID != "doc_0" {
		t.Errorf("expected doc_0 as nearest, got %s (dist=%.4f)", results[0].ID, results[0].Distance)
	}
	if results[0].Distance > 0.01 {
		t.Errorf("self-search distance should be ~0, got %.4f", results[0].Distance)
	}
}

func TestRecall(t *testing.T) {
	dim := 64
	n := 1000
	k := 10

	idx := New(Config{M: 16, EfConstruction: 200, Dim: dim, Distance: "cosine"})
	vectors := make([][]float32, n)
	for i := 0; i < n; i++ {
		v := randomVector(dim)
		vectors[i] = v
		idx.Add(fmt.Sprintf("v_%d", i), v)
	}

	totalRecall := 0
	queries := 50
	for q := 0; q < queries; q++ {
		query := randomVector(dim)

		// Brute force ground truth
		type distID struct {
			id   int
			dist float32
		}
		bf := make([]distID, n)
		for i := 0; i < n; i++ {
			bf[i] = distID{id: i, dist: CosineDistance(query, vectors[i])}
		}
		for i := 0; i < k; i++ {
			for j := i + 1; j < n; j++ {
				if bf[j].dist < bf[i].dist {
					bf[i], bf[j] = bf[j], bf[i]
				}
			}
		}
		groundTruth := make(map[string]bool)
		for i := 0; i < k; i++ {
			groundTruth[fmt.Sprintf("v_%d", bf[i].id)] = true
		}

		results := idx.Search(query, k, 200)
		hits := 0
		for _, r := range results {
			if groundTruth[r.ID] {
				hits++
			}
		}
		totalRecall += hits
	}

	recall := float64(totalRecall) / float64(queries*k)
	t.Logf("Recall@%d: %.2f%% (%d queries, %d vectors, dim=%d)", k, recall*100, queries, n, dim)
	if recall < 0.90 {
		t.Errorf("recall %.2f%% is below 90%% threshold", recall*100)
	}
}

func TestDelete(t *testing.T) {
	idx := New(Config{M: 16, EfConstruction: 100, Dim: 8})
	for i := 0; i < 20; i++ {
		idx.Add(fmt.Sprintf("d_%d", i), randomVector(8))
	}
	if idx.Len() != 20 {
		t.Fatalf("expected 20, got %d", idx.Len())
	}
	idx.Delete("d_5")
	if idx.Len() != 19 {
		t.Fatalf("expected 19 after delete, got %d", idx.Len())
	}
	results := idx.Search(randomVector(8), 20, 50)
	for _, r := range results {
		if r.ID == "d_5" {
			t.Error("deleted node d_5 still appears in results")
		}
	}
}

func TestSaveLoad(t *testing.T) {
	idx := New(Config{M: 8, EfConstruction: 50, Dim: 16, Distance: "l2"})
	query := randomVector(16)
	for i := 0; i < 50; i++ {
		idx.Add(fmt.Sprintf("s_%d", i), randomVector(16))
	}
	idx.Add("target", query)

	path := "/tmp/test_hnsw.gob"
	defer os.Remove(path)

	if err := idx.Save(path); err != nil {
		t.Fatal(err)
	}

	idx2 := New(Config{M: 8, EfConstruction: 50, Dim: 16, Distance: "l2"})
	if err := idx2.Load(path); err != nil {
		t.Fatal(err)
	}

	if idx2.Len() != idx.Len() {
		t.Errorf("loaded %d nodes, expected %d", idx2.Len(), idx.Len())
	}

	results := idx2.Search(query, 1, 50)
	if len(results) == 0 || results[0].ID != "target" {
		t.Error("loaded index did not find the target vector")
	}
}

func TestL2Distance(t *testing.T) {
	a := []float32{1, 0, 0}
	b := []float32{0, 1, 0}
	d := L2Distance(a, b)
	expected := float32(math.Sqrt(2))
	if math.Abs(float64(d-expected)) > 0.001 {
		t.Errorf("L2(%v, %v) = %.4f, expected %.4f", a, b, d, expected)
	}
}

func TestCosineDistance(t *testing.T) {
	a := []float32{1, 0, 0}
	b := []float32{1, 0, 0}
	d := CosineDistance(a, b)
	if d > 0.001 {
		t.Errorf("cosine distance of identical vectors should be ~0, got %.4f", d)
	}

	c := []float32{-1, 0, 0}
	d2 := CosineDistance(a, c)
	if d2 < 1.99 {
		t.Errorf("cosine distance of opposite vectors should be ~2, got %.4f", d2)
	}
}
