// Package hybrid provides combined vector + keyword search using
// Reciprocal Rank Fusion (RRF) to merge HNSW and BM25 results.
// This is the primary search interface for the Sigma Ecosystem.
package hybrid

import (
	"sort"

	"github.com/iamthegreatdestroyer/sigma-index/pkg/bm25"
	"github.com/iamthegreatdestroyer/sigma-index/pkg/hnsw"
)

type SearchResult struct {
	ID       string
	Score    float64
	VecDist  float32
	TextRank int
}

type Index struct {
	vectors *hnsw.Index
	text    *bm25.Index
}

func New(dim int) *Index {
	return &Index{
		vectors: hnsw.New(hnsw.Config{M: 16, EfConstruction: 200, Dim: dim, Distance: "cosine"}),
		text:    bm25.New(),
	}
}

func (idx *Index) Add(id string, vector []float32, text string) {
	idx.vectors.Add(id, vector)
	idx.text.Add(id, text)
}

func (idx *Index) Delete(id string) {
	idx.vectors.Delete(id)
	idx.text.Delete(id)
}

func (idx *Index) SearchVector(query []float32, k int) []SearchResult {
	raw := idx.vectors.Search(query, k, 0)
	results := make([]SearchResult, len(raw))
	for i, r := range raw {
		results[i] = SearchResult{ID: r.ID, VecDist: r.Distance, Score: float64(1 / (1 + r.Distance))}
	}
	return results
}

func (idx *Index) SearchText(query string, k int) []SearchResult {
	raw := idx.text.Search(query, k)
	results := make([]SearchResult, len(raw))
	for i, r := range raw {
		results[i] = SearchResult{ID: r.ID, Score: r.Score, TextRank: i + 1}
	}
	return results
}

func (idx *Index) Search(vectorQuery []float32, textQuery string, k int) []SearchResult {
	vecResults := idx.vectors.Search(vectorQuery, k*2, 0)
	textResults := idx.text.Search(textQuery, k*2)

	scores := make(map[string]float64)
	vecDists := make(map[string]float32)
	textRanks := make(map[string]int)

	const rrfK = 60.0

	for rank, r := range vecResults {
		scores[r.ID] += 1.0 / (rrfK + float64(rank+1))
		vecDists[r.ID] = r.Distance
	}

	for rank, r := range textResults {
		scores[r.ID] += 1.0 / (rrfK + float64(rank+1))
		textRanks[r.ID] = rank + 1
	}

	results := make([]SearchResult, 0, len(scores))
	for id, score := range scores {
		results = append(results, SearchResult{
			ID:       id,
			Score:    score,
			VecDist:  vecDists[id],
			TextRank: textRanks[id],
		})
	}

	sort.Slice(results, func(i, j int) bool { return results[i].Score > results[j].Score })
	if len(results) > k {
		results = results[:k]
	}
	return results
}

func (idx *Index) Len() int {
	return idx.vectors.Len()
}
