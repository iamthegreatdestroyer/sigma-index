// Package bm25 implements the Okapi BM25 ranking function for full-text search.
package bm25

import (
	"math"
	"sort"
	"strings"
	"sync"
	"unicode"
)

type Document struct {
	ID   string
	Text string
}

type SearchResult struct {
	ID    string
	Score float64
}

type Index struct {
	mu       sync.RWMutex
	docs     map[string]string    // id -> original text
	termFreq map[string]map[string]int // id -> term -> count
	docFreq  map[string]int       // term -> number of docs containing it
	docLen   map[string]int       // id -> token count
	avgDL    float64
	n        int
	k1       float64
	b        float64
}

func New() *Index {
	return &Index{
		docs:     make(map[string]string),
		termFreq: make(map[string]map[string]int),
		docFreq:  make(map[string]int),
		docLen:   make(map[string]int),
		k1:       1.2,
		b:        0.75,
	}
}

func tokenize(text string) []string {
	text = strings.ToLower(text)
	words := strings.FieldsFunc(text, func(r rune) bool {
		return !unicode.IsLetter(r) && !unicode.IsDigit(r)
	})
	filtered := make([]string, 0, len(words))
	for _, w := range words {
		if len(w) >= 2 {
			filtered = append(filtered, w)
		}
	}
	return filtered
}

func (idx *Index) Add(id, text string) {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	if _, exists := idx.docs[id]; exists {
		idx.removeLocked(id)
	}

	tokens := tokenize(text)
	idx.docs[id] = text
	idx.docLen[id] = len(tokens)

	tf := make(map[string]int)
	seen := make(map[string]bool)
	for _, t := range tokens {
		tf[t]++
		if !seen[t] {
			idx.docFreq[t]++
			seen[t] = true
		}
	}
	idx.termFreq[id] = tf
	idx.n++
	idx.recalcAvgDL()
}

func (idx *Index) Delete(id string) {
	idx.mu.Lock()
	defer idx.mu.Unlock()
	idx.removeLocked(id)
}

func (idx *Index) removeLocked(id string) {
	tf, ok := idx.termFreq[id]
	if !ok {
		return
	}
	for term := range tf {
		idx.docFreq[term]--
		if idx.docFreq[term] <= 0 {
			delete(idx.docFreq, term)
		}
	}
	delete(idx.docs, id)
	delete(idx.termFreq, id)
	delete(idx.docLen, id)
	idx.n--
	idx.recalcAvgDL()
}

func (idx *Index) recalcAvgDL() {
	if idx.n == 0 {
		idx.avgDL = 0
		return
	}
	total := 0
	for _, l := range idx.docLen {
		total += l
	}
	idx.avgDL = float64(total) / float64(idx.n)
}

func (idx *Index) Search(query string, k int) []SearchResult {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	queryTerms := tokenize(query)
	if len(queryTerms) == 0 {
		return nil
	}

	scores := make(map[string]float64)

	for _, term := range queryTerms {
		df, ok := idx.docFreq[term]
		if !ok {
			continue
		}
		idf := math.Log(1 + (float64(idx.n)-float64(df)+0.5)/(float64(df)+0.5))

		for id, tf := range idx.termFreq {
			freq := float64(tf[term])
			if freq == 0 {
				continue
			}
			dl := float64(idx.docLen[id])
			denom := freq + idx.k1*(1-idx.b+idx.b*dl/idx.avgDL)
			score := idf * (freq * (idx.k1 + 1)) / denom
			scores[id] += score
		}
	}

	results := make([]SearchResult, 0, len(scores))
	for id, score := range scores {
		results = append(results, SearchResult{ID: id, Score: score})
	}
	sort.Slice(results, func(i, j int) bool { return results[i].Score > results[j].Score })
	if len(results) > k {
		results = results[:k]
	}
	return results
}

// Doc returns the original text stored for id, or "" if the id is unknown.
func (idx *Index) Doc(id string) string {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	return idx.docs[id]
}

func (idx *Index) Len() int {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	return idx.n
}
