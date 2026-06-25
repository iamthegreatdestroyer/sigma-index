// Package hnsw implements Hierarchical Navigable Small World graphs for
// approximate nearest neighbor search on dense vectors.
package hnsw

import (
	"encoding/gob"
	"math"
	"math/rand"
	"os"
	"sort"
	"sync"
)

type DistanceFunc func(a, b []float32) float32

func CosineDistance(a, b []float32) float32 {
	var dot, normA, normB float32
	for i := range a {
		dot += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}
	if normA == 0 || normB == 0 {
		return 1
	}
	return 1 - dot/float32(math.Sqrt(float64(normA*normB)))
}

func L2Distance(a, b []float32) float32 {
	var sum float32
	for i := range a {
		d := a[i] - b[i]
		sum += d * d
	}
	return float32(math.Sqrt(float64(sum)))
}

type node struct {
	ID      string
	Vector  []float32
	Friends [][]string // friends[level] = list of neighbor IDs
}

type searchCandidate struct {
	ID   string
	Dist float32
}

type Index struct {
	mu             sync.RWMutex
	nodes          map[string]*node
	entryPoint     string
	maxLevel       int
	m              int     // max connections per layer
	mMax0          int     // max connections at layer 0
	efConstruction int
	ml             float64 // level multiplier
	dim            int
	dist           DistanceFunc
}

type Config struct {
	M              int
	EfConstruction int
	Dim            int
	Distance       string // "cosine" or "l2"
}

func New(cfg Config) *Index {
	if cfg.M == 0 {
		cfg.M = 16
	}
	if cfg.EfConstruction == 0 {
		cfg.EfConstruction = 200
	}
	dist := CosineDistance
	if cfg.Distance == "l2" {
		dist = L2Distance
	}
	return &Index{
		nodes:          make(map[string]*node),
		m:              cfg.M,
		mMax0:          cfg.M * 2,
		efConstruction: cfg.EfConstruction,
		ml:             1 / math.Log(float64(cfg.M)),
		dim:            cfg.Dim,
		dist:           dist,
		maxLevel:       -1,
	}
}

func (idx *Index) randomLevel() int {
	return int(math.Floor(-math.Log(rand.Float64()) * idx.ml))
}

func (idx *Index) Add(id string, vector []float32) {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	level := idx.randomLevel()
	n := &node{
		ID:      id,
		Vector:  vector,
		Friends: make([][]string, level+1),
	}
	for i := range n.Friends {
		n.Friends[i] = []string{}
	}
	idx.nodes[id] = n

	if idx.maxLevel == -1 {
		idx.entryPoint = id
		idx.maxLevel = level
		return
	}

	ep := idx.entryPoint
	for l := idx.maxLevel; l > level; l-- {
		ep = idx.greedyClosest(vector, ep, l)
	}

	for l := min(level, idx.maxLevel); l >= 0; l-- {
		neighbors := idx.searchLayer(vector, ep, idx.efConstruction, l)

		maxConn := idx.m
		if l == 0 {
			maxConn = idx.mMax0
		}
		selected := idx.selectNeighbors(neighbors, maxConn)

		n.Friends[l] = make([]string, len(selected))
		for i, s := range selected {
			n.Friends[l][i] = s.ID
		}

		for _, s := range selected {
			neighbor := idx.nodes[s.ID]
			if l < len(neighbor.Friends) {
				neighbor.Friends[l] = append(neighbor.Friends[l], id)
				if len(neighbor.Friends[l]) > maxConn {
					cands := make([]searchCandidate, len(neighbor.Friends[l]))
					for i, fid := range neighbor.Friends[l] {
						cands[i] = searchCandidate{ID: fid, Dist: idx.dist(neighbor.Vector, idx.nodes[fid].Vector)}
					}
					trimmed := idx.selectNeighbors(cands, maxConn)
					neighbor.Friends[l] = make([]string, len(trimmed))
					for i, t := range trimmed {
						neighbor.Friends[l][i] = t.ID
					}
				}
			}
		}

		if len(neighbors) > 0 {
			ep = neighbors[0].ID
		}
	}

	if level > idx.maxLevel {
		idx.maxLevel = level
		idx.entryPoint = id
	}
}

func (idx *Index) Search(query []float32, k int, efSearch int) []SearchResult {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	if len(idx.nodes) == 0 {
		return nil
	}
	if efSearch == 0 {
		efSearch = max(k, 100)
	}

	ep := idx.entryPoint
	for l := idx.maxLevel; l > 0; l-- {
		ep = idx.greedyClosest(query, ep, l)
	}

	candidates := idx.searchLayer(query, ep, efSearch, 0)
	if len(candidates) > k {
		candidates = candidates[:k]
	}

	results := make([]SearchResult, len(candidates))
	for i, c := range candidates {
		results[i] = SearchResult{ID: c.ID, Distance: c.Dist}
	}
	return results
}

type SearchResult struct {
	ID       string
	Distance float32
}

func (idx *Index) Delete(id string) {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	n, ok := idx.nodes[id]
	if !ok {
		return
	}
	for l, friends := range n.Friends {
		for _, fid := range friends {
			f := idx.nodes[fid]
			if l < len(f.Friends) {
				newFriends := make([]string, 0, len(f.Friends[l]))
				for _, ff := range f.Friends[l] {
					if ff != id {
						newFriends = append(newFriends, ff)
					}
				}
				f.Friends[l] = newFriends
			}
		}
	}
	delete(idx.nodes, id)

	if id == idx.entryPoint && len(idx.nodes) > 0 {
		for nid := range idx.nodes {
			idx.entryPoint = nid
			break
		}
	}
}

func (idx *Index) Len() int {
	idx.mu.RLock()
	defer idx.mu.RUnlock()
	return len(idx.nodes)
}

func (idx *Index) greedyClosest(query []float32, ep string, level int) string {
	best := ep
	bestDist := idx.dist(query, idx.nodes[ep].Vector)
	changed := true
	for changed {
		changed = false
		n := idx.nodes[best]
		if level < len(n.Friends) {
			for _, fid := range n.Friends[level] {
				d := idx.dist(query, idx.nodes[fid].Vector)
				if d < bestDist {
					bestDist = d
					best = fid
					changed = true
				}
			}
		}
	}
	return best
}

func (idx *Index) searchLayer(query []float32, ep string, ef int, level int) []searchCandidate {
	visited := map[string]bool{ep: true}
	candidates := []searchCandidate{{ID: ep, Dist: idx.dist(query, idx.nodes[ep].Vector)}}
	results := []searchCandidate{{ID: ep, Dist: idx.dist(query, idx.nodes[ep].Vector)}}

	for len(candidates) > 0 {
		sort.Slice(candidates, func(i, j int) bool { return candidates[i].Dist < candidates[j].Dist })
		closest := candidates[0]
		candidates = candidates[1:]

		worst := results[len(results)-1]
		if closest.Dist > worst.Dist && len(results) >= ef {
			break
		}

		n := idx.nodes[closest.ID]
		if level < len(n.Friends) {
			for _, fid := range n.Friends[level] {
				if visited[fid] {
					continue
				}
				visited[fid] = true
				d := idx.dist(query, idx.nodes[fid].Vector)

				if len(results) < ef || d < results[len(results)-1].Dist {
					candidates = append(candidates, searchCandidate{ID: fid, Dist: d})
					results = append(results, searchCandidate{ID: fid, Dist: d})
					sort.Slice(results, func(i, j int) bool { return results[i].Dist < results[j].Dist })
					if len(results) > ef {
						results = results[:ef]
					}
				}
			}
		}
	}
	return results
}

func (idx *Index) selectNeighbors(candidates []searchCandidate, m int) []searchCandidate {
	sort.Slice(candidates, func(i, j int) bool { return candidates[i].Dist < candidates[j].Dist })
	if len(candidates) > m {
		return candidates[:m]
	}
	return candidates
}

func (idx *Index) Save(path string) error {
	idx.mu.RLock()
	defer idx.mu.RUnlock()

	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()

	type exportData struct {
		Nodes      map[string]*node
		EntryPoint string
		MaxLevel   int
		M          int
		Dim        int
	}

	return gob.NewEncoder(f).Encode(exportData{
		Nodes:      idx.nodes,
		EntryPoint: idx.entryPoint,
		MaxLevel:   idx.maxLevel,
		M:          idx.m,
		Dim:        idx.dim,
	})
}

func (idx *Index) Load(path string) error {
	idx.mu.Lock()
	defer idx.mu.Unlock()

	f, err := os.Open(path)
	if err != nil {
		return err
	}
	defer f.Close()

	type exportData struct {
		Nodes      map[string]*node
		EntryPoint string
		MaxLevel   int
		M          int
		Dim        int
	}

	var data exportData
	if err := gob.NewDecoder(f).Decode(&data); err != nil {
		return err
	}

	idx.nodes = data.Nodes
	idx.entryPoint = data.EntryPoint
	idx.maxLevel = data.MaxLevel
	idx.m = data.M
	idx.mMax0 = data.M * 2
	idx.dim = data.Dim
	return nil
}
