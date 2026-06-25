// Package server exposes sigma-index as an HTTP API for cross-language integration.
// Python projects (sigmalang, In-My-Head, sigma-harvest) call this instead of
// maintaining their own vector indices.
package server

import (
	"encoding/json"
	"log"
	"net/http"
	"sync"

	"github.com/iamthegreatdestroyer/sigma-index/pkg/bm25"
	"github.com/iamthegreatdestroyer/sigma-index/pkg/hnsw"
	"github.com/iamthegreatdestroyer/sigma-index/pkg/hybrid"
)

type IndexServer struct {
	mu        sync.RWMutex
	indices   map[string]*hybrid.Index
	bm25Only  map[string]*bm25.Index
	hnswOnly  map[string]*hnsw.Index
	defaultDim int
}

func New(defaultDim int) *IndexServer {
	return &IndexServer{
		indices:    make(map[string]*hybrid.Index),
		bm25Only:   make(map[string]*bm25.Index),
		hnswOnly:   make(map[string]*hnsw.Index),
		defaultDim: defaultDim,
	}
}

func (s *IndexServer) getOrCreate(namespace string) *hybrid.Index {
	s.mu.Lock()
	defer s.mu.Unlock()
	if idx, ok := s.indices[namespace]; ok {
		return idx
	}
	idx := hybrid.New(s.defaultDim)
	s.indices[namespace] = idx
	return idx
}

type addRequest struct {
	Namespace string    `json:"namespace"`
	ID        string    `json:"id"`
	Vector    []float32 `json:"vector"`
	Text      string    `json:"text"`
}

type searchRequest struct {
	Namespace string    `json:"namespace"`
	Vector    []float32 `json:"vector,omitempty"`
	Text      string    `json:"text,omitempty"`
	K         int       `json:"k"`
}

type deleteRequest struct {
	Namespace string `json:"namespace"`
	ID        string `json:"id"`
}

func (s *IndexServer) Handler() http.Handler {
	mux := http.NewServeMux()

	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		s.mu.RLock()
		count := len(s.indices)
		s.mu.RUnlock()
		json.NewEncoder(w).Encode(map[string]interface{}{
			"status":     "healthy",
			"namespaces": count,
			"service":    "sigma-index",
		})
	})

	mux.HandleFunc("/add", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "POST only", 405)
			return
		}
		var req addRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, err.Error(), 400)
			return
		}
		if req.Namespace == "" {
			req.Namespace = "default"
		}
		idx := s.getOrCreate(req.Namespace)
		idx.Add(req.ID, req.Vector, req.Text)
		json.NewEncoder(w).Encode(map[string]string{"status": "ok", "id": req.ID})
	})

	mux.HandleFunc("/search", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "POST only", 405)
			return
		}
		var req searchRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, err.Error(), 400)
			return
		}
		if req.Namespace == "" {
			req.Namespace = "default"
		}
		if req.K == 0 {
			req.K = 10
		}

		idx := s.getOrCreate(req.Namespace)
		var results []hybrid.SearchResult

		if len(req.Vector) > 0 && req.Text != "" {
			results = idx.Search(req.Vector, req.Text, req.K)
		} else if len(req.Vector) > 0 {
			results = idx.SearchVector(req.Vector, req.K)
		} else if req.Text != "" {
			results = idx.SearchText(req.Text, req.K)
		}

		json.NewEncoder(w).Encode(map[string]interface{}{
			"results": results,
			"count":   len(results),
		})
	})

	mux.HandleFunc("/delete", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "POST only", 405)
			return
		}
		var req deleteRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, err.Error(), 400)
			return
		}
		if req.Namespace == "" {
			req.Namespace = "default"
		}
		idx := s.getOrCreate(req.Namespace)
		idx.Delete(req.ID)
		json.NewEncoder(w).Encode(map[string]string{"status": "deleted", "id": req.ID})
	})

	return mux
}

func (s *IndexServer) ListenAndServe(addr string) error {
	log.Printf("sigma-index server listening on %s", addr)
	return http.ListenAndServe(addr, s.Handler())
}
