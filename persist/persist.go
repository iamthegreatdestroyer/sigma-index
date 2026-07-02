// Package persist adds disk persistence to sigma-index.
//
// The in-memory hybrid indices (HNSW vectors + BM25 text) are volatile — they
// are lost when the server restarts. This package wraps the same HTTP API as
// pkg/server but keeps an authoritative record store of every (namespace, id,
// vector, text) triple that has been added. Durability is two-tier:
//
//  1. Write-ahead log (wal.jsonl): every /add and /delete is appended to an
//     append-only JSONL log before the request is acknowledged, so a hard
//     crash loses nothing that was acknowledged (modulo OS page cache unless
//     Fsync is enabled).
//  2. Snapshots (<namespace>.json): the full record store is written
//     atomically (temp file + rename) per namespace — periodically, on
//     graceful shutdown, and on demand via POST /snapshot. A successful
//     full snapshot truncates the WAL.
//
// On startup the snapshots are loaded, the WAL is replayed on top, and the
// indices are rebuilt exactly by feeding every record through hybrid.Add
// (rebuild-from-triples). This deliberately avoids relying on HNSW/BM25
// internal serialization: correctness and simplicity over cleverness.
//
// It is API-compatible with pkg/server (POST /add, POST /search, POST /delete,
// GET /health) and adds POST /snapshot.
package persist

import (
	"bufio"
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"regexp"
	"sort"
	"sync"

	"github.com/iamthegreatdestroyer/sigma-index/pkg/hybrid"
)

// namespaceRe restricts namespaces to filesystem-safe names so that the
// snapshot file <namespace>.json cannot escape the data dir and two distinct
// namespaces can never collide onto the same file.
var namespaceRe = regexp.MustCompile(`^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$`)

const walName = "wal.jsonl"

// record is the reconstructable state of a single document.
type record struct {
	ID     string    `json:"id"`
	Vector []float32 `json:"vector"`
	Text   string    `json:"text"`
}

// namespaceSnapshot is the on-disk form of one namespace: an ordered list of
// records. Rebuilding a hybrid.Index from these is exact.
type namespaceSnapshot struct {
	Dim     int      `json:"dim"`
	Records []record `json:"records"`
}

// walEntry is one logged mutation. Op is "add" or "del".
type walEntry struct {
	Op        string    `json:"op"`
	Namespace string    `json:"ns"`
	ID        string    `json:"id"`
	Vector    []float32 `json:"vector,omitempty"`
	Text      string    `json:"text,omitempty"`
}

// Options configures Open.
type Options struct {
	Dim     int    // default vector dimension for new namespaces
	DataDir string // persistence root; "" disables persistence entirely
	Fsync   bool   // fsync the WAL after every acknowledged write
}

// IndexServer is a persistence-aware drop-in for pkg/server.IndexServer.
type IndexServer struct {
	mu         sync.RWMutex
	indices    map[string]*hybrid.Index
	records    map[string]map[string]record // namespace -> id -> record
	defaultDim int
	dataDir    string
	fsync      bool
	wal        *os.File
}

// New returns a server whose data is persisted under dataDir. If dataDir
// already contains snapshots and/or a WAL, they are loaded and the indices
// rebuilt. An empty dataDir yields a purely in-memory server.
func New(defaultDim int, dataDir string) (*IndexServer, error) {
	return Open(Options{Dim: defaultDim, DataDir: dataDir})
}

// Open is New with full options.
func Open(opts Options) (*IndexServer, error) {
	s := &IndexServer{
		indices:    make(map[string]*hybrid.Index),
		records:    make(map[string]map[string]record),
		defaultDim: opts.Dim,
		dataDir:    opts.DataDir,
		fsync:      opts.Fsync,
	}
	if s.dataDir == "" {
		return s, nil
	}
	if err := os.MkdirAll(s.dataDir, 0o755); err != nil {
		return nil, err
	}
	if err := s.loadSnapshots(); err != nil {
		return nil, err
	}
	if err := s.replayWAL(); err != nil {
		return nil, err
	}
	wal, err := os.OpenFile(s.walPath(), os.O_APPEND|os.O_CREATE|os.O_WRONLY, 0o644)
	if err != nil {
		return nil, err
	}
	s.wal = wal
	return s, nil
}

// Close snapshots all namespaces and closes the WAL. The server must not be
// used afterwards.
func (s *IndexServer) Close() error {
	if _, err := s.Snapshot(); err != nil {
		return err
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	if s.wal != nil {
		err := s.wal.Close()
		s.wal = nil
		return err
	}
	return nil
}

func (s *IndexServer) walPath() string {
	return filepath.Join(s.dataDir, walName)
}

// snapshotPath returns the JSON file path for a namespace. Namespaces are
// validated against namespaceRe before they exist, so this cannot escape
// dataDir.
func (s *IndexServer) snapshotPath(namespace string) string {
	return filepath.Join(s.dataDir, namespace+".json")
}

// loadSnapshots reads every *.json snapshot in dataDir and rebuilds the indices.
func (s *IndexServer) loadSnapshots() error {
	entries, err := os.ReadDir(s.dataDir)
	if err != nil {
		return err
	}
	loaded := 0
	for _, e := range entries {
		if e.IsDir() || filepath.Ext(e.Name()) != ".json" {
			continue
		}
		path := filepath.Join(s.dataDir, e.Name())
		data, err := os.ReadFile(path)
		if err != nil {
			return err
		}
		var snap namespaceSnapshot
		if err := json.Unmarshal(data, &snap); err != nil {
			log.Printf("persist: skipping corrupt snapshot %s: %v", path, err)
			continue
		}
		namespace := e.Name()[:len(e.Name())-len(".json")]
		if !namespaceRe.MatchString(namespace) {
			log.Printf("persist: skipping snapshot with invalid namespace name %q", e.Name())
			continue
		}
		dim := snap.Dim
		if dim == 0 {
			dim = s.defaultDim
		}
		idx := hybrid.New(dim)
		recs := make(map[string]record, len(snap.Records))
		for _, r := range snap.Records {
			idx.Add(r.ID, r.Vector, r.Text)
			recs[r.ID] = r
		}
		s.indices[namespace] = idx
		s.records[namespace] = recs
		loaded++
	}
	if loaded > 0 {
		log.Printf("persist: loaded %d namespace(s) from %s", loaded, s.dataDir)
	}
	return nil
}

// replayWAL applies WAL entries on top of the loaded snapshots. Malformed
// lines (e.g. a torn final write from a crash) are skipped with a warning.
func (s *IndexServer) replayWAL() error {
	f, err := os.Open(s.walPath())
	if os.IsNotExist(err) {
		return nil
	}
	if err != nil {
		return err
	}
	defer f.Close()

	sc := bufio.NewScanner(f)
	sc.Buffer(make([]byte, 0, 64*1024), 16*1024*1024)
	applied, skipped, lineNo := 0, 0, 0
	for sc.Scan() {
		lineNo++
		line := sc.Bytes()
		if len(line) == 0 {
			continue
		}
		var e walEntry
		if err := json.Unmarshal(line, &e); err != nil {
			skipped++
			log.Printf("persist: skipping malformed WAL line %d: %v", lineNo, err)
			continue
		}
		switch e.Op {
		case "add":
			idx := s.getOrCreateLocked(e.Namespace)
			idx.Add(e.ID, e.Vector, e.Text)
			s.records[e.Namespace][e.ID] = record{ID: e.ID, Vector: e.Vector, Text: e.Text}
			applied++
		case "del":
			if idx, ok := s.indices[e.Namespace]; ok {
				idx.Delete(e.ID)
				delete(s.records[e.Namespace], e.ID)
			}
			applied++
		default:
			skipped++
			log.Printf("persist: skipping WAL line %d with unknown op %q", lineNo, e.Op)
		}
	}
	if err := sc.Err(); err != nil {
		return fmt.Errorf("persist: reading WAL: %w", err)
	}
	if applied > 0 || skipped > 0 {
		log.Printf("persist: replayed %d WAL entrie(s), skipped %d", applied, skipped)
	}
	return nil
}

// appendWAL logs one mutation. Caller must hold s.mu for writing.
func (s *IndexServer) appendWAL(e walEntry) error {
	if s.wal == nil {
		return nil
	}
	data, err := json.Marshal(e)
	if err != nil {
		return err
	}
	if _, err := s.wal.Write(append(data, '\n')); err != nil {
		return err
	}
	if s.fsync {
		return s.wal.Sync()
	}
	return nil
}

// saveNamespaceLocked writes one namespace snapshot atomically (temp file +
// rename). Caller must hold s.mu.
func (s *IndexServer) saveNamespaceLocked(namespace string) error {
	recs := s.records[namespace]
	snap := namespaceSnapshot{Dim: s.defaultDim, Records: make([]record, 0, len(recs))}
	for _, r := range recs {
		snap.Records = append(snap.Records, r)
	}
	// Deterministic order keeps snapshots diff-friendly and reproducible.
	sort.Slice(snap.Records, func(i, j int) bool { return snap.Records[i].ID < snap.Records[j].ID })

	data, err := json.Marshal(snap)
	if err != nil {
		return err
	}
	final := s.snapshotPath(namespace)
	tmp := final + ".tmp"
	if err := os.WriteFile(tmp, data, 0o644); err != nil {
		return err
	}
	return os.Rename(tmp, final)
}

// Snapshot persists all namespaces to disk and, on full success, truncates the
// WAL (everything it contained is now captured by the snapshots). Safe to call
// concurrently with reads and writes.
func (s *IndexServer) Snapshot() (int, error) {
	if s.dataDir == "" {
		return 0, nil
	}
	// Full write lock: mutations must not land between a namespace being
	// snapshotted and the WAL being truncated, or they would be lost.
	s.mu.Lock()
	defer s.mu.Unlock()
	count := 0
	for ns := range s.records {
		if err := s.saveNamespaceLocked(ns); err != nil {
			return count, err
		}
		count++
	}
	if s.wal != nil {
		if err := s.wal.Truncate(0); err != nil {
			return count, err
		}
		if s.fsync {
			if err := s.wal.Sync(); err != nil {
				return count, err
			}
		}
	}
	return count, nil
}

// getOrCreateLocked returns the index for namespace, creating it if needed.
// Caller must hold s.mu for writing.
func (s *IndexServer) getOrCreateLocked(namespace string) *hybrid.Index {
	if idx, ok := s.indices[namespace]; ok {
		return idx
	}
	idx := hybrid.New(s.defaultDim)
	s.indices[namespace] = idx
	s.records[namespace] = make(map[string]record)
	return idx
}

// lookup returns the index for namespace or nil. Unlike getOrCreateLocked it
// never creates namespaces, so reads cannot pollute memory or disk.
func (s *IndexServer) lookup(namespace string) *hybrid.Index {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.indices[namespace]
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

// Handler returns the HTTP mux. The routes match pkg/server exactly and add
// POST /snapshot.
func (s *IndexServer) Handler() http.Handler {
	mux := http.NewServeMux()

	mux.HandleFunc("/health", func(w http.ResponseWriter, r *http.Request) {
		s.mu.RLock()
		namespaces := len(s.indices)
		docs := 0
		for _, recs := range s.records {
			docs += len(recs)
		}
		s.mu.RUnlock()
		json.NewEncoder(w).Encode(map[string]interface{}{
			"status":     "healthy",
			"namespaces": namespaces,
			"documents":  docs,
			"service":    "sigma-index",
			"persistent": s.dataDir != "",
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
		if !namespaceRe.MatchString(req.Namespace) {
			http.Error(w, "invalid namespace: must match "+namespaceRe.String(), 400)
			return
		}

		// Index mutation, record bookkeeping, and WAL append happen under one
		// write lock so a concurrent add/delete of the same id cannot leave
		// the index and the record store disagreeing.
		s.mu.Lock()
		idx := s.getOrCreateLocked(req.Namespace)
		idx.Add(req.ID, req.Vector, req.Text)
		s.records[req.Namespace][req.ID] = record{ID: req.ID, Vector: req.Vector, Text: req.Text}
		err := s.appendWAL(walEntry{Op: "add", Namespace: req.Namespace, ID: req.ID, Vector: req.Vector, Text: req.Text})
		s.mu.Unlock()
		if err != nil {
			log.Printf("persist: WAL append failed: %v", err)
			http.Error(w, "write accepted in memory but not persisted: "+err.Error(), 500)
			return
		}
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

		var results []hybrid.SearchResult
		if idx := s.lookup(req.Namespace); idx != nil {
			if len(req.Vector) > 0 && req.Text != "" {
				results = idx.Search(req.Vector, req.Text, req.K)
			} else if len(req.Vector) > 0 {
				results = idx.SearchVector(req.Vector, req.K)
			} else if req.Text != "" {
				results = idx.SearchText(req.Text, req.K)
			}
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

		s.mu.Lock()
		var err error
		if idx, ok := s.indices[req.Namespace]; ok {
			idx.Delete(req.ID)
			delete(s.records[req.Namespace], req.ID)
			err = s.appendWAL(walEntry{Op: "del", Namespace: req.Namespace, ID: req.ID})
		}
		s.mu.Unlock()
		if err != nil {
			log.Printf("persist: WAL append failed: %v", err)
			http.Error(w, "delete applied in memory but not persisted: "+err.Error(), 500)
			return
		}
		json.NewEncoder(w).Encode(map[string]string{"status": "deleted", "id": req.ID})
	})

	mux.HandleFunc("/snapshot", func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "POST only", 405)
			return
		}
		n, err := s.Snapshot()
		if err != nil {
			http.Error(w, err.Error(), 500)
			return
		}
		json.NewEncoder(w).Encode(map[string]interface{}{"status": "ok", "namespaces": n})
	})

	return mux
}
