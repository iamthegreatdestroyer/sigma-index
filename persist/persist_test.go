package persist

import (
	"bytes"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"testing"
)

// crash simulates a hard crash: it releases the WAL file handle without
// snapshotting, leaving the on-disk state exactly as an abrupt process exit
// would. (Releasing the handle matters on Windows, where open files cannot be
// deleted by the TempDir cleanup.)
func crash(s *IndexServer) {
	if s.wal != nil {
		s.wal.Close()
		s.wal = nil
	}
}

// doJSON drives the HTTP handler the same way real clients do.
func doJSON(t *testing.T, h http.Handler, method, path string, body interface{}) (int, map[string]interface{}) {
	t.Helper()
	var buf bytes.Buffer
	if body != nil {
		if err := json.NewEncoder(&buf).Encode(body); err != nil {
			t.Fatalf("encode request: %v", err)
		}
	}
	req := httptest.NewRequest(method, path, &buf)
	rec := httptest.NewRecorder()
	h.ServeHTTP(rec, req)
	var out map[string]interface{}
	if rec.Body.Len() > 0 {
		_ = json.Unmarshal(rec.Body.Bytes(), &out)
	}
	return rec.Code, out
}

func addDoc(t *testing.T, h http.Handler, ns, id string, vec []float32, text string) {
	t.Helper()
	code, _ := doJSON(t, h, "POST", "/add", map[string]interface{}{
		"namespace": ns, "id": id, "vector": vec, "text": text,
	})
	if code != 200 {
		t.Fatalf("add %s/%s: status %d", ns, id, code)
	}
}

// searchIDs returns the result ids of a text search.
func searchIDs(t *testing.T, h http.Handler, ns, text string, k int) []string {
	t.Helper()
	code, out := doJSON(t, h, "POST", "/search", map[string]interface{}{
		"namespace": ns, "text": text, "k": k,
	})
	if code != 200 {
		t.Fatalf("search %s %q: status %d", ns, text, code)
	}
	var ids []string
	if results, ok := out["results"].([]interface{}); ok {
		for _, r := range results {
			m := r.(map[string]interface{})
			ids = append(ids, m["ID"].(string))
		}
	}
	return ids
}

// TestSnapshotRoundTrip proves that data added through the HTTP API survives a
// Snapshot + fresh Open from the same dir (simulating a clean restart).
func TestSnapshotRoundTrip(t *testing.T) {
	dir := t.TempDir()

	s1, err := New(4, dir)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	h1 := s1.Handler()

	addDoc(t, h1, "docs", "a", []float32{1, 0, 0, 0}, "alpha apple")
	addDoc(t, h1, "docs", "b", []float32{0, 1, 0, 0}, "beta banana")
	addDoc(t, h1, "notes", "x", []float32{0, 0, 1, 0}, "gamma grape")

	if n, err := s1.Snapshot(); err != nil || n != 2 {
		t.Fatalf("Snapshot: n=%d err=%v", n, err)
	}
	if err := s1.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}

	// Fresh instance from the same dir = restart.
	s2, err := New(4, dir)
	if err != nil {
		t.Fatalf("reload New: %v", err)
	}
	defer s2.Close()
	h2 := s2.Handler()

	if got := len(s2.indices); got != 2 {
		t.Fatalf("namespaces after reload = %d, want 2", got)
	}

	// Vector search must return the same nearest neighbor.
	code, out := doJSON(t, h2, "POST", "/search", map[string]interface{}{
		"namespace": "docs", "vector": []float32{1, 0, 0, 0}, "k": 1,
	})
	if code != 200 {
		t.Fatalf("vector search: status %d", code)
	}
	results := out["results"].([]interface{})
	if len(results) == 0 || results[0].(map[string]interface{})["ID"] != "a" {
		t.Fatalf("vector search after reload = %+v, want id=a", results)
	}

	// Text search must still find the reconstructed BM25 docs.
	if ids := searchIDs(t, h2, "docs", "banana", 1); len(ids) == 0 || ids[0] != "b" {
		t.Fatalf("text search after reload = %v, want [b]", ids)
	}
	if ids := searchIDs(t, h2, "notes", "grape", 1); len(ids) == 0 || ids[0] != "x" {
		t.Fatalf("notes search after reload = %v, want [x]", ids)
	}
}

// TestWALReplaySurvivesCrash proves that acknowledged writes survive a crash
// even when no snapshot was ever taken: the WAL alone rebuilds the state.
func TestWALReplaySurvivesCrash(t *testing.T) {
	dir := t.TempDir()

	s1, err := New(4, dir)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	h1 := s1.Handler()
	addDoc(t, h1, "docs", "a", []float32{1, 0, 0, 0}, "alpha apple")
	addDoc(t, h1, "docs", "b", []float32{0, 1, 0, 0}, "beta banana")
	// No Snapshot, no Close: simulate a hard crash.
	crash(s1)

	s2, err := New(4, dir)
	if err != nil {
		t.Fatalf("reopen: %v", err)
	}
	h2 := s2.Handler()
	if ids := searchIDs(t, h2, "docs", "apple", 1); len(ids) == 0 || ids[0] != "a" {
		t.Fatalf("after WAL replay, search apple = %v, want [a]", ids)
	}
	if ids := searchIDs(t, h2, "docs", "banana", 1); len(ids) == 0 || ids[0] != "b" {
		t.Fatalf("after WAL replay, search banana = %v, want [b]", ids)
	}

	// Deletes must replay too.
	if code, _ := doJSON(t, h2, "POST", "/delete", map[string]interface{}{
		"namespace": "docs", "id": "a",
	}); code != 200 {
		t.Fatalf("delete: status %d", code)
	}
	// Crash again.
	crash(s2)
	s3, err := New(4, dir)
	if err != nil {
		t.Fatalf("second reopen: %v", err)
	}
	defer s3.Close()
	h3 := s3.Handler()
	if ids := searchIDs(t, h3, "docs", "apple", 5); len(ids) != 0 {
		t.Fatalf("deleted doc came back after replay: %v", ids)
	}
	if ids := searchIDs(t, h3, "docs", "banana", 1); len(ids) == 0 || ids[0] != "b" {
		t.Fatalf("surviving doc lost after replay: %v", ids)
	}
}

// TestSnapshotTruncatesWAL: once a full snapshot succeeds the WAL must be
// empty, and a reload from snapshot+empty WAL must still be complete.
func TestSnapshotTruncatesWAL(t *testing.T) {
	dir := t.TempDir()

	s1, err := New(4, dir)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	h1 := s1.Handler()
	addDoc(t, h1, "docs", "a", []float32{1, 0, 0, 0}, "alpha apple")

	walPath := filepath.Join(dir, walName)
	if fi, err := os.Stat(walPath); err != nil || fi.Size() == 0 {
		t.Fatalf("WAL should be non-empty before snapshot: fi=%v err=%v", fi, err)
	}
	if _, err := s1.Snapshot(); err != nil {
		t.Fatalf("Snapshot: %v", err)
	}
	if fi, err := os.Stat(walPath); err != nil || fi.Size() != 0 {
		t.Fatalf("WAL should be empty after snapshot: size=%d err=%v", fi.Size(), err)
	}
	crash(s1)

	s2, err := New(4, dir)
	if err != nil {
		t.Fatalf("reopen: %v", err)
	}
	defer s2.Close()
	if ids := searchIDs(t, s2.Handler(), "docs", "apple", 1); len(ids) == 0 || ids[0] != "a" {
		t.Fatalf("doc lost after snapshot+truncate: %v", ids)
	}
}

// TestTornWALLine: a partially-written final WAL line (torn write during a
// crash) must not prevent recovery of the intact entries before it.
func TestTornWALLine(t *testing.T) {
	dir := t.TempDir()

	s1, err := New(4, dir)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	addDoc(t, s1.Handler(), "docs", "a", []float32{1, 0, 0, 0}, "alpha apple")
	crash(s1)

	// Simulate a torn write: append garbage that is not valid JSON.
	f, err := os.OpenFile(filepath.Join(dir, walName), os.O_APPEND|os.O_WRONLY, 0o644)
	if err != nil {
		t.Fatalf("open wal: %v", err)
	}
	if _, err := f.WriteString(`{"op":"add","ns":"docs","id":"tor`); err != nil {
		t.Fatalf("write torn line: %v", err)
	}
	f.Close()

	s2, err := New(4, dir)
	if err != nil {
		t.Fatalf("reopen with torn WAL: %v", err)
	}
	defer s2.Close()
	if ids := searchIDs(t, s2.Handler(), "docs", "apple", 1); len(ids) == 0 || ids[0] != "a" {
		t.Fatalf("intact entry lost due to torn line: %v", ids)
	}
}

// TestCorruptSnapshotSkipped: one corrupt snapshot file must not take down the
// whole server or the healthy namespaces.
func TestCorruptSnapshotSkipped(t *testing.T) {
	dir := t.TempDir()

	s1, err := New(4, dir)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	addDoc(t, s1.Handler(), "good", "a", []float32{1, 0, 0, 0}, "alpha apple")
	if _, err := s1.Snapshot(); err != nil {
		t.Fatalf("Snapshot: %v", err)
	}
	crash(s1)
	if err := os.WriteFile(filepath.Join(dir, "bad.json"), []byte("{not json"), 0o644); err != nil {
		t.Fatalf("write corrupt snapshot: %v", err)
	}

	s2, err := New(4, dir)
	if err != nil {
		t.Fatalf("reopen with corrupt snapshot: %v", err)
	}
	defer s2.Close()
	if got := len(s2.indices); got != 1 {
		t.Fatalf("namespaces = %d, want 1 (bad.json skipped)", got)
	}
	if ids := searchIDs(t, s2.Handler(), "good", "apple", 1); len(ids) == 0 || ids[0] != "a" {
		t.Fatalf("healthy namespace lost: %v", ids)
	}
}

// TestInvalidNamespaceRejected: names that could escape the data dir or
// collide on disk are rejected at the API boundary.
func TestInvalidNamespaceRejected(t *testing.T) {
	s, err := New(4, t.TempDir())
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer s.Close()
	h := s.Handler()

	for _, ns := range []string{"../evil", "a/b", "a\\b", ".hidden", "x y", "wal.jsonl/../z"} {
		code, _ := doJSON(t, h, "POST", "/add", map[string]interface{}{
			"namespace": ns, "id": "1", "vector": []float32{1, 0, 0, 0}, "text": "t",
		})
		if code != 400 {
			t.Errorf("namespace %q: status %d, want 400", ns, code)
		}
	}
	// Sane names still work.
	for _, ns := range []string{"default", "in-my-head", "sigma_lang.v2", "A1"} {
		code, _ := doJSON(t, h, "POST", "/add", map[string]interface{}{
			"namespace": ns, "id": "1", "vector": []float32{1, 0, 0, 0}, "text": "t",
		})
		if code != 200 {
			t.Errorf("namespace %q: status %d, want 200", ns, code)
		}
	}
}

// TestSearchDoesNotCreateNamespace: reads must not pollute memory or disk with
// empty namespaces.
func TestSearchDoesNotCreateNamespace(t *testing.T) {
	s, err := New(4, t.TempDir())
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer s.Close()
	h := s.Handler()

	code, out := doJSON(t, h, "POST", "/search", map[string]interface{}{
		"namespace": "ghost", "text": "anything", "k": 5,
	})
	if code != 200 {
		t.Fatalf("search: status %d", code)
	}
	if out["count"].(float64) != 0 {
		t.Fatalf("search of unknown namespace returned results: %v", out)
	}
	if _, ok := s.indices["ghost"]; ok {
		t.Fatal("search created a namespace")
	}
	if _, out := doJSON(t, h, "GET", "/health", nil); out["namespaces"].(float64) != 0 {
		t.Fatalf("health reports phantom namespaces: %v", out)
	}
}

// TestPersistenceDisabled: empty data dir means pure in-memory operation.
func TestPersistenceDisabled(t *testing.T) {
	s, err := New(4, "")
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	h := s.Handler()
	addDoc(t, h, "docs", "a", []float32{1, 0, 0, 0}, "alpha apple")
	if ids := searchIDs(t, h, "docs", "apple", 1); len(ids) == 0 || ids[0] != "a" {
		t.Fatalf("in-memory search failed: %v", ids)
	}
	if n, err := s.Snapshot(); err != nil || n != 0 {
		t.Fatalf("Snapshot with persistence disabled: n=%d err=%v", n, err)
	}
	if _, out := doJSON(t, h, "GET", "/health", nil); out["persistent"].(bool) != false {
		t.Fatalf("health should report persistent=false: %v", out)
	}
}

// TestHybridSearchAfterReload: the RRF hybrid path works on rebuilt indices.
func TestHybridSearchAfterReload(t *testing.T) {
	dir := t.TempDir()
	s1, err := New(4, dir)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	h1 := s1.Handler()
	addDoc(t, h1, "docs", "a", []float32{1, 0, 0, 0}, "alpha apple pie recipe")
	addDoc(t, h1, "docs", "b", []float32{0, 1, 0, 0}, "beta banana bread recipe")
	s1.Close()

	s2, err := New(4, dir)
	if err != nil {
		t.Fatalf("reopen: %v", err)
	}
	defer s2.Close()
	code, out := doJSON(t, s2.Handler(), "POST", "/search", map[string]interface{}{
		"namespace": "docs", "vector": []float32{1, 0, 0, 0}, "text": "apple", "k": 2,
	})
	if code != 200 {
		t.Fatalf("hybrid search: status %d", code)
	}
	results := out["results"].([]interface{})
	if len(results) == 0 || results[0].(map[string]interface{})["ID"] != "a" {
		t.Fatalf("hybrid search after reload = %+v, want a first", results)
	}
}
