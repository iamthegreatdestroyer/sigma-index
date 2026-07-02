// sigma-index server — exposes HNSW + BM25 hybrid search as HTTP API.
// Used by sigmalang, In-My-Head, sigma-harvest, and any Python project
// needing vector + text search.
//
// Data is persistent by default: every /add and /delete is written to a WAL
// before it is acknowledged, full snapshots are taken periodically and on
// graceful shutdown, and state is rebuilt on startup. Routes are unchanged
// (POST /add, /search, /delete, GET /health) plus POST /snapshot.
//
// Usage: sigma-index-server [-port 8200] [-dim 768] [-data <dir>]
//                           [-snapshot-interval 60s] [-fsync] [-no-persist]
//
// The default data dir is chosen writable: $SIGMA_INDEX_DATA if set, else
// /var/lib/sigma-index if writable, else ~/.sigma-index.
package main

import (
	"context"
	"flag"
	"log"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"syscall"
	"time"

	"github.com/iamthegreatdestroyer/sigma-index/persist"
)

// defaultDataDir picks a writable default: env override, then
// /var/lib/sigma-index, then ~/.sigma-index.
func defaultDataDir() string {
	if d := os.Getenv("SIGMA_INDEX_DATA"); d != "" {
		return d
	}
	const sys = "/var/lib/sigma-index"
	if err := os.MkdirAll(sys, 0o755); err == nil {
		// Confirm we can actually write there.
		probe := filepath.Join(sys, ".wprobe")
		if f, err := os.Create(probe); err == nil {
			f.Close()
			os.Remove(probe)
			return sys
		}
	}
	if home, err := os.UserHomeDir(); err == nil {
		return filepath.Join(home, ".sigma-index")
	}
	return "./sigma-index-data"
}

func main() {
	port := flag.String("port", "8200", "Listen port")
	dim := flag.Int("dim", 768, "Default vector dimension")
	dataDir := flag.String("data", "", "Persistence data dir (default: $SIGMA_INDEX_DATA, /var/lib/sigma-index if writable, else ~/.sigma-index)")
	interval := flag.Duration("snapshot-interval", 60*time.Second, "Periodic snapshot interval (0 disables periodic snapshots)")
	fsync := flag.Bool("fsync", false, "fsync the WAL after every write (maximum durability, slower)")
	noPersist := flag.Bool("no-persist", false, "Disable persistence entirely (in-memory only)")
	flag.Parse()

	dir := ""
	if !*noPersist {
		dir = *dataDir
		if dir == "" {
			dir = defaultDataDir()
		}
	}

	srv, err := persist.Open(persist.Options{Dim: *dim, DataDir: dir, Fsync: *fsync})
	if err != nil {
		log.Fatalf("sigma-index: failed to init persistence at %s: %v", dir, err)
	}
	if dir != "" {
		log.Printf("sigma-index: persistence data dir = %s (fsync=%v)", dir, *fsync)
	} else {
		log.Printf("sigma-index: persistence DISABLED — data will not survive restart")
	}

	httpSrv := &http.Server{Addr: ":" + *port, Handler: srv.Handler()}

	// Periodic snapshots keep the WAL short and startup replay fast.
	stopTicker := make(chan struct{})
	if dir != "" && *interval > 0 {
		go func() {
			t := time.NewTicker(*interval)
			defer t.Stop()
			for {
				select {
				case <-t.C:
					if _, err := srv.Snapshot(); err != nil {
						log.Printf("sigma-index: periodic snapshot error: %v", err)
					}
				case <-stopTicker:
					return
				}
			}
		}()
	}

	// Graceful shutdown: final snapshot, close WAL, then drain connections.
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, syscall.SIGINT, syscall.SIGTERM)
	go func() {
		sig := <-sigCh
		log.Printf("sigma-index: received %s, snapshotting before exit", sig)
		close(stopTicker)
		if err := srv.Close(); err != nil {
			log.Printf("sigma-index: shutdown snapshot error: %v", err)
		} else {
			log.Printf("sigma-index: shutdown snapshot complete")
		}
		ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
		defer cancel()
		_ = httpSrv.Shutdown(ctx)
	}()

	log.Printf("sigma-index server listening on :%s (dim=%d)", *port, *dim)
	if err := httpSrv.ListenAndServe(); err != nil && err != http.ErrServerClosed {
		log.Fatal(err)
	}
	log.Printf("sigma-index: stopped")
}
