// sigma-index server — exposes HNSW + BM25 hybrid search as HTTP API.
// Used by sigmalang, In-My-Head, sigma-harvest, and any Python project
// needing vector + text search.
//
// Usage: sigma-index-server [-port 8200] [-dim 256]
package main

import (
	"flag"
	"log"

	"github.com/iamthegreatdestroyer/sigma-index/pkg/server"
)

func main() {
	port := flag.String("port", "8200", "Listen port")
	dim := flag.Int("dim", 256, "Default vector dimension")
	flag.Parse()

	srv := server.New(*dim)
	addr := ":" + *port
	log.Fatal(srv.ListenAndServe(addr))
}
