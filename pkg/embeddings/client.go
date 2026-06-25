// Package embeddings provides a client for generating vector embeddings
// via the Ryzanstein API (/v1/embeddings endpoint).
// This wires sigma-index to the ecosystem's central inference service.
package embeddings

import (
	"bytes"
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"time"
)

type Client struct {
	baseURL    string
	model      string
	httpClient *http.Client
}

type embeddingRequest struct {
	Input string `json:"input"`
	Model string `json:"model"`
}

type embeddingResponse struct {
	Data []struct {
		Embedding []float32 `json:"embedding"`
		Index     int       `json:"index"`
	} `json:"data"`
}

func NewClient() *Client {
	baseURL := os.Getenv("RYZANSTEIN_URL")
	if baseURL == "" {
		baseURL = "http://localhost:8000"
	}
	model := os.Getenv("EMBEDDING_MODEL")
	if model == "" {
		model = "ryzanstein-bitnet-7b"
	}
	return &Client{
		baseURL:    baseURL,
		model:      model,
		httpClient: &http.Client{Timeout: 30 * time.Second},
	}
}

func (c *Client) Embed(text string) ([]float32, error) {
	payload, _ := json.Marshal(embeddingRequest{Input: text, Model: c.model})

	resp, err := c.httpClient.Post(
		c.baseURL+"/v1/embeddings",
		"application/json",
		bytes.NewReader(payload),
	)
	if err != nil {
		return nil, fmt.Errorf("ryzanstein request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("ryzanstein returned %d", resp.StatusCode)
	}

	var result embeddingResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, fmt.Errorf("decode error: %w", err)
	}

	if len(result.Data) == 0 || len(result.Data[0].Embedding) == 0 {
		return nil, fmt.Errorf("empty embedding response")
	}

	return result.Data[0].Embedding, nil
}

func (c *Client) EmbedBatch(texts []string) ([][]float32, error) {
	results := make([][]float32, len(texts))
	for i, t := range texts {
		emb, err := c.Embed(t)
		if err != nil {
			return nil, fmt.Errorf("embed[%d] failed: %w", i, err)
		}
		results[i] = emb
	}
	return results, nil
}

func (c *Client) Available() bool {
	resp, err := c.httpClient.Get(c.baseURL + "/v1/models")
	if err != nil {
		return false
	}
	defer resp.Body.Close()
	return resp.StatusCode == 200
}
