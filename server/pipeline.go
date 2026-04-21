package main

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"time"
)

// DetectedSpan is a single PII entity found in the text.
type DetectedSpan struct {
	Text        string  `json:"text"`
	Type        string  `json:"type"`
	Source      string  `json:"source"`
	Replacement string  `json:"replacement"`
	Score       float64 `json:"score,omitempty"`
}

// inferRequest is the payload sent to the inference backend.
type inferRequest struct {
	Text      string `json:"text"`
	SessionID string `json:"session_id,omitempty"`
}

// inferResponse is what the inference backend returns.
type inferResponse struct {
	Text      string         `json:"text"`
	Detected  []DetectedSpan `json:"detected"`
	UsedModel bool           `json:"used_model"`
}

// Pipeline calls an HTTP inference backend for obfuscation.
// Fast-path stages (bloom, user table) will be added here later.
type Pipeline struct {
	inferenceURL string
	client       *http.Client
}

func NewPipeline(inferenceURL string) *Pipeline {
	return &Pipeline{
		inferenceURL: inferenceURL,
		client: &http.Client{
			Timeout: 30 * time.Second,
		},
	}
}

func (p *Pipeline) Process(ctx context.Context, text, sessionID string) (*obfuscateResponse, error) {
	body, err := json.Marshal(inferRequest{Text: text, SessionID: sessionID})
	if err != nil {
		return nil, err
	}

	req, err := http.NewRequestWithContext(ctx, http.MethodPost, p.inferenceURL, bytes.NewReader(body))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")

	resp, err := p.client.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("inference backend returned %d", resp.StatusCode)
	}

	var result inferResponse
	if err := json.NewDecoder(resp.Body).Decode(&result); err != nil {
		return nil, err
	}

	return &obfuscateResponse{
		Text:      result.Text,
		Detected:  result.Detected,
		UsedModel: result.UsedModel,
	}, nil
}
