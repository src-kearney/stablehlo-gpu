package main

import (
	"encoding/json"
	"flag"
	"log"
	"net/http"
	"os"
)

func main() {
	addr := flag.String("addr", ":8080", "listen address")
	inferenceURL := flag.String("inference-url", envOr("INFERENCE_URL", "http://localhost:8000/infer"), "inference backend URL")
	flag.Parse()

	pipeline := NewPipeline(*inferenceURL)
	mux := http.NewServeMux()
	mux.HandleFunc("POST /obfuscate", handleObfuscate(pipeline))

	log.Printf("listening on %s  inference=%s", *addr, *inferenceURL)
	if err := http.ListenAndServe(*addr, mux); err != nil {
		log.Fatal(err)
	}
}

type obfuscateRequest struct {
	Text      string `json:"text"`
	SessionID string `json:"session_id"`
}

type obfuscateResponse struct {
	Text      string          `json:"text"`
	Detected  []DetectedSpan  `json:"detected"`
	UsedModel bool            `json:"used_model"`
}

func handleObfuscate(p *Pipeline) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		var req obfuscateRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			http.Error(w, "bad request", http.StatusBadRequest)
			return
		}
		if req.Text == "" {
			http.Error(w, "text is required", http.StatusBadRequest)
			return
		}

		result, err := p.Process(r.Context(), req.Text, req.SessionID)
		if err != nil {
			log.Printf("pipeline error: %v", err)
			http.Error(w, "inference error", http.StatusInternalServerError)
			return
		}

		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(result)
	}
}

func envOr(key, fallback string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return fallback
}
