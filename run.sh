#!/usr/bin/env bash
# One-command launch: ensure Qdrant is up, then build & run the server.
set -e

# 1. Start Qdrant if it isn't already running on :6333.
if ! curl -s http://localhost:6333/healthz >/dev/null 2>&1; then
  echo "Starting Qdrant (Docker)..."
  if command -v docker >/dev/null 2>&1; then
    docker run -d --name agent-toolkit-qdrant -p 6333:6333 qdrant/qdrant >/dev/null
    # Wait for it to come up.
    for i in {1..30}; do
      if curl -s http://localhost:6333/healthz >/dev/null 2>&1; then break; fi
      sleep 0.5
    done
    echo "Qdrant is up."
  else
    echo "Docker not found. Install Docker, or run Qdrant another way, then re-run."
    echo "  docker run -p 6333:6333 qdrant/qdrant"
    exit 1
  fi
else
  echo "Qdrant already running."
fi

# 2. Build (release) and run.
echo "Building (first build downloads crates, ~1-2 min)..."
cargo run --release
