#!/usr/bin/env bash
# One-command launch. No Docker, no external services — the vector store is
# embedded in the binary and persists to vectors.json.
set -e

echo "Building (first build downloads crates, ~1-2 min)..."
echo "Then open http://localhost:3000 and complete the setup screen."
cargo run --release
