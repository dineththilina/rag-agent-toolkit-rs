# agent-toolkit (Rust)

A custom AI agent with **local-first hybrid vector RAG**, tool-calling,
conversational memory, and a web UI — a single self-contained Rust binary.
No Python, no pip, **no Docker**, **no Postgres**, and **no cloud vector
database**. Documents are chunked, embedded locally, stored in a local SQLite
vector database, and retrieved by semantic similarity fused with BM25 keyword
search.

Point the chat model at **any OpenAI-compatible API** — OpenAI, Groq, Together,
OpenRouter, Ollama, LM Studio, and others — and it fetches the live model list
from that endpoint. Retrieval, however, runs entirely on your machine.

---

## What retrieval looks like now

Uploaded documents are turned into a searchable knowledge base entirely on your
machine:

1. **Chunk** — each document is split with a deterministic, boundary-aware
   chunker (~3,500 characters with ~600 characters of overlap), so chunks don't
   cut through sentences and context carries across boundaries.
2. **Embed (locally)** — each chunk is embedded with
   [`fastembed`](https://crates.io/crates/fastembed) running a small English
   retrieval model (`all-MiniLM-L6-v2`, 384 dimensions) via local ONNX
   inference. **No embedding API key.** The model downloads once on first use
   and is cached for offline use afterwards.
3. **Store (locally, on disk)** — chunks and their vectors are written to a
   local SQLite database at **`./data/rag_vectors.sqlite`** using the
   [`sqlite-vec`](https://crates.io/crates/sqlite-vec) extension. The SQLite
   engine and the vector extension are compiled from vendored C source — there
   is no server to run.
4. **Retrieve (hybrid)** — at query time the question is embedded, the top‑k
   nearest chunks come from the vector store, the top‑k keyword chunks come from
   the preserved **BM25** index, and the two lists are merged with **Reciprocal
   Rank Fusion**, deduplicated by `document_id + chunk_id`, and returned with
   source metadata and scores.
5. **Ground** — the top retrieved chunks (with citations) are fed into the chat
   model's context. If nothing relevant is found, the assistant is told to say
   it couldn't find enough evidence in your files rather than guessing.

> Vectors are stored **locally** on disk. Embeddings are generated **locally**
> by default. **BM25 keyword search is preserved** as part of hybrid retrieval.
> **No cloud vector database is required.**

---

## Why this design

- **One binary, zero services.** `cargo build --release` produces a single
  executable with the UI embedded. The vector index is a local SQLite file.
  Nothing else to install or run — no Docker, no Postgres, no managed cloud DB.
- **Local-first retrieval.** Embeddings are computed locally by fastembed and
  vectors live in a local SQLite file. Your document text never has to leave the
  machine to be indexed or searched.
- **Hybrid, not either/or.** Semantic (vector) recall plus BM25 keyword
  precision, fused with RRF. BM25 also acts as a graceful fallback if the
  embedding model can't load.
- **Provider-agnostic chat.** The chat model talks the OpenAI-compatible wire
  format, so any provider that implements it works. The setup screen calls
  `/models` on whatever API base you give it.

---

## Architecture

```
                ┌────────────────────────────────────────────┐
   browser ───► │  axum server (single Rust binary)           │
                │                                              │
                │  GET  /            embedded HTML UI          │
                │  GET  /api/models  live model fetch ─────────┼─► any OpenAI-compatible
                │  POST /api/config  save settings             │        /models
                │  POST /api/chat    grounded chat ────────────┼─► /chat/completions
                │  POST /api/rag     hybrid retrieval          │
                │  POST /api/upload  chunk + embed + store     │
                │  POST /api/rebuild recreate the index        │
                └───────────────┬──────────────────────────────┘
                                │
        ┌───────────────────────┴──────────────────────────┐
        │                 RAG subsystem                      │
        │                                                    │
        │  chunk ─► embed (fastembed, local ONNX) ─┐         │
        │                                          ▼         │
        │   ┌──────────────────┐        ┌────────────────────┐
        │   │  BM25 keyword     │        │  sqlite-vec store  │
        │   │  index (in-mem,   │        │  data/             │
        │   │  rebuilt on boot) │        │  rag_vectors.sqlite│
        │   └─────────┬─────────┘        └─────────┬──────────┘
        │             │   Reciprocal Rank Fusion   │
        │             └───────────► merge ◄────────┘
        │                            │ dedup by document_id + chunk_id
        │                            ▼
        │                  ranked, cited chunks → chat context
        └────────────────────────────────────────────────────┘
```

SQLite is the single source of truth. On startup the in-memory BM25 index is
rebuilt from the on-disk chunks, so **previously indexed documents stay
searchable across restarts** with no separate index file.

### Data model

Each chunk is stored with: `document_id`, `chunk_id`, source filename, the
original text chunk, a normalized text chunk, character/token counts, the
embedding vector, a `created_at` timestamp, and optional metadata JSON — enough
to trace every retrieved answer back to its exact source document and chunk.

### Retrieval settings (in `config.toml`)

| Setting           | Default                | Meaning                                       |
|-------------------|------------------------|-----------------------------------------------|
| `retrieval_mode`  | `hybrid`               | `keyword`, `vector`, or `hybrid`              |
| `top_k`           | `8`                    | how many chunks to retrieve                   |
| `min_similarity`  | `0.2`                  | cosine floor below which evidence is "weak"   |
| `embedding_model` | `all-MiniLM-L6-v2`     | local fastembed model                         |
| `embedding_dim`   | `384`                  | vector dimension (must match the model)       |
| `vector_db_path`  | `data/rag_vectors.sqlite` | local vector database file                 |

---

## Prerequisites

1. **Rust** (stable). Install from https://rustup.rs:
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   ```
2. **A free Groq key** for chat (the default). Get one in under a minute at
   [console.groq.com/keys](https://console.groq.com/keys) — no credit card.
   Paste it into Settings on first launch. Prefer another provider? Settings has
   presets for OpenAI, a local Ollama (no key), and LM Studio, or any
   OpenAI-compatible API address.
3. **The ONNX Runtime** for local embeddings. The default build links it
   *dynamically*, so it builds anywhere offline but needs the runtime present at
   launch — see [Local embeddings & the ONNX Runtime](#local-embeddings--the-onnx-runtime).
   (Even without it, the app still works: it falls back to BM25 keyword search.)

Retrieval needs **no cloud service and no embedding API key** — embeddings run
locally and vectors are stored in a local SQLite file.

---

## Run

```bash
cargo run --release
# or: bash run.sh
```

Then open **http://localhost:3000**. The app boots straight into the chat.

On first launch you'll be prompted to paste a free Groq key for the chat model.
Document indexing starts in the background: the embedding model downloads once
(then it's cached for offline use), documents are chunked, embedded, and written
to `./data/rag_vectors.sqlite`. Sample documents are preloaded so retrieval
works right away.

### Local embeddings & the ONNX Runtime

fastembed runs the embedding model with the ONNX Runtime. There are two build
modes:

- **Default (`local-embeddings`)** — builds offline everywhere by loading the
  ONNX Runtime *dynamically* at runtime. Make the runtime available to `ort`,
  for example:
  ```bash
  # Download the official ONNX Runtime, then point ort at the shared library:
  export ORT_DYLIB_PATH=/path/to/libonnxruntime.so   # libonnxruntime.dylib on macOS
  cargo run --release
  ```
- **Managed download (`local-embeddings-managed`)** — Cargo downloads a prebuilt
  ONNX Runtime at *build* time, so there's nothing to install at runtime
  (needs network to the ort CDN while building):
  ```bash
  cargo run --release --no-default-features --features local-embeddings-managed
  ```

If the runtime or the model can't be loaded, indexing and retrieval **fall back
to BM25 keyword search** automatically, and a clear message is logged — the app
keeps working.

> **Offline / keyword-only build.** For a fast build with no embedding stack at
> all, use `cargo build --no-default-features`. Retrieval then uses BM25 over the
> same local SQLite store.

### Uploading your own documents

Click **Add** in the Documents panel. Pick PDF, TXT, or MD files. PDFs are read
in your browser with Mozilla's pdf.js (only the extracted text is sent to be
indexed). Each document is chunked, embedded, and stored locally; documents
appear immediately and **persist across restarts**. Re-uploading the same
filename replaces its chunks rather than duplicating them.

### Rebuilding the index

To recreate the local vector index — for example after changing the embedding
model, or to repair it — call:

```bash
curl -X POST http://localhost:3000/api/rebuild
```

Rebuild re-reads on-disk documents and re-embeds previously uploaded ones, then
recreates the local vector index. Nothing you uploaded is lost.

---

## Using it

**Agent Chat tab** — grounded chat: the most relevant retrieved chunks (with
citations) are placed in the model's context, plus tool-calling and memory.

**Docs Q&A tab** — direct hybrid retrieval over your documents. Type a query,
get ranked passages with relevance scores and source metadata, click one to read
the full excerpt. No LLM call — just retrieval.

Try:
- `How long does the H1 battery last?`
- `Annual cost of 10 robots on Growth with the discount?`
- `What's the uptime guarantee on the Premium plan?`

---

## Build & test status

This project builds and its tests pass on stable Rust:

```bash
cargo fmt --check     # formatting
cargo clippy          # lints (clean)
cargo test            # unit + integration tests
cargo build           # debug build
```

The test suite covers chunking, the embedding-provider interface (via a
deterministic mock so tests need no model download), vector-store insert/search,
hybrid merge + deduplication, retrieval-mode selection, and **persistence across
a simulated restart** using a temporary on-disk database.

---

## Project layout

```
agent-toolkit-rs/
├── Cargo.toml              # pinned dependencies + feature flags
├── config.example.toml     # config template (UI writes config.toml for you)
├── run.sh                  # cargo run --release
├── src/
│   ├── main.rs             # axum server, routes, startup, embeds the UI
│   ├── lib.rs              # library surface (used by integration tests)
│   ├── config.rs           # load/save config.toml (incl. retrieval settings)
│   ├── models.rs           # live model-list fetch from /models
│   ├── agent.rs            # grounded chat + per-session memory
│   ├── rag.rs              # RAG orchestrator (chunk → embed → store → retrieve)
│   ├── rag/
│   │   ├── types.rs        # chunk record, hit, retrieval mode, config
│   │   ├── chunk.rs        # deterministic boundary-aware chunker
│   │   ├── bm25.rs         # BM25 keyword index (preserved)
│   │   ├── embed.rs        # EmbeddingProvider trait + deterministic mock
│   │   ├── fastembed_provider.rs  # local fastembed embeddings
│   │   ├── store.rs        # sqlite-vec on-disk vector store
│   │   └── hybrid.rs       # Reciprocal Rank Fusion + dedup
│   └── api/
│       ├── config.rs       # /api/config, /api/models, AppState
│       ├── chat.rs         # /api/chat
│       ├── rag.rs          # /api/rag
│       └── upload.rs       # /api/upload, /api/sources, /api/remove, /api/rebuild
├── templates/
│   └── index.html          # single-file UI, embedded at compile time
├── tests/
│   └── rag_integration.rs  # black-box persistence + retrieval tests
└── data/                   # sample docs (Helios Robotics, fictional)
                            # + rag_vectors.sqlite (created at runtime, gitignored)
```

---

## Endpoints

| Method | Path           | Purpose                                            |
|--------|----------------|----------------------------------------------------|
| GET    | `/`            | Web UI                                              |
| GET    | `/health`      | Liveness check                                     |
| GET    | `/api/config`  | Current config (key redacted)                      |
| POST   | `/api/config`  | Save config                                        |
| GET    | `/api/models`  | Live model list: `?base=<url>&key=<key>`           |
| POST   | `/api/chat`    | Grounded chat: `{message, session_id}`             |
| POST   | `/api/rag`     | Hybrid retrieval: `{query}`                        |
| POST   | `/api/upload`  | Upload PDF/TXT/MD text into the index              |
| GET    | `/api/sources` | List indexed documents with chunk counts          |
| POST   | `/api/remove`  | Remove a document: `{name}`                        |
| POST   | `/api/rebuild` | Recreate the local vector index                    |

---

## Notes

- The vector database is a local SQLite file at `data/rag_vectors.sqlite`
  (gitignored). Delete it to force a full re-index, or call `/api/rebuild`.
- Embeddings are generated locally; the model is cached under
  `./.fastembed_cache` after the first download (override with
  `FASTEMBED_CACHE_DIR`).
- The chat transcript is saved locally in your browser (localStorage), so it
  survives page reloads and app restarts; use the **✎ New chat** button to clear
  it and start fresh. The backend conversation memory is in-process (a `DashMap`)
  and resets when the server restarts, but the document index persists.
- When a request fails (e.g. a provider rate-limit / 429), the error bubble shows
  a **↻ Retry** button that re-runs the same prompt in place.
- `config.toml` holds your chat API key in plaintext and is gitignored. Treat it
  like any local secrets file.
