# agent-toolkit (Rust)

A custom AI agent with RAG, tool-calling, conversational memory, and a web UI —
rewritten in Rust as a single self-contained binary. No Python, no pip, no
gigabytes of ML wheels. The frontend is embedded into the binary at compile
time, so the release build is one file plus a local Qdrant instance.

Point it at **any OpenAI-compatible API** — OpenAI, Groq, Ollama, LM Studio,
Anthropic, NVIDIA NIM — and it fetches the live model list from that endpoint.
No hardcoded model names.

---

## Why this design

- **One binary.** `cargo build --release` produces a single executable with the
  HTML UI baked in. Nothing scattered across your system.
- **Live model list.** The setup screen calls `/models` on whatever API base you
  give it and shows you exactly what that endpoint offers, today. No stale lists.
- **Minimal dependencies.** Talks to the model API and Qdrant over plain HTTP
  with `reqwest`. No embedding model weights downloaded locally — embeddings are
  computed by whatever API you point at (or a local Ollama embed model).
- **Qdrant for vectors.** Runs as a ~50 MB container or single binary. No Python.

---

## Architecture

```
                ┌──────────────────────────────────────┐
   browser ───► │  axum server (single Rust binary)     │
                │                                        │
                │  GET  /            embedded HTML UI    │
                │  GET  /api/models  live model fetch ───┼──► OpenAI-compatible
                │  POST /api/config  save + build index  │        /models
                │  POST /api/chat    agent loop ─────────┼──► /chat/completions
                │  POST /api/rag     direct retrieval    │        /embeddings
                └───────────────┬────────────────────────┘
                                │
                  ┌─────────────┴───────────────┐
                  │                             │
            ┌─────▼──────┐              ┌───────▼────────┐
            │  Qdrant    │              │  data/ (docs)  │
            │  (vectors) │◄─ embed ─────│  txt, md, pdf  │
            └────────────┘              └────────────────┘

   Agent loop: user msg ─► model ─► (tool_calls?) ─► dispatch tools ─► model ─► answer
   Tools: search_knowledge_base (RAG) · calculator · country_facts (HTTP)
```

---

## Prerequisites

1. **Rust** (stable). Install from https://rustup.rs:
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   ```
2. **Qdrant** (the vector store). Easiest via Docker:
   ```bash
   docker run -p 6333:6333 qdrant/qdrant
   ```
   Or download the single binary from https://github.com/qdrant/qdrant/releases.
3. **An OpenAI-compatible API.** Any of:
   - OpenAI key, base `https://api.openai.com/v1`
   - Groq (free) key, base `https://api.groq.com/openai/v1`
   - Ollama (fully local, no key), base `http://localhost:11434/v1`
     — run `ollama pull llama3.2` and `ollama pull nomic-embed-text` first
   - LM Studio (local), base `http://localhost:1234/v1`

---

## Run

```bash
# Easiest — starts Qdrant via Docker, then builds and runs:
bash run.sh
```

or manually:

```bash
docker run -d -p 6333:6333 qdrant/qdrant
cargo run --release
```

Then open **http://localhost:3000**. The setup screen appears on first launch.
Enter your API base + key, click **Fetch models**, pick one, and hit
**Save & start**. The chat UI loads.

---

## Using it

**Agent Chat tab** — full agent with tool-calling and memory. It decides per
message whether to search the docs, run a calculation, or call the country API,
and shows which tools it used below each reply.

**Docs Q&A tab** — direct semantic search over your documents. Type a query, get
ranked passages with relevance scores, click one to read the full excerpt. No
LLM call, just embeddings + vector search.

Both hit the same Rust backend. Try:
- `How long does the H1 battery last?` (RAG)
- `Annual cost of 10 robots on Growth with the discount?` (RAG + calculator)
- `What's the capital of Portugal?` then `And its population?` (HTTP tool + memory)

---

## Build verification status — read this

This project was written and reviewed carefully, but it was **authored in an
environment without a Rust toolchain**, so it has **not been through `cargo
build` yet**. The module structure, types, and API call shapes are internally
consistent, but Rust is strict and a few third-party crate APIs
(`text-splitter`, `pdf-extract`) may have minor signature differences from what's
written here.

**Expect to run `cargo build` once and fix 0–5 small errors** (a renamed method,
a changed argument). This is normal for a fresh Rust project pinned to specific
crate versions. To make that fast:

```bash
cargo build 2>&1 | head -40    # see the first errors with file:line
```

The most likely spots, in order:
1. `src/rag.rs` — the `text-splitter` chunking call (`TextSplitter::new` /
   `.chunks`). If the API differs, the fix is one or two lines; the crate docs
   show the current signature.
2. `src/rag.rs` — `pdf_extract::extract_text_from_mem`. If renamed, check the
   `pdf-extract` docs for the current function name.

Everything else (axum routes, serde types, the agent loop, the calculator
parser, the Qdrant REST calls) uses stable, well-established APIs.

If you'd rather I harden any specific module against a known crate version, say
which and I'll revise it.

---

## Project layout

```
agent-toolkit-rs/
├── Cargo.toml              # pinned dependencies
├── config.example.toml     # config template (UI writes config.toml for you)
├── run.sh                  # start Qdrant + build + run
├── src/
│   ├── main.rs             # axum server, routes, startup, embeds the UI
│   ├── config.rs           # load/save config.toml
│   ├── models.rs           # live model-list fetch from /models
│   ├── rag.rs              # load docs, chunk, embed, Qdrant store, retrieve
│   ├── agent.rs            # tool-calling loop + per-session memory
│   ├── tools.rs            # RAG tool, calculator (parser), country HTTP tool
│   └── api/
│       ├── mod.rs
│       ├── config.rs       # /api/config, /api/models, AppState
│       ├── chat.rs         # /api/chat
│       └── rag.rs          # /api/rag
├── templates/
│   └── index.html          # single-file UI, embedded at compile time
└── data/                   # sample docs (Helios Robotics, fictional)
    ├── company_overview.md
    ├── product_specs.txt
    ├── pricing.md
    └── support_faq.txt
```

---

## Endpoints

| Method | Path           | Purpose                                            |
|--------|----------------|----------------------------------------------------|
| GET    | `/`            | Web UI (setup screen or chat, depending on config) |
| GET    | `/health`      | Liveness check                                     |
| GET    | `/api/config`  | Current config (key redacted)                      |
| POST   | `/api/config`  | Save config, build the RAG index                   |
| GET    | `/api/models`  | Live model list: `?base=<url>&key=<key>`           |
| POST   | `/api/chat`    | Agent chat: `{message, session_id}`                |
| POST   | `/api/rag`     | Direct retrieval: `{query}`                        |

---

## Notes

- Session memory is in-process (a `DashMap`). Restarting the server clears it.
- The RAG index persists in Qdrant, so it survives restarts; delete the
  collection or restart Qdrant to force a re-ingest.
- `config.toml` holds your API key in plaintext and is gitignored. Treat it like
  any local secrets file.
