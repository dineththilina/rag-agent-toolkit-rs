# agent-toolkit (Rust)

A custom AI agent with RAG, tool-calling, conversational memory, and a web UI —
a single self-contained Rust binary. No Python, no pip, no Docker, no external
database. The vector store is embedded in the process and the HTML UI is baked
into the binary at compile time.

Point it at **any OpenAI-compatible API** — OpenAI, Groq, Together, OpenRouter,
Ollama, LM Studio, and others — and it fetches the live model list from that
endpoint. No hardcoded model names.

---

## Why this design

- **One binary, zero services.** `cargo build --release` produces a single
  executable with the UI embedded. The vector index lives in-process and
  persists to a `vectors.json` file. Nothing else to install or run.
- **Provider-agnostic.** Talks the OpenAI-compatible wire format, so any
  provider that implements it works. Models that don't support tool-calling
  (common with small local models) are detected and fall back to plain chat
  automatically.
- **Live model list.** The setup screen calls `/models` on whatever API base you
  give it and shows exactly what that endpoint offers right now.
- **Tiny dependency tree.** No ML model weights downloaded locally — embeddings
  are computed by whatever API you point at (or a local Ollama embed model).

---

## Architecture

```
                ┌──────────────────────────────────────┐
   browser ───► │  axum server (single Rust binary)     │
                │                                        │
                │  GET  /            embedded HTML UI    │
                │  GET  /api/models  live model fetch ───┼──► any OpenAI-compatible
                │  POST /api/config  save + build index  │        /models
                │  POST /api/chat    agent loop ─────────┼──► /chat/completions
                │  POST /api/rag     direct retrieval    │        /embeddings
                └───────────────┬────────────────────────┘
                                │
                  ┌─────────────┴───────────────┐
                  │                             │
         ┌────────▼─────────┐          ┌────────▼────────┐
         │ embedded vectors │          │  data/ (docs)   │
         │ Vec + cosine     │◄─ embed ─│  txt, md, pdf   │
         │ → vectors.json   │          └─────────────────┘
         └──────────────────┘

   Agent loop: user msg ─► model ─► (tool_calls?) ─► run tools ─► model ─► answer
   Tools: search_knowledge_base (RAG) · calculator · country_facts (HTTP)
```

The vector store is a plain `Vec` of `{text, source, embedding}` scored by
brute-force cosine similarity. For a demo-scale corpus this is faster than a
network round-trip to a vector DB and needs no setup. It persists to
`vectors.json` and rebuilds automatically when the documents or embedding model
change.

---

## Prerequisites

1. **Rust** (stable). Install from https://rustup.rs:
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   ```
2. **A free Groq key** (the default). Get one in under a minute at
   [console.groq.com/keys](https://console.groq.com/keys) — no credit card. Paste
   it into Settings on first launch. That's the only thing you need.

   Prefer another provider? Settings also has presets for OpenAI, a local Ollama
   (no key), and LM Studio, or any OpenAI-compatible API address.

Document search needs nothing extra: embeddings use a built-in local method, so
RAG works with just the free chat key. No Docker, no database, no model
downloads — the vector store is embedded in the binary.

---

## Run

```bash
cargo run --release
# or: bash run.sh
```

Then open **http://localhost:3000**. The app boots **straight into the chat**.

It uses **Groq** (a free hosted AI) by default. On first launch you'll see a
one-line prompt to paste a free Groq key — get one in under a minute at
[console.groq.com/keys](https://console.groq.com/keys) (no credit card), paste
it in Settings, and you're chatting. Document search works even before that,
because embeddings run with a built-in local method that needs no key.

### Uploading your own PDFs

Click **Add** in the Documents panel (left side). Pick one or more PDF, TXT, or
MD files. PDFs are read in your browser with Mozilla's pdf.js — the same engine
Firefox uses — which handles real-world PDFs far more reliably than server-side
parsers, and means your files never leave your machine as raw bytes (only the
extracted text is sent to be indexed). Documents appear in the panel immediately
and persist across restarts. Scanned/image-only PDFs are detected and reported
clearly (they'd need OCR first). Sample documents are preloaded so search works
right away.

### Switching providers

Click the gear icon for Settings. Quick presets cover Groq, OpenAI, Ollama
(local), and LM Studio, or type any OpenAI-compatible API address and key. The
panel slides over the chat and never blocks it.

---

## Embeddings across providers

Embeddings always use the OpenAI-compatible `/embeddings` endpoint. Most
providers serve it on the same base URL as chat, so you usually leave the
embeddings fields blank and they reuse your chat API.

If your chat provider has **no** embeddings endpoint (Groq is the common case),
open **"Embeddings on a different provider"** in setup and point it at:
- OpenAI: base `https://api.openai.com/v1`, model `text-embedding-3-small`, your OpenAI key, or
- A local Ollama: base `http://localhost:11434/v1`, model `nomic-embed-text`, no key.

---

## Using it

**Agent Chat tab** — the full agent: tool-calling, memory, and a panel under each
reply showing which tools ran and which documents were retrieved.

**Docs Q&A tab** — direct semantic search over your documents. Type a query, get
ranked passages with relevance scores, click one to read the full excerpt. No
LLM call, just embeddings + cosine search.

Try:
- `How long does the H1 battery last?` (RAG)
- `Annual cost of 10 robots on Growth with the discount?` (RAG + calculator)
- `What's the capital of Portugal?` then `And its population?` (HTTP tool + memory)

---

## Build verification status — read this

This project was authored in an environment **without a Rust toolchain**, so it
has **not been through `cargo build` yet**. The module structure, types, and API
shapes are internally consistent and cross-checked by hand, but Rust is strict
and a couple of third-party crate APIs may differ slightly from what's written.

**Expect to run `cargo build` once and fix 0–5 small errors.** To make that fast:

```bash
cargo build 2>&1 | head -40
```

Most likely spots, in order:
1. `src/rag.rs` — the `text-splitter` chunking call (`TextSplitter::new` /
   `.chunks`). If the API differs, it's a one-line fix; check the crate docs.
2. `src/rag.rs` — `pdf_extract::extract_text_from_mem`. If renamed, check the
   `pdf-extract` docs for the current function name.

Everything else (axum routes, serde types, the agent loop, the calculator
parser, the embedded vector store) uses stable, well-established APIs.

---

## Project layout

```
agent-toolkit-rs/
├── Cargo.toml              # pinned dependencies
├── config.example.toml     # config template (UI writes config.toml for you)
├── run.sh                  # cargo run --release
├── src/
│   ├── main.rs             # axum server, routes, startup, embeds the UI
│   ├── config.rs           # load/save config.toml
│   ├── models.rs           # live model-list fetch from /models
│   ├── rag.rs              # load docs, chunk, embed, embedded vector store
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
| POST   | `/api/config`  | Save config, build the embedded index              |
| GET    | `/api/models`  | Live model list: `?base=<url>&key=<key>`           |
| POST   | `/api/chat`    | Agent chat: `{message, session_id}`                |
| POST   | `/api/rag`     | Direct retrieval: `{query}`                        |
| POST   | `/api/upload`  | Upload PDF/TXT/MD files (multipart) into the index |
| GET    | `/api/sources` | List indexed documents with chunk counts          |

---

## Notes

- Session memory is in-process (a `DashMap`). Restarting the server clears it.
- The vector index persists to `vectors.json` and survives restarts. Delete it
  (or change the documents / embedding model) to force a rebuild.
- `config.toml` holds your API key in plaintext and is gitignored. Treat it like
  any local secrets file.
- Models without tool-calling support still work for plain chat; the agent
  detects the lack of tool support and retries without tools.
