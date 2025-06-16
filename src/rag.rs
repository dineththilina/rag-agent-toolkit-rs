// src/rag.rs
//
// RAG pipeline with an EMBEDDED vector store — no external database, no Docker.
//
//   1. Load documents from data/ (txt, md, pdf)
//   2. Chunk with text-splitter (pure Rust, no ML weights)
//   3. Embed chunks via the configured embeddings endpoint
//   4. Store vectors in-process (Vec) and persist to vectors.json
//   5. retrieve(query) -> brute-force cosine similarity -> Vec<Hit>
//
// For a demo-scale corpus (hundreds to low-thousands of chunks), brute-force
// cosine over an in-memory Vec is faster than a network round-trip to a vector
// DB and needs zero setup. The index persists to a single JSON file so it
// survives restarts; delete vectors.json (or change the docs) to force a
// rebuild.

use anyhow::{Context, Result};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::path::Path;
use std::sync::Arc;
use text_splitter::TextSplitter;
use tokio::sync::RwLock;
use tracing::{info, warn};

use crate::config::Config;

// Where the persisted index lives, next to the binary.
const STORE_PATH: &str = "vectors.json";
// text-splitter sizes chunks by character count. 700 chars keeps each passage
// focused while staying well under typical embedding token limits.
const CHUNK_SIZE: usize = 700;

// ── Public types ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize)]
pub struct Hit {
    pub text:   String,
    pub source: String,
    pub score:  f32,
}

/// One stored chunk: its text, source file, content hash, and embedding vector.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct StoredChunk {
    text:      String,
    source:    String,
    hash:      String,
    embedding: Vec<f32>,
}

/// The on-disk / in-memory index. `fingerprint` identifies the corpus + embed
/// model so we know whether a persisted index is still valid for the current
/// config and documents. Public so the `SharedStore` alias can cross module
/// boundaries; its fields stay private to this module.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct VectorStore {
    fingerprint: String,
    chunks:      Vec<StoredChunk>,
}

/// Shared, hot-swappable store. Built once at startup (or after config save),
/// then read concurrently by every /chat and /rag request.
pub type SharedStore = Arc<RwLock<VectorStore>>;

pub fn new_shared_store() -> SharedStore {
    Arc::new(RwLock::new(VectorStore::default()))
}

// ── Document loading ─────────────────────────────────────────────────────────

fn load_txt(path: &Path) -> Result<String> {
    Ok(std::fs::read_to_string(path)?)
}

fn load_pdf(path: &Path) -> Result<String> {
    let bytes = std::fs::read(path)?;
    let text  = pdf_extract::extract_text_from_mem(&bytes)
        .context("extracting PDF text")?;
    Ok(text)
}

/// Extract text from uploaded file bytes based on the filename's extension.
/// Supports pdf, txt, and md. Used by the /api/upload endpoint.
pub fn extract_text(filename: &str, bytes: &[u8]) -> Result<String> {
    let ext = filename.rsplit('.').next().unwrap_or("").to_lowercase();
    match ext.as_str() {
        "pdf" => pdf_extract::extract_text_from_mem(bytes).context("reading PDF text"),
        "txt" | "md" | "markdown" | "text" => {
            String::from_utf8(bytes.to_vec()).context("reading text file (not valid UTF-8)")
        }
        other => anyhow::bail!("Unsupported file type '.{other}'. Use PDF, TXT, or MD."),
    }
}

struct Doc {
    source:  String,
    content: String,
}

fn load_all(data_dir: &str) -> Result<Vec<Doc>> {
    let mut docs = Vec::new();
    let entries  = std::fs::read_dir(data_dir)
        .with_context(|| format!("opening data dir '{data_dir}'"))?;

    for entry in entries {
        let entry = entry?;
        let path  = entry.path();
        if !path.is_file() { continue; }

        let ext    = path.extension().and_then(|e| e.to_str()).unwrap_or("").to_lowercase();
        let source = path.file_name().unwrap().to_string_lossy().into_owned();

        let content = match ext.as_str() {
            "txt" | "md" => load_txt(&path)?,
            "pdf"        => load_pdf(&path)?,
            _            => { warn!("skipping unsupported file: {source}"); continue; }
        };

        docs.push(Doc { source, content });
    }

    docs.sort_by(|a, b| a.source.cmp(&b.source));
    Ok(docs)
}

// ── Chunking ─────────────────────────────────────────────────────────────────

struct Chunk {
    source: String,
    text:   String,
    hash:   String,
}

fn chunk_docs(docs: Vec<Doc>) -> Vec<Chunk> {
    let splitter = TextSplitter::new(CHUNK_SIZE);
    let mut chunks = Vec::new();

    for doc in docs {
        for piece in splitter.chunks(&doc.content) {
            let trimmed = piece.trim().to_string();
            if trimmed.len() < 40 { continue; }   // skip tiny fragments

            let mut hasher = Sha256::new();
            hasher.update(trimmed.as_bytes());
            let hash = format!("{:x}", hasher.finalize())[..16].to_string();

            chunks.push(Chunk { source: doc.source.clone(), text: trimmed, hash });
        }
    }
    chunks
}

// ── Embeddings ───────────────────────────────────────────────────────────────
//
// Two paths:
//   * model == "local"  → built-in pure-Rust embedder (no API, no key, no
//                         download). Hashes character n-grams into a fixed
//                         vector with TF weighting. Not as strong as a neural
//                         embedding, but works well for keyword-style search
//                         over a small document set, and needs nothing.
//   * otherwise         → OpenAI-compatible /embeddings endpoint.
//
// This is what lets the app do document search with only a free Groq chat key
// (Groq has no embeddings endpoint of its own).

const LOCAL_EMBED_DIM: usize = 512;

/// Deterministic local embedding: character 3-gram hashing with term-frequency
/// weighting, then L2-normalised. Same text always yields the same vector, and
/// similar texts yield similar vectors, which is all cosine search needs.
fn local_embed(text: &str) -> Vec<f32> {
    let mut v = vec![0.0f32; LOCAL_EMBED_DIM];
    let lower = text.to_lowercase();
    let chars: Vec<char> = lower.chars().collect();

    // Character 3-grams capture word fragments and survive minor variations.
    if chars.len() >= 3 {
        for w in chars.windows(3) {
            let mut h: u64 = 1469598103934665603; // FNV offset
            for &c in w {
                h ^= c as u64;
                h = h.wrapping_mul(1099511628211); // FNV prime
            }
            let idx = (h as usize) % LOCAL_EMBED_DIM;
            v[idx] += 1.0;
        }
    }
    // Also hash whole words so exact term matches score strongly.
    for word in lower.split(|c: char| !c.is_alphanumeric()).filter(|w| !w.is_empty()) {
        let mut h: u64 = 1469598103934665603;
        for b in word.bytes() {
            h ^= b as u64;
            h = h.wrapping_mul(1099511628211);
        }
        let idx = (h as usize) % LOCAL_EMBED_DIM;
        v[idx] += 2.0; // weight exact words higher than n-grams
    }

    // L2 normalise so cosine similarity is well-behaved.
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 0.0 {
        for x in v.iter_mut() { *x /= norm; }
    }
    v
}

fn is_local_embed(model: &str) -> bool {
    let m = model.trim().to_lowercase();
    m == "local" || m.is_empty()
}

async fn embed_batch(
    client: &Client,
    base:   &str,
    key:    &str,
    model:  &str,
    texts:  &[String],
) -> Result<Vec<Vec<f32>>> {
    // Built-in local path: no network at all.
    if is_local_embed(model) {
        return Ok(texts.iter().map(|t| local_embed(t)).collect());
    }

    let url  = format!("{}/embeddings", base.trim_end_matches('/'));
    let body = json!({ "model": model, "input": texts });

    let mut req = client.post(&url).json(&body);
    if !key.is_empty() { req = req.bearer_auth(key); }

    let resp = req.send().await.context("calling embeddings endpoint")?;
    if !resp.status().is_success() {
        let status = resp.status();
        let text   = resp.text().await.unwrap_or_default();
        anyhow::bail!("embeddings API {status}: {text}");
    }

    #[derive(Deserialize)]
    struct EmbResp { data: Vec<EmbItem> }
    #[derive(Deserialize)]
    struct EmbItem { embedding: Vec<f32> }

    let parsed: EmbResp = resp.json().await.context("parsing embeddings response")?;
    Ok(parsed.data.into_iter().map(|e| e.embedding).collect())
}

async fn embed_one(
    client: &Client,
    base:   &str,
    key:    &str,
    model:  &str,
    text:   &str,
) -> Result<Vec<f32>> {
    let mut batch = embed_batch(client, base, key, model, &[text.to_string()]).await?;
    batch.pop().context("empty embedding response")
}

// ── Cosine similarity ────────────────────────────────────────────────────────

fn cosine(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let mut dot = 0.0f32;
    let mut na  = 0.0f32;
    let mut nb  = 0.0f32;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        na  += a[i] * a[i];
        nb  += b[i] * b[i];
    }
    if na == 0.0 || nb == 0.0 {
        return 0.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

// ── Persistence ──────────────────────────────────────────────────────────────

fn load_persisted() -> Option<VectorStore> {
    let text = std::fs::read_to_string(STORE_PATH).ok()?;
    serde_json::from_str(&text).ok()
}

fn save_persisted(store: &VectorStore) -> Result<()> {
    let text = serde_json::to_string(store).context("serializing vector store")?;
    std::fs::write(STORE_PATH, text).context("writing vectors.json")?;
    Ok(())
}

/// A fingerprint of the corpus + embed model. If this changes, the persisted
/// index is stale and we rebuild. It combines the embed model name and a hash
/// of every chunk's content hash, so adding/editing docs invalidates the index.
fn fingerprint(embed_model: &str, chunks: &[Chunk]) -> String {
    let mut hasher = Sha256::new();
    hasher.update(embed_model.as_bytes());
    for c in chunks {
        hasher.update(c.hash.as_bytes());
    }
    format!("{:x}", hasher.finalize())[..24].to_string()
}

// ── Public: build index ──────────────────────────────────────────────────────

/// Build the in-memory vector store, reusing a persisted index when the corpus
/// and embed model are unchanged. Populates the shared store in place.
pub async fn build_index(client: &Client, cfg: &Config, store: &SharedStore) -> Result<()> {
    info!("Loading documents from '{}'", cfg.data_dir);
    let docs   = load_all(&cfg.data_dir)?;
    let chunks = chunk_docs(docs);
    info!("Chunked into {} pieces", chunks.len());

    if chunks.is_empty() {
        anyhow::bail!("No chunks produced — is '{}' empty?", cfg.data_dir);
    }

    let fp = fingerprint(&cfg.embeddings_model, &chunks);

    // Reuse a persisted index if it matches the current corpus + embed model,
    // OR if it contains user uploads (a "custom-" fingerprint). We never want a
    // background rebuild to wipe documents the user added through the UI.
    if let Some(persisted) = load_persisted() {
        let has_uploads = persisted.fingerprint.starts_with("custom-");
        if (persisted.fingerprint == fp || has_uploads) && !persisted.chunks.is_empty() {
            info!("Reusing persisted index ({} chunks)", persisted.chunks.len());
            *store.write().await = persisted;
            return Ok(());
        }
        info!("Persisted index is stale, rebuilding");
    }

    // Embed all chunks in batches.
    let emb_base  = cfg.effective_embeddings_base();
    let emb_key   = cfg.effective_embeddings_key();
    let emb_model = &cfg.embeddings_model;
    let texts: Vec<String> = chunks.iter().map(|c| c.text.clone()).collect();

    let mut all_vecs: Vec<Vec<f32>> = Vec::with_capacity(texts.len());
    for batch in texts.chunks(32) {
        let vecs = embed_batch(client, emb_base, emb_key, emb_model, batch).await?;
        all_vecs.extend(vecs);
    }

    let dim = all_vecs.first().map(|v| v.len()).unwrap_or(0);
    info!("Embedded {} chunks (dim {dim})", all_vecs.len());

    let stored: Vec<StoredChunk> = chunks
        .into_iter()
        .zip(all_vecs.into_iter())
        .map(|(c, embedding)| StoredChunk {
            text: c.text, source: c.source, hash: c.hash, embedding,
        })
        .collect();

    let new_store = VectorStore { fingerprint: fp, chunks: stored };

    // Persist for next launch, then swap into shared state.
    if let Err(e) = save_persisted(&new_store) {
        warn!("Could not persist vectors.json: {e}");
    }
    *store.write().await = new_store;

    info!("Index ready");
    Ok(())
}

/// Add one uploaded document to the live store: chunk it, embed the chunks,
/// append them, and re-persist. Returns the number of chunks added. Used by
/// the /api/upload endpoint so users can add their own PDFs from the UI.
pub async fn add_file(
    client: &Client,
    cfg:    &Config,
    store:  &SharedStore,
    source: &str,
    content: &str,
) -> Result<usize> {
    let doc = Doc { source: source.to_string(), content: content.to_string() };
    let chunks = chunk_docs(vec![doc]);
    if chunks.is_empty() {
        anyhow::bail!("No readable text found in '{source}'.");
    }

    let emb_base  = cfg.effective_embeddings_base();
    let emb_key   = cfg.effective_embeddings_key();
    let emb_model = &cfg.embeddings_model;
    let texts: Vec<String> = chunks.iter().map(|c| c.text.clone()).collect();

    let mut all_vecs: Vec<Vec<f32>> = Vec::with_capacity(texts.len());
    for batch in texts.chunks(32) {
        let vecs = embed_batch(client, emb_base, emb_key, emb_model, batch).await?;
        all_vecs.extend(vecs);
    }

    let added = chunks.len();
    {
        let mut guard = store.write().await;
        // Remove any existing chunks from a file with the same name (re-upload).
        guard.chunks.retain(|c| c.source != source);
        for (c, embedding) in chunks.into_iter().zip(all_vecs.into_iter()) {
            guard.chunks.push(StoredChunk {
                text: c.text, source: c.source, hash: c.hash, embedding,
            });
        }
        // Invalidate the fingerprint so a future build_index doesn't clobber
        // these user uploads with a stale persisted index.
        guard.fingerprint = format!("custom-{}", guard.chunks.len());
        let snapshot = guard.clone();
        if let Err(e) = save_persisted(&snapshot) {
            warn!("Could not persist after upload: {e}");
        }
    }

    info!("Added {added} chunks from uploaded file '{source}'");
    Ok(added)
}

/// List the distinct source documents currently in the store, with chunk counts.
pub async fn list_sources(store: &SharedStore) -> Vec<(String, usize)> {
    let guard = store.read().await;
    let mut counts: std::collections::BTreeMap<String, usize> = Default::default();
    for c in &guard.chunks {
        *counts.entry(c.source.clone()).or_insert(0) += 1;
    }
    counts.into_iter().collect()
}

// ── Public: retrieve ─────────────────────────────────────────────────────────

pub async fn retrieve(
    client: &Client,
    cfg:    &Config,
    store:  &SharedStore,
    query:  &str,
    k:      usize,
) -> Result<Vec<Hit>> {
    let emb_base  = cfg.effective_embeddings_base();
    let emb_key   = cfg.effective_embeddings_key();
    let query_vec = embed_one(client, emb_base, emb_key, &cfg.embeddings_model, query).await?;

    let guard = store.read().await;
    if guard.chunks.is_empty() {
        anyhow::bail!("Vector store is empty — index may still be building.");
    }

    // Score every chunk by cosine similarity, then take the top k.
    let mut scored: Vec<Hit> = guard.chunks.iter().map(|c| Hit {
        text:   c.text.clone(),
        source: c.source.clone(),
        score:  cosine(&query_vec, &c.embedding),
    }).collect();

    scored.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(k);
    Ok(scored)
}
