// src/rag.rs
//
// Document search using BM25 — the ranking function real search engines use.
// No embeddings, no API key, no external service, no model download. Pure Rust.
//
//   1. Load documents from data/ (txt, md) — PDFs are read in the browser
//   2. Chunk with text-splitter
//   3. Tokenize each chunk (lowercase, split on non-alphanumeric, drop stopwords)
//   4. Store chunks + token stats in-process, persisted to index.json
//   5. retrieve(query) -> BM25 score every chunk -> top-k Hits
//
// Why BM25 instead of vector embeddings: a hand-rolled local embedding is too
// weak to separate relevant from irrelevant text, and requiring a hosted
// embedding API means a second key. BM25 is keyword-based, needs nothing, and
// gives sharp, correct rankings for the document-Q&A this app does.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;
use text_splitter::TextSplitter;
use tokio::sync::RwLock;
use tracing::{info, warn};

use crate::config::Config;

const STORE_PATH: &str = "index.json";
const CHUNK_SIZE: usize = 700;

// BM25 tuning constants (standard defaults).
const BM25_K1: f32 = 1.5;
const BM25_B:  f32 = 0.75;

// Common English words that add noise to keyword matching.
const STOPWORDS: &[&str] = &[
    "a","an","the","of","to","in","on","at","for","and","or","but","is","are",
    "was","were","be","been","being","this","that","these","those","it","its",
    "as","by","with","from","into","over","under","do","does","did","have","has",
    "had","will","would","can","could","should","i","you","he","she","we","they",
    "my","your","his","her","our","their","what","which","who","when","where",
    "why","how","not","no","yes","if","then","else","than","there","here","about",
    "me","him","them","us","so","up","out","off","all","any","some","more","most",
];

fn is_stopword(w: &str) -> bool {
    STOPWORDS.contains(&w)
}

/// Tokenize: lowercase, split on non-alphanumeric, drop stopwords and 1-char tokens.
fn tokenize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.len() > 1 && !is_stopword(w))
        .map(|w| w.to_string())
        .collect()
}

// ── Public types ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize)]
pub struct Hit {
    pub text:   String,
    pub source: String,
    pub score:  f32,
}

/// One stored chunk: its text, source file, and precomputed token statistics.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct StoredChunk {
    text:      String,
    source:    String,
    /// term -> frequency within this chunk
    term_freq: HashMap<String, u32>,
    /// total token count of this chunk (for BM25 length normalisation)
    length:    u32,
}

/// The in-memory / on-disk index.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct VectorStore {
    fingerprint: String,
    chunks:      Vec<StoredChunk>,
}

pub type SharedStore = Arc<RwLock<VectorStore>>;

pub fn new_shared_store() -> SharedStore {
    Arc::new(RwLock::new(VectorStore::default()))
}

// ── BM25 scoring ─────────────────────────────────────────────────────────────

/// Score every chunk against the query tokens and return the top k as Hits.
fn bm25_search(chunks: &[StoredChunk], query: &str, k: usize) -> Vec<Hit> {
    let q_tokens: Vec<String> = {
        let mut t = tokenize(query);
        t.sort();
        t.dedup();
        t
    };
    if q_tokens.is_empty() || chunks.is_empty() {
        return Vec::new();
    }

    let n = chunks.len() as f32;
    let avgdl = chunks.iter().map(|c| c.length as f32).sum::<f32>() / n;

    // Document frequency: how many chunks contain each query term.
    let mut df: HashMap<&str, u32> = HashMap::new();
    for qt in &q_tokens {
        let count = chunks.iter().filter(|c| c.term_freq.contains_key(qt)).count() as u32;
        df.insert(qt.as_str(), count);
    }

    let mut scored: Vec<Hit> = chunks.iter().map(|c| {
        let mut score = 0.0f32;
        for qt in &q_tokens {
            let dfq = *df.get(qt.as_str()).unwrap_or(&0);
            if dfq == 0 { continue; }
            let tf = *c.term_freq.get(qt).unwrap_or(&0) as f32;
            if tf == 0.0 { continue; }
            // BM25 IDF (with +1 to keep it positive).
            let idf = ((n - dfq as f32 + 0.5) / (dfq as f32 + 0.5) + 1.0).ln();
            let denom = tf + BM25_K1 * (1.0 - BM25_B + BM25_B * c.length as f32 / avgdl);
            score += idf * (tf * (BM25_K1 + 1.0)) / denom;
        }
        Hit { text: c.text.clone(), source: c.source.clone(), score }
    }).collect();

    // Keep only chunks that actually matched something.
    scored.retain(|h| h.score > 0.0);
    scored.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
    scored.truncate(k);

    // Normalise scores to 0..1 for a friendly "% match" in the UI.
    if let Some(max) = scored.first().map(|h| h.score) {
        if max > 0.0 {
            for h in scored.iter_mut() { h.score /= max; }
        }
    }
    scored
}

// ── Document loading ─────────────────────────────────────────────────────────

fn load_txt(path: &Path) -> Result<String> {
    Ok(std::fs::read_to_string(path)?)
}

// PDF text extraction happens in the browser (pdf.js) and is sent as plain text
// to /api/upload, so the backend never parses PDF bytes.

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
            _            => { warn!("skipping unsupported file: {source}"); continue; }
        };
        docs.push(Doc { source, content });
    }
    docs.sort_by(|a, b| a.source.cmp(&b.source));
    Ok(docs)
}

// ── Chunking ─────────────────────────────────────────────────────────────────

fn make_chunk(source: &str, text: &str) -> StoredChunk {
    let tokens = tokenize(text);
    let length = tokens.len() as u32;
    let mut term_freq: HashMap<String, u32> = HashMap::new();
    for t in tokens {
        *term_freq.entry(t).or_insert(0) += 1;
    }
    StoredChunk { text: text.to_string(), source: source.to_string(), term_freq, length }
}

fn chunk_docs(docs: Vec<Doc>) -> Vec<StoredChunk> {
    let splitter = TextSplitter::new(CHUNK_SIZE);
    let mut chunks = Vec::new();
    for doc in docs {
        for piece in splitter.chunks(&doc.content) {
            let trimmed = piece.trim();
            if trimmed.len() < 40 { continue; }
            chunks.push(make_chunk(&doc.source, trimmed));
        }
    }
    chunks
}

// ── Persistence ──────────────────────────────────────────────────────────────

fn load_persisted() -> Option<VectorStore> {
    let text = std::fs::read_to_string(STORE_PATH).ok()?;
    serde_json::from_str(&text).ok()
}

fn save_persisted(store: &VectorStore) -> Result<()> {
    let text = serde_json::to_string(store).context("serializing index")?;
    std::fs::write(STORE_PATH, text).context("writing index.json")?;
    Ok(())
}

fn fingerprint(chunks: &[StoredChunk]) -> String {
    // Cheap content fingerprint: chunk count + total length + a sample of sources.
    let total_len: u32 = chunks.iter().map(|c| c.length).sum();
    let mut srcs: Vec<&str> = chunks.iter().map(|c| c.source.as_str()).collect();
    srcs.sort();
    srcs.dedup();
    format!("{}-{}-{}", chunks.len(), total_len, srcs.join(","))
}

// ── Public: build index ──────────────────────────────────────────────────────

/// Build the in-memory index from the documents in data/. Reuses a persisted
/// index when the corpus is unchanged, and never clobbers user uploads.
pub async fn build_index(_client: &reqwest::Client, cfg: &Config, store: &SharedStore) -> Result<()> {
    info!("Loading documents from '{}'", cfg.data_dir);
    let docs   = load_all(&cfg.data_dir)?;
    let chunks = chunk_docs(docs);
    info!("Chunked into {} pieces", chunks.len());

    if chunks.is_empty() {
        // Not fatal — the user may rely entirely on uploads.
        warn!("No documents found in '{}'", cfg.data_dir);
    }

    let fp = fingerprint(&chunks);

    if let Some(persisted) = load_persisted() {
        let has_uploads = persisted.fingerprint.starts_with("custom-");
        if (persisted.fingerprint == fp || has_uploads) && !persisted.chunks.is_empty() {
            info!("Reusing persisted index ({} chunks)", persisted.chunks.len());
            *store.write().await = persisted;
            return Ok(());
        }
        info!("Persisted index is stale, rebuilding");
    }

    let new_store = VectorStore { fingerprint: fp, chunks };
    if let Err(e) = save_persisted(&new_store) {
        warn!("Could not persist index.json: {e}");
    }
    *store.write().await = new_store;
    info!("Index ready");
    Ok(())
}

/// Add one uploaded document to the live index: chunk it, tokenize, append,
/// re-persist. Returns the number of chunks added.
pub async fn add_file(
    _client: &reqwest::Client,
    _cfg:    &Config,
    store:   &SharedStore,
    source:  &str,
    content: &str,
) -> Result<usize> {
    let doc = Doc { source: source.to_string(), content: content.to_string() };
    let chunks = chunk_docs(vec![doc]);
    if chunks.is_empty() {
        anyhow::bail!("No readable text found in '{source}'.");
    }
    let added = chunks.len();
    {
        let mut guard = store.write().await;
        guard.chunks.retain(|c| c.source != source);   // replace on re-upload
        guard.chunks.extend(chunks);
        guard.fingerprint = format!("custom-{}", guard.chunks.len());
        let snapshot = guard.clone();
        if let Err(e) = save_persisted(&snapshot) {
            warn!("Could not persist after upload: {e}");
        }
    }
    info!("Added {added} chunks from uploaded file '{source}'");
    Ok(added)
}

/// List distinct source documents with chunk counts.
pub async fn list_sources(store: &SharedStore) -> Vec<(String, usize)> {
    let guard = store.read().await;
    let mut counts: std::collections::BTreeMap<String, usize> = Default::default();
    for c in &guard.chunks {
        *counts.entry(c.source.clone()).or_insert(0) += 1;
    }
    counts.into_iter().collect()
}

/// Remove all chunks belonging to a named source. Returns true if any removed.
pub async fn remove_source(store: &SharedStore, name: &str) -> bool {
    let mut guard = store.write().await;
    let before = guard.chunks.len();
    guard.chunks.retain(|c| c.source != name);
    let removed = guard.chunks.len() != before;
    if removed {
        guard.fingerprint = format!("custom-{}", guard.chunks.len());
        let snapshot = guard.clone();
        if let Err(e) = save_persisted(&snapshot) {
            warn!("Could not persist after removal: {e}");
        }
    }
    removed
}

// ── Public: retrieve ─────────────────────────────────────────────────────────

pub async fn retrieve(
    _client: &reqwest::Client,
    _cfg:    &Config,
    store:   &SharedStore,
    query:   &str,
    k:       usize,
) -> Result<Vec<Hit>> {
    let guard = store.read().await;
    if guard.chunks.is_empty() {
        anyhow::bail!("No documents are loaded yet. Add a document first.");
    }
    Ok(bm25_search(&guard.chunks, query, k))
}
