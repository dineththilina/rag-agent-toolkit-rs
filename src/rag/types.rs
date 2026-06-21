// src/rag/types.rs
//
// Shared data types for the RAG subsystem: the chunk record that is persisted
// to the local vector store, the retrieval hit returned to callers, the
// retrieval mode, and the tunable configuration.

use serde::{Deserialize, Serialize};

use crate::config::Config;

/// How a query is answered against the corpus.
///
///   * `Keyword` — BM25 only (the original behaviour, always available).
///   * `Vector`  — semantic similarity only (needs a working embedder).
///   * `Hybrid`  — merge keyword + vector with Reciprocal Rank Fusion.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum RetrievalMode {
    Keyword,
    Vector,
    #[default]
    Hybrid,
}

impl RetrievalMode {
    /// Parse from a config string. Unknown values fall back to the default
    /// (hybrid) rather than failing — retrieval should never be a hard error.
    pub fn parse(s: &str) -> Self {
        match s.trim().to_lowercase().as_str() {
            "keyword" | "bm25" | "lexical" => RetrievalMode::Keyword,
            "vector" | "semantic" | "dense" => RetrievalMode::Vector,
            "hybrid" | "" => RetrievalMode::Hybrid,
            _ => RetrievalMode::Hybrid,
        }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            RetrievalMode::Keyword => "keyword",
            RetrievalMode::Vector => "vector",
            RetrievalMode::Hybrid => "hybrid",
        }
    }
}

/// Which layer produced a hit (for transparency / debugging in the UI).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum HitSource {
    Keyword,
    Vector,
    Hybrid,
}

/// One chunk of a document, with everything needed to (a) embed it, (b) rank it
/// with BM25, and (c) trace a retrieved answer back to its exact source.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StoredChunk {
    /// Stable id for the whole document (derived from the source filename), so a
    /// re-upload of the same file replaces its chunks instead of duplicating.
    pub document_id: String,
    /// 0-based index of this chunk within its document.
    pub chunk_id: usize,
    /// Source filename (e.g. `pricing.md`).
    pub source: String,
    /// The original chunk text, verbatim.
    pub text: String,
    /// Lowercased / whitespace-collapsed text used for keyword indexing.
    pub normalized: String,
    /// Number of characters in `text`.
    pub char_count: usize,
    /// Approximate token count (whitespace-ish split) for chunk-size budgeting.
    pub token_count: usize,
    /// Unix timestamp (seconds) when the chunk was indexed.
    pub created_at: i64,
    /// Optional free-form metadata, stored as JSON.
    pub metadata: Option<serde_json::Value>,
}

/// A retrieval result returned to callers (API + prompt builder).
///
/// Keeps the legacy `text` / `source` / `score` fields the UI already consumes,
/// and adds provenance so every answer can be cited back to a document + chunk.
#[derive(Debug, Clone, Serialize)]
pub struct Hit {
    pub text: String,
    pub source: String,
    /// Normalised 0..1 relevance score (higher is better).
    pub score: f32,
    pub document_id: String,
    pub chunk_id: usize,
    /// Which retrieval layer surfaced this hit.
    pub retrieval: HitSource,
}

/// Tunable RAG configuration, derived from the app [`Config`] with safe
/// defaults. Centralised here so chunking, embedding and retrieval all read the
/// same knobs.
///
/// Note: per-query knobs (mode / top_k / min_similarity) are read live from the
/// app [`Config`] at query time so they can change at runtime; this struct holds
/// the settings fixed at store-open time (paths, dimension, chunk geometry).
#[derive(Debug, Clone)]
pub struct RagConfig {
    /// Directory of source documents to ingest.
    pub data_dir: String,
    /// Path to the on-disk SQLite vector database.
    pub db_path: String,
    /// fastembed model identifier (the variant is mapped in the fastembed provider).
    pub embedding_model: String,
    /// Embedding dimension. Must match the model; validated before insert/query.
    pub embedding_dim: usize,
    /// Target chunk size in characters.
    pub chunk_chars: usize,
    /// Overlap between consecutive chunks in characters.
    pub overlap_chars: usize,
}

impl RagConfig {
    /// Default location of the local vector database, under `data/`.
    pub const DEFAULT_DB_PATH: &'static str = "data/rag_vectors.sqlite";

    pub fn from_app_config(cfg: &Config) -> Self {
        RagConfig {
            data_dir: cfg.data_dir.clone(),
            db_path: if cfg.vector_db_path.trim().is_empty() {
                Self::DEFAULT_DB_PATH.to_string()
            } else {
                cfg.vector_db_path.clone()
            },
            embedding_model: cfg.embedding_model.clone(),
            embedding_dim: if cfg.embedding_dim == 0 {
                384
            } else {
                cfg.embedding_dim
            },
            // ~3,500 chars ≈ 800–900 tokens; ~600 chars ≈ 130 tokens overlap.
            chunk_chars: 3_500,
            overlap_chars: 600,
        }
    }
}

/// Deterministic FNV-1a 64-bit hash → short hex string. Used to derive a stable
/// `document_id` from a source filename (stable across restarts and Rust
/// versions, unlike `DefaultHasher`).
pub fn stable_id(input: &str) -> String {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    for b in input.as_bytes() {
        hash ^= *b as u64;
        hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
    }
    format!("{hash:016x}")
}

/// Current Unix time in seconds (0 if the clock is before the epoch).
pub fn now_unix() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs() as i64)
        .unwrap_or(0)
}
