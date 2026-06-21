// src/config.rs
//
// Configuration is stored in a single config.toml next to the binary.
// On first launch the file doesn't exist; the frontend's /api/config POST
// creates it. All fields have sensible defaults so partial configs work.
//
// Secrets (api_key, embeddings_key): prefer the AGENT_API_KEY /
// AGENT_EMBEDDINGS_KEY environment variables over config.toml. An env var
// always wins over whatever is on disk, and — because it's already available
// at process start — is never written back to config.toml, so the secret
// never lands in plaintext on disk just because the app loaded it. Falling
// back to storing a pasted key in config.toml is kept for local/dev
// convenience (the file is gitignored and chmod'd 600 on write).

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;
use tracing::warn;

// The path we look for / write config to.
pub const CONFIG_PATH: &str = "config.toml";

const ENV_API_KEY: &str = "AGENT_API_KEY";
const ENV_EMBEDDINGS_KEY: &str = "AGENT_EMBEDDINGS_KEY";

fn env_key(var: &str) -> Option<String> {
    std::env::var(var)
        .ok()
        .map(|v| v.trim().to_string())
        .filter(|v| !v.is_empty())
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Config {
    /// Base URL of the OpenAI-compatible API.
    /// Examples:
    ///   https://api.openai.com/v1
    ///   https://api.anthropic.com/v1        (OpenAI-compat endpoint)
    ///   https://api.groq.com/openai/v1
    ///   http://localhost:11434/v1           (Ollama)
    ///   http://localhost:1234/v1            (LM Studio)
    pub api_base: String,

    /// API key. Leave empty for Ollama / LM Studio (they don't require one).
    #[serde(default)]
    pub api_key: String,

    /// Model ID chosen by the user from the live model list.
    pub model: String,

    /// Base URL for embeddings. Defaults to api_base if not set.
    /// Most providers serve embeddings on the same base URL. For providers
    /// without embeddings (e.g. Groq), point this at OpenAI or a local Ollama.
    #[serde(default)]
    pub embeddings_base: String,

    /// Embedding model. Defaults to text-embedding-3-small for OpenAI;
    /// for Ollama use e.g. "nomic-embed-text".
    #[serde(default = "default_embed_model")]
    pub embeddings_model: String,

    /// API key for the embeddings endpoint. Falls back to api_key if empty.
    /// Set this when embeddings use a different provider than chat.
    #[serde(default)]
    pub embeddings_key: String,

    /// Directory containing documents to ingest.
    #[serde(default = "default_data_dir")]
    pub data_dir: String,

    // ── Local vector RAG settings ────────────────────────────────────────────
    /// Retrieval mode: "keyword", "vector", or "hybrid". Default: hybrid.
    #[serde(default = "default_retrieval_mode")]
    pub retrieval_mode: String,

    /// How many chunks to retrieve. Default: 8.
    #[serde(default = "default_top_k")]
    pub top_k: usize,

    /// Minimum cosine similarity (0..1) for a vector hit to count as confident
    /// evidence. Below this and the assistant says it lacks evidence.
    #[serde(default = "default_min_similarity")]
    pub min_similarity: f32,

    /// Local embedding model used by fastembed. Default: all-MiniLM-L6-v2.
    #[serde(default = "default_embedding_model")]
    pub embedding_model: String,

    /// Embedding vector dimension (must match the model). Default: 384.
    #[serde(default = "default_embedding_dim")]
    pub embedding_dim: usize,

    /// Path to the on-disk SQLite vector database. Default: data/rag_vectors.sqlite.
    #[serde(default = "default_vector_db_path")]
    pub vector_db_path: String,

    /// Path to the on-disk SQLite session-memory database. Default: data/sessions.sqlite.
    #[serde(default = "default_session_db_path")]
    pub session_db_path: String,
}

fn default_embed_model() -> String {
    "text-embedding-3-small".into()
}
fn default_data_dir() -> String {
    "data".into()
}

fn default_retrieval_mode() -> String {
    "hybrid".into()
}
fn default_top_k() -> usize {
    8
}
fn default_min_similarity() -> f32 {
    0.2
}
fn default_embedding_model() -> String {
    "all-MiniLM-L6-v2".into()
}
fn default_embedding_dim() -> usize {
    384
}
fn default_vector_db_path() -> String {
    "data/rag_vectors.sqlite".into()
}
fn default_session_db_path() -> String {
    "data/sessions.sqlite".into()
}

impl Config {
    /// The out-of-the-box default. Chat goes through Groq (free, one key);
    /// document search is BM25 (in-process, needs no key or service). So pasting
    /// one free Groq key makes everything work with nothing else to install.
    pub fn default_local() -> Self {
        let mut cfg = Self {
            api_base: "https://api.groq.com/openai/v1".into(),
            api_key: String::new(), // user pastes a free Groq key, or set AGENT_API_KEY
            model: "llama-3.3-70b-versatile".into(),
            embeddings_base: String::new(),
            embeddings_model: "local".into(), // unused; kept for config compat
            embeddings_key: String::new(),
            data_dir: default_data_dir(),
            retrieval_mode: default_retrieval_mode(),
            top_k: default_top_k(),
            min_similarity: default_min_similarity(),
            embedding_model: default_embedding_model(),
            embedding_dim: default_embedding_dim(),
            vector_db_path: default_vector_db_path(),
            session_db_path: default_session_db_path(),
        };
        cfg.apply_env_secrets();
        cfg
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            api_base: "https://api.openai.com/v1".into(),
            api_key: String::new(),
            model: String::new(),
            embeddings_base: String::new(),
            embeddings_model: default_embed_model(),
            embeddings_key: String::new(),
            data_dir: default_data_dir(),
            retrieval_mode: default_retrieval_mode(),
            top_k: default_top_k(),
            min_similarity: default_min_similarity(),
            embedding_model: default_embedding_model(),
            embedding_dim: default_embedding_dim(),
            vector_db_path: default_vector_db_path(),
            session_db_path: default_session_db_path(),
        }
    }
}

impl Config {
    /// Apply AGENT_API_KEY / AGENT_EMBEDDINGS_KEY overrides, if set. An env
    /// var always wins over whatever is in config.toml or posted by the UI.
    pub fn apply_env_secrets(&mut self) {
        if let Some(k) = env_key(ENV_API_KEY) {
            self.api_key = k;
        }
        if let Some(k) = env_key(ENV_EMBEDDINGS_KEY) {
            self.embeddings_key = k;
        }
    }

    /// Human-readable source of the chat API key, for startup logging. Never
    /// reveals the key itself.
    pub fn api_key_source(&self) -> &'static str {
        if env_key(ENV_API_KEY).is_some() {
            "AGENT_API_KEY environment variable"
        } else if !self.api_key.is_empty() {
            "config.toml (consider moving it to the AGENT_API_KEY env var instead)"
        } else {
            "none set"
        }
    }
}

// ── Persistence ─────────────────────────────────────────────────────────────

pub fn load() -> Result<Option<Config>> {
    let path = PathBuf::from(CONFIG_PATH);
    if !path.exists() {
        return Ok(None);
    }
    let text = std::fs::read_to_string(&path).context("reading config.toml")?;
    let mut cfg: Config = toml::from_str(&text).context("parsing config.toml")?;
    cfg.apply_env_secrets();
    Ok(Some(cfg))
}

/// Persist config to disk. Secrets sourced from an environment variable are
/// never written back to the file — only a key the user actually pasted into
/// config.toml (or the setup UI, with no env var set) is persisted there.
pub fn save(cfg: &Config) -> Result<()> {
    let mut on_disk = cfg.clone();
    if env_key(ENV_API_KEY).is_some() {
        on_disk.api_key = String::new();
    }
    if env_key(ENV_EMBEDDINGS_KEY).is_some() {
        on_disk.embeddings_key = String::new();
    }

    let text = toml::to_string_pretty(&on_disk).context("serializing config")?;
    std::fs::write(CONFIG_PATH, text).context("writing config.toml")?;
    restrict_permissions(CONFIG_PATH);
    Ok(())
}

/// Best-effort lockdown of config.toml to owner-only (it may contain a
/// plaintext API key). No-op on platforms without Unix permission bits.
fn restrict_permissions(path: &str) {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if let Err(e) = std::fs::set_permissions(path, std::fs::Permissions::from_mode(0o600)) {
            warn!("Could not restrict permissions on {path}: {e}");
        }
    }
    #[cfg(not(unix))]
    {
        let _ = path;
    }
}

// ── Shared mutable state ─────────────────────────────────────────────────────

/// App-wide config wrapped in an async RwLock so the /api/config POST can
/// update it at runtime without a restart.
pub type SharedConfig = Arc<RwLock<Option<Config>>>;

pub fn new_shared(initial: Option<Config>) -> SharedConfig {
    Arc::new(RwLock::new(initial))
}
