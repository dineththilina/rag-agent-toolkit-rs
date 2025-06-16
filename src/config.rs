// src/config.rs
//
// Configuration is stored in a single config.toml next to the binary.
// On first launch the file doesn't exist; the frontend's /api/config POST
// creates it. All fields have sensible defaults so partial configs work.

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::Arc;
use tokio::sync::RwLock;

// The path we look for / write config to.
pub const CONFIG_PATH: &str = "config.toml";

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
}

fn default_embed_model() -> String { "text-embedding-3-small".into() }
fn default_data_dir()     -> String { "data".into() }

impl Config {
    /// Effective embeddings base URL: falls back to api_base if not explicitly set.
    pub fn effective_embeddings_base(&self) -> &str {
        if self.embeddings_base.is_empty() {
            &self.api_base
        } else {
            &self.embeddings_base
        }
    }

    /// Effective embeddings key: falls back to api_key if not explicitly set.
    pub fn effective_embeddings_key(&self) -> &str {
        if self.embeddings_key.is_empty() {
            &self.api_key
        } else {
            &self.embeddings_key
        }
    }

    /// The out-of-the-box default. Chat goes through Groq (free, one key) and
    /// document search uses the built-in local embedder ("local") which needs
    /// no key or service. This means: paste one free Groq key and everything
    /// works — chat AND PDF search — with nothing else to install.
    pub fn default_local() -> Self {
        Self {
            api_base:         "https://api.groq.com/openai/v1".into(),
            api_key:          String::new(),               // user pastes a free Groq key
            model:            "llama-3.3-70b-versatile".into(),
            embeddings_base:  String::new(),
            embeddings_model: "local".into(),              // built-in, no key needed
            embeddings_key:   String::new(),
            data_dir:         default_data_dir(),
        }
    }
}

impl Default for Config {
    fn default() -> Self {
        Self {
            api_base:         "https://api.openai.com/v1".into(),
            api_key:          String::new(),
            model:            String::new(),
            embeddings_base:  String::new(),
            embeddings_model: default_embed_model(),
            embeddings_key:   String::new(),
            data_dir:         default_data_dir(),
        }
    }
}

// ── Persistence ─────────────────────────────────────────────────────────────

pub fn load() -> Result<Option<Config>> {
    let path = PathBuf::from(CONFIG_PATH);
    if !path.exists() {
        return Ok(None);
    }
    let text = std::fs::read_to_string(&path)
        .context("reading config.toml")?;
    let cfg: Config = toml::from_str(&text)
        .context("parsing config.toml")?;
    Ok(Some(cfg))
}

pub fn save(cfg: &Config) -> Result<()> {
    let text = toml::to_string_pretty(cfg)
        .context("serializing config")?;
    std::fs::write(CONFIG_PATH, text)
        .context("writing config.toml")?;
    Ok(())
}

// ── Shared mutable state ─────────────────────────────────────────────────────

/// App-wide config wrapped in an async RwLock so the /api/config POST can
/// update it at runtime without a restart.
pub type SharedConfig = Arc<RwLock<Option<Config>>>;

pub fn new_shared(initial: Option<Config>) -> SharedConfig {
    Arc::new(RwLock::new(initial))
}
