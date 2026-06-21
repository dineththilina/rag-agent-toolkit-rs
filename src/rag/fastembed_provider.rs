// src/rag/fastembed_provider.rs
//
// Local neural embeddings via the `fastembed` crate (ONNX inference, no API
// key, no cloud). Compiled only when the `local-embeddings` feature is on.
//
// The model is downloaded once from the Hugging Face hub on first use and then
// cached under `./.fastembed_cache` (override with `FASTEMBED_CACHE_DIR`), so
// subsequent runs work offline. At runtime `ort` loads the ONNX Runtime shared
// library dynamically (see the README for how to make it available).
//
// `TextEmbedding::embed` takes `&mut self`, so the model lives behind a Mutex;
// the trait exposes an immutable `&self` interface to callers.

use std::sync::Mutex;

use anyhow::{Context, Result};
use fastembed::{EmbeddingModel, TextEmbedding, TextInitOptions};
use tracing::info;

use crate::rag::embed::EmbeddingProvider;

/// Default local English retrieval model: `all-MiniLM-L6-v2` (384 dims). Small,
/// fast, well-supported, and a sensible default for document Q&A.
pub const DEFAULT_MODEL: &str = "all-MiniLM-L6-v2";
pub const DEFAULT_DIM: usize = 384;

/// Map a config string to a fastembed model + its dimension. Unknown values
/// fall back to the default model with a logged note.
fn resolve_model(name: &str) -> (EmbeddingModel, usize) {
    match name.trim().to_lowercase().as_str() {
        "all-minilm-l6-v2" | "minilm" | "minilm-l6" | "allminilml6v2" | "" => {
            (EmbeddingModel::AllMiniLML6V2, 384)
        }
        "bge-small-en-v1.5" | "bge-small" | "bgesmallenv15" => (EmbeddingModel::BGESmallENV15, 384),
        "bge-base-en-v1.5" | "bge-base" | "bgebaseenv15" => (EmbeddingModel::BGEBaseENV15, 768),
        "all-minilm-l12-v2" | "minilm-l12" => (EmbeddingModel::AllMiniLML12V2, 384),
        other => {
            tracing::warn!(
                "unknown embedding model '{other}', falling back to {DEFAULT_MODEL} ({DEFAULT_DIM}d)"
            );
            (EmbeddingModel::AllMiniLML6V2, 384)
        }
    }
}

/// fastembed-backed embedding provider.
pub struct LocalFastEmbedProvider {
    model: Mutex<TextEmbedding>,
    dim: usize,
    label: String,
}

impl LocalFastEmbedProvider {
    /// Load (and on first use, download) the embedding model. Returns a clear
    /// error if the model or ONNX Runtime cannot be loaded, so callers can fall
    /// back to keyword retrieval instead of crashing.
    pub fn new(model_name: &str) -> Result<Self> {
        let (variant, dim) = resolve_model(model_name);
        info!("Loading local embedding model '{model_name}' ({dim} dims) via fastembed…");

        let model =
            TextEmbedding::try_new(TextInitOptions::new(variant.clone())).with_context(|| {
                format!(
                    "could not load the local embedding model '{model_name}'. The first run \
                     downloads it from the Hugging Face hub (needs network access), and at \
                     runtime the ONNX Runtime shared library must be available to `ort` \
                     (set ORT_DYLIB_PATH or install onnxruntime). See the README for details."
                )
            })?;

        info!("Local embedding model ready: {model_name}");
        Ok(Self {
            model: Mutex::new(model),
            dim,
            label: model_name.to_string(),
        })
    }
}

impl EmbeddingProvider for LocalFastEmbedProvider {
    fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(Vec::new());
        }
        let mut model = self
            .model
            .lock()
            .map_err(|_| anyhow::anyhow!("embedding model mutex was poisoned"))?;
        let vectors = model
            .embed(texts, None)
            .context("fastembed failed to generate embeddings")?;

        // Validate dimensions before they ever reach the vector store.
        for v in &vectors {
            if v.len() != self.dim {
                anyhow::bail!(
                    "embedding dimension mismatch: model returned {} dims, expected {}",
                    v.len(),
                    self.dim
                );
            }
        }
        Ok(vectors)
    }

    fn dimension(&self) -> usize {
        self.dim
    }

    fn name(&self) -> String {
        format!("fastembed:{}", self.label)
    }
}
