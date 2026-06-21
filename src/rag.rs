// src/rag.rs
//
// Retrieval-augmented generation: the orchestrator that ties together
// deterministic chunking, local embeddings, the on-disk sqlite-vec vector
// store, and the preserved BM25 keyword index — fused into one ranking.
//
// Pipeline:
//   upload/ingest → chunk (chunk.rs) → embed (embed.rs / fastembed_provider.rs)
//                 → store on disk (store.rs, sqlite-vec) + BM25 (bm25.rs)
//   query         → embed query → vector KNN + BM25 → fuse (hybrid.rs) → cited
//                   chunks fed into the chat context
//
// SQLite is the single source of truth; the BM25 index is rebuilt from it on
// startup, so indexed documents stay searchable across restarts. If the local
// embedding model can't load, the system degrades cleanly to keyword retrieval.

mod bm25;
mod chunk;
pub mod embed;
#[cfg(feature = "local-embeddings")]
mod fastembed_provider;
mod hybrid;
mod store;
pub mod types;

use std::path::Path;
use std::sync::{Arc, RwLock};

use anyhow::{Context, Result};
use tracing::{info, warn};

use crate::config::Config;

use self::bm25::Bm25Index;
use self::embed::EmbeddingProvider;
use self::store::{SqliteVecStore, VectorStore};
use self::types::HitSource;

pub use self::types::{Hit, RagConfig, RetrievalMode};

/// Roughly 4 characters per token. We budget context by characters to stay
/// simple. ~32k chars ≈ ~8k tokens, comfortably inside common free-tier limits.
const CONTEXT_CHAR_BUDGET: usize = 32_000;

// ── The system ───────────────────────────────────────────────────────────────

/// The whole retrieval system. Held behind an `Arc` and shared across requests;
/// all interior state uses its own locks, so no outer lock is needed.
pub struct RagSystem {
    cfg: RagConfig,
    store: SqliteVecStore,
    bm25: RwLock<Bm25Index>,
    embedder: RwLock<Option<Arc<dyn EmbeddingProvider>>>,
}

pub type SharedStore = Arc<RagSystem>;

impl RagSystem {
    /// Open the system against the on-disk store described by `cfg`, optionally
    /// with an embedding provider already loaded (tests inject a mock). The
    /// store dimension comes from the embedder if present, else `cfg.embedding_dim`.
    pub fn open(cfg: RagConfig, embedder: Option<Arc<dyn EmbeddingProvider>>) -> Result<Self> {
        let dim = embedder
            .as_ref()
            .map(|e| e.dimension())
            .unwrap_or(cfg.embedding_dim);
        let store = SqliteVecStore::open(&cfg.db_path, dim)?;

        // Rebuild the in-memory BM25 index from whatever is already on disk.
        let mut bm25 = Bm25Index::new();
        for chunk in store.all_chunks()? {
            bm25.add(&chunk);
        }
        if !bm25.is_empty() {
            info!("Rebuilt BM25 index from disk ({} chunks)", bm25.len());
        }

        Ok(Self {
            cfg,
            store,
            bm25: RwLock::new(bm25),
            embedder: RwLock::new(embedder),
        })
    }

    /// Clone out the current embedding provider, if one is loaded.
    fn embedder(&self) -> Option<Arc<dyn EmbeddingProvider>> {
        self.embedder
            .read()
            .expect("embedder lock poisoned")
            .clone()
    }

    /// Install an embedding provider (called after the model loads in the
    /// background). Refuses a provider whose dimension doesn't match the store.
    pub fn set_embedder(&self, provider: Arc<dyn EmbeddingProvider>) {
        if provider.dimension() != self.store.dimension() {
            warn!(
                "Embedding model '{}' has dimension {} but the vector store is {}; \
                 keeping keyword-only retrieval. Run rebuild after fixing the model.",
                provider.name(),
                provider.dimension(),
                self.store.dimension()
            );
            return;
        }
        info!("Embedding model active: {}", provider.name());
        *self.embedder.write().expect("embedder lock poisoned") = Some(provider);
    }

    /// Embed texts off the async runtime (embedding is blocking CPU work).
    async fn embed_texts(&self, texts: Vec<String>) -> Result<Vec<Vec<f32>>> {
        let emb = self
            .embedder()
            .ok_or_else(|| anyhow::anyhow!("no embedding model is available"))?;
        tokio::task::spawn_blocking(move || emb.embed(&texts))
            .await
            .context("embedding task failed")?
    }

    /// Public entry point to chunk, embed, and index one document. Replaces any
    /// existing chunks for the same source (idempotent re-upload).
    pub async fn add_document(&self, source: &str, content: &str) -> Result<usize> {
        self.ingest(source, content).await
    }

    /// Chunk + embed (or store text-only) + index one document. Replaces any
    /// existing chunks for the same source (idempotent re-upload).
    async fn ingest(&self, source: &str, content: &str) -> Result<usize> {
        let chunks = chunk::chunk_document(&self.cfg, source, content);
        if chunks.is_empty() {
            anyhow::bail!("No readable text found in '{source}'.");
        }
        let n = chunks.len();

        if self.embedder().is_some() {
            let texts: Vec<String> = chunks.iter().map(|c| c.text.clone()).collect();
            info!("Embedding {n} chunks from '{source}'…");
            let vectors = self.embed_texts(texts).await?;
            self.store
                .upsert(&chunks, &vectors)
                .with_context(|| format!("storing vectors for '{source}'"))?;
        } else {
            warn!("No embedding model available — indexing '{source}' for keyword search only.");
            self.store
                .insert_text_only(&chunks)
                .with_context(|| format!("storing chunks for '{source}'"))?;
        }

        // Refresh the in-memory BM25 index for this source.
        {
            let mut bm25 = self.bm25.write().expect("bm25 lock poisoned");
            bm25.remove_source(source);
            for c in &chunks {
                bm25.add(c);
            }
        }
        info!("Indexed '{source}' ({n} chunks)");
        Ok(n)
    }

    /// Index every supported document in the data directory. Returns the sources
    /// that were (re)indexed.
    async fn index_data_dir(&self) -> Result<Vec<String>> {
        let docs = load_all(&self.cfg.data_dir)?;
        if docs.is_empty() {
            warn!("No documents found in '{}'", self.cfg.data_dir);
        }
        let mut sources = Vec::new();
        for doc in docs {
            match self.ingest(&doc.source, &doc.content).await {
                Ok(_) => sources.push(doc.source),
                // Never silently swallow indexing failures.
                Err(e) => warn!("Skipping '{}': {e:#}", doc.source),
            }
        }
        Ok(sources)
    }

    /// Run a retrieval and report the effective mode + whether the evidence is
    /// confident enough to answer from.
    pub async fn retrieve(
        &self,
        query: &str,
        top_k: usize,
        requested: RetrievalMode,
        min_similarity: f32,
    ) -> Result<RetrievalResult> {
        let have_vectors = self.embedder().is_some() && self.store.count().unwrap_or(0) > 0;

        // Degrade gracefully when no embeddings are available.
        let mode = match requested {
            RetrievalMode::Keyword => RetrievalMode::Keyword,
            RetrievalMode::Vector if have_vectors => RetrievalMode::Vector,
            RetrievalMode::Hybrid if have_vectors => RetrievalMode::Hybrid,
            other => {
                warn!(
                    "{} retrieval requested but no embeddings available; using keyword",
                    other.as_str()
                );
                RetrievalMode::Keyword
            }
        };

        let keyword_hits = if mode != RetrievalMode::Vector {
            self.bm25
                .read()
                .expect("bm25 lock poisoned")
                .search(query, top_k)
        } else {
            Vec::new()
        };
        let keyword_found = !keyword_hits.is_empty();

        let vector_hits = if mode != RetrievalMode::Keyword {
            let qv = self
                .embed_texts(vec![query.to_string()])
                .await?
                .pop()
                .ok_or_else(|| anyhow::anyhow!("query embedding produced no vector"))?;
            self.store.search(&qv, top_k)?
        } else {
            Vec::new()
        };
        let best_vec_sim = vector_hits.first().map(|h| h.score).unwrap_or(0.0);

        let hits = match mode {
            RetrievalMode::Keyword => keyword_hits,
            RetrievalMode::Vector => vector_hits,
            RetrievalMode::Hybrid => {
                hybrid::reciprocal_rank_fusion(&[keyword_hits, vector_hits], top_k)
            }
        };

        // Confidence: a keyword hit means real term overlap; a vector hit counts
        // only if it clears the similarity threshold.
        let confident = !hits.is_empty()
            && match mode {
                RetrievalMode::Keyword => true,
                RetrievalMode::Vector => best_vec_sim >= min_similarity,
                RetrievalMode::Hybrid => keyword_found || best_vec_sim >= min_similarity,
            };

        Ok(RetrievalResult {
            hits,
            mode,
            confident,
        })
    }
}

/// Result of a retrieval, with provenance for the prompt builder and callers.
pub struct RetrievalResult {
    /// Ranked, deduplicated hits with metadata and scores.
    pub hits: Vec<Hit>,
    /// The mode actually used (may differ from the request if it had to
    /// degrade to keyword because no embeddings were available).
    pub mode: RetrievalMode,
    /// Whether there is enough relevant evidence to answer confidently.
    pub confident: bool,
}

// ── Public API (kept compatible with the rest of the app) ────────────────────

/// Create the shared store, opening the on-disk vector database. No embedding
/// model is loaded yet — call [`init_embedder`] (usually in the background).
pub fn new_shared_store(cfg: &Config) -> Result<SharedStore> {
    let rag_cfg = RagConfig::from_app_config(cfg);
    let system = RagSystem::open(rag_cfg, None)?;
    Ok(Arc::new(system))
}

/// Attempt to load the local embedding model and install it on the store. Best
/// effort: on failure the system keeps working in keyword-only mode. This may
/// download the model on first run (then it is cached for offline use).
pub async fn init_embedder(_cfg: &Config, store: &SharedStore) {
    #[cfg(feature = "local-embeddings")]
    {
        let model = store.cfg.embedding_model.clone();
        let built = tokio::task::spawn_blocking(move || {
            fastembed_provider::LocalFastEmbedProvider::new(&model)
        })
        .await;
        match built {
            Ok(Ok(provider)) => store.set_embedder(Arc::new(provider)),
            Ok(Err(e)) => warn!(
                "Local embedding model unavailable, falling back to keyword (BM25) search: {e:#}"
            ),
            Err(e) => warn!("Embedding model load task failed: {e}"),
        }
    }
    #[cfg(not(feature = "local-embeddings"))]
    {
        let _ = store;
        info!("Built without the 'local-embeddings' feature — using keyword (BM25) retrieval.");
    }
}

/// Build the index from the documents in the data directory. Reuses whatever is
/// already persisted on disk and adds/refreshes the data-dir documents.
pub async fn build_index(
    _client: &reqwest::Client,
    _cfg: &Config,
    store: &SharedStore,
) -> Result<()> {
    info!("Indexing documents from '{}'", store.cfg.data_dir);
    let sources = store.index_data_dir().await?;
    info!(
        "Index ready ({} sources, {} chunks)",
        sources.len(),
        store.store.count().unwrap_or(0)
    );
    Ok(())
}

/// Add one uploaded document to the live index. Returns chunks added.
pub async fn add_file(
    _client: &reqwest::Client,
    _cfg: &Config,
    store: &SharedStore,
    source: &str,
    content: &str,
) -> Result<usize> {
    store.ingest(source, content).await
}

/// List distinct source documents with chunk counts.
pub async fn list_sources(store: &SharedStore) -> Vec<(String, usize)> {
    store.store.list_sources().unwrap_or_default()
}

/// Remove all chunks for a named source from both the vector store and BM25.
pub async fn remove_source(store: &SharedStore, name: &str) -> bool {
    let removed = store.store.delete_source(name).unwrap_or(0) > 0;
    if removed {
        store
            .bm25
            .write()
            .expect("bm25 lock poisoned")
            .remove_source(name);
    }
    removed
}

/// Safe rebuild: recreate the local vector index from everything currently
/// indexed (data-dir documents are re-read from disk; uploaded documents are
/// re-embedded from their stored chunks). Returns the chunk count afterwards.
pub async fn rebuild(
    _client: &reqwest::Client,
    _cfg: &Config,
    store: &SharedStore,
) -> Result<usize> {
    info!("Rebuilding local vector index…");
    // Snapshot what's indexed, then wipe and recreate.
    let snapshot = store.store.all_chunks()?;
    store.store.clear()?;
    store.bm25.write().expect("bm25 lock poisoned").clear();

    // Re-read on-disk documents (picks up edits / additions / removals).
    let fresh_sources: std::collections::BTreeSet<String> =
        store.index_data_dir().await?.into_iter().collect();

    // Re-index uploaded documents (those not backed by a data-dir file) from
    // their stored chunks, so nothing the user uploaded is lost.
    let mut upload_chunks: Vec<types::StoredChunk> = snapshot
        .into_iter()
        .filter(|c| !fresh_sources.contains(&c.source))
        .collect();
    upload_chunks.sort_by(|a, b| a.source.cmp(&b.source).then(a.chunk_id.cmp(&b.chunk_id)));

    if !upload_chunks.is_empty() {
        if store.embedder().is_some() {
            let texts: Vec<String> = upload_chunks.iter().map(|c| c.text.clone()).collect();
            let vectors = store.embed_texts(texts).await?;
            store.store.upsert(&upload_chunks, &vectors)?;
        } else {
            store.store.insert_text_only(&upload_chunks)?;
        }
        let mut bm25 = store.bm25.write().expect("bm25 lock poisoned");
        for c in &upload_chunks {
            bm25.add(c);
        }
    }

    let count = store.store.count()?;
    info!("Rebuild complete ({count} chunks)");
    Ok(count)
}

// ── Retrieve (used by /api/rag) ──────────────────────────────────────────────

pub async fn retrieve(
    _client: &reqwest::Client,
    cfg: &Config,
    store: &SharedStore,
    query: &str,
    k: usize,
) -> Result<Vec<Hit>> {
    if store.store.count().unwrap_or(0) == 0
        && store.bm25.read().expect("bm25 lock poisoned").is_empty()
    {
        anyhow::bail!("No documents are loaded yet. Add a document first.");
    }
    let mode = RetrievalMode::parse(&cfg.retrieval_mode);
    let top_k = if k == 0 { cfg.top_k.max(1) } else { k };
    let result = store
        .retrieve(query, top_k, mode, cfg.min_similarity)
        .await?;
    Ok(result.hits)
}

// ── Context building for the chat/agent layer ────────────────────────────────

/// If the user's message clearly refers to ONE specific loaded document, return
/// that document's source name (so e.g. "summarize pricing.md" focuses on it).
fn matched_document(message: &str, sources: &[String]) -> Option<String> {
    let msg = message.to_lowercase();
    let mut best: Option<(usize, &String)> = None;
    for src in sources {
        let lower = src.to_lowercase();
        let stem = lower.rsplit_once('.').map(|(s, _)| s).unwrap_or(&lower);
        let stem_spaced = stem.replace(['_', '-'], " ");
        for key in [lower.as_str(), stem, stem_spaced.as_str()] {
            if key.len() >= 3
                && msg.contains(key)
                && best.map(|(len, _)| key.len() > len).unwrap_or(true)
            {
                best = Some((key.len(), src));
            }
        }
    }
    best.map(|(_, s)| s.clone())
}

/// Build the document context placed in front of the model each turn.
///
///   * If the user names a specific document, include only that one (in full).
///   * Else if the whole corpus fits the budget, include it all (so summaries
///     and broad questions just work — preserved NotebookLM behaviour).
///   * Else run hybrid retrieval and include the top cited chunks. If retrieval
///     confidence is low, instruct the model to say it lacks evidence rather
///     than guessing.
///
/// Returns (context_text, included_sources, truncated).
pub async fn build_context(
    store: &SharedStore,
    cfg: &Config,
    message: &str,
) -> (String, Vec<String>, bool) {
    let all_chunks = store.bm25.read().expect("bm25 lock poisoned").all_chunks();
    if all_chunks.is_empty() {
        return (String::new(), Vec::new(), false);
    }

    // Group chunks back into documents, preserving first-seen order.
    let mut doc_order: Vec<String> = Vec::new();
    let mut by_source: std::collections::HashMap<String, Vec<String>> = Default::default();
    for (source, text) in &all_chunks {
        if !by_source.contains_key(source) {
            doc_order.push(source.clone());
        }
        by_source
            .entry(source.clone())
            .or_default()
            .push(text.clone());
    }

    // 1) User named a specific document.
    if let Some(target) = matched_document(message, &doc_order) {
        let mut out = format!("\n===== DOCUMENT: {target} =====\n");
        let mut used = 0usize;
        let mut truncated = false;
        for text in &by_source[&target] {
            if used + text.len() > CONTEXT_CHAR_BUDGET {
                truncated = true;
                break;
            }
            out.push_str(text);
            out.push('\n');
            used += text.len();
        }
        return (out, vec![target], truncated);
    }

    // 2) Whole corpus fits — include everything.
    let total_chars: usize = all_chunks.iter().map(|(_, t)| t.len()).sum();
    if total_chars <= CONTEXT_CHAR_BUDGET {
        let mut out = String::new();
        let mut sources = Vec::new();
        for src in &doc_order {
            out.push_str(&format!("\n===== DOCUMENT: {src} =====\n"));
            for text in &by_source[src] {
                out.push_str(text);
                out.push('\n');
            }
            sources.push(src.clone());
        }
        return (out, sources, false);
    }

    // 3) Large corpus — hybrid retrieval of the most relevant cited chunks.
    let mode = RetrievalMode::parse(&cfg.retrieval_mode);
    let top_k = cfg.top_k.max(1);
    let retrieval = match store
        .retrieve(message, top_k, mode, cfg.min_similarity)
        .await
    {
        Ok(r) => r,
        Err(e) => {
            warn!("Retrieval failed while building context: {e:#}");
            RetrievalResult {
                hits: Vec::new(),
                mode,
                confident: false,
            }
        }
    };

    if !retrieval.confident || retrieval.hits.is_empty() {
        let note = "\n(No sufficiently relevant passages were found in the uploaded documents for \
                    this question. Tell the user you could not find enough evidence in their files \
                    to answer confidently, rather than guessing.)\n";
        return (note.to_string(), Vec::new(), true);
    }

    let mut out = format!(
        "\n(The most relevant excerpts from your documents, retrieved by {} search. Each is \
         labelled with its source so you can cite it.)\n",
        retrieval.mode.as_str()
    );
    let mut used = 0usize;
    let mut sources_set: std::collections::BTreeSet<String> = Default::default();
    for h in &retrieval.hits {
        if used + h.text.len() > CONTEXT_CHAR_BUDGET {
            break;
        }
        let tag = match h.retrieval {
            HitSource::Keyword => "keyword",
            HitSource::Vector => "semantic",
            HitSource::Hybrid => "hybrid",
        };
        out.push_str(&format!(
            "\n----- from {} (chunk {}, {} match, score {:.2}) -----\n{}\n",
            h.source, h.chunk_id, tag, h.score, h.text
        ));
        used += h.text.len();
        sources_set.insert(h.source.clone());
    }
    (out, sources_set.into_iter().collect(), true)
}

// ── Document loading ─────────────────────────────────────────────────────────

struct Doc {
    source: String,
    content: String,
}

fn load_all(data_dir: &str) -> Result<Vec<Doc>> {
    let mut docs = Vec::new();
    let entries = match std::fs::read_dir(data_dir) {
        Ok(e) => e,
        Err(e) => {
            warn!("Could not read data dir '{data_dir}': {e}");
            return Ok(docs);
        }
    };

    for entry in entries {
        let entry = entry?;
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let ext = path
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();
        let source = path.file_name().unwrap().to_string_lossy().into_owned();

        // Skip the vector database files that live alongside the documents.
        if matches!(ext.as_str(), "sqlite" | "db" | "sqlite-wal" | "sqlite-shm")
            || source.starts_with("rag_vectors")
        {
            continue;
        }

        let content = match ext.as_str() {
            "txt" | "md" | "markdown" | "text" => match load_txt(&path) {
                Ok(c) => c,
                Err(e) => {
                    warn!("Could not read '{source}': {e}");
                    continue;
                }
            },
            _ => {
                warn!("Skipping unsupported file: {source}");
                continue;
            }
        };
        docs.push(Doc { source, content });
    }
    docs.sort_by(|a, b| a.source.cmp(&b.source));
    Ok(docs)
}

fn load_txt(path: &Path) -> Result<String> {
    Ok(std::fs::read_to_string(path)?)
}
