// src/rag.rs
//
// RAG pipeline:
//   1. Load documents from data/ (txt, md, pdf)
//   2. Chunk with text-splitter (pure Rust, no ML weights)
//   3. Embed chunks via the configured embeddings endpoint
//   4. Upsert into Qdrant (local Docker or binary)
//   5. retrieve(query) -> Vec<Hit>
//
// Qdrant is used because it runs as a single ~50 MB binary or tiny Docker
// image, has a clean REST API, and needs zero Python. The Qdrant Rust SDK
// is intentionally avoided here; we talk to the REST API directly with
// reqwest to keep the dependency tree small.

use anyhow::{Context, Result};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::path::Path;
use text_splitter::TextSplitter;
use tracing::{info, warn};

use crate::config::Config;

const COLLECTION: &str = "agent_toolkit";
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
    id:     String,   // sha256 of content for dedup
}

fn chunk_docs(docs: Vec<Doc>) -> Vec<Chunk> {
    // text-splitter uses character count as the size unit.
    let splitter = TextSplitter::new(CHUNK_SIZE);
    let mut chunks = Vec::new();

    for doc in docs {
        for piece in splitter.chunks(&doc.content) {
            let trimmed = piece.trim().to_string();
            if trimmed.len() < 40 { continue; }   // skip tiny fragments

            let mut hasher = Sha256::new();
            hasher.update(trimmed.as_bytes());
            let id = format!("{:x}", hasher.finalize())[..16].to_string();

            chunks.push(Chunk {
                source: doc.source.clone(),
                text:   trimmed,
                id,
            });
        }
    }
    chunks
}

// ── Embeddings ───────────────────────────────────────────────────────────────

async fn embed_batch(
    client:  &Client,
    base:    &str,
    key:     &str,
    model:   &str,
    texts:   &[String],
) -> Result<Vec<Vec<f32>>> {
    let url = format!("{}/embeddings", base.trim_end_matches('/'));
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

// ── Qdrant helpers ───────────────────────────────────────────────────────────

async fn qdrant_collection_exists(client: &Client, base: &str) -> bool {
    let url  = format!("{}/collections/{}", base.trim_end_matches('/'), COLLECTION);
    client.get(&url).send().await.map(|r| r.status().is_success()).unwrap_or(false)
}

async fn qdrant_create_collection(client: &Client, base: &str, dim: usize) -> Result<()> {
    let url  = format!("{}/collections/{}", base.trim_end_matches('/'), COLLECTION);
    let body = json!({
        "vectors": {
            "size": dim,
            "distance": "Cosine"
        }
    });
    let resp = client.put(&url).json(&body).send().await?;
    if !resp.status().is_success() {
        let text = resp.text().await.unwrap_or_default();
        anyhow::bail!("creating Qdrant collection: {text}");
    }
    Ok(())
}

async fn qdrant_upsert(
    client: &Client,
    base:   &str,
    points: Vec<Value>,
) -> Result<()> {
    let url  = format!("{}/collections/{}/points", base.trim_end_matches('/'), COLLECTION);
    let body = json!({ "points": points });
    let resp = client.put(&url).json(&body).send().await?;
    if !resp.status().is_success() {
        let text = resp.text().await.unwrap_or_default();
        anyhow::bail!("Qdrant upsert: {text}");
    }
    Ok(())
}

async fn qdrant_count(client: &Client, base: &str) -> Result<u64> {
    let url  = format!("{}/collections/{}", base.trim_end_matches('/'), COLLECTION);
    let resp = client.get(&url).send().await?;
    if !resp.status().is_success() { return Ok(0); }
    let v: Value = resp.json().await?;
    Ok(v["result"]["vectors_count"].as_u64().unwrap_or(0))
}

// ── Public: build index ──────────────────────────────────────────────────────

/// Ingest all documents in data_dir into Qdrant.
/// Skips the ingest if the collection already has vectors (idempotent).
pub async fn build_index(client: &Client, cfg: &Config) -> Result<()> {
    let qdrant = &cfg.qdrant_url;

    // If the collection already has content, don't re-embed.
    if qdrant_collection_exists(client, qdrant).await {
        let count = qdrant_count(client, qdrant).await?;
        if count > 0 {
            info!("Qdrant collection already has {count} vectors, skipping ingest");
            return Ok(());
        }
    }

    info!("Loading documents from '{}'", cfg.data_dir);
    let docs   = load_all(&cfg.data_dir)?;
    let chunks = chunk_docs(docs);
    info!("Chunked into {} pieces", chunks.len());

    if chunks.is_empty() {
        anyhow::bail!("No chunks produced — is '{}' empty?", cfg.data_dir);
    }

    // Embed in batches of 32.
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
    info!("Embedding dim: {dim}");

    // Create or recreate the collection.
    if qdrant_collection_exists(client, qdrant).await {
        // Delete and recreate to ensure correct dimensions.
        let url = format!("{}/collections/{}", qdrant.trim_end_matches('/'), COLLECTION);
        client.delete(&url).send().await.ok();
    }
    qdrant_create_collection(client, qdrant, dim).await?;

    // Build Qdrant point objects.
    let points: Vec<Value> = chunks
        .iter()
        .zip(all_vecs.iter())
        .enumerate()
        .map(|(i, (chunk, vec))| {
            json!({
                "id": i,
                "vector": vec,
                "payload": {
                    "text":   chunk.text,
                    "source": chunk.source,
                    "hash":   chunk.id,
                }
            })
        })
        .collect();

    // Upsert in batches of 100.
    for batch in points.chunks(100) {
        qdrant_upsert(client, qdrant, batch.to_vec()).await?;
    }

    info!("Indexed {} chunks into Qdrant", chunks.len());
    Ok(())
}

// ── Public: retrieve ─────────────────────────────────────────────────────────

pub async fn retrieve(
    client:  &Client,
    cfg:     &Config,
    query:   &str,
    k:       usize,
) -> Result<Vec<Hit>> {
    let emb_base  = cfg.effective_embeddings_base();
    let emb_key   = cfg.effective_embeddings_key();
    let query_vec = embed_one(client, emb_base, emb_key, &cfg.embeddings_model, query).await?;

    let url  = format!("{}/collections/{}/points/search", cfg.qdrant_url.trim_end_matches('/'), COLLECTION);
    let body = json!({
        "vector": query_vec,
        "limit":  k,
        "with_payload": true,
    });

    let resp = client.post(&url).json(&body).send().await
        .context("Qdrant search")?;

    if !resp.status().is_success() {
        let text = resp.text().await.unwrap_or_default();
        anyhow::bail!("Qdrant search error: {text}");
    }

    #[derive(Deserialize)]
    struct SearchResp { result: Vec<SearchHit> }
    #[derive(Deserialize)]
    struct SearchHit  { score: f32, payload: HitPayload }
    #[derive(Deserialize)]
    struct HitPayload { text: String, source: String }

    let parsed: SearchResp = resp.json().await.context("parsing Qdrant search response")?;
    Ok(parsed.result.into_iter().map(|h| Hit {
        text:   h.payload.text,
        source: h.payload.source,
        score:  h.score,
    }).collect())
}
