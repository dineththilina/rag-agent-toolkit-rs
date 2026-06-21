// src/api/config.rs
//
// GET  /api/config          — return current config (key redacted)
// POST /api/config          — save new config, rebuild index
// GET  /api/models?base=...&key=...  — fetch live model list from the API

use axum::{
    extract::{Query, State},
    http::StatusCode,
    Json,
};
use reqwest::Client;
use serde::Deserialize;
use serde_json::{json, Value};
use tracing::{error, info};

use crate::config::{self, Config, SharedConfig};
use crate::models;
use crate::rag::{self, SharedStore};

#[derive(Clone)]
pub struct AppState {
    pub cfg: SharedConfig,
    pub client: Client,
    pub sessions: crate::sessions::SessionStore,
    pub store: SharedStore,
}

// ── GET /api/config ──────────────────────────────────────────────────────────

pub async fn get_config(State(state): State<AppState>) -> Json<Value> {
    let cfg = state.cfg.read().await;
    match &*cfg {
        None => Json(json!({ "configured": false })),
        Some(c) => Json(json!({
            "configured":       true,
            "api_base":         c.api_base,
            // Never send the key back to the browser.
            "api_key_set":      !c.api_key.is_empty(),
            "model":            c.model,
            "embeddings_model": c.embeddings_model,
            "data_dir":         c.data_dir,
            // Local vector RAG settings.
            "retrieval_mode":   c.retrieval_mode,
            "top_k":            c.top_k,
            "min_similarity":   c.min_similarity,
            "embedding_model":  c.embedding_model,
            "embedding_dim":    c.embedding_dim,
        })),
    }
}

// ── POST /api/config ─────────────────────────────────────────────────────────

#[derive(Deserialize)]
pub struct ConfigPayload {
    pub api_base: String,
    pub api_key: Option<String>,
    pub model: String,
    pub embeddings_base: Option<String>,
    pub embeddings_model: Option<String>,
    pub embeddings_key: Option<String>,
    // Optional local vector RAG settings; omitted fields keep their current value.
    pub retrieval_mode: Option<String>,
    pub top_k: Option<usize>,
    pub min_similarity: Option<f32>,
    pub embedding_model: Option<String>,
}

pub async fn post_config(
    State(state): State<AppState>,
    Json(payload): Json<ConfigPayload>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    // Carry over existing RAG settings (and data_dir / db path / dim) so saving
    // provider settings from the UI never silently resets them.
    let base = state.cfg.read().await.clone().unwrap_or_default();

    let mut new_cfg = Config {
        api_base: payload.api_base.trim().to_string(),
        api_key: payload.api_key.unwrap_or_default().trim().to_string(),
        model: payload.model.trim().to_string(),
        embeddings_base: payload
            .embeddings_base
            .unwrap_or_default()
            .trim()
            .to_string(),
        embeddings_model: payload
            .embeddings_model
            .unwrap_or_else(|| "text-embedding-3-small".into()),
        embeddings_key: payload
            .embeddings_key
            .unwrap_or_default()
            .trim()
            .to_string(),
        data_dir: if base.data_dir.trim().is_empty() {
            "data".into()
        } else {
            base.data_dir.clone()
        },
        retrieval_mode: payload.retrieval_mode.unwrap_or(base.retrieval_mode),
        top_k: payload.top_k.unwrap_or(base.top_k),
        min_similarity: payload.min_similarity.unwrap_or(base.min_similarity),
        embedding_model: payload.embedding_model.unwrap_or(base.embedding_model),
        embedding_dim: base.embedding_dim,
        vector_db_path: base.vector_db_path.clone(),
        session_db_path: base.session_db_path.clone(),
    };
    // An AGENT_API_KEY / AGENT_EMBEDDINGS_KEY env var always wins over
    // whatever the setup UI just posted.
    new_cfg.apply_env_secrets();

    // Validate: try fetching models to confirm the API is reachable.
    if let Err(e) = models::fetch(&state.client, &new_cfg.api_base, &new_cfg.api_key).await {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(json!({ "error": format!("Cannot reach API: {e}") })),
        ));
    }

    // Build the embedded vector index. May take a few seconds while embedding.
    info!("Building RAG index...");
    if let Err(e) = rag::build_index(&state.client, &new_cfg, &state.store).await {
        error!("Index build failed: {e}");
        return Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": format!("Index build failed: {e}") })),
        ));
    }

    // Persist config to disk.
    if let Err(e) = config::save(&new_cfg) {
        error!("Failed to save config: {e}");
    }

    // Update shared state.
    *state.cfg.write().await = Some(new_cfg);

    Ok(Json(json!({ "ok": true })))
}

// ── GET /api/models ──────────────────────────────────────────────────────────

#[derive(Deserialize)]
pub struct ModelsQuery {
    pub base: String,
    pub key: Option<String>,
}

pub async fn get_models(
    State(state): State<AppState>,
    Query(q): Query<ModelsQuery>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let key = q.key.as_deref().unwrap_or("");
    match models::fetch(&state.client, &q.base, key).await {
        Ok(list) => Ok(Json(json!({ "models": list }))),
        Err(e) => Err((
            StatusCode::BAD_GATEWAY,
            Json(json!({ "error": e.to_string() })),
        )),
    }
}
