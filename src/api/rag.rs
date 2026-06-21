// src/api/rag.rs
//
// POST /api/rag
//   Request:  { "query": "..." }
//   Response: { "results": [{ "text", "source", "score" }, ...] }
//
// Direct retrieval without agent overhead. Used by the Docs Q&A tab.

use axum::{extract::State, http::StatusCode, Json};
use serde::Deserialize;
use serde_json::{json, Value};

use crate::api::config::AppState;
use crate::rag;

#[derive(Deserialize)]
pub struct RagRequest {
    pub query: String,
}

pub async fn post_rag(
    State(state): State<AppState>,
    Json(req): Json<RagRequest>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let cfg_guard = state.cfg.read().await;
    let cfg = cfg_guard.as_ref().ok_or_else(|| {
        (
            StatusCode::PRECONDITION_FAILED,
            Json(json!({ "error": "Not configured." })),
        )
    })?;

    let query = req.query.trim().to_string();
    if query.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(json!({ "error": "query is required" })),
        ));
    }

    let top_k = cfg.top_k.max(1);
    match rag::retrieve(&state.client, cfg, &state.store, &query, top_k).await {
        Ok(hits) => Ok(Json(json!({
            "results": hits,
            "query": query,
            "mode": cfg.retrieval_mode,
        }))),
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": e.to_string() })),
        )),
    }
}
