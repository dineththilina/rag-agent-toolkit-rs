// src/api/chat.rs
//
// POST /api/chat
//   Request:  { "message": "...", "session_id": "..." }
//   Response: { "answer": "...", "tools_used": [...], "sources": [...], "session_id": "..." }

use axum::{extract::State, http::StatusCode, Json};
use serde::Deserialize;
use serde_json::{json, Value};
use uuid::Uuid;

use crate::agent;
use crate::api::config::AppState;

#[derive(Deserialize)]
pub struct ChatRequest {
    pub message: String,
    pub session_id: Option<String>,
}

pub async fn post_chat(
    State(state): State<AppState>,
    Json(req): Json<ChatRequest>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let cfg_guard = state.cfg.read().await;
    let cfg = cfg_guard.as_ref().ok_or_else(|| {
        (
            StatusCode::PRECONDITION_FAILED,
            Json(json!({ "error": "Not configured. Complete setup first." })),
        )
    })?;

    let message = req.message.trim().to_string();
    let session_id = req.session_id.unwrap_or_else(|| Uuid::new_v4().to_string());

    if message.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(json!({ "error": "message is required" })),
        ));
    }

    match agent::chat(
        &message,
        &session_id,
        &state.client,
        cfg,
        &state.sessions,
        &state.store,
    )
    .await
    {
        Ok(result) => Ok(Json(json!({
            "answer":     result.answer,
            "tools_used": result.tools_used,
            "sources":    result.sources,
            "session_id": session_id,
        }))),
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": e.to_string() })),
        )),
    }
}
