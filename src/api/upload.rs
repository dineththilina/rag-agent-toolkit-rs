// src/api/upload.rs
//
// POST /api/upload   — JSON { filename, text }. The browser extracts text from
//                      PDFs (via pdf.js) and plain files, so the backend never
//                      parses PDF bytes. We just chunk, embed, and index.
// GET  /api/sources  — list indexed documents with chunk counts.
// POST /api/remove   — JSON { name }. Remove a document from the index.

use axum::{extract::State, http::StatusCode, Json};
use serde::Deserialize;
use serde_json::{json, Value};
use tracing::info;

use crate::api::config::AppState;
use crate::rag;

#[derive(Deserialize)]
pub struct UploadPayload {
    pub filename: String,
    pub text: String,
}

pub async fn post_upload(
    State(state): State<AppState>,
    Json(payload): Json<UploadPayload>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let cfg = {
        let guard = state.cfg.read().await;
        guard
            .as_ref()
            .ok_or_else(|| {
                (
                    StatusCode::PRECONDITION_FAILED,
                    Json(json!({ "error": "Not ready yet. Try again in a moment." })),
                )
            })?
            .clone()
    };

    let filename = payload.filename.trim();
    let text = payload.text.trim();

    if filename.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(json!({ "error": "Missing file name." })),
        ));
    }
    if text.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(
                json!({ "error": format!("'{filename}' has no readable text. If it's a scanned PDF, it would need OCR first.") }),
            ),
        ));
    }

    match rag::add_file(&state.client, &cfg, &state.store, filename, text).await {
        Ok(n) => {
            info!("Indexed uploaded '{filename}' ({n} chunks)");
            Ok(Json(json!({ "ok": true, "name": filename, "chunks": n })))
        }
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": format!("Could not index '{filename}': {e}") })),
        )),
    }
}

pub async fn get_sources(State(state): State<AppState>) -> Json<Value> {
    let sources = rag::list_sources(&state.store).await;
    let list: Vec<Value> = sources
        .into_iter()
        .map(|(name, chunks)| json!({ "name": name, "chunks": chunks }))
        .collect();
    Json(json!({ "sources": list }))
}

#[derive(Deserialize)]
pub struct RemovePayload {
    pub name: String,
}

pub async fn post_remove(
    State(state): State<AppState>,
    Json(payload): Json<RemovePayload>,
) -> Json<Value> {
    let removed = rag::remove_source(&state.store, &payload.name).await;
    Json(json!({ "ok": true, "removed": removed }))
}

/// POST /api/rebuild — safely recreate the local vector index: re-read on-disk
/// documents and re-embed previously uploaded ones. Useful after changing the
/// embedding model or to repair the index.
pub async fn post_rebuild(
    State(state): State<AppState>,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let cfg = {
        let guard = state.cfg.read().await;
        guard
            .as_ref()
            .ok_or_else(|| {
                (
                    StatusCode::PRECONDITION_FAILED,
                    Json(json!({ "error": "Not configured." })),
                )
            })?
            .clone()
    };

    match rag::rebuild(&state.client, &cfg, &state.store).await {
        Ok(n) => {
            info!("Rebuilt local vector index ({n} chunks)");
            Ok(Json(json!({ "ok": true, "chunks": n })))
        }
        Err(e) => Err((
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": format!("Rebuild failed: {e}") })),
        )),
    }
}
