// src/api/upload.rs
//
// POST /api/upload   — multipart file upload (PDF / TXT / MD). Extracts text,
//                      chunks, embeds, and adds it to the live searchable store.
// GET  /api/sources  — list the documents currently searchable, with chunk counts.

use axum::{
    extract::{Multipart, State},
    http::StatusCode,
    Json,
};
use serde_json::{json, Value};
use tracing::info;

use crate::api::config::AppState;
use crate::rag;

pub async fn post_upload(
    State(state): State<AppState>,
    mut multipart: Multipart,
) -> Result<Json<Value>, (StatusCode, Json<Value>)> {
    let cfg_guard = state.cfg.read().await;
    let cfg = cfg_guard.as_ref().ok_or_else(|| (
        StatusCode::PRECONDITION_FAILED,
        Json(json!({ "error": "Not ready yet. Try again in a moment." })),
    ))?.clone();
    drop(cfg_guard);

    let mut added_files: Vec<Value> = Vec::new();
    let mut total_chunks = 0usize;

    // Iterate over each uploaded file part.
    loop {
        let field = match multipart.next_field().await {
            Ok(Some(f)) => f,
            Ok(None)    => break,
            Err(e)      => return Err((
                StatusCode::BAD_REQUEST,
                Json(json!({ "error": format!("Upload was interrupted: {e}") })),
            )),
        };

        let filename = field.file_name().map(|s| s.to_string())
            .unwrap_or_else(|| "upload".to_string());

        let bytes = match field.bytes().await {
            Ok(b)  => b,
            Err(e) => return Err((
                StatusCode::BAD_REQUEST,
                Json(json!({ "error": format!("Could not read '{filename}': {e}") })),
            )),
        };

        if bytes.is_empty() {
            continue;
        }

        // Extract text by file type.
        let text = match rag::extract_text(&filename, &bytes) {
            Ok(t)  => t,
            Err(e) => return Err((
                StatusCode::BAD_REQUEST,
                Json(json!({ "error": format!("{e}") })),
            )),
        };

        if text.trim().is_empty() {
            return Err((
                StatusCode::BAD_REQUEST,
                Json(json!({ "error": format!("'{filename}' has no readable text. If it's a scanned PDF, it needs OCR first.") })),
            ));
        }

        // Chunk + embed + add to the live store.
        match rag::add_file(&state.client, &cfg, &state.store, &filename, &text).await {
            Ok(n) => {
                total_chunks += n;
                added_files.push(json!({ "name": filename, "chunks": n }));
                info!("Uploaded '{filename}' ({n} chunks)");
            }
            Err(e) => return Err((
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(json!({ "error": format!("Could not index '{filename}': {e}") })),
            )),
        }
    }

    if added_files.is_empty() {
        return Err((
            StatusCode::BAD_REQUEST,
            Json(json!({ "error": "No files received." })),
        ));
    }

    Ok(Json(json!({
        "ok": true,
        "files": added_files,
        "total_chunks": total_chunks,
    })))
}

pub async fn get_sources(State(state): State<AppState>) -> Json<Value> {
    let sources = rag::list_sources(&state.store).await;
    let list: Vec<Value> = sources.into_iter()
        .map(|(name, chunks)| json!({ "name": name, "chunks": chunks }))
        .collect();
    Json(json!({ "sources": list }))
}
