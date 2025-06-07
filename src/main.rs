// src/main.rs
//
// Entry point. Starts the axum server and wires all routes.
//
// On first launch (no config.toml) the server still starts, but GET /
// serves the setup screen. Once the user completes setup via the UI,
// POST /api/config saves config.toml and the chat UI loads.

mod agent;
mod api;
mod config;
mod models;
mod rag;
mod tools;

use std::net::SocketAddr;
use std::sync::Arc;

use axum::{
    http::StatusCode,
    response::{Html, IntoResponse},
    routing::{get, post},
    Router,
};
use reqwest::Client;
use tower_http::cors::{Any, CorsLayer};
use tower_http::trace::TraceLayer;
use tracing::{info, warn};
use tracing_subscriber::{fmt, EnvFilter};

use api::config::{AppState, get_config, get_models, post_config};
use api::chat::post_chat;
use api::rag::post_rag;

// The single HTML file is embedded into the binary at compile time so the
// binary is truly self-contained — no templates directory needed at runtime.
const INDEX_HTML: &str = include_str!("../templates/index.html");

async fn serve_index() -> impl IntoResponse {
    Html(INDEX_HTML)
}

async fn health() -> impl IntoResponse {
    (StatusCode::OK, "ok")
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Logging: RUST_LOG=info by default, override with e.g. RUST_LOG=debug
    tracing_subscriber::fmt()
        .with_env_filter(
            EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| EnvFilter::new("info,agent_toolkit=debug")),
        )
        .init();

    // Load saved config if it exists.
    let initial_cfg = match config::load() {
        Ok(Some(cfg)) => {
            info!("Loaded config: api_base={} model={}", cfg.api_base, cfg.model);
            Some(cfg)
        }
        Ok(None) => {
            info!("No config.toml found — serving setup screen");
            None
        }
        Err(e) => {
            warn!("Could not load config.toml: {e} — serving setup screen");
            None
        }
    };

    let http_client = Client::builder()
        .timeout(std::time::Duration::from_secs(120))
        .build()?;

    // If we have a config, pre-build the Qdrant index in the background.
    if let Some(ref cfg) = initial_cfg {
        let client = http_client.clone();
        let cfg    = cfg.clone();
        tokio::spawn(async move {
            if let Err(e) = rag::build_index(&client, &cfg).await {
                warn!("Background index build failed: {e}");
            }
        });
    }

    let state = AppState {
        cfg:      config::new_shared(initial_cfg),
        client:   http_client,
        sessions: agent::new_session_store(),
    };

    let cors = CorsLayer::new().allow_origin(Any).allow_methods(Any).allow_headers(Any);

    let app = Router::new()
        // UI
        .route("/",              get(serve_index))
        .route("/health",        get(health))
        // Config + models
        .route("/api/config",    get(get_config).post(post_config))
        .route("/api/models",    get(get_models))
        // Chat + RAG
        .route("/api/chat",      post(post_chat))
        .route("/api/rag",       post(post_rag))
        .layer(cors)
        .layer(TraceLayer::new_for_http())
        .with_state(state);

    let port: u16 = std::env::var("PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(3000);
    let addr = SocketAddr::from(([0, 0, 0, 0], port));

    info!("Listening on http://localhost:{port}");
    let listener = tokio::net::TcpListener::bind(addr).await?;
    axum::serve(listener, app).await?;
    Ok(())
}
