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
use tracing_subscriber::EnvFilter;

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

    // Load saved config, or fall back to a sensible default that points at a
    // local Ollama. This means the app boots straight into a working chat UI:
    // if Ollama is running, the user can chat with zero setup. If it isn't,
    // the chat UI still loads and a single inline message guides them.
    let initial_cfg = match config::load() {
        Ok(Some(cfg)) => {
            info!("Loaded config: api_base={} model={}", cfg.api_base, cfg.model);
            cfg
        }
        Ok(None) => {
            info!("No config.toml — defaulting to local Ollama (llama3.2)");
            config::Config::default_local()
        }
        Err(e) => {
            warn!("Could not load config.toml: {e} — defaulting to local Ollama");
            config::Config::default_local()
        }
    };

    let http_client = Client::builder()
        .timeout(std::time::Duration::from_secs(120))
        .build()?;

    // Shared, in-process vector store (no external DB).
    let store = rag::new_shared_store();

    let state = AppState {
        cfg:      config::new_shared(Some(initial_cfg.clone())),
        client:   http_client.clone(),
        sessions: agent::new_session_store(),
        store:    store.clone(),
    };

    // Build the embedded index in the background so the first chat/RAG request
    // is fast. Reuses vectors.json if valid. Failures here are non-fatal — they
    // just mean the first RAG query waits, or surfaces a clear error.
    {
        let client = http_client.clone();
        let store  = store.clone();
        let cfg    = initial_cfg;
        tokio::spawn(async move {
            if let Err(e) = rag::build_index(&client, &cfg, &store).await {
                warn!("Background index build failed (will retry on demand): {e}");
            }
        });
    }

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
