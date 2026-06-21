// src/main.rs
//
// Entry point. Starts the axum server and wires all routes.
//
// On first launch (no config.toml) the server still starts, but GET /
// serves the setup screen. Once the user completes setup via the UI,
// POST /api/config saves config.toml and the chat UI loads.

use std::net::SocketAddr;

use axum::{
    extract::Request,
    http::{HeaderName, StatusCode},
    response::{Html, IntoResponse},
    routing::{get, post},
    Router,
};
use reqwest::Client;
use tower_http::cors::{Any, CorsLayer};
use tower_http::request_id::{MakeRequestUuid, PropagateRequestIdLayer, SetRequestIdLayer};
use tower_http::trace::TraceLayer;
use tracing::{error, info, warn};
use tracing_subscriber::EnvFilter;

use agent_toolkit::{api, config, metrics, rag, sessions};
use api::chat::post_chat;
use api::config::{get_config, get_models, post_config, AppState};
use api::rag::post_rag;

const REQUEST_ID_HEADER: &str = "x-request-id";

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
    // Logging: RUST_LOG=info by default, override with e.g. RUST_LOG=debug.
    // LOG_FORMAT=json switches to structured JSON lines for log aggregators
    // (the default human-readable format is friendlier for local dev).
    let env_filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info,agent_toolkit=debug"));
    let json_logs = std::env::var("LOG_FORMAT").is_ok_and(|v| v.eq_ignore_ascii_case("json"));
    if json_logs {
        tracing_subscriber::fmt()
            .json()
            .with_env_filter(env_filter)
            .init();
    } else {
        tracing_subscriber::fmt().with_env_filter(env_filter).init();
    }

    // Load saved config, or fall back to a sensible default that points at a
    // local Ollama. This means the app boots straight into a working chat UI:
    // if Ollama is running, the user can chat with zero setup. If it isn't,
    // the chat UI still loads and a single inline message guides them.
    let initial_cfg = match config::load() {
        Ok(Some(cfg)) => {
            info!(
                "Loaded config: api_base={} model={}",
                cfg.api_base, cfg.model
            );
            cfg
        }
        Ok(None) => {
            info!("No config.toml — defaulting to Groq (paste a free key in Settings)");
            config::Config::default_local()
        }
        Err(e) => {
            warn!("Could not load config.toml: {e} — defaulting to Groq");
            config::Config::default_local()
        }
    };
    info!("Chat API key source: {}", initial_cfg.api_key_source());

    let http_client = Client::builder()
        .timeout(std::time::Duration::from_secs(120))
        .build()?;

    // Shared, on-disk vector store: local SQLite + sqlite-vec. No external DB,
    // no Docker. Opening it is fast; the embedding model loads separately below.
    let store = match rag::new_shared_store(&initial_cfg) {
        Ok(s) => s,
        Err(e) => {
            error!("Could not open the local vector store: {e:#}");
            return Err(e);
        }
    };

    // Persistent session memory: conversation turns survive a server restart
    // (the backend used to keep them only in an in-process map).
    let sessions = match sessions::open(&initial_cfg.session_db_path) {
        Ok(s) => s,
        Err(e) => {
            error!("Could not open the session store: {e:#}");
            return Err(e);
        }
    };

    let state = AppState {
        cfg: config::new_shared(Some(initial_cfg.clone())),
        client: http_client.clone(),
        sessions,
        store: store.clone(),
    };

    // In the background: load the local embedding model (the first run may
    // download it; afterwards it's cached for offline use), then build the
    // index. Both steps are non-fatal — if the model can't load, the app falls
    // back to keyword (BM25) retrieval, and indexing retries on demand.
    {
        let client = http_client.clone();
        let store = store.clone();
        let cfg = initial_cfg;
        tokio::spawn(async move {
            rag::init_embedder(&cfg, &store).await;
            if let Err(e) = rag::build_index(&client, &cfg, &store).await {
                warn!("Background index build failed (will retry on demand): {e:#}");
            }
        });
    }

    // Prometheus metrics, exposed at GET /metrics. The HTTP-level counter and
    // histogram are recorded by `metrics::track` below; app-level gauges
    // (e.g. rag_indexed_chunks) are updated directly where the RAG index
    // changes.
    let metrics_handle = metrics::install();

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let request_id_header = HeaderName::from_static(REQUEST_ID_HEADER);

    let app_routes = Router::new()
        // UI
        .route("/", get(serve_index))
        .route("/health", get(health))
        // Config + models
        .route("/api/config", get(get_config).post(post_config))
        .route("/api/models", get(get_models))
        // Chat + RAG
        .route("/api/chat", post(post_chat))
        .route("/api/rag", post(post_rag))
        // Document upload + listing
        .route("/api/upload", post(api::upload::post_upload))
        .route("/api/sources", get(api::upload::get_sources))
        .route("/api/remove", post(api::upload::post_remove))
        .route("/api/rebuild", post(api::upload::post_rebuild))
        .route_layer(axum::middleware::from_fn(metrics::track))
        .with_state(state);

    let metrics_routes = Router::new()
        .route("/metrics", get(metrics::handler))
        .with_state(metrics_handle);

    let app = app_routes
        .merge(metrics_routes)
        .layer(PropagateRequestIdLayer::new(request_id_header.clone()))
        .layer(
            TraceLayer::new_for_http().make_span_with(move |req: &Request| {
                let request_id = req
                    .headers()
                    .get(REQUEST_ID_HEADER)
                    .and_then(|v| v.to_str().ok())
                    .unwrap_or("-");
                tracing::info_span!(
                    "http_request",
                    method = %req.method(),
                    path = %req.uri().path(),
                    request_id = %request_id,
                )
            }),
        )
        .layer(SetRequestIdLayer::new(request_id_header, MakeRequestUuid))
        .layer(cors);

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
