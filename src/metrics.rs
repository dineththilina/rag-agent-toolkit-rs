// src/metrics.rs
//
// Prometheus metrics, exposed at GET /metrics. `track` is mounted as a
// route_layer so every request to a matched route is counted and timed by
// method, route pattern, and status code; `rag_indexed_chunks` is a gauge the
// RAG layer updates whenever the index changes.

use std::time::Instant;

use axum::{
    extract::{MatchedPath, Request, State},
    middleware::Next,
    response::IntoResponse,
};
use metrics_exporter_prometheus::{PrometheusBuilder, PrometheusHandle};

/// Install the global Prometheus recorder. Call once at startup.
pub fn install() -> PrometheusHandle {
    PrometheusBuilder::new()
        .install_recorder()
        .expect("failed to install Prometheus recorder")
}

/// Axum middleware: records request count and latency, labelled by method,
/// route (the matched pattern, e.g. `/api/chat`, not the raw path — keeps
/// label cardinality bounded), and status code.
pub async fn track(req: Request, next: Next) -> impl IntoResponse {
    let method = req.method().to_string();
    let path = req
        .extensions()
        .get::<MatchedPath>()
        .map(|p| p.as_str().to_string())
        .unwrap_or_else(|| req.uri().path().to_string());

    let start = Instant::now();
    let resp = next.run(req).await;
    let elapsed = start.elapsed().as_secs_f64();
    let status = resp.status().as_u16().to_string();

    let labels = [("method", method), ("path", path), ("status", status)];
    metrics::counter!("http_requests_total", &labels).increment(1);
    metrics::histogram!("http_request_duration_seconds", &labels).record(elapsed);

    resp
}

/// GET /metrics — Prometheus text exposition format.
pub async fn handler(State(handle): State<PrometheusHandle>) -> String {
    handle.render()
}
