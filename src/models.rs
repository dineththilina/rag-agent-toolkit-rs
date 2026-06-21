// src/models.rs
//
// Fetches the live model list from /models on whatever API base the user
// configured. Works with OpenAI, Groq, Ollama, LM Studio, and any other
// provider that implements the OpenAI-compatible models endpoint.

use anyhow::{Context, Result};
use reqwest::Client;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Model {
    pub id: String,
    /// Optional human-readable name some providers include.
    #[serde(default)]
    pub name: String,
}

#[derive(Debug, Deserialize)]
struct ModelsResponse {
    data: Vec<ModelRaw>,
}

#[derive(Debug, Deserialize)]
struct ModelRaw {
    id: String,
    #[serde(default)]
    name: Option<String>,
}

/// Fetch models from `{base_url}/models`.
/// Returns them sorted by id so the list is stable across calls.
pub async fn fetch(client: &Client, base_url: &str, api_key: &str) -> Result<Vec<Model>> {
    let url = format!("{}/models", base_url.trim_end_matches('/'));

    let mut req = client.get(&url);
    if !api_key.is_empty() {
        req = req.bearer_auth(api_key);
    }

    let resp = req.send().await.context("reaching the models endpoint")?;

    if !resp.status().is_success() {
        let status = resp.status();
        let body = resp.text().await.unwrap_or_default();
        anyhow::bail!("models endpoint returned {}: {}", status, body);
    }

    let parsed: ModelsResponse = resp.json().await.context("parsing models response")?;

    let mut models: Vec<Model> = parsed
        .data
        .into_iter()
        .map(|m| Model {
            name: m.name.unwrap_or_else(|| m.id.clone()),
            id: m.id,
        })
        .collect();

    models.sort_by(|a, b| a.id.cmp(&b.id));
    Ok(models)
}
