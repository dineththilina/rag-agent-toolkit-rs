// src/agent.rs
//
// Grounded document chat (NotebookLM-style).
//
// Each turn:
//   1. Build document context from the store (all docs in full if they fit,
//      otherwise the most relevant excerpts by BM25)
//   2. Assemble: system prompt + documents + prior conversation + new message
//   3. POST to /chat/completions (a single call — the docs are already present,
//      so no tool/search round-trip is needed)
//   4. Save the user+assistant turns to session memory and return the answer
//
// The documents are rebuilt every turn, so newly uploaded files are available
// immediately. Session memory holds only the conversation turns (not the doc
// context) and is persisted to a local SQLite database (see `crate::sessions`),
// so conversations survive a server restart.

use anyhow::{Context, Result};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};

use crate::config::Config;
use crate::sessions::SessionStore;

// ── Message types ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role: String,
    pub content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls: Option<Vec<ToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id: String,
    #[serde(rename = "type")]
    pub kind: String,
    pub function: ToolFunction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolFunction {
    pub name: String,
    pub arguments: String, // JSON string from the model
}

// ── Per-turn result ───────────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
pub struct TurnResult {
    pub answer: String,
    pub tools_used: Vec<ToolUsed>,
    pub sources: Vec<String>, // document filenames from RAG hits
}

#[derive(Debug, Serialize)]
pub struct ToolUsed {
    pub tool: String,
    pub detail: Value,
}

// ── System prompt ─────────────────────────────────────────────────────────────

const SYSTEM: &str = "You are a knowledgeable assistant that answers questions about the user's documents. \
The user's documents are provided to you in full below, under a DOCUMENTS section. \
Treat their contents as your working knowledge for this conversation.\
\n\nRules:\
\n- Answer using the information in the provided documents. You already have them — do not say you need to search or look anything up.\
\n- If asked to summarize, summarize the documents. If asked about the subject matter, describe what the documents cover. If asked a specific question, answer it from the documents.\
\n- When a fact comes from a document, you may mention which document it came from (the names are shown as 'DOCUMENT: <name>').\
\n- If the documents genuinely do not contain the answer, say so plainly rather than inventing facts.\
\n- For arithmetic, you can compute it directly. Be accurate.\
\n- Maintain the thread of the conversation; the user may ask follow-up questions that refer back to earlier messages.";

// ── Main chat function ────────────────────────────────────────────────────────

// ── Main chat function ────────────────────────────────────────────────────────
//
// NotebookLM-style: the user's documents are placed directly in the model's
// context every turn, so the model genuinely knows their contents. "Summarize",
// "what's the subject", and specific questions all work with no search step.
// Conversation history is preserved for memory/follow-ups.

pub async fn chat(
    message: &str,
    session_id: &str,
    client: &Client,
    cfg: &Config,
    sessions: &SessionStore,
    store: &crate::rag::SharedStore,
) -> Result<TurnResult> {
    // Guard: a hosted provider (anything not on localhost) needs a key.
    let is_local = cfg.api_base.contains("localhost") || cfg.api_base.contains("127.0.0.1");
    if cfg.api_key.is_empty() && !is_local {
        anyhow::bail!(
            "NO_KEY: I need a free key to start. Open Settings (the gear icon) and paste your Groq key — \
             you can get one free in under a minute at console.groq.com/keys."
        );
    }

    // Conversation history (memory) for this session — user/assistant turns
    // only, loaded from the persistent session store.
    let history: Vec<Message> = sessions
        .history(session_id)
        .context("loading session history")?;

    // Build the document context fresh each turn so newly uploaded documents are
    // immediately available. This is the heart of the NotebookLM behaviour.
    let (doc_context, sources, truncated) = crate::rag::build_context(store, cfg, message).await;

    // Assemble the messages sent to the model:
    //   1. system instructions
    //   2. the documents (as a system message, rebuilt every turn)
    //   3. prior conversation turns (memory)
    //   4. the new user message
    let mut messages: Vec<Message> = Vec::new();
    messages.push(sys_msg(SYSTEM));

    if doc_context.is_empty() {
        messages.push(sys_msg(
            "The user has not added any documents yet. If they ask about documents, \
             let them know they can add one with the “Add” button, and otherwise help as best you can."
        ));
    } else {
        let note = if truncated {
            "Below are the most relevant excerpts from the user's documents (their library is large). \
             Treat these as your knowledge for this conversation.\n"
        } else {
            "Below are the user's documents in full. Treat their contents as your knowledge for this conversation.\n"
        };
        messages.push(sys_msg(&format!(
            "{note}\n===== DOCUMENTS =====\n{doc_context}\n===== END DOCUMENTS ====="
        )));
    }

    // Prior conversation (already excludes any system/doc messages — see save below).
    messages.extend(history.iter().cloned());

    // New user message.
    let user_turn = user_msg(message);
    messages.push(user_turn.clone());

    // Single model call — no tool loop needed; the documents are already present.
    let url = format!("{}/chat/completions", cfg.api_base.trim_end_matches('/'));
    let body = json!({ "model": cfg.model, "messages": messages });

    // Send, with automatic retry on 429 (rate limit). Free tiers cap tokens per
    // minute; when we hit that, the provider tells us how long to wait, so we
    // wait and resend instead of surfacing an error to the user.
    let mut attempt = 0;
    let resp = loop {
        attempt += 1;

        let mut req = client.post(&url).json(&body);
        if !cfg.api_key.is_empty() {
            req = req.bearer_auth(&cfg.api_key);
        }

        let r = match req.send().await {
            Ok(r) => r,
            Err(e) => {
                if is_local {
                    anyhow::bail!("CONN: I couldn't reach the local AI. Open Settings and switch to Groq (free), or start your local model. (details: {e})");
                }
                anyhow::bail!("CONN: I couldn't reach the AI service. Check your internet connection. (details: {e})");
            }
        };

        // Rate limited: wait the suggested time (capped) and retry, up to 3 times.
        if r.status().as_u16() == 429 && attempt <= 3 {
            let body_text = r.text().await.unwrap_or_default();
            let wait = parse_retry_secs(&body_text).unwrap_or(3.0).min(10.0);
            tracing::warn!("Rate limited (attempt {attempt}); waiting {wait:.1}s then retrying");
            tokio::time::sleep(std::time::Duration::from_secs_f64(wait + 0.3)).await;
            continue;
        }

        break r;
    };

    if !resp.status().is_success() {
        let status = resp.status();
        let text = resp.text().await.unwrap_or_default();
        let lc = text.to_lowercase();

        if status.as_u16() == 401
            || status.as_u16() == 403
            || lc.contains("invalid api key")
            || lc.contains("incorrect api key")
            || lc.contains("unauthorized")
        {
            anyhow::bail!("BAD_KEY: That API key was rejected. Open Settings and paste a valid Groq key (free at console.groq.com/keys).");
        }
        if lc.contains("model")
            && (lc.contains("not found")
                || lc.contains("does not exist")
                || lc.contains("decommissioned"))
        {
            anyhow::bail!("BAD_MODEL: The selected model isn't available. Open Settings, click 'See list', and pick one.");
        }
        if lc.contains("context")
            && (lc.contains("length")
                || lc.contains("maximum")
                || lc.contains("too large")
                || lc.contains("token"))
        {
            anyhow::bail!("TOO_BIG: Those documents are too large to fit all at once. Try removing some, or ask a specific question so I can focus on the relevant parts.");
        }
        anyhow::bail!(
            "The AI service returned an error ({status}). {}",
            if text.len() > 300 {
                &text[..300]
            } else {
                &text
            }
        );
    }

    #[derive(Deserialize)]
    struct CompResp {
        choices: Vec<Choice>,
    }
    #[derive(Deserialize)]
    struct Choice {
        message: AssistantMsg,
    }
    #[derive(Deserialize)]
    struct AssistantMsg {
        #[serde(default)]
        content: Option<String>,
    }

    let parsed: CompResp = resp.json().await.context("reading the AI's reply")?;
    let answer = parsed
        .choices
        .into_iter()
        .next()
        .and_then(|c| c.message.content)
        .unwrap_or_default();

    // Save memory: append this user turn and the assistant reply to the
    // persistent session store. We deliberately do NOT store the
    // system/document messages — those are rebuilt every turn so
    // uploads/removals are always reflected. The store itself caps how many
    // messages it keeps per session.
    sessions
        .append(session_id, &[user_turn, assistant_msg(&answer)])
        .context("saving session history")?;

    // Report which documents informed the answer (all in-context ones).
    let tools_used = if !sources.is_empty() {
        vec![ToolUsed {
            tool: "documents".into(),
            detail: json!({ "in_context": sources.clone(), "truncated": truncated }),
        }]
    } else {
        Vec::new()
    };

    Ok(TurnResult {
        answer,
        tools_used,
        sources,
    })
}

// ── Message constructors ──────────────────────────────────────────────────────

fn sys_msg(text: &str) -> Message {
    Message {
        role: "system".into(),
        content: Some(text.into()),
        tool_calls: None,
        tool_call_id: None,
        name: None,
    }
}
fn user_msg(text: &str) -> Message {
    Message {
        role: "user".into(),
        content: Some(text.into()),
        tool_calls: None,
        tool_call_id: None,
        name: None,
    }
}
fn assistant_msg(text: &str) -> Message {
    Message {
        role: "assistant".into(),
        content: Some(text.into()),
        tool_calls: None,
        tool_call_id: None,
        name: None,
    }
}

/// Extract the suggested wait time (in seconds) from a rate-limit response body,
/// e.g. "...Please try again in 2.225s..." or "...in 850ms...". None if absent.
fn parse_retry_secs(body: &str) -> Option<f64> {
    let lower = body.to_lowercase();
    let idx = lower.find("try again in")?;
    let tail = lower[idx + "try again in".len()..].trim_start();

    // The number.
    let num: String = tail
        .chars()
        .take_while(|c| c.is_ascii_digit() || *c == '.')
        .collect();
    let val: f64 = num.parse().ok()?;

    // The unit immediately after the number.
    let unit = tail[num.len()..].trim_start();
    if unit.starts_with("ms") {
        Some(val / 1000.0)
    } else {
        Some(val) // assume seconds
    }
}
