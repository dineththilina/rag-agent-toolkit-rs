// src/agent.rs
//
// Tool-calling agent loop.
//
// Each turn:
//   1. Append the user message to session history
//   2. POST to /chat/completions with tools attached
//   3. If the model returns tool_calls, dispatch each, append results, loop
//   4. Return the final text response + list of tool calls made
//
// Session memory is stored in a DashMap<session_id, Vec<Message>> so each
// browser session has its own conversation history. Memory lives in-process;
// a server restart clears it (fine for a demo; swap in Redis for production).

use anyhow::{Context, Result};
use dashmap::DashMap;
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::sync::Arc;
use tracing::debug;

use crate::config::Config;
use crate::tools;

// ── Message types ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Message {
    pub role:         String,
    pub content:      Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_calls:   Option<Vec<ToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name:         Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCall {
    pub id:       String,
    #[serde(rename = "type")]
    pub kind:     String,
    pub function: ToolFunction,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolFunction {
    pub name:      String,
    pub arguments: String,  // JSON string from the model
}

// ── Per-turn result ───────────────────────────────────────────────────────────

#[derive(Debug, Serialize)]
pub struct TurnResult {
    pub answer:     String,
    pub tools_used: Vec<ToolUsed>,
    pub sources:    Vec<String>,    // document filenames from RAG hits
}

#[derive(Debug, Serialize)]
pub struct ToolUsed {
    pub tool:   String,
    pub detail: Value,
}

// ── Session store ─────────────────────────────────────────────────────────────

pub type SessionStore = Arc<DashMap<String, Vec<Message>>>;

pub fn new_session_store() -> SessionStore {
    Arc::new(DashMap::new())
}

// ── System prompt ─────────────────────────────────────────────────────────────

const SYSTEM: &str = "You are the Helios Robotics assistant. You have three tools:\
\n- search_knowledge_base: for anything about Helios, the H1 robot, pricing, or support. \
Always use this for company questions.\
\n- calculator: for arithmetic — totals, discounts, conversions.\
\n- country_facts: for general geography questions unrelated to Helios.\
\n\nDecide which tool each question needs. You may chain tools. \
Cite source document names when you use the knowledge base. \
If the knowledge base has no answer, say so — do not invent facts.";

// ── Main chat function ────────────────────────────────────────────────────────

pub async fn chat(
    message:    &str,
    session_id: &str,
    client:     &Client,
    cfg:        &Config,
    sessions:   &SessionStore,
) -> Result<TurnResult> {
    // Retrieve or create session history.
    let mut history: Vec<Message> = sessions
        .get(session_id)
        .map(|r| r.value().clone())
        .unwrap_or_default();

    // System message always first.
    if history.is_empty() {
        history.push(Message {
            role:         "system".into(),
            content:      Some(SYSTEM.into()),
            tool_calls:   None,
            tool_call_id: None,
            name:         None,
        });
    }

    // Append the user turn.
    history.push(Message {
        role:         "user".into(),
        content:      Some(message.into()),
        tool_calls:   None,
        tool_call_id: None,
        name:         None,
    });

    let mut tools_used: Vec<ToolUsed> = Vec::new();
    let mut sources:    Vec<String>   = Vec::new();
    let     tool_defs                 = tools::tool_definitions();

    // Agentic loop: keep calling the model until it gives a plain text reply.
    let answer = loop {
        let body = json!({
            "model":    cfg.model,
            "messages": history,
            "tools":    tool_defs,
        });

        let url = format!("{}/chat/completions", cfg.api_base.trim_end_matches('/'));
        let mut req = client.post(&url).json(&body);
        if !cfg.api_key.is_empty() { req = req.bearer_auth(&cfg.api_key); }

        let resp = req.send().await.context("calling chat completions")?;
        if !resp.status().is_success() {
            let status = resp.status();
            let text   = resp.text().await.unwrap_or_default();
            anyhow::bail!("chat completions {status}: {text}");
        }

        #[derive(Deserialize)]
        struct CompResp { choices: Vec<Choice> }
        #[derive(Deserialize)]
        struct Choice   { message: AssistantMsg }
        #[derive(Deserialize)]
        struct AssistantMsg {
            #[serde(default)]
            content:               Option<String>,
            #[serde(default)]
            tool_calls:            Vec<ToolCall>,
        }

        let parsed: CompResp = resp.json().await.context("parsing completions response")?;
        let msg = parsed.choices.into_iter().next()
            .context("empty choices")?
            .message;

        if msg.tool_calls.is_empty() {
            // Plain text answer — we're done.
            // Append the assistant message to history before saving.
            history.push(Message {
                role:         "assistant".into(),
                content:      msg.content.clone(),
                tool_calls:   None,
                tool_call_id: None,
                name:         None,
            });
            break msg.content.unwrap_or_default();
        }

        // Model wants to call tools. Append the assistant turn (with tool_calls).
        history.push(Message {
            role:         "assistant".into(),
            content:      msg.content.clone(),
            tool_calls:   Some(msg.tool_calls.clone()),
            tool_call_id: None,
            name:         None,
        });

        // Execute each tool call and append results.
        for tc in &msg.tool_calls {
            let args: Value = serde_json::from_str(&tc.function.arguments)
                .unwrap_or(json!({}));

            debug!("tool call: {} args={}", tc.function.name, args);

            let result = tools::dispatch(&tc.function.name, &args, client, cfg).await;

            // Track which tools ran and extract RAG sources.
            if tc.function.name == "search_knowledge_base" {
                let query = args["query"].as_str().unwrap_or("").to_string();
                // Parse source filenames out of the result text.
                let srcs: Vec<String> = result
                    .lines()
                    .filter(|l| l.starts_with("[source:"))
                    .filter_map(|l| {
                        let after = l.strip_prefix("[source:")?;
                        Some(after.split('|').next()?.trim().to_string())
                    })
                    .collect();
                sources.extend(srcs.clone());
                tools_used.push(ToolUsed {
                    tool:   tc.function.name.clone(),
                    detail: json!({ "query": query, "sources": srcs }),
                });
            } else {
                tools_used.push(ToolUsed {
                    tool:   tc.function.name.clone(),
                    detail: args.clone(),
                });
            }

            history.push(Message {
                role:         "tool".into(),
                content:      Some(result),
                tool_calls:   None,
                tool_call_id: Some(tc.id.clone()),
                name:         Some(tc.function.name.clone()),
            });
        }
        // Loop back to ask the model what to do next.
    };

    // Deduplicate sources.
    sources.sort(); sources.dedup();

    // Save updated history.
    sessions.insert(session_id.to_string(), history);

    Ok(TurnResult { answer, tools_used, sources })
}
