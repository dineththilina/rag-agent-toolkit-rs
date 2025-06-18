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

const SYSTEM: &str = "You are a helpful research assistant with access to the user's uploaded documents through tools. \
Follow these rules strictly:\
\n\n1. For ANY question that might be answered by the user's documents, you MUST call the search_knowledge_base tool. \
Do this immediately and silently — never say 'I will search' or 'I need to use my tool'. Just call it.\
\n2. NEVER invent, assume, or guess document names, file names, or document contents. You do not know what documents exist until you search. \
The search tool searches everything the user has uploaded; you do not need to know file names to search.\
\n3. For arithmetic (totals, discounts, conversions), call the calculator tool.\
\n4. For general geography facts about a country, call the country_facts tool.\
\n5. After a tool returns results, answer using ONLY what the results contain. If the search returns nothing relevant, say plainly that the documents don't cover it. Do not fill gaps with made-up information.\
\n6. When you used the knowledge base, mention which source documents the answer came from (use the exact source names returned by the tool).\
\n\nThe user may ask follow-up questions like 'summarize' or 'tell me more' — interpret these in the context of the conversation and the documents, and search again if needed.";

// ── Main chat function ────────────────────────────────────────────────────────

pub async fn chat(
    message:    &str,
    session_id: &str,
    client:     &Client,
    cfg:        &Config,
    sessions:   &SessionStore,
    store:      &crate::rag::SharedStore,
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

    // Friendly guard: a hosted provider (anything not on localhost) needs a key.
    // Catch this before making a doomed request so the user gets clear guidance.
    let is_local = cfg.api_base.contains("localhost") || cfg.api_base.contains("127.0.0.1");
    if cfg.api_key.is_empty() && !is_local {
        anyhow::bail!(
            "NO_KEY: I need a free key to start. Open Settings (the gear icon) and paste your Groq key — \
             you can get one free in under a minute at console.groq.com/keys."
        );
    }

    // Some open models (especially via Ollama) don't support tool-calling.
    // We start with tools enabled and, if the API rejects the tools parameter,
    // fall back to a plain-chat request so the model can still answer.
    let mut tools_enabled = true;
    // Guard against infinite tool loops.
    let mut iterations = 0;
    // Track whether we've already nudged a model that narrated a tool call
    // instead of emitting one (a known quirk of some hosted models).
    let mut nudged = false;

    // Agentic loop: keep calling the model until it gives a plain text reply.
    let answer = loop {
        iterations += 1;
        if iterations > 6 {
            // Safety valve: don't loop forever. Return whatever we have.
            break "I wasn't able to complete that with the tools available. Please try rephrasing.".to_string();
        }

        let body = if tools_enabled {
            json!({
                "model": cfg.model,
                "messages": history,
                "tools": tool_defs,
                "tool_choice": "auto"
            })
        } else {
            json!({ "model": cfg.model, "messages": history })
        };

        let url = format!("{}/chat/completions", cfg.api_base.trim_end_matches('/'));
        let mut req = client.post(&url).json(&body);
        if !cfg.api_key.is_empty() { req = req.bearer_auth(&cfg.api_key); }

        // A failed *network* call (provider unreachable) lands here. Give a
        // plain-English message instead of a raw reqwest error.
        let resp = match req.send().await {
            Ok(r)  => r,
            Err(e) => {
                if is_local {
                    anyhow::bail!(
                        "CONN: I couldn't reach the local AI. Open Settings and switch to Groq \
                         (free), or start your local model. (details: {e})"
                    );
                }
                anyhow::bail!("CONN: I couldn't reach the AI service. Check your internet connection. (details: {e})");
            }
        };

        if !resp.status().is_success() {
            let status = resp.status();
            let text   = resp.text().await.unwrap_or_default();
            let lc = text.to_lowercase();

            // Tools unsupported → retry without them.
            if tools_enabled && (
                lc.contains("tool") || lc.contains("function") ||
                lc.contains("does not support") || lc.contains("not supported")
            ) {
                tracing::warn!("Model rejected tools ({status}); retrying without tool-calling");
                tools_enabled = false;
                continue;
            }

            // Bad/expired key.
            if status.as_u16() == 401 || status.as_u16() == 403
                || lc.contains("invalid api key") || lc.contains("incorrect api key")
                || lc.contains("unauthorized") {
                anyhow::bail!(
                    "BAD_KEY: That API key was rejected. Open Settings and paste a valid Groq key \
                     (free at console.groq.com/keys)."
                );
            }

            // Unknown model name.
            if lc.contains("model") && (lc.contains("not found") || lc.contains("does not exist")
                || lc.contains("decommissioned")) {
                anyhow::bail!(
                    "BAD_MODEL: The selected model isn't available. Open Settings, click \
                     'See available', and pick one from the list."
                );
            }

            anyhow::bail!("The AI service returned an error ({status}). {}",
                if text.len() > 300 { &text[..300] } else { &text });
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

        let parsed: CompResp = resp.json().await.context("reading the AI's reply")?;
        let msg = parsed.choices.into_iter().next()
            .context("empty choices")?
            .message;

        if msg.tool_calls.is_empty() {
            let text = msg.content.clone().unwrap_or_default();

            // Safety net for the known failure where a model *says* it will use
            // a tool but doesn't emit a tool call. If the reply reads like a
            // promise to search and we haven't nudged yet, push it once to
            // actually call the tool instead of accepting the narration.
            let lc = text.to_lowercase();
            let looks_like_narration = tools_enabled && !nudged && (
                (lc.contains("search") || lc.contains("look up") || lc.contains("knowledge base")
                 || lc.contains("my tool") || lc.contains("use the") || lc.contains("i'll use")
                 || lc.contains("i will use") || lc.contains("i need to"))
                && (lc.contains("tool") || lc.contains("search") || lc.contains("find"))
            );

            if looks_like_narration {
                nudged = true;
                // Add the model's narration to history, then a firm instruction.
                history.push(Message {
                    role: "assistant".into(),
                    content: msg.content.clone(),
                    tool_calls: None, tool_call_id: None, name: None,
                });
                history.push(Message {
                    role: "user".into(),
                    content: Some("Call the search_knowledge_base tool now with an appropriate query. Do not describe what you will do — just make the tool call.".into()),
                    tool_calls: None, tool_call_id: None, name: None,
                });
                continue;
            }

            // Genuine plain-text answer — we're done.
            history.push(Message {
                role:         "assistant".into(),
                content:      msg.content.clone(),
                tool_calls:   None,
                tool_call_id: None,
                name:         None,
            });
            break text;
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

            let result = tools::dispatch(&tc.function.name, &args, client, cfg, store).await;

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
