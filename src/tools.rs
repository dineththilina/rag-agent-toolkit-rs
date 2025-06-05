// src/tools.rs
//
// The three tools the agent can call. Each tool is a plain async function.
// Tool selection is driven by the model's JSON tool_calls response; we match
// on the function name and dispatch here.
//
//  1. search_knowledge_base — RAG retrieval over the local document corpus
//  2. calculator            — safe arithmetic via a recursive descent parser
//  3. country_facts         — live HTTP call to the free REST Countries API

use anyhow::Result;
use reqwest::Client;
use serde_json::{json, Value};
use tracing::debug;

use crate::config::Config;
use crate::rag;

// ── Tool schemas (sent to the model in every chat request) ───────────────────

pub fn tool_definitions() -> Value {
    json!([
        {
            "type": "function",
            "function": {
                "name": "search_knowledge_base",
                "description": "Search the Helios Robotics internal knowledge base for facts about the company, the H1 robot specifications, Helios Fleet pricing and plans, and support FAQs. Use this for ANY question about Helios, its products, pricing, specs, or support.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "The search query"
                        }
                    },
                    "required": ["query"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "calculator",
                "description": "Evaluate a basic arithmetic expression. Supports + - * / ** % and parentheses. Use this for totals, discounts, or unit conversions.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "A plain math expression like '90 * 12 * 0.85'"
                        }
                    },
                    "required": ["expression"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "country_facts",
                "description": "Look up live facts about a country (capital, population, region, currency) using the public REST Countries API. Use this for general geography questions NOT about Helios.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "country": {
                            "type": "string",
                            "description": "Country name, e.g. 'Portugal'"
                        }
                    },
                    "required": ["country"]
                }
            }
        }
    ])
}

// ── Dispatch ─────────────────────────────────────────────────────────────────

/// Call the named tool with the given JSON arguments.
/// Returns a string to be placed in the tool_call result message.
pub async fn dispatch(
    name:   &str,
    args:   &Value,
    client: &Client,
    cfg:    &Config,
) -> String {
    debug!("tool dispatch: {name} args={args}");
    match name {
        "search_knowledge_base" => {
            let query = args["query"].as_str().unwrap_or("").to_string();
            run_rag(&query, client, cfg).await
        }
        "calculator" => {
            let expr = args["expression"].as_str().unwrap_or("").to_string();
            run_calculator(&expr)
        }
        "country_facts" => {
            let country = args["country"].as_str().unwrap_or("").to_string();
            run_country_facts(&country, client).await
        }
        other => format!("Unknown tool '{other}'."),
    }
}

// ── Tool 1: RAG ──────────────────────────────────────────────────────────────

async fn run_rag(query: &str, client: &Client, cfg: &Config) -> String {
    match rag::retrieve(client, cfg, query, 4).await {
        Err(e) => format!("Knowledge base search failed: {e}"),
        Ok(hits) if hits.is_empty() => "No relevant documents found.".into(),
        Ok(hits) => hits
            .iter()
            .map(|h| format!("[source: {} | score: {:.3}]\n{}", h.source, h.score, h.text))
            .collect::<Vec<_>>()
            .join("\n\n---\n\n"),
    }
}

// ── Tool 2: Calculator ───────────────────────────────────────────────────────
//
// Implemented as a recursive descent parser so we never call eval() on
// arbitrary input. Supports: numbers (int and float), +, -, *, /, **, %, ()

fn run_calculator(expr: &str) -> String {
    match calc_parse(expr.trim()) {
        Ok(v)  => format!("{v}"),
        Err(e) => format!("Could not evaluate '{expr}': {e}"),
    }
}

struct CalcParser<'a> {
    input: &'a [u8],
    pos:   usize,
}

impl<'a> CalcParser<'a> {
    fn new(s: &'a str) -> Self { Self { input: s.as_bytes(), pos: 0 } }
    fn peek(&self) -> Option<u8> { self.input.get(self.pos).copied() }
    fn consume(&mut self)        { self.pos += 1; }
    fn skip_ws(&mut self)        { while self.peek().map(|c| c == b' ' || c == b'\t').unwrap_or(false) { self.consume(); } }

    fn parse_expr(&mut self) -> Result<f64> { self.parse_add() }

    fn parse_add(&mut self) -> Result<f64> {
        let mut v = self.parse_mul()?;
        loop {
            self.skip_ws();
            match self.peek() {
                Some(b'+') => { self.consume(); v += self.parse_mul()?; }
                Some(b'-') => { self.consume(); v -= self.parse_mul()?; }
                _          => break,
            }
        }
        Ok(v)
    }

    fn parse_mul(&mut self) -> Result<f64> {
        let mut v = self.parse_pow()?;
        loop {
            self.skip_ws();
            if self.input.get(self.pos..self.pos+2) == Some(b"**") {
                break;   // handled in parse_pow
            }
            match self.peek() {
                Some(b'*') => { self.consume(); v *= self.parse_pow()?; }
                Some(b'/') => {
                    self.consume();
                    let d = self.parse_pow()?;
                    if d == 0.0 { anyhow::bail!("division by zero"); }
                    v /= d;
                }
                Some(b'%') => { self.consume(); v %= self.parse_pow()?; }
                _          => break,
            }
        }
        Ok(v)
    }

    fn parse_pow(&mut self) -> Result<f64> {
        let base = self.parse_unary()?;
        self.skip_ws();
        if self.input.get(self.pos..self.pos+2) == Some(b"**") {
            self.pos += 2;
            let exp = self.parse_pow()?;   // right-associative
            return Ok(base.powf(exp));
        }
        Ok(base)
    }

    fn parse_unary(&mut self) -> Result<f64> {
        self.skip_ws();
        if self.peek() == Some(b'-') { self.consume(); return Ok(-self.parse_atom()?); }
        if self.peek() == Some(b'+') { self.consume(); }
        self.parse_atom()
    }

    fn parse_atom(&mut self) -> Result<f64> {
        self.skip_ws();
        if self.peek() == Some(b'(') {
            self.consume();
            let v = self.parse_expr()?;
            self.skip_ws();
            if self.peek() != Some(b')') { anyhow::bail!("missing closing )"); }
            self.consume();
            return Ok(v);
        }
        self.parse_number()
    }

    fn parse_number(&mut self) -> Result<f64> {
        let start = self.pos;
        while self.peek().map(|c| c.is_ascii_digit() || c == b'.').unwrap_or(false) {
            self.consume();
        }
        if self.pos == start { anyhow::bail!("expected number at position {}", self.pos); }
        let s = std::str::from_utf8(&self.input[start..self.pos])?;
        Ok(s.parse::<f64>()?)
    }
}

fn calc_parse(expr: &str) -> Result<f64> {
    let mut p = CalcParser::new(expr);
    let v = p.parse_expr()?;
    p.skip_ws();
    if p.pos != p.input.len() {
        anyhow::bail!("unexpected character '{}' at position {}", p.peek().unwrap_or(b'?') as char, p.pos);
    }
    Ok(v)
}

// ── Tool 3: Country facts ────────────────────────────────────────────────────

async fn run_country_facts(country: &str, client: &Client) -> String {
    let encoded = urlencoding::encode(country);
    let url     = format!(
        "https://restcountries.com/v3.1/name/{}?fields=name,capital,population,region,currencies",
        encoded
    );

    let resp = match client.get(&url).send().await {
        Err(e) => return format!("Could not reach country API: {e}. Check your internet connection."),
        Ok(r)  => r,
    };

    if !resp.status().is_success() {
        return format!("Country '{country}' not found.");
    }

    let data: Value = match resp.json().await {
        Err(e) => return format!("Country API parse error: {e}"),
        Ok(v)  => v,
    };

    let c = match data.as_array().and_then(|a| a.first()) {
        None    => return format!("No data for '{country}'."),
        Some(c) => c.clone(),
    };

    let name      = c["name"]["common"].as_str().unwrap_or(country);
    let capital   = c["capital"].as_array()
                      .and_then(|a| a.first())
                      .and_then(|v| v.as_str())
                      .unwrap_or("unknown");
    let pop       = c["population"].as_u64()
                      .map(|p| format!("{p:,}"))
                      .unwrap_or_else(|| "unknown".into());
    let region    = c["region"].as_str().unwrap_or("unknown");
    let currency  = c["currencies"].as_object()
                      .and_then(|m| m.values().next())
                      .and_then(|v| v["name"].as_str())
                      .unwrap_or("unknown");

    format!("{name}: capital {capital}, population ~{pop}, region {region}, currency {currency}.")
}

// We need this crate for URL encoding.
mod urlencoding {
    pub fn encode(s: &str) -> String {
        s.chars().map(|c| match c {
            'A'..='Z' | 'a'..='z' | '0'..='9' | '-' | '_' | '.' | '~' => c.to_string(),
            ' ' => "%20".into(),
            c   => format!("%{:02X}", c as u32),
        }).collect()
    }
}
