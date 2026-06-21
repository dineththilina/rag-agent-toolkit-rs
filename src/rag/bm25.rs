// src/rag/bm25.rs
//
// BM25 keyword retrieval — the ranking function classic search engines use.
// Pure Rust, no model, no network. This is the original retrieval engine,
// preserved as the keyword half of hybrid retrieval (and as the fallback when
// the embedding model is unavailable).
//
// The index is held in memory and rebuilt from the SQLite store on startup, so
// SQLite remains the single source of truth for chunk data.

use std::collections::HashMap;

use crate::rag::types::{Hit, HitSource, StoredChunk};

// BM25 tuning constants (standard defaults).
const BM25_K1: f32 = 1.5;
const BM25_B: f32 = 0.75;

// Common English words that add noise to keyword matching.
const STOPWORDS: &[&str] = &[
    "a", "an", "the", "of", "to", "in", "on", "at", "for", "and", "or", "but", "is", "are", "was",
    "were", "be", "been", "being", "this", "that", "these", "those", "it", "its", "as", "by",
    "with", "from", "into", "over", "under", "do", "does", "did", "have", "has", "had", "will",
    "would", "can", "could", "should", "i", "you", "he", "she", "we", "they", "my", "your", "his",
    "her", "our", "their", "what", "which", "who", "when", "where", "why", "how", "not", "no",
    "yes", "if", "then", "else", "than", "there", "here", "about", "me", "him", "them", "us", "so",
    "up", "out", "off", "all", "any", "some", "more", "most",
];

fn is_stopword(w: &str) -> bool {
    STOPWORDS.contains(&w)
}

/// Tokenize: lowercase, split on non-alphanumeric, drop stopwords and 1-char tokens.
pub fn tokenize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| w.len() > 1 && !is_stopword(w))
        .map(|w| w.to_string())
        .collect()
}

/// One indexed chunk with its precomputed term statistics.
#[derive(Clone)]
struct Doc {
    document_id: String,
    chunk_id: usize,
    source: String,
    text: String,
    term_freq: HashMap<String, u32>,
    length: u32,
}

/// In-memory BM25 index over all chunks.
#[derive(Default)]
pub struct Bm25Index {
    docs: Vec<Doc>,
}

impl Bm25Index {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn clear(&mut self) {
        self.docs.clear();
    }

    pub fn len(&self) -> usize {
        self.docs.len()
    }

    pub fn is_empty(&self) -> bool {
        self.docs.is_empty()
    }

    /// Add one chunk to the index.
    pub fn add(&mut self, chunk: &StoredChunk) {
        let tokens = tokenize(&chunk.text);
        let length = tokens.len() as u32;
        let mut term_freq: HashMap<String, u32> = HashMap::new();
        for t in tokens {
            *term_freq.entry(t).or_insert(0) += 1;
        }
        self.docs.push(Doc {
            document_id: chunk.document_id.clone(),
            chunk_id: chunk.chunk_id,
            source: chunk.source.clone(),
            text: chunk.text.clone(),
            term_freq,
            length,
        });
    }

    /// Remove every chunk belonging to a source filename. Returns count removed.
    pub fn remove_source(&mut self, source: &str) -> usize {
        let before = self.docs.len();
        self.docs.retain(|d| d.source != source);
        before - self.docs.len()
    }

    /// All chunk (source, text) pairs in insertion order — used by the prompt
    /// context builder.
    pub fn all_chunks(&self) -> Vec<(String, String)> {
        self.docs
            .iter()
            .map(|d| (d.source.clone(), d.text.clone()))
            .collect()
    }

    /// Score every chunk against the query and return the top `k` as hits,
    /// with scores normalised to 0..1.
    pub fn search(&self, query: &str, k: usize) -> Vec<Hit> {
        let q_tokens: Vec<String> = {
            let mut t = tokenize(query);
            t.sort();
            t.dedup();
            t
        };
        if q_tokens.is_empty() || self.docs.is_empty() || k == 0 {
            return Vec::new();
        }

        let n = self.docs.len() as f32;
        let avgdl = self.docs.iter().map(|c| c.length as f32).sum::<f32>() / n;
        let avgdl = if avgdl <= 0.0 { 1.0 } else { avgdl };

        // Document frequency: how many chunks contain each query term.
        let mut df: HashMap<&str, u32> = HashMap::new();
        for qt in &q_tokens {
            let count = self
                .docs
                .iter()
                .filter(|c| c.term_freq.contains_key(qt))
                .count() as u32;
            df.insert(qt.as_str(), count);
        }

        let mut scored: Vec<Hit> = self
            .docs
            .iter()
            .map(|c| {
                let mut score = 0.0f32;
                for qt in &q_tokens {
                    let dfq = *df.get(qt.as_str()).unwrap_or(&0);
                    if dfq == 0 {
                        continue;
                    }
                    let tf = *c.term_freq.get(qt).unwrap_or(&0) as f32;
                    if tf == 0.0 {
                        continue;
                    }
                    let idf = ((n - dfq as f32 + 0.5) / (dfq as f32 + 0.5) + 1.0).ln();
                    let denom = tf + BM25_K1 * (1.0 - BM25_B + BM25_B * c.length as f32 / avgdl);
                    score += idf * (tf * (BM25_K1 + 1.0)) / denom;
                }
                Hit {
                    text: c.text.clone(),
                    source: c.source.clone(),
                    score,
                    document_id: c.document_id.clone(),
                    chunk_id: c.chunk_id,
                    retrieval: HitSource::Keyword,
                }
            })
            .collect();

        scored.retain(|h| h.score > 0.0);
        scored.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        scored.truncate(k);

        if let Some(max) = scored.first().map(|h| h.score) {
            if max > 0.0 {
                for h in scored.iter_mut() {
                    h.score /= max;
                }
            }
        }
        scored
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rag::types::{now_unix, stable_id};

    fn chunk(source: &str, id: usize, text: &str) -> StoredChunk {
        StoredChunk {
            document_id: stable_id(source),
            chunk_id: id,
            source: source.to_string(),
            text: text.to_string(),
            normalized: text.to_lowercase(),
            char_count: text.chars().count(),
            token_count: text.split_whitespace().count(),
            created_at: now_unix(),
            metadata: None,
        }
    }

    #[test]
    fn ranks_keyword_matches_first() {
        let mut idx = Bm25Index::new();
        idx.add(&chunk(
            "a.txt",
            0,
            "The Helios H1 battery lasts eight hours per charge.",
        ));
        idx.add(&chunk(
            "b.txt",
            0,
            "Pricing for the Growth plan includes a discount.",
        ));
        idx.add(&chunk(
            "c.txt",
            0,
            "Support hours are weekdays nine to five.",
        ));

        let hits = idx.search("how long does the battery last", 3);
        assert!(!hits.is_empty());
        assert_eq!(hits[0].source, "a.txt");
        assert!(
            (hits[0].score - 1.0).abs() < 1e-6,
            "top score should normalise to 1.0"
        );
    }

    #[test]
    fn remove_source_drops_chunks() {
        let mut idx = Bm25Index::new();
        idx.add(&chunk("a.txt", 0, "alpha battery"));
        idx.add(&chunk("a.txt", 1, "beta battery"));
        idx.add(&chunk("b.txt", 0, "gamma pricing"));
        assert_eq!(idx.len(), 3);
        let removed = idx.remove_source("a.txt");
        assert_eq!(removed, 2);
        assert_eq!(idx.len(), 1);
        let remaining = idx.all_chunks();
        assert_eq!(remaining.len(), 1);
        assert_eq!(remaining[0].0, "b.txt");
    }

    #[test]
    fn empty_query_returns_nothing() {
        let mut idx = Bm25Index::new();
        idx.add(&chunk("a.txt", 0, "alpha battery"));
        assert!(idx.search("", 5).is_empty());
        assert!(idx.search("the and of", 5).is_empty()); // all stopwords
    }
}
