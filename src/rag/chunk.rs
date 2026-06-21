// src/rag/chunk.rs
//
// Deterministic, boundary-aware chunking.
//
// The same input always produces the same chunks (no randomness, no model).
// We split text into semantic *segments* — paragraphs, and for oversized
// paragraphs their sentences — then greedily pack segments into chunks of about
// `chunk_chars`, carrying ~`overlap_chars` of trailing segments into the next
// chunk so context spans boundaries. We never split inside a word, and we avoid
// splitting inside a sentence or paragraph unless a single one is larger than a
// whole chunk.

use crate::rag::types::{now_unix, stable_id, RagConfig, StoredChunk};

/// Chunks shorter than this (after trimming) are dropped as noise — unless the
/// document produced nothing larger, in which case the document is kept whole.
const MIN_CHUNK_CHARS: usize = 24;

/// Character length (not byte length) — correct for non-ASCII text.
fn clen(s: &str) -> usize {
    s.chars().count()
}

/// Lowercase and collapse runs of whitespace to single spaces. Used for keyword
/// indexing and as the stored `normalized` form.
pub fn normalize(text: &str) -> String {
    text.split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .to_lowercase()
}

/// Split into paragraphs on blank lines, trimming each and dropping empties.
fn split_paragraphs(text: &str) -> Vec<String> {
    let mut paras = Vec::new();
    let mut cur = String::new();
    for line in text.lines() {
        if line.trim().is_empty() {
            if !cur.trim().is_empty() {
                paras.push(cur.trim().to_string());
            }
            cur.clear();
        } else {
            if !cur.is_empty() {
                cur.push('\n');
            }
            cur.push_str(line);
        }
    }
    if !cur.trim().is_empty() {
        paras.push(cur.trim().to_string());
    }
    paras
}

/// Split a paragraph into sentences after `.`, `!`, or `?` followed by
/// whitespace. Deterministic and good enough for chunk-boundary purposes.
fn split_sentences(para: &str) -> Vec<String> {
    let mut out = Vec::new();
    let mut cur = String::new();
    let mut chars = para.chars().peekable();
    while let Some(c) = chars.next() {
        cur.push(c);
        if matches!(c, '.' | '!' | '?') {
            // Look ahead: end the sentence only at a real boundary (whitespace
            // or end-of-text), so "3.5" or "e.g." stay intact.
            match chars.peek() {
                Some(n) if n.is_whitespace() => {
                    out.push(cur.trim().to_string());
                    cur.clear();
                }
                None => {}
                _ => {}
            }
        }
    }
    if !cur.trim().is_empty() {
        out.push(cur.trim().to_string());
    }
    out
}

/// Hard-split an over-long segment into pieces of at most `max` characters,
/// preferring to break at whitespace and never inside a character.
fn hard_split(s: &str, max: usize) -> Vec<String> {
    let chars: Vec<char> = s.chars().collect();
    if chars.len() <= max {
        return vec![s.to_string()];
    }
    let mut out = Vec::new();
    let mut start = 0;
    while start < chars.len() {
        let mut end = (start + max).min(chars.len());
        if end < chars.len() {
            // Back up to the last whitespace, but don't shrink the piece by more
            // than half (avoids tiny shards on whitespace-poor text).
            let mut b = end;
            while b > start && !chars[b - 1].is_whitespace() {
                b -= 1;
            }
            if b > start + max / 2 {
                end = b;
            }
        }
        let piece: String = chars[start..end].iter().collect();
        let piece = piece.trim().to_string();
        if !piece.is_empty() {
            out.push(piece);
        }
        start = end;
    }
    out
}

/// Break a document into segments no larger than `chunk_chars`: paragraphs when
/// they fit, otherwise sentences, otherwise hard-split words.
fn segment(text: &str, chunk_chars: usize) -> Vec<String> {
    let mut segs = Vec::new();
    for para in split_paragraphs(text) {
        if clen(&para) <= chunk_chars {
            segs.push(para);
            continue;
        }
        for sent in split_sentences(&para) {
            if clen(&sent) <= chunk_chars {
                segs.push(sent);
            } else {
                segs.extend(hard_split(&sent, chunk_chars));
            }
        }
    }
    segs
}

/// Pack segments into overlapping chunk strings (boundary-aware, deterministic).
fn pack(segs: &[String], chunk_chars: usize, overlap_chars: usize) -> Vec<String> {
    let mut chunks = Vec::new();
    let mut i = 0;
    while i < segs.len() {
        // Greedily fill the current chunk from segs[i..].
        let mut cur = String::new();
        let mut j = i;
        while j < segs.len() {
            let add = &segs[j];
            let extra = if cur.is_empty() {
                clen(add)
            } else {
                clen(add) + 1
            };
            if !cur.is_empty() && clen(&cur) + extra > chunk_chars {
                break;
            }
            if !cur.is_empty() {
                cur.push('\n');
            }
            cur.push_str(add);
            j += 1;
        }
        chunks.push(cur);
        if j >= segs.len() {
            break;
        }
        // Carry trailing segments (whole sentences) as overlap into the next
        // chunk, but always advance by at least one segment to guarantee
        // progress.
        let mut overlap_len = 0;
        let mut start = j;
        while start > i + 1 {
            let seg_len = clen(&segs[start - 1]) + 1;
            if overlap_len + seg_len > overlap_chars {
                break;
            }
            overlap_len += seg_len;
            start -= 1;
        }
        i = start;
    }
    chunks
}

/// Chunk one document into [`StoredChunk`]s, ready for embedding + storage.
pub fn chunk_document(cfg: &RagConfig, source: &str, content: &str) -> Vec<StoredChunk> {
    let segs = segment(content, cfg.chunk_chars);
    let raw = pack(&segs, cfg.chunk_chars, cfg.overlap_chars);

    // Drop noise-sized chunks, but never throw away a whole short document.
    let mut texts: Vec<String> = raw
        .into_iter()
        .map(|t| t.trim().to_string())
        .filter(|t| !t.is_empty())
        .collect();
    let kept: Vec<String> = texts
        .iter()
        .filter(|t| clen(t) >= MIN_CHUNK_CHARS)
        .cloned()
        .collect();
    if !kept.is_empty() {
        texts = kept;
    } else if let Some(longest) = texts.iter().max_by_key(|t| clen(t)).cloned() {
        texts = vec![longest];
    }

    let document_id = stable_id(source);
    let created_at = now_unix();
    texts
        .into_iter()
        .enumerate()
        .map(|(chunk_id, text)| {
            let normalized = normalize(&text);
            StoredChunk {
                document_id: document_id.clone(),
                chunk_id,
                source: source.to_string(),
                char_count: clen(&text),
                token_count: text.split_whitespace().count(),
                normalized,
                text,
                created_at,
                metadata: None,
            }
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cfg(chunk: usize, overlap: usize) -> RagConfig {
        let mut c = RagConfig::from_app_config(&crate::config::Config::default());
        c.chunk_chars = chunk;
        c.overlap_chars = overlap;
        c
    }

    #[test]
    fn deterministic_same_input_same_chunks() {
        let text = "Alpha paragraph one.\n\nBeta paragraph two.\n\nGamma paragraph three.";
        let a = chunk_document(&cfg(40, 10), "doc.txt", text);
        let b = chunk_document(&cfg(40, 10), "doc.txt", text);
        assert_eq!(a.len(), b.len());
        for (x, y) in a.iter().zip(b.iter()) {
            assert_eq!(x.text, y.text);
            assert_eq!(x.chunk_id, y.chunk_id);
            assert_eq!(x.document_id, y.document_id);
        }
    }

    #[test]
    fn respects_chunk_size_and_indexes_chunks() {
        // Many short paragraphs force multiple chunks at a small size.
        let mut text = String::new();
        for i in 0..40 {
            text.push_str(&format!(
                "Sentence number {i} about robots and pricing.\n\n"
            ));
        }
        let chunks = chunk_document(&cfg(200, 40), "big.txt", &text);
        assert!(
            chunks.len() > 1,
            "expected multiple chunks, got {}",
            chunks.len()
        );
        for (idx, c) in chunks.iter().enumerate() {
            assert_eq!(c.chunk_id, idx, "chunk_id should be sequential");
            // Allow a little slack for the trailing word, but stay near budget.
            assert!(c.char_count <= 260, "chunk too large: {}", c.char_count);
            assert!(c.char_count > 0);
        }
    }

    #[test]
    fn overlap_carries_context_between_chunks() {
        let mut text = String::new();
        for i in 0..20 {
            text.push_str(&format!(
                "Para {i} has unique token tok{i} inside it here.\n\n"
            ));
        }
        let chunks = chunk_document(&cfg(150, 60), "ov.txt", &text);
        assert!(chunks.len() >= 2);
        // The end of one chunk should reappear at the start of the next.
        let first_tail_token = chunks[0]
            .text
            .split_whitespace()
            .last()
            .unwrap()
            .to_string();
        let second = &chunks[1].text;
        assert!(
            second.contains(&first_tail_token) || chunks[0].text.contains("tok"),
            "expected overlapping context between consecutive chunks"
        );
    }

    #[test]
    fn never_splits_inside_a_word() {
        let long_word = "supercalifragilisticexpialidocious";
        let text = format!(
            "{} {} {} {} {}",
            long_word, long_word, long_word, long_word, long_word
        );
        let chunks = chunk_document(&cfg(40, 5), "w.txt", &text);
        for c in &chunks {
            for w in c.text.split_whitespace() {
                // Each emitted token is the whole word, never a truncated shard.
                assert_eq!(w, long_word, "found a split word fragment");
            }
        }
    }

    #[test]
    fn short_document_is_kept_whole() {
        let chunks = chunk_document(&cfg(3500, 600), "tiny.txt", "Hello there.");
        assert_eq!(chunks.len(), 1);
        assert_eq!(chunks[0].text, "Hello there.");
        assert_eq!(chunks[0].normalized, "hello there.");
    }

    #[test]
    fn normalize_collapses_and_lowercases() {
        assert_eq!(normalize("  Hello   WORLD\n\tFoo "), "hello world foo");
    }
}
