// src/rag/hybrid.rs
//
// Merge keyword (BM25) and vector (semantic) result lists into one ranking with
// Reciprocal Rank Fusion (RRF), then deduplicate by (document_id, chunk_id).
//
// RRF is rank-based, so it combines two scores that live on different scales
// (BM25 magnitudes vs cosine similarity) without any tuning: a chunk's fused
// score is the sum over lists of 1 / (k + rank). Chunks that rank well in both
// lists rise to the top; a chunk found by only one layer still contributes.

use std::collections::HashMap;

use crate::rag::types::{Hit, HitSource};

/// Standard RRF damping constant.
pub const RRF_K: f32 = 60.0;

/// Fuse several ranked lists into one, deduplicating by (document_id, chunk_id).
/// The returned hits carry a fused score normalised to 0..1 and `retrieval =
/// Hybrid` when a chunk was found by more than one layer.
pub fn reciprocal_rank_fusion(lists: &[Vec<Hit>], top_k: usize) -> Vec<Hit> {
    let mut acc: HashMap<(String, usize), Fused> = HashMap::new();

    for list in lists {
        for (rank, hit) in list.iter().enumerate() {
            let key = (hit.document_id.clone(), hit.chunk_id);
            let contribution = 1.0 / (RRF_K + (rank as f32 + 1.0));
            let entry = acc.entry(key).or_insert_with(|| Fused {
                hit: hit.clone(),
                score: 0.0,
                layers: 0,
            });
            entry.score += contribution;
            entry.layers += 1;
            // Keep the display text/source from the higher individually-scored
            // instance (they refer to the same chunk, so this is cosmetic).
            if hit.score > entry.hit.score {
                entry.hit = hit.clone();
            }
        }
    }

    let mut fused: Vec<Fused> = acc.into_values().collect();
    // Sort by fused score (desc), breaking ties deterministically.
    fused.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.hit.source.cmp(&b.hit.source))
            .then_with(|| a.hit.chunk_id.cmp(&b.hit.chunk_id))
    });

    let max = fused.first().map(|f| f.score).unwrap_or(0.0);
    fused
        .into_iter()
        .take(top_k)
        .map(|mut f| {
            f.hit.score = if max > 0.0 { f.score / max } else { 0.0 };
            f.hit.retrieval = if f.layers > 1 {
                HitSource::Hybrid
            } else {
                f.hit.retrieval
            };
            f.hit
        })
        .collect()
}

struct Fused {
    hit: Hit,
    score: f32,
    layers: u32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rag::types::{stable_id, HitSource};

    fn hit(source: &str, chunk_id: usize, score: f32, retrieval: HitSource) -> Hit {
        Hit {
            text: format!("{source}#{chunk_id}"),
            source: source.to_string(),
            score,
            document_id: stable_id(source),
            chunk_id,
            retrieval,
        }
    }

    #[test]
    fn deduplicates_by_document_and_chunk() {
        // Same chunk (b.txt#0) appears in both lists; must collapse to one hit.
        let keyword = vec![
            hit("a.txt", 0, 1.0, HitSource::Keyword),
            hit("b.txt", 0, 0.5, HitSource::Keyword),
        ];
        let vector = vec![
            hit("b.txt", 0, 0.9, HitSource::Vector),
            hit("c.txt", 0, 0.7, HitSource::Vector),
        ];

        let merged = reciprocal_rank_fusion(&[keyword, vector], 10);
        let keys: Vec<_> = merged
            .iter()
            .map(|h| (h.source.clone(), h.chunk_id))
            .collect();
        let mut unique = keys.clone();
        unique.sort();
        unique.dedup();
        assert_eq!(keys.len(), unique.len(), "results must be deduplicated");
        assert_eq!(merged.len(), 3); // a, b, c — b merged once
    }

    #[test]
    fn chunk_in_both_lists_ranks_first_and_is_hybrid() {
        let keyword = vec![
            hit("shared.txt", 0, 0.4, HitSource::Keyword),
            hit("kw.txt", 0, 0.9, HitSource::Keyword),
        ];
        let vector = vec![
            hit("shared.txt", 0, 0.4, HitSource::Vector),
            hit("vec.txt", 0, 0.9, HitSource::Vector),
        ];

        let merged = reciprocal_rank_fusion(&[keyword, vector], 10);
        assert_eq!(
            merged[0].source, "shared.txt",
            "chunk found by both layers should win"
        );
        assert_eq!(merged[0].retrieval, HitSource::Hybrid);
        assert!(
            (merged[0].score - 1.0).abs() < 1e-6,
            "top fused score normalises to 1.0"
        );
        // Single-layer hits keep their own provenance.
        let kw = merged.iter().find(|h| h.source == "kw.txt").unwrap();
        assert_eq!(kw.retrieval, HitSource::Keyword);
    }

    #[test]
    fn respects_top_k() {
        let a = vec![
            hit("a.txt", 0, 1.0, HitSource::Keyword),
            hit("b.txt", 0, 0.9, HitSource::Keyword),
        ];
        let b = vec![
            hit("c.txt", 0, 1.0, HitSource::Vector),
            hit("d.txt", 0, 0.9, HitSource::Vector),
        ];
        let merged = reciprocal_rank_fusion(&[a, b], 2);
        assert_eq!(merged.len(), 2);
    }

    #[test]
    fn empty_lists_yield_empty() {
        let merged = reciprocal_rank_fusion(&[vec![], vec![]], 5);
        assert!(merged.is_empty());
    }
}
