// src/rag/embed.rs
//
// The embedding abstraction.
//
// `EmbeddingProvider` is the seam between the retrieval engine and whatever
// generates vectors. Production uses `LocalFastEmbedProvider` (see
// `fastembed_provider.rs`); tests use `MockEmbeddingProvider`, which is
// deterministic, dependency-free, and fast — so test runs never download a
// model.
//
// The interface is synchronous and blocking by design (embedding is CPU work);
// the async orchestration layer wraps heavy calls in `spawn_blocking`.

use anyhow::Result;

/// Anything that can turn text into fixed-length vectors.
pub trait EmbeddingProvider: Send + Sync {
    /// Embed a batch of texts. Each returned vector has length [`dimension`].
    fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>>;

    /// The dimension of every vector this provider returns.
    fn dimension(&self) -> usize;

    /// A short human-readable name for logging.
    fn name(&self) -> String;

    /// Convenience: embed a single string.
    fn embed_one(&self, text: &str) -> Result<Vec<f32>> {
        let mut v = self.embed(&[text.to_string()])?;
        v.pop()
            .ok_or_else(|| anyhow::anyhow!("embedding provider returned no vector"))
    }
}

/// Deterministic, model-free embedding for tests and offline builds.
///
/// It hashes tokens into a fixed number of dimensions (a hashing bag-of-words),
/// then L2-normalises. Texts that share words get similar vectors, which is
/// enough to exercise vector storage, hybrid merge, and persistence without a
/// heavyweight ML model.
///
/// This is intentionally NOT wired in as a production default — real semantic
/// quality comes from `LocalFastEmbedProvider`. It exists so tests are fast and
/// hermetic.
pub struct MockEmbeddingProvider {
    dim: usize,
}

impl MockEmbeddingProvider {
    pub fn new(dim: usize) -> Self {
        assert!(dim > 0, "embedding dimension must be > 0");
        Self { dim }
    }

    fn token_bucket(token: &str, dim: usize) -> usize {
        // FNV-1a → bucket index. Deterministic across runs and platforms.
        let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
        for b in token.as_bytes() {
            hash ^= *b as u64;
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
        (hash % dim as u64) as usize
    }
}

impl EmbeddingProvider for MockEmbeddingProvider {
    fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        let mut out = Vec::with_capacity(texts.len());
        for text in texts {
            let mut v = vec![0.0f32; self.dim];
            for token in text.to_lowercase().split(|c: char| !c.is_alphanumeric()) {
                if token.len() < 2 {
                    continue;
                }
                let idx = Self::token_bucket(token, self.dim);
                v[idx] += 1.0;
            }
            // L2-normalise so cosine distance is meaningful; keep a tiny floor so
            // empty text still yields a valid (non-NaN) unit-ish vector.
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for x in v.iter_mut() {
                    *x /= norm;
                }
            } else {
                v[0] = 1.0;
            }
            out.push(v);
        }
        Ok(out)
    }

    fn dimension(&self) -> usize {
        self.dim
    }

    fn name(&self) -> String {
        format!("mock-hash-{}d", self.dim)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_and_correct_dimension() {
        let p = MockEmbeddingProvider::new(64);
        let a = p.embed_one("the quick brown fox").unwrap();
        let b = p.embed_one("the quick brown fox").unwrap();
        assert_eq!(a.len(), 64);
        assert_eq!(a, b, "mock embeddings must be deterministic");
    }

    fn cosine(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(x, y)| x * y).sum::<f32>()
    }

    #[test]
    fn similar_text_is_closer_than_unrelated_text() {
        let p = MockEmbeddingProvider::new(256);
        let q = p.embed_one("battery life of the robot").unwrap();
        let related = p.embed_one("the robot battery lasts a long time").unwrap();
        let unrelated = p
            .embed_one("quarterly pricing discounts for enterprise")
            .unwrap();
        assert!(
            cosine(&q, &related) > cosine(&q, &unrelated),
            "related text should have higher cosine similarity"
        );
    }

    #[test]
    fn batch_matches_single() {
        let p = MockEmbeddingProvider::new(32);
        let batch = p.embed(&["alpha".to_string(), "beta".to_string()]).unwrap();
        assert_eq!(batch.len(), 2);
        assert_eq!(batch[0], p.embed_one("alpha").unwrap());
    }
}
