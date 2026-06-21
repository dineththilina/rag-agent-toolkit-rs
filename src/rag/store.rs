// src/rag/store.rs
//
// Local, disk-backed vector store built on SQLite + the sqlite-vec extension.
//
//   * Everything lives in one file on disk (default `data/rag_vectors.sqlite`).
//   * No server, no Docker, no cloud — the SQLite engine and the vec0 extension
//     are compiled from vendored C source.
//   * `chunks` holds the full chunk record (text + metadata); `vec_chunks` is a
//     vec0 virtual table holding the embeddings, joined by rowid. Cosine
//     distance is used so scores are comparable across queries.
//
// SQLite is the single source of truth: on startup the BM25 index is rebuilt
// from `chunks`, so previously indexed documents stay searchable across
// restarts with no separate index file.

use std::path::Path;
use std::sync::{Mutex, Once};

use anyhow::{Context, Result};
use rusqlite::{ffi::sqlite3_auto_extension, Connection};
use sqlite_vec::sqlite3_vec_init;
use tracing::{debug, info};

use crate::rag::types::{Hit, HitSource, StoredChunk};

/// The vector store interface. Implemented by [`SqliteVecStore`]; kept as a
/// trait so the storage backend can be swapped without touching retrieval.
pub trait VectorStore: Send + Sync {
    /// Embedding dimension this store was created with.
    fn dimension(&self) -> usize;

    /// Insert chunks with their embeddings. Any existing chunks for the same
    /// `document_id`s are replaced first, so re-indexing a document is
    /// idempotent (handles duplicate uploads gracefully).
    fn upsert(&self, chunks: &[StoredChunk], vectors: &[Vec<f32>]) -> Result<()>;

    /// Insert chunks WITHOUT embeddings (keyword-only fallback used when no
    /// embedding model is available). The chunks remain BM25-searchable and
    /// persist across restarts; a later rebuild fills in their vectors.
    fn insert_text_only(&self, chunks: &[StoredChunk]) -> Result<()>;

    /// K-nearest-neighbour search by cosine similarity. Returns up to `top_k`
    /// hits with a 0..1 similarity score (higher is better).
    fn search(&self, query: &[f32], top_k: usize) -> Result<Vec<Hit>>;

    /// Delete every chunk for a source filename. Returns the number removed.
    fn delete_source(&self, source: &str) -> Result<usize>;

    /// Distinct sources with chunk counts, sorted by name.
    fn list_sources(&self) -> Result<Vec<(String, usize)>>;

    /// Every stored chunk (used to rebuild the BM25 index on startup).
    fn all_chunks(&self) -> Result<Vec<StoredChunk>>;

    /// Total number of stored chunks.
    fn count(&self) -> Result<usize>;

    /// Delete all data (used by the rebuild command).
    fn clear(&self) -> Result<()>;
}

/// Register sqlite-vec as a SQLite auto-extension exactly once per process, so
/// every connection opened afterwards has the `vec0` virtual table available.
fn register_sqlite_vec() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        // SAFETY: `sqlite3_vec_init` is a valid SQLite extension entry point;
        // this is the registration pattern documented by sqlite-vec.
        #[allow(clippy::missing_transmute_annotations)]
        unsafe {
            sqlite3_auto_extension(Some(std::mem::transmute(sqlite3_vec_init as *const ())));
        }
    });
}

/// Serialise a vector as little-endian f32 bytes for sqlite-vec.
fn vec_to_blob(v: &[f32]) -> Vec<u8> {
    let mut b = Vec::with_capacity(v.len() * 4);
    for &x in v {
        b.extend_from_slice(&x.to_le_bytes());
    }
    b
}

pub struct SqliteVecStore {
    conn: Mutex<Connection>,
    dim: usize,
}

impl SqliteVecStore {
    /// Open (creating if needed) the vector database at `path` with the given
    /// embedding dimension. The parent directory is created automatically.
    pub fn open(path: &str, dim: usize) -> Result<Self> {
        anyhow::ensure!(dim > 0, "embedding dimension must be greater than zero");
        register_sqlite_vec();

        if let Some(parent) = Path::new(path).parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent).with_context(|| {
                    format!("creating vector-store directory '{}'", parent.display())
                })?;
            }
        }

        let conn =
            Connection::open(path).with_context(|| format!("opening vector database '{path}'"))?;

        // Confirm the extension loaded.
        let vec_version: String = conn
            .query_row("SELECT vec_version()", [], |r| r.get(0))
            .context("sqlite-vec extension failed to load")?;
        debug!("sqlite-vec {vec_version} loaded for '{path}'");

        let store = Self {
            conn: Mutex::new(conn),
            dim,
        };
        store.init_schema()?;
        info!(
            "Vector store ready at '{path}' ({} chunks, {dim} dims)",
            store.count().unwrap_or(0)
        );
        Ok(store)
    }

    fn init_schema(&self) -> Result<()> {
        let conn = self.conn.lock().expect("vector store mutex poisoned");
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS meta(key TEXT PRIMARY KEY, value TEXT);
             CREATE TABLE IF NOT EXISTS chunks(
                rowid       INTEGER PRIMARY KEY,
                document_id TEXT    NOT NULL,
                chunk_id    INTEGER NOT NULL,
                source      TEXT    NOT NULL,
                text        TEXT    NOT NULL,
                normalized  TEXT    NOT NULL,
                char_count  INTEGER NOT NULL,
                token_count INTEGER NOT NULL,
                created_at  INTEGER NOT NULL,
                metadata    TEXT
             );
             CREATE INDEX IF NOT EXISTS idx_chunks_source ON chunks(source);
             CREATE INDEX IF NOT EXISTS idx_chunks_docid  ON chunks(document_id);",
        )
        .context("creating metadata tables")?;

        // The vec0 table fixes its dimension at creation. If an existing DB was
        // built with a different dimension, fail loudly with a clear message
        // rather than corrupting the index — the rebuild command recreates it.
        let stored_dim: Option<i64> = conn
            .query_row(
                "SELECT value FROM meta WHERE key = 'embedding_dim'",
                [],
                |r| r.get::<_, String>(0),
            )
            .ok()
            .and_then(|s| s.parse().ok());

        match stored_dim {
            Some(d) if d as usize != self.dim => {
                anyhow::bail!(
                    "vector database dimension mismatch: stored index is {d}-dim but the current \
                     embedding model is {}-dim. Run the rebuild command (POST /api/rebuild) to \
                     recreate the local index.",
                    self.dim
                );
            }
            Some(_) => {}
            None => {
                conn.execute(
                    "INSERT OR REPLACE INTO meta(key, value) VALUES ('embedding_dim', ?1)",
                    rusqlite::params![self.dim.to_string()],
                )?;
            }
        }

        conn.execute(
            &format!(
                "CREATE VIRTUAL TABLE IF NOT EXISTS vec_chunks USING vec0(embedding float[{}] distance_metric=cosine)",
                self.dim
            ),
            [],
        )
        .context("creating vec0 virtual table")?;
        Ok(())
    }
}

impl VectorStore for SqliteVecStore {
    fn dimension(&self) -> usize {
        self.dim
    }

    fn upsert(&self, chunks: &[StoredChunk], vectors: &[Vec<f32>]) -> Result<()> {
        anyhow::ensure!(
            chunks.len() == vectors.len(),
            "chunk/vector count mismatch: {} chunks vs {} vectors",
            chunks.len(),
            vectors.len()
        );
        if chunks.is_empty() {
            return Ok(());
        }
        // Validate every vector's dimension before touching the database.
        for (i, v) in vectors.iter().enumerate() {
            anyhow::ensure!(
                v.len() == self.dim,
                "vector {i} has {} dims, expected {}",
                v.len(),
                self.dim
            );
        }

        let mut conn = self.conn.lock().expect("vector store mutex poisoned");
        let tx = conn.transaction().context("starting insert transaction")?;

        // Replace any existing chunks for the documents being upserted.
        let mut seen_docs = std::collections::BTreeSet::new();
        for c in chunks {
            if seen_docs.insert(c.document_id.clone()) {
                tx.execute(
                    "DELETE FROM vec_chunks WHERE rowid IN (SELECT rowid FROM chunks WHERE document_id = ?1)",
                    rusqlite::params![c.document_id],
                )?;
                tx.execute(
                    "DELETE FROM chunks WHERE document_id = ?1",
                    rusqlite::params![c.document_id],
                )?;
            }
        }

        for (c, v) in chunks.iter().zip(vectors.iter()) {
            let metadata = c.metadata.as_ref().map(|m| m.to_string());
            tx.execute(
                "INSERT INTO chunks(document_id, chunk_id, source, text, normalized, char_count, token_count, created_at, metadata)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
                rusqlite::params![
                    c.document_id,
                    c.chunk_id as i64,
                    c.source,
                    c.text,
                    c.normalized,
                    c.char_count as i64,
                    c.token_count as i64,
                    c.created_at,
                    metadata,
                ],
            )?;
            let rowid = tx.last_insert_rowid();
            tx.execute(
                "INSERT INTO vec_chunks(rowid, embedding) VALUES (?1, ?2)",
                rusqlite::params![rowid, vec_to_blob(v)],
            )?;
        }

        tx.commit().context("committing insert transaction")?;
        debug!("Upserted {} chunks into the vector store", chunks.len());
        Ok(())
    }

    fn insert_text_only(&self, chunks: &[StoredChunk]) -> Result<()> {
        if chunks.is_empty() {
            return Ok(());
        }
        let mut conn = self.conn.lock().expect("vector store mutex poisoned");
        let tx = conn
            .transaction()
            .context("starting text-only insert transaction")?;
        let mut seen_docs = std::collections::BTreeSet::new();
        for c in chunks {
            if seen_docs.insert(c.document_id.clone()) {
                tx.execute(
                    "DELETE FROM vec_chunks WHERE rowid IN (SELECT rowid FROM chunks WHERE document_id = ?1)",
                    rusqlite::params![c.document_id],
                )?;
                tx.execute(
                    "DELETE FROM chunks WHERE document_id = ?1",
                    rusqlite::params![c.document_id],
                )?;
            }
        }
        for c in chunks {
            let metadata = c.metadata.as_ref().map(|m| m.to_string());
            tx.execute(
                "INSERT INTO chunks(document_id, chunk_id, source, text, normalized, char_count, token_count, created_at, metadata)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
                rusqlite::params![
                    c.document_id,
                    c.chunk_id as i64,
                    c.source,
                    c.text,
                    c.normalized,
                    c.char_count as i64,
                    c.token_count as i64,
                    c.created_at,
                    metadata,
                ],
            )?;
        }
        tx.commit()
            .context("committing text-only insert transaction")?;
        Ok(())
    }

    fn search(&self, query: &[f32], top_k: usize) -> Result<Vec<Hit>> {
        anyhow::ensure!(
            query.len() == self.dim,
            "query vector has {} dims, expected {}",
            query.len(),
            self.dim
        );
        if top_k == 0 || self.count()? == 0 {
            return Ok(Vec::new());
        }

        let conn = self.conn.lock().expect("vector store mutex poisoned");
        let mut stmt = conn.prepare(
            "SELECT c.document_id, c.chunk_id, c.source, c.text, v.distance
             FROM vec_chunks v
             JOIN chunks c ON c.rowid = v.rowid
             WHERE v.embedding MATCH ?1 AND k = ?2
             ORDER BY v.distance",
        )?;
        let blob = vec_to_blob(query);
        let rows = stmt.query_map(rusqlite::params![blob, top_k as i64], |r| {
            let document_id: String = r.get(0)?;
            let chunk_id: i64 = r.get(1)?;
            let source: String = r.get(2)?;
            let text: String = r.get(3)?;
            let distance: f64 = r.get(4)?;
            Ok((document_id, chunk_id as usize, source, text, distance))
        })?;

        let mut hits = Vec::new();
        for row in rows {
            let (document_id, chunk_id, source, text, distance) = row?;
            // cosine distance ∈ [0, 2]; similarity = 1 − distance, clamped 0..1.
            let score = (1.0 - distance as f32).clamp(0.0, 1.0);
            hits.push(Hit {
                text,
                source,
                score,
                document_id,
                chunk_id,
                retrieval: HitSource::Vector,
            });
        }
        Ok(hits)
    }

    fn delete_source(&self, source: &str) -> Result<usize> {
        let conn = self.conn.lock().expect("vector store mutex poisoned");
        conn.execute(
            "DELETE FROM vec_chunks WHERE rowid IN (SELECT rowid FROM chunks WHERE source = ?1)",
            rusqlite::params![source],
        )?;
        let n = conn.execute(
            "DELETE FROM chunks WHERE source = ?1",
            rusqlite::params![source],
        )?;
        Ok(n)
    }

    fn list_sources(&self) -> Result<Vec<(String, usize)>> {
        let conn = self.conn.lock().expect("vector store mutex poisoned");
        let mut stmt =
            conn.prepare("SELECT source, COUNT(*) FROM chunks GROUP BY source ORDER BY source")?;
        let rows = stmt.query_map([], |r| {
            let source: String = r.get(0)?;
            let count: i64 = r.get(1)?;
            Ok((source, count as usize))
        })?;
        Ok(rows.collect::<Result<Vec<_>, _>>()?)
    }

    fn all_chunks(&self) -> Result<Vec<StoredChunk>> {
        let conn = self.conn.lock().expect("vector store mutex poisoned");
        let mut stmt = conn.prepare(
            "SELECT document_id, chunk_id, source, text, normalized, char_count, token_count, created_at, metadata
             FROM chunks ORDER BY source, chunk_id",
        )?;
        let rows = stmt.query_map([], |r| {
            let metadata: Option<String> = r.get(8)?;
            Ok(StoredChunk {
                document_id: r.get(0)?,
                chunk_id: r.get::<_, i64>(1)? as usize,
                source: r.get(2)?,
                text: r.get(3)?,
                normalized: r.get(4)?,
                char_count: r.get::<_, i64>(5)? as usize,
                token_count: r.get::<_, i64>(6)? as usize,
                created_at: r.get(7)?,
                metadata: metadata.and_then(|s| serde_json::from_str(&s).ok()),
            })
        })?;
        Ok(rows.collect::<Result<Vec<_>, _>>()?)
    }

    fn count(&self) -> Result<usize> {
        let conn = self.conn.lock().expect("vector store mutex poisoned");
        let n: i64 = conn.query_row("SELECT COUNT(*) FROM chunks", [], |r| r.get(0))?;
        Ok(n as usize)
    }

    fn clear(&self) -> Result<()> {
        let conn = self.conn.lock().expect("vector store mutex poisoned");
        conn.execute_batch("DELETE FROM vec_chunks; DELETE FROM chunks;")?;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rag::embed::{EmbeddingProvider, MockEmbeddingProvider};
    use crate::rag::types::{now_unix, stable_id};

    struct TempDb(String);
    impl TempDb {
        fn new(tag: &str) -> Self {
            let p = std::env::temp_dir().join(format!(
                "ragvec-{tag}-{}-{}.sqlite",
                std::process::id(),
                now_unix()
            ));
            TempDb(p.to_string_lossy().into_owned())
        }
    }
    impl Drop for TempDb {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

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
    fn insert_and_search_returns_nearest() {
        let p = MockEmbeddingProvider::new(128);
        let db = TempDb::new("search");
        let store = SqliteVecStore::open(&db.0, p.dimension()).unwrap();

        let chunks = vec![
            chunk(
                "specs.txt",
                0,
                "The Helios H1 battery lasts eight hours per charge",
            ),
            chunk(
                "pricing.md",
                0,
                "Growth plan annual pricing with volume discount",
            ),
            chunk(
                "faq.txt",
                0,
                "Support is available weekdays from nine to five",
            ),
        ];
        let vectors: Vec<Vec<f32>> = chunks
            .iter()
            .map(|c| p.embed_one(&c.text).unwrap())
            .collect();
        store.upsert(&chunks, &vectors).unwrap();
        assert_eq!(store.count().unwrap(), 3);

        let q = p.embed_one("how many hours does the battery last").unwrap();
        let hits = store.search(&q, 2).unwrap();
        assert!(!hits.is_empty());
        assert_eq!(hits[0].source, "specs.txt");
        assert!(hits[0].score > 0.0);
    }

    #[test]
    fn delete_source_removes_vectors_too() {
        let p = MockEmbeddingProvider::new(64);
        let db = TempDb::new("delete");
        let store = SqliteVecStore::open(&db.0, p.dimension()).unwrap();
        let chunks = vec![
            chunk("a.txt", 0, "alpha battery"),
            chunk("b.txt", 0, "beta pricing"),
        ];
        let vecs: Vec<Vec<f32>> = chunks
            .iter()
            .map(|c| p.embed_one(&c.text).unwrap())
            .collect();
        store.upsert(&chunks, &vecs).unwrap();
        assert_eq!(store.delete_source("a.txt").unwrap(), 1);
        assert_eq!(store.count().unwrap(), 1);
        assert_eq!(
            store.list_sources().unwrap(),
            vec![("b.txt".to_string(), 1)]
        );
    }

    #[test]
    fn upsert_same_document_replaces() {
        let p = MockEmbeddingProvider::new(64);
        let db = TempDb::new("replace");
        let store = SqliteVecStore::open(&db.0, p.dimension()).unwrap();
        let v1 = vec![chunk("a.txt", 0, "first version of the text")];
        store
            .upsert(&v1, &[p.embed_one(&v1[0].text).unwrap()])
            .unwrap();
        let v2 = vec![
            chunk("a.txt", 0, "second version part one"),
            chunk("a.txt", 1, "second version part two"),
        ];
        let vecs: Vec<Vec<f32>> = v2.iter().map(|c| p.embed_one(&c.text).unwrap()).collect();
        store.upsert(&v2, &vecs).unwrap();
        // Re-uploading the same document replaces, not appends.
        assert_eq!(store.count().unwrap(), 2);
    }

    #[test]
    fn dimension_mismatch_is_rejected() {
        let db = TempDb::new("dim");
        {
            let _store = SqliteVecStore::open(&db.0, 64).unwrap();
        }
        // Reopening with a different dimension must fail clearly.
        let result = SqliteVecStore::open(&db.0, 128);
        assert!(
            result.is_err(),
            "reopening with a different dimension should fail"
        );
        let err = result.err().unwrap();
        assert!(err.to_string().contains("dimension mismatch"), "got: {err}");
    }
}
