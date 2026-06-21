// tests/rag_integration.rs
//
// Black-box integration tests driving the real RAG system through its public
// library API. They use the deterministic `MockEmbeddingProvider` so they run
// fast and offline (no model download), while exercising the genuine on-disk
// sqlite-vec store, the BM25 index, hybrid fusion, and persistence.

use std::sync::Arc;

use agent_toolkit::config::Config;
use agent_toolkit::rag::embed::MockEmbeddingProvider;
use agent_toolkit::rag::{self, RagConfig, RagSystem, RetrievalMode};

const DIM: usize = 256;

/// A unique temp database path that is cleaned up on drop.
struct TempDb(String);
impl TempDb {
    fn new(tag: &str) -> Self {
        let nanos = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos();
        let p =
            std::env::temp_dir().join(format!("ragit-{tag}-{}-{nanos}.sqlite", std::process::id()));
        TempDb(p.to_string_lossy().into_owned())
    }
}
impl Drop for TempDb {
    fn drop(&mut self) {
        let _ = std::fs::remove_file(&self.0);
        let _ = std::fs::remove_file(format!("{}-wal", self.0));
        let _ = std::fs::remove_file(format!("{}-shm", self.0));
    }
}

fn open_system(db_path: &str) -> Arc<RagSystem> {
    let cfg = Config {
        vector_db_path: db_path.to_string(),
        embedding_dim: DIM,
        // Point the data dir somewhere empty so tests don't ingest the repo's
        // sample documents.
        data_dir: std::env::temp_dir()
            .join("ragit-empty")
            .to_string_lossy()
            .into_owned(),
        ..Config::default()
    };
    let _ = std::fs::create_dir_all(&cfg.data_dir);
    let rag_cfg = RagConfig::from_app_config(&cfg);
    let embedder = Arc::new(MockEmbeddingProvider::new(DIM));
    Arc::new(RagSystem::open(rag_cfg, Some(embedder)).expect("open RagSystem"))
}

const ROBOT_DOC: &str = "The Helios H1 robot battery lasts eight hours on a single charge. \
    Recharging the robot battery to full takes ninety minutes.";
const PRICING_DOC: &str = "Enterprise pricing offers volume discounts for annual contracts. \
    The Growth plan is billed monthly with no setup fee.";

#[tokio::test]
async fn ingests_persists_and_retrieves_across_restart() {
    let db = TempDb::new("restart");

    // First "run": ingest two documents, then drop the system.
    {
        let sys = open_system(&db.0);
        let n1 = sys.add_document("robot.txt", ROBOT_DOC).await.unwrap();
        let n2 = sys.add_document("pricing.txt", PRICING_DOC).await.unwrap();
        assert!(n1 >= 1 && n2 >= 1);
        let sources = rag::list_sources(&sys).await;
        assert_eq!(sources.len(), 2, "both documents should be indexed");
    } // system dropped — simulates app shutdown

    // Second "run": reopen the SAME database; nothing re-ingested.
    let sys = open_system(&db.0);
    let sources = rag::list_sources(&sys).await;
    assert_eq!(sources.len(), 2, "documents must persist across restart");

    // Semantic query should surface the robot document, not pricing.
    let result = sys
        .retrieve(
            "how long does the robot battery last",
            5,
            RetrievalMode::Vector,
            0.0,
        )
        .await
        .unwrap();
    assert!(
        !result.hits.is_empty(),
        "expected vector hits after restart"
    );
    assert_eq!(
        result.hits[0].source, "robot.txt",
        "most relevant chunk should be the robot doc"
    );
}

#[tokio::test]
async fn retrieval_mode_selection_keyword_vector_hybrid() {
    let db = TempDb::new("modes");
    let sys = open_system(&db.0);
    sys.add_document("robot.txt", ROBOT_DOC).await.unwrap();
    sys.add_document("pricing.txt", PRICING_DOC).await.unwrap();

    let query = "robot battery charge";

    let kw = sys
        .retrieve(query, 5, RetrievalMode::Keyword, 0.0)
        .await
        .unwrap();
    assert_eq!(kw.mode, RetrievalMode::Keyword);
    assert!(!kw.hits.is_empty());
    assert!(kw
        .hits
        .iter()
        .all(|h| matches!(h.retrieval, rag::types::HitSource::Keyword)));

    let vec = sys
        .retrieve(query, 5, RetrievalMode::Vector, 0.0)
        .await
        .unwrap();
    assert_eq!(vec.mode, RetrievalMode::Vector);
    assert!(!vec.hits.is_empty());

    let hyb = sys
        .retrieve(query, 5, RetrievalMode::Hybrid, 0.0)
        .await
        .unwrap();
    assert_eq!(hyb.mode, RetrievalMode::Hybrid);
    assert!(!hyb.hits.is_empty());
    // The robot doc is the relevant one across every mode.
    assert_eq!(hyb.hits[0].source, "robot.txt");
}

#[tokio::test]
async fn hybrid_results_are_deduplicated() {
    let db = TempDb::new("dedup");
    let sys = open_system(&db.0);
    sys.add_document("robot.txt", ROBOT_DOC).await.unwrap();
    sys.add_document("pricing.txt", PRICING_DOC).await.unwrap();

    let result = sys
        .retrieve("robot battery charge hours", 8, RetrievalMode::Hybrid, 0.0)
        .await
        .unwrap();

    // No (document_id, chunk_id) pair appears twice after fusion.
    let mut keys: Vec<(String, usize)> = result
        .hits
        .iter()
        .map(|h| (h.document_id.clone(), h.chunk_id))
        .collect();
    let total = keys.len();
    keys.sort();
    keys.dedup();
    assert_eq!(
        keys.len(),
        total,
        "hybrid results must be deduplicated by document+chunk"
    );
}

#[tokio::test]
async fn low_confidence_when_nothing_matches() {
    let db = TempDb::new("lowconf");
    let sys = open_system(&db.0);
    sys.add_document("robot.txt", ROBOT_DOC).await.unwrap();

    // A query with no lexical overlap and a high similarity threshold should
    // not be considered confident evidence.
    let result = sys
        .retrieve(
            "photosynthesis chlorophyll wavelength",
            5,
            RetrievalMode::Vector,
            0.95,
        )
        .await
        .unwrap();
    assert!(
        !result.confident,
        "unrelated query should not be confident at a high threshold"
    );
}

#[tokio::test]
async fn duplicate_upload_replaces_rather_than_appends() {
    let db = TempDb::new("dupe");
    let sys = open_system(&db.0);
    sys.add_document("robot.txt", ROBOT_DOC).await.unwrap();
    let before = rag::list_sources(&sys).await;
    let before_count: usize = before.iter().map(|(_, n)| n).sum();

    // Re-upload the same filename — chunks should be replaced, not duplicated.
    sys.add_document("robot.txt", ROBOT_DOC).await.unwrap();
    let after = rag::list_sources(&sys).await;
    assert_eq!(after.len(), 1, "still a single source");
    let after_count: usize = after.iter().map(|(_, n)| n).sum();
    assert_eq!(
        before_count, after_count,
        "re-upload must not duplicate chunks"
    );
}
