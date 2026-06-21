// src/sessions.rs
//
// Persistent conversation memory. Each session's turns are stored in a local
// SQLite file (default data/sessions.sqlite), so conversations survive a
// server restart — previously this lived only in an in-process DashMap and
// was lost every time the process exited.

use std::path::Path;
use std::sync::{Arc, Mutex};

use anyhow::{Context, Result};
use rusqlite::Connection;
use tracing::info;

use crate::agent::{Message, ToolCall};

/// Keep at most this many messages (user+assistant turns) per session, so a
/// long-running conversation can't grow the database without bound.
const MAX_MESSAGES_PER_SESSION: i64 = 40;

/// Sessions with no activity for this long are dropped, once, at startup.
const SESSION_TTL_SECS: i64 = 60 * 60 * 24 * 30; // 30 days

pub struct SqliteSessionStore {
    conn: Mutex<Connection>,
}

pub type SessionStore = Arc<SqliteSessionStore>;

/// Open (creating if needed) the session database at `path`.
pub fn open(path: &str) -> Result<SessionStore> {
    Ok(Arc::new(SqliteSessionStore::open(path)?))
}

impl SqliteSessionStore {
    fn open(path: &str) -> Result<Self> {
        if let Some(parent) = Path::new(path).parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent).with_context(|| {
                    format!("creating session-store directory '{}'", parent.display())
                })?;
            }
        }
        let conn =
            Connection::open(path).with_context(|| format!("opening session database '{path}'"))?;
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS messages(
                id           INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id   TEXT    NOT NULL,
                seq          INTEGER NOT NULL,
                role         TEXT    NOT NULL,
                content      TEXT,
                tool_calls   TEXT,
                tool_call_id TEXT,
                name         TEXT,
                created_at   INTEGER NOT NULL
             );
             CREATE INDEX IF NOT EXISTS idx_messages_session ON messages(session_id, seq);",
        )
        .context("creating session table")?;

        let store = Self {
            conn: Mutex::new(conn),
        };
        store.prune_stale()?;
        info!("Session store ready at '{path}'");
        Ok(store)
    }

    /// Drop every message belonging to a session whose most recent message is
    /// older than the TTL. Runs once at startup.
    fn prune_stale(&self) -> Result<()> {
        let cutoff = now_unix() - SESSION_TTL_SECS;
        let conn = self.conn.lock().expect("session store mutex poisoned");
        let removed = conn.execute(
            "DELETE FROM messages WHERE session_id IN (
                SELECT session_id FROM messages GROUP BY session_id HAVING MAX(created_at) < ?1
             )",
            rusqlite::params![cutoff],
        )?;
        if removed > 0 {
            info!("Pruned {removed} message(s) from sessions inactive for 30+ days");
        }
        Ok(())
    }

    /// Full conversation history for a session, oldest first.
    pub fn history(&self, session_id: &str) -> Result<Vec<Message>> {
        let conn = self.conn.lock().expect("session store mutex poisoned");
        let mut stmt = conn.prepare(
            "SELECT role, content, tool_calls, tool_call_id, name
             FROM messages WHERE session_id = ?1 ORDER BY seq",
        )?;
        let rows = stmt.query_map(rusqlite::params![session_id], |r| {
            let role: String = r.get(0)?;
            let content: Option<String> = r.get(1)?;
            let tool_calls: Option<String> = r.get(2)?;
            let tool_call_id: Option<String> = r.get(3)?;
            let name: Option<String> = r.get(4)?;
            Ok((role, content, tool_calls, tool_call_id, name))
        })?;

        let mut out = Vec::new();
        for row in rows {
            let (role, content, tool_calls, tool_call_id, name) = row?;
            let tool_calls: Option<Vec<ToolCall>> =
                tool_calls.and_then(|s| serde_json::from_str(&s).ok());
            out.push(Message {
                role,
                content,
                tool_calls,
                tool_call_id,
                name,
            });
        }
        Ok(out)
    }

    /// Append new turns to a session, then trim to the most recent
    /// `MAX_MESSAGES_PER_SESSION` messages.
    pub fn append(&self, session_id: &str, messages: &[Message]) -> Result<()> {
        if messages.is_empty() {
            return Ok(());
        }
        let mut conn = self.conn.lock().expect("session store mutex poisoned");
        let tx = conn
            .transaction()
            .context("starting session append transaction")?;
        let next_seq: i64 = tx.query_row(
            "SELECT COALESCE(MAX(seq), -1) + 1 FROM messages WHERE session_id = ?1",
            rusqlite::params![session_id],
            |r| r.get(0),
        )?;
        let now = now_unix();
        for (i, m) in messages.iter().enumerate() {
            let tool_calls = m
                .tool_calls
                .as_ref()
                .map(serde_json::to_string)
                .transpose()?;
            tx.execute(
                "INSERT INTO messages(session_id, seq, role, content, tool_calls, tool_call_id, name, created_at)
                 VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
                rusqlite::params![
                    session_id,
                    next_seq + i as i64,
                    m.role,
                    m.content,
                    tool_calls,
                    m.tool_call_id,
                    m.name,
                    now,
                ],
            )?;
        }
        tx.execute(
            "DELETE FROM messages WHERE session_id = ?1 AND id NOT IN (
                SELECT id FROM messages WHERE session_id = ?1 ORDER BY seq DESC LIMIT ?2
             )",
            rusqlite::params![session_id, MAX_MESSAGES_PER_SESSION],
        )?;
        tx.commit()
            .context("committing session append transaction")?;
        Ok(())
    }

    /// Delete a session's history entirely.
    pub fn clear(&self, session_id: &str) -> Result<()> {
        let conn = self.conn.lock().expect("session store mutex poisoned");
        conn.execute(
            "DELETE FROM messages WHERE session_id = ?1",
            rusqlite::params![session_id],
        )?;
        Ok(())
    }
}

fn now_unix() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TempDb(String);
    impl TempDb {
        fn new(tag: &str) -> Self {
            let nanos = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            let p = std::env::temp_dir().join(format!(
                "sessions-{tag}-{}-{nanos}.sqlite",
                std::process::id()
            ));
            TempDb(p.to_string_lossy().into_owned())
        }
    }
    impl Drop for TempDb {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }

    fn msg(role: &str, text: &str) -> Message {
        Message {
            role: role.into(),
            content: Some(text.into()),
            tool_calls: None,
            tool_call_id: None,
            name: None,
        }
    }

    #[test]
    fn history_persists_across_reopen() {
        let db = TempDb::new("persist");
        {
            let store = SqliteSessionStore::open(&db.0).unwrap();
            store
                .append("s1", &[msg("user", "hi"), msg("assistant", "hello")])
                .unwrap();
        }
        let store = SqliteSessionStore::open(&db.0).unwrap();
        let history = store.history("s1").unwrap();
        assert_eq!(history.len(), 2);
        assert_eq!(history[0].role, "user");
        assert_eq!(history[1].content.as_deref(), Some("hello"));
    }

    #[test]
    fn sessions_are_isolated() {
        let db = TempDb::new("isolated");
        let store = SqliteSessionStore::open(&db.0).unwrap();
        store.append("a", &[msg("user", "from a")]).unwrap();
        store.append("b", &[msg("user", "from b")]).unwrap();
        assert_eq!(store.history("a").unwrap().len(), 1);
        assert_eq!(store.history("b").unwrap().len(), 1);
    }

    #[test]
    fn history_is_capped() {
        let db = TempDb::new("capped");
        let store = SqliteSessionStore::open(&db.0).unwrap();
        for i in 0..60 {
            store
                .append("s1", &[msg("user", &format!("turn {i}"))])
                .unwrap();
        }
        let history = store.history("s1").unwrap();
        assert_eq!(history.len() as i64, MAX_MESSAGES_PER_SESSION);
        assert_eq!(history.last().unwrap().content.as_deref(), Some("turn 59"));
    }

    #[test]
    fn clear_removes_history() {
        let db = TempDb::new("clear");
        let store = SqliteSessionStore::open(&db.0).unwrap();
        store.append("s1", &[msg("user", "hi")]).unwrap();
        store.clear("s1").unwrap();
        assert!(store.history("s1").unwrap().is_empty());
    }
}
