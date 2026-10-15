//! SQLite implementation of MetadataStore.

use super::metadata_store::{DocumentRecord, MetadataStore};
use rusqlite::{params, Connection};
use std::path::PathBuf;
use std::sync::Mutex;

/// SQLite-backed metadata store with a single persistent connection guarded
/// by a `Mutex`.
///
/// # Concurrency semantics (honest)
///
/// Within one process there is exactly ONE connection shared under a `Mutex`:
/// operations serialize — readers never observe torn state and there is no
/// `SQLITE_BUSY` between threads of this process.
///
/// Across processes (CLI + MCP server over the same `.db` file) SQLite's
/// file locking applies: WAL mode allows one writer + N concurrent readers;
/// the rollback journal (fallback when WAL is unavailable, e.g. on filesystems
/// without shared-memory support) allows a single accessor at a time. Either
/// way, `busy_timeout=5000ms` makes lock contention retry for up to 5s
/// instead of failing immediately with `SQLITE_BUSY`. Durability:
/// `synchronous=NORMAL` under WAL (commits are durable on checkpoint / power
/// loss may lose the last transactions but the DB stays consistent); FULL
/// default fsync per commit when running on the rollback journal.
pub struct SqliteMetadataStore {
    conn: Mutex<Connection>,
}

impl SqliteMetadataStore {
    pub fn new(db_path: PathBuf) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        std::fs::create_dir_all(db_path.parent().unwrap_or(&db_path))?;
        let conn = Connection::open(&db_path)?;
        conn.busy_timeout(std::time::Duration::from_millis(5000))?;
        // WAL can silently fail (returns the old mode) on some filesystems
        // (e.g. network mounts without shared-memory support). Read the mode
        // back and fall back to the default journal if it did not stick.
        let journal_mode: String =
            conn.query_row("PRAGMA journal_mode=WAL", [], |row| row.get(0))?;
        if journal_mode.eq_ignore_ascii_case("wal") {
            // Only enable NORMAL durability under WAL: with the rollback
            // journal, NORMAL still fsyncs on every commit.
            conn.pragma_update(None, "synchronous", "NORMAL")?;
        } else {
            eprintln!(
                "[sqlite] WAL not available (mode={journal_mode:?}), keeping rollback journal"
            );
        }
        conn.pragma_update(None, "cache_size", -64000)?;
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS documents (
                 id TEXT PRIMARY KEY,
                 shard_idx INTEGER,
                 vector_idx INTEGER,
                 metadata TEXT,
                 document TEXT,
                 deleted INTEGER DEFAULT 0,
                 created_at REAL,
                 updated_at REAL
             );
             CREATE INDEX IF NOT EXISTS idx_deleted ON documents(deleted);
             CREATE INDEX IF NOT EXISTS idx_vector ON documents(vector_idx);",
        )?;
        Ok(Self {
            conn: Mutex::new(conn),
        })
    }

    fn row_to_record(row: &rusqlite::Row) -> rusqlite::Result<DocumentRecord> {
        let meta_str: Option<String> = row.get(3)?;
        Ok(DocumentRecord {
            id: row.get(0)?,
            shard_idx: row.get(1)?,
            vector_idx: row.get(2)?,
            metadata: meta_str.and_then(|s| serde_json::from_str(&s).ok()),
            document: row.get(4)?,
            deleted: row.get::<_, i32>(5)? != 0,
            created_at: row.get(6)?,
            updated_at: row.get(7)?,
        })
    }

    /// Current journal mode (e.g. "wal", "delete") — exposed for tests.
    pub fn journal_mode(&self) -> Result<String, Box<dyn std::error::Error + Send + Sync>> {
        let conn = self.conn.lock().map_err(|e| format!("mutex: {e}"))?;
        let mode: String = conn.query_row("PRAGMA journal_mode", [], |row| row.get(0))?;
        Ok(mode)
    }
}

impl MetadataStore for SqliteMetadataStore {
    fn upsert(
        &self,
        record: &DocumentRecord,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let conn = self.conn.lock().map_err(|e| format!("mutex: {e}"))?;
        conn.execute(
            "INSERT OR REPLACE INTO documents (id, shard_idx, vector_idx, metadata, document, deleted, created_at, updated_at)
             VALUES (?1,?2,?3,?4,?5,?6,?7,?8)",
            params![
                record.id,
                record.shard_idx,
                record.vector_idx,
                record.metadata.as_ref().map(|m| m.to_string()),
                record.document,
                record.deleted as i32,
                record.created_at,
                record.updated_at,
            ],
        )?;
        Ok(())
    }

    fn get(
        &self,
        doc_id: &str,
    ) -> Result<Option<DocumentRecord>, Box<dyn std::error::Error + Send + Sync>> {
        let conn = self.conn.lock().map_err(|e| format!("mutex: {e}"))?;
        let mut stmt = conn.prepare(
            "SELECT id, shard_idx, vector_idx, metadata, document, deleted, created_at, updated_at
             FROM documents WHERE id=?1 AND deleted=0",
        )?;
        let result = stmt.query_row(params![doc_id], Self::row_to_record);
        match result {
            Ok(r) => Ok(Some(r)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    fn soft_delete(&self, doc_id: &str) -> Result<bool, Box<dyn std::error::Error + Send + Sync>> {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_secs_f64();
        let conn = self.conn.lock().map_err(|e| format!("mutex: {e}"))?;
        let affected = conn.execute(
            "UPDATE documents SET deleted=1, updated_at=?1 WHERE id=?2",
            params![now, doc_id],
        )?;
        Ok(affected > 0)
    }

    fn hard_delete(&self, doc_id: &str) -> Result<bool, Box<dyn std::error::Error + Send + Sync>> {
        let conn = self.conn.lock().map_err(|e| format!("mutex: {e}"))?;
        let affected = conn.execute("DELETE FROM documents WHERE id=?1", params![doc_id])?;
        Ok(affected > 0)
    }

    fn list_ids(
        &self,
        include_deleted: bool,
    ) -> Result<Vec<String>, Box<dyn std::error::Error + Send + Sync>> {
        let conn = self.conn.lock().map_err(|e| format!("mutex: {e}"))?;
        let sql = if include_deleted {
            "SELECT id FROM documents"
        } else {
            "SELECT id FROM documents WHERE deleted=0"
        };
        let mut stmt = conn.prepare(sql)?;
        let ids: Vec<String> = stmt
            .query_map([], |row| row.get(0))?
            .filter_map(|r| r.ok())
            .collect();
        Ok(ids)
    }

    fn count(
        &self,
        include_deleted: bool,
    ) -> Result<usize, Box<dyn std::error::Error + Send + Sync>> {
        let conn = self.conn.lock().map_err(|e| format!("mutex: {e}"))?;
        let sql = if include_deleted {
            "SELECT COUNT(*) FROM documents"
        } else {
            "SELECT COUNT(*) FROM documents WHERE deleted=0"
        };
        let count: usize = conn.query_row(sql, [], |row| row.get(0))?;
        Ok(count)
    }

    fn flush(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let conn = self.conn.lock().map_err(|e| format!("mutex: {e}"))?;
        conn.execute_batch("PRAGMA wal_checkpoint(TRUNCATE);")?;
        Ok(())
    }

    fn backend_name(&self) -> &'static str {
        "sqlite"
    }

    /// Indexed lookup via `idx_vector`: O(log N + k) where k = duplicates of
    /// this `vector_idx`, instead of the O(N·log N) default scan.
    ///
    /// `vector_idx` is NOT unique (a soft-deleted doc keeps its slot and
    /// `n_active()` can hand the same index to a new doc), so duplicates are
    /// possible. Parity with the default trait scan is guaranteed by
    /// construction: `ORDER BY id LIMIT 1` picks the lexicographically-first
    /// matching non-deleted row, which is exactly the row the default scan
    /// hits first — `list_ids` is served by the covering index on the TEXT
    /// PRIMARY KEY, which returns `id`s in ascending lexicographic order
    /// (verified empirically; see `find_by_vector_idx_sql_matches_scan`).
    fn find_by_vector_idx(
        &self,
        vector_idx: i64,
    ) -> Result<Option<DocumentRecord>, Box<dyn std::error::Error + Send + Sync>> {
        let conn = self.conn.lock().map_err(|e| format!("mutex: {e}"))?;
        let mut stmt = conn.prepare(
            "SELECT id, shard_idx, vector_idx, metadata, document, deleted, created_at, updated_at
             FROM documents
             WHERE vector_idx = ?1 AND deleted = 0
             ORDER BY id
             LIMIT 1",
        )?;
        let result = stmt.query_row(params![vector_idx], Self::row_to_record);
        match result {
            Ok(r) => Ok(Some(r)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Reference O(N) scan, verbatim copy of the default trait method — the
    /// SQL path must match this exactly (same doc) on every query.
    fn scan_reference(store: &SqliteMetadataStore, vector_idx: i64) -> Option<DocumentRecord> {
        let ids = store.list_ids(true).unwrap();
        for id in ids {
            if let Some(record) = store.get(&id).unwrap() {
                if record.vector_idx == vector_idx && !record.deleted {
                    return Some(record);
                }
            }
        }
        None
    }

    fn temp_store(tag: &str) -> SqliteMetadataStore {
        let dir = std::env::temp_dir().join(format!(
            "splatsdb-sqlite-test-{}-{}",
            tag,
            std::process::id()
        ));
        SqliteMetadataStore::new(dir.join("meta.db")).unwrap()
    }

    /// P1 parity: indexed SQL path vs O(N) scan over a random dataset with
    /// deliberate vector_idx collisions and soft-deletes — same doc, every time.
    #[test]
    fn find_by_vector_idx_sql_matches_scan() {
        use rand::prelude::SliceRandom;
        use rand::Rng;
        use rand::SeedableRng;
        let store = temp_store("parity");
        let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(0xC0FFEE);

        // 300 docs, vector_idx in [0,50) → guaranteed collisions; random ids
        // (not insertion-ordered) so lexicographic order differs from insert
        // order; ~20% soft-deleted so ties can be broken by deleted flags.
        let mut ids: Vec<String> = (0..300).map(|i| format!("doc_{i:05}")).collect();
        ids.shuffle(&mut rng);
        for (i, id) in ids.iter().enumerate() {
            let record = DocumentRecord {
                id: id.clone(),
                shard_idx: 0,
                vector_idx: rng.gen_range(0..50),
                metadata: Some(serde_json::json!({ "n": i })),
                document: Some(format!("content {i}")),
                deleted: false,
                created_at: i as f64,
                updated_at: i as f64,
            };
            store.upsert(&record).unwrap();
            if rng.gen_bool(0.2) {
                store.soft_delete(id).unwrap();
            }
        }

        // Every idx value incl. values with collisions, plus misses (50..60).
        for target in 0..60 {
            let sql = store.find_by_vector_idx(target).unwrap();
            let scan = scan_reference(&store, target);
            assert_eq!(
                sql, scan,
                "parity broken for vector_idx={target}: SQL={sql:?} scan={scan:?}"
            );
        }
    }

    /// P2: journal mode must be read back after setting — WAL if the
    /// filesystem supports it, otherwise the rollback journal, but never a
    /// silently-assumed mode.
    #[test]
    fn journal_mode_is_verified() {
        let store = temp_store("journal");
        let mode = store.journal_mode().unwrap().to_ascii_lowercase();
        assert!(
            mode == "wal" || mode == "delete",
            "unexpected journal mode {mode:?}"
        );
        if mode == "wal" {
            // Round-trip still works under WAL.
            let rec = DocumentRecord {
                id: "wal_probe".into(),
                shard_idx: 0,
                vector_idx: 7,
                metadata: None,
                document: Some("ok".into()),
                deleted: false,
                created_at: 1.0,
                updated_at: 1.0,
            };
            store.upsert(&rec).unwrap();
            assert_eq!(store.get("wal_probe").unwrap().unwrap().vector_idx, 7);
        }
    }
}
