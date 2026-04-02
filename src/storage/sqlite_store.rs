//! SQLite implementation of MetadataStore.

use super::metadata_store::{DocumentRecord, MetadataStore};
use rusqlite::{Connection, params};
use std::path::PathBuf;

/// SQLite-backed metadata store.
pub struct SqliteMetadataStore {
    db_path: PathBuf,
}

impl SqliteMetadataStore {
    pub fn new(db_path: PathBuf) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        std::fs::create_dir_all(db_path.parent().unwrap_or(&db_path))?;
        let conn = Connection::open(&db_path)?;
        conn.execute_batch(
            "PRAGMA journal_mode=WAL;
             CREATE TABLE IF NOT EXISTS documents (
                 id TEXT PRIMARY KEY,
                 shard_idx INTEGER,
                 vector_idx INTEGER,
                 metadata TEXT,
                 document TEXT,
                 deleted INTEGER DEFAULT 0,
                 created_at REAL,
                 updated_at REAL
             );
             CREATE INDEX IF NOT EXISTS idx_deleted ON documents(deleted);"
        )?;
        drop(conn);
        Ok(Self { db_path })
    }

    fn conn(&self) -> Result<Connection, rusqlite::Error> {
        Connection::open(&self.db_path)
    }

    fn row_to_record(&self, row: &rusqlite::Row) -> rusqlite::Result<DocumentRecord> {
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
}

impl MetadataStore for SqliteMetadataStore {
    fn upsert(&self, record: &DocumentRecord) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let conn = self.conn()?;
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

    fn get(&self, doc_id: &str) -> Result<Option<DocumentRecord>, Box<dyn std::error::Error + Send + Sync>> {
        let conn = self.conn()?;
        let mut stmt = conn.prepare(
            "SELECT id, shard_idx, vector_idx, metadata, document, deleted, created_at, updated_at
             FROM documents WHERE id=?1 AND deleted=0"
        )?;
        let result = stmt.query_row(params![doc_id], |row| self.row_to_record(row));
        match result {
            Ok(r) => Ok(Some(r)),
            Err(rusqlite::Error::QueryReturnedNoRows) => Ok(None),
            Err(e) => Err(e.into()),
        }
    }

    fn soft_delete(&self, doc_id: &str) -> Result<bool, Box<dyn std::error::Error + Send + Sync>> {
        let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_secs_f64();
        let conn = self.conn()?;
        let affected = conn.execute(
            "UPDATE documents SET deleted=1, updated_at=?1 WHERE id=?2",
            params![now, doc_id],
        )?;
        Ok(affected > 0)
    }

    fn hard_delete(&self, doc_id: &str) -> Result<bool, Box<dyn std::error::Error + Send + Sync>> {
        let conn = self.conn()?;
        let affected = conn.execute("DELETE FROM documents WHERE id=?1", params![doc_id])?;
        Ok(affected > 0)
    }

    fn list_ids(&self, include_deleted: bool) -> Result<Vec<String>, Box<dyn std::error::Error + Send + Sync>> {
        let conn = self.conn()?;
        let sql = if include_deleted {
            "SELECT id FROM documents"
        } else {
            "SELECT id FROM documents WHERE deleted=0"
        };
        let mut stmt = conn.prepare(sql)?;
        let ids: Vec<String> = stmt.query_map([], |row| row.get(0))?.filter_map(|r| r.ok()).collect();
        Ok(ids)
    }

    fn count(&self, include_deleted: bool) -> Result<usize, Box<dyn std::error::Error + Send + Sync>> {
        let conn = self.conn()?;
        let sql = if include_deleted {
            "SELECT COUNT(*) FROM documents"
        } else {
            "SELECT COUNT(*) FROM documents WHERE deleted=0"
        };
        let count: usize = conn.query_row(sql, [], |row| row.get(0))?;
        Ok(count)
    }

    fn flush(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let conn = self.conn()?;
        conn.execute_batch("PRAGMA wal_checkpoint(TRUNCATE);")?;
        Ok(())
    }

    fn backend_name(&self) -> &'static str { "sqlite" }
}
