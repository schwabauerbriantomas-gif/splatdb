//! Abstract metadata store trait — database-agnostic persistence.
//!
//! Implement this trait to support any structured database:
//! SQLite, PostgreSQL, MySQL, MongoDB, JSON-file, etc.

use std::collections::HashMap;

/// A single document record.
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct DocumentRecord {
    pub id: String,
    pub shard_idx: i64,
    pub vector_idx: i64,
    pub metadata: Option<serde_json::Value>,
    pub document: Option<String>,
    pub deleted: bool,
    pub created_at: f64,
    pub updated_at: f64,
}

/// Abstract interface for structured document storage.
///
/// All metadata operations go through this trait.
/// Binary vector storage (shards) is handled separately by `VectorStore`.
pub trait MetadataStore: Send + Sync {
    /// Insert or replace a document record.
    fn upsert(
        &self,
        record: &DocumentRecord,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>>;

    /// Get a non-deleted document by ID.
    fn get(
        &self,
        doc_id: &str,
    ) -> Result<Option<DocumentRecord>, Box<dyn std::error::Error + Send + Sync>>;

    /// Soft-delete (mark as deleted).
    fn soft_delete(&self, doc_id: &str) -> Result<bool, Box<dyn std::error::Error + Send + Sync>>;

    /// Hard-delete (permanent removal).
    fn hard_delete(&self, doc_id: &str) -> Result<bool, Box<dyn std::error::Error + Send + Sync>>;

    /// List all document IDs, optionally including soft-deleted ones.
    fn list_ids(
        &self,
        include_deleted: bool,
    ) -> Result<Vec<String>, Box<dyn std::error::Error + Send + Sync>>;

    /// Count documents, optionally including soft-deleted ones.
    fn count(
        &self,
        include_deleted: bool,
    ) -> Result<usize, Box<dyn std::error::Error + Send + Sync>>;

    /// Find the first non-deleted document with the given `vector_idx`.
    ///
    /// Default: generic O(N) scan in `list_ids` order. Backends with an index
    /// on `vector_idx` (e.g. SQLite `idx_vector`) override this with an
    /// indexed lookup — see `SqliteMetadataStore::find_by_vector_idx` and its
    /// parity test.
    ///
    /// On duplicated `vector_idx` values the winner is the first matching
    /// non-deleted record in `list_ids` order (for SQLite: ascending `id`,
    /// because the covering index on the TEXT PRIMARY KEY yields rows in
    /// lexicographic `id` order).
    fn find_by_vector_idx(
        &self,
        vector_idx: i64,
    ) -> Result<Option<DocumentRecord>, Box<dyn std::error::Error + Send + Sync>> {
        let ids = self.list_ids(true)?;
        for id in ids {
            if let Some(record) = self.get(&id)? {
                if record.vector_idx == vector_idx && !record.deleted {
                    return Ok(Some(record));
                }
            }
        }
        Ok(None)
    }

    /// Flush/sync pending writes (for databases with write buffers).
    fn flush(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        Ok(())
    }

    /// Human-readable name of the backend.
    fn backend_name(&self) -> &'static str;
}

// ---- Convenience: convert DocumentRecord to the HashMap format used by the CLI ----

impl DocumentRecord {
    pub fn to_hashmap(&self) -> HashMap<String, serde_json::Value> {
        let mut map = HashMap::new();
        map.insert("id".into(), serde_json::json!(&self.id));
        map.insert("shard_idx".into(), serde_json::json!(self.shard_idx));
        map.insert("vector_idx".into(), serde_json::json!(self.vector_idx));
        if let Some(ref meta) = self.metadata {
            map.insert("metadata".into(), meta.clone());
        }
        if let Some(ref doc) = self.document {
            map.insert("document".into(), serde_json::json!(doc));
        }
        map.insert("deleted".into(), serde_json::json!(self.deleted));
        map
    }
}
