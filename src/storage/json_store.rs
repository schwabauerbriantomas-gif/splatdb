//! JSON-file implementation of MetadataStore.
//!
//! Stores all documents in a single JSON file. Suitable for:
//! - Development/testing
//! - Small deployments (<10K documents)
//! - Environments without SQLite available

use super::metadata_store::{DocumentRecord, MetadataStore};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::RwLock;

/// JSON-file backed metadata store.
pub struct JsonMetadataStore {
    path: PathBuf,
    data: RwLock<HashMap<String, DocumentRecord>>,
}

impl JsonMetadataStore {
    pub fn new(path: PathBuf) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        std::fs::create_dir_all(path.parent().unwrap_or(&path))?;
        let data = if path.exists() {
            let content = std::fs::read_to_string(&path)?;
            serde_json::from_str(&content).unwrap_or_default()
        } else {
            HashMap::new()
        };
        Ok(Self {
            path,
            data: RwLock::new(data),
        })
    }

    fn persist(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let data = self.data.read().expect("metadata store lock poisoned");
        let json = serde_json::to_string_pretty(&*data)?;
        std::fs::write(&self.path, json)?;
        Ok(())
    }
}

impl MetadataStore for JsonMetadataStore {
    fn upsert(&self, record: &DocumentRecord) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        {
            let mut data = self.data.write().expect("metadata store lock poisoned");
            data.insert(record.id.clone(), record.clone());
        }
        self.persist()
    }

    fn get(&self, doc_id: &str) -> Result<Option<DocumentRecord>, Box<dyn std::error::Error + Send + Sync>> {
        let data = self.data.read().expect("metadata store lock poisoned");
        match data.get(doc_id) {
            Some(r) if !r.deleted => Ok(Some(r.clone())),
            _ => Ok(None),
        }
    }

    fn soft_delete(&self, doc_id: &str) -> Result<bool, Box<dyn std::error::Error + Send + Sync>> {
        let now = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH)?.as_secs_f64();
        {
            let mut data = self.data.write().expect("metadata store lock poisoned");
            if let Some(record) = data.get_mut(doc_id) {
                record.deleted = true;
                record.updated_at = now;
            } else {
                return Ok(false);
            }
        }
        self.persist()?;
        Ok(true)
    }

    fn hard_delete(&self, doc_id: &str) -> Result<bool, Box<dyn std::error::Error + Send + Sync>> {
        let mut data = self.data.write().expect("metadata store lock poisoned");
        let existed = data.remove(doc_id).is_some();
        drop(data);
        if existed { self.persist()?; }
        Ok(existed)
    }

    fn list_ids(&self, include_deleted: bool) -> Result<Vec<String>, Box<dyn std::error::Error + Send + Sync>> {
        let data = self.data.read().expect("metadata store lock poisoned");
        let ids: Vec<String> = data.values()
            .filter(|r| include_deleted || !r.deleted)
            .map(|r| r.id.clone())
            .collect();
        Ok(ids)
    }

    fn count(&self, include_deleted: bool) -> Result<usize, Box<dyn std::error::Error + Send + Sync>> {
        let data = self.data.read().expect("metadata store lock poisoned");
        let count = data.values()
            .filter(|r| include_deleted || !r.deleted)
            .count();
        Ok(count)
    }

    fn flush(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.persist()
    }

    fn backend_name(&self) -> &'static str { "json-file" }
}
