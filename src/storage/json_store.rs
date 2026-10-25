//! JSON-file implementation of MetadataStore.
//!
//! Stores all documents in a single JSON file. Suitable for:
//! - Development/testing
//! - Small deployments (<10K documents)
//! - Environments without SQLite available

use super::metadata_store::{DocumentRecord, MetadataStore};
use parking_lot::RwLock;
use std::collections::HashMap;
use std::path::PathBuf;

/// JSON-file backed metadata store.
pub struct JsonMetadataStore {
    path: PathBuf,
    data: RwLock<HashMap<String, DocumentRecord>>,
}

impl JsonMetadataStore {
    pub fn new(path: PathBuf) -> Result<Self, Box<dyn std::error::Error + Send + Sync>> {
        std::fs::create_dir_all(path.parent().unwrap_or(&path))?;
        let data: HashMap<String, DocumentRecord> = if path.exists() {
            // Never panic and never silently discard: if the file cannot be
            // loaded (unreadable, invalid UTF-8, corrupt JSON) quarantine it
            // under a `.corrupt-<timestamp>` sibling and start empty.
            match std::fs::read(&path)
                .map_err(|e| format!("read failed: {}", e))
                .and_then(|bytes| {
                    String::from_utf8(bytes).map_err(|e| format!("invalid UTF-8: {}", e))
                })
                .and_then(|content| {
                    serde_json::from_str(&content).map_err(|e| format!("corrupt JSON: {}", e))
                }) {
                Ok(data) => data,
                Err(reason) => {
                    let ts = std::time::SystemTime::now()
                        .duration_since(std::time::UNIX_EPOCH)
                        .unwrap_or_default()
                        .as_millis();
                    let quarantine = path.with_file_name(format!(
                        "{}.corrupt-{}",
                        path.file_name()
                            .map(|n| n.to_string_lossy().into_owned())
                            .unwrap_or_else(|| "metadata.json".to_string()),
                        ts
                    ));
                    match std::fs::rename(&path, &quarantine) {
                        Ok(()) => eprintln!(
                            "[json_store] WARNING: metadata file {} is unusable ({}); \
                             quarantined to {} and starting empty",
                            path.display(),
                            reason,
                            quarantine.display()
                        ),
                        Err(e) => eprintln!(
                            "[json_store] WARNING: metadata file {} is unusable ({}); \
                             could not quarantine it ({}); starting empty and it will be \
                             replaced on next persist",
                            path.display(),
                            reason,
                            e
                        ),
                    }
                    HashMap::new()
                }
            }
        } else {
            HashMap::new()
        };
        Ok(Self {
            path,
            data: RwLock::new(data),
        })
    }

    fn persist(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let data = self.data.read();
        let json = serde_json::to_string_pretty(&*data)?;
        // Atomic (temp + fsync + rename): a crash mid-persist leaves the
        // previous metadata file intact instead of a half-written one.
        super::persistence::atomic_write(&self.path, json.as_bytes())?;
        Ok(())
    }
}

impl MetadataStore for JsonMetadataStore {
    fn upsert(
        &self,
        record: &DocumentRecord,
    ) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        {
            let mut data = self.data.write();
            data.insert(record.id.clone(), record.clone());
        }
        self.persist()
    }

    fn get(
        &self,
        doc_id: &str,
    ) -> Result<Option<DocumentRecord>, Box<dyn std::error::Error + Send + Sync>> {
        let data = self.data.read();
        match data.get(doc_id) {
            Some(r) if !r.deleted => Ok(Some(r.clone())),
            _ => Ok(None),
        }
    }

    fn soft_delete(&self, doc_id: &str) -> Result<bool, Box<dyn std::error::Error + Send + Sync>> {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_secs_f64();
        {
            let mut data = self.data.write();
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
        let mut data = self.data.write();
        let existed = data.remove(doc_id).is_some();
        drop(data);
        if existed {
            self.persist()?;
        }
        Ok(existed)
    }

    fn list_ids(
        &self,
        include_deleted: bool,
    ) -> Result<Vec<String>, Box<dyn std::error::Error + Send + Sync>> {
        let data = self.data.read();
        let ids: Vec<String> = data
            .values()
            .filter(|r| include_deleted || !r.deleted)
            .map(|r| r.id.clone())
            .collect();
        Ok(ids)
    }

    fn count(
        &self,
        include_deleted: bool,
    ) -> Result<usize, Box<dyn std::error::Error + Send + Sync>> {
        let data = self.data.read();
        let count = data
            .values()
            .filter(|r| include_deleted || !r.deleted)
            .count();
        Ok(count)
    }

    fn flush(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        self.persist()
    }

    fn backend_name(&self) -> &'static str {
        "json-file"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::metadata_store::{DocumentRecord, MetadataStore};

    fn temp_dir(tag: &str) -> PathBuf {
        std::env::temp_dir().join(format!(
            "splatdb-json-store-{}-{}-{}",
            tag,
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ))
    }

    fn sample_record(id: &str) -> DocumentRecord {
        DocumentRecord {
            id: id.to_string(),
            shard_idx: 0,
            vector_idx: 0,
            metadata: None,
            document: None,
            deleted: false,
            created_at: 0.0,
            updated_at: 0.0,
        }
    }

    #[test]
    fn corrupt_metadata_is_quarantined_and_store_starts_empty() {
        let dir = temp_dir("corrupt");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("metadata.json");
        std::fs::write(&path, "{ definitely not valid json").unwrap();

        let store = JsonMetadataStore::new(path.clone()).unwrap();
        assert_eq!(store.count(false).unwrap(), 0);

        // Original file was moved aside under a `.corrupt-<timestamp>` name.
        let quarantined: Vec<String> = std::fs::read_dir(&dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().into_owned())
            .filter(|n| n.starts_with("metadata.json.corrupt-"))
            .collect();
        assert_eq!(quarantined.len(), 1, "quarantine files: {:?}", quarantined);
        assert!(!path.exists(), "corrupt file must be moved out of the way");

        // Store remains fully usable afterwards (persist goes through
        // atomic_write and must recreate the file).
        store.upsert(&sample_record("d1")).unwrap();
        assert_eq!(store.count(false).unwrap(), 1);

        let reloaded = JsonMetadataStore::new(path).unwrap();
        assert_eq!(reloaded.count(false).unwrap(), 1);

        std::fs::remove_dir_all(&dir).ok();
    }

    #[test]
    fn valid_metadata_loads_normally() {
        let dir = temp_dir("valid");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("metadata.json");

        let store = JsonMetadataStore::new(path.clone()).unwrap();
        store.upsert(&sample_record("d1")).unwrap();
        store.upsert(&sample_record("d2")).unwrap();

        let reloaded = JsonMetadataStore::new(path).unwrap();
        assert_eq!(reloaded.count(false).unwrap(), 2);
        assert!(reloaded.get("d1").unwrap().is_some());

        std::fs::remove_dir_all(&dir).ok();
    }
}
