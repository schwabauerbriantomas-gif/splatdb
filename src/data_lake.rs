//! Data Lake for M2M — dataset management and batch operations.
//!
//! Simple dataset storage for training and evaluation.
//! Ported from m2m-vector-search Python.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io::{Read, Write};
use std::path::Path;

/// A dataset entry in the data lake.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DatasetEntry {
    pub id: String,
    pub name: String,
    pub n_vectors: usize,
    pub dim: usize,
    pub description: Option<String>,
}

/// Data Lake — manages datasets for training and evaluation.
pub struct DataLake {
    entries: HashMap<String, DatasetEntry>,
    base_path: String,
}

impl DataLake {
    /// New.
    pub fn new(base_path: &str) -> Self {
        Self {
            entries: HashMap::new(),
            base_path: base_path.to_string(),
        }
    }

    /// Register a dataset in the lake.
    pub fn register(&mut self, id: &str, name: &str, n_vectors: usize, dim: usize, description: Option<&str>) {
        self.entries.insert(id.to_string(), DatasetEntry {
            id: id.to_string(),
            name: name.to_string(),
            n_vectors,
            dim,
            description: description.map(|s| s.to_string()),
        });
    }

    /// List all registered datasets.
    pub fn list(&self) -> Vec<&DatasetEntry> {
        self.entries.values().collect()
    }

    /// Get a dataset entry by ID.
    pub fn get(&self, id: &str) -> Option<&DatasetEntry> {
        self.entries.get(id)
    }

    /// Remove a dataset registration.
    pub fn remove(&mut self, id: &str) -> bool {
        self.entries.remove(id).is_some()
    }

    /// Save data lake index to disk.
    pub fn save(&self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let path = Path::new(&self.base_path).join("data_lake_index.json");
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let json = serde_json::to_string_pretty(&self.entries)?;
        let mut file = std::fs::File::create(&path)?;
        file.write_all(json.as_bytes())?;
        Ok(())
    }

    /// Load data lake index from disk.
    pub fn load(&mut self) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
        let path = Path::new(&self.base_path).join("data_lake_index.json");
        if !path.exists() {
            return Ok(());
        }
        let mut file = std::fs::File::open(&path)?;
        let mut content = String::new();
        file.read_to_string(&mut content)?;
        self.entries = serde_json::from_str(&content)?;
        Ok(())
    }

    /// Total number of registered datasets.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Check if data lake is empty.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_register_and_list() {
        let mut lake = DataLake::new("./test_lake");
        lake.register("ds1", "Dataset 1", 1000, 384, Some("Test dataset"));
        lake.register("ds2", "Dataset 2", 5000, 768, None);
        assert_eq!(lake.len(), 2);
        let list = lake.list();
        assert_eq!(list.len(), 2);
    }

    #[test]
    fn test_get_and_remove() {
        let mut lake = DataLake::new("./test_lake");
        lake.register("ds1", "Test", 100, 64, None);
        assert!(lake.get("ds1").is_some());
        assert!(lake.remove("ds1"));
        assert!(lake.get("ds1").is_none());
    }
}
