//! Edge API — REST endpoints for edge node operations.
//! Request/response types and handler logic (framework-agnostic).
//! Ready to plug into actix-web, axum, or warp.
//! Ported from splatsdb Python.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

// ─── Request Types ───

/// Create collection request.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct CreateCollectionRequest {
    pub name: String,
    pub dimension: usize,
    pub mode: String, // "edge", "standard", "ebm"
    pub enable_ebm: bool,
    pub storage_path: Option<String>,
    pub metadata_schema: Option<HashMap<String, String>>,
}

/// Insert vectors request.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct InsertVectorsRequest {
    pub vectors: Vec<Vec<f32>>,
    pub ids: Option<Vec<String>>,
    pub metadata: Option<Vec<HashMap<String, serde_json::Value>>>,
    pub documents: Option<Vec<String>>,
}

/// Update vector request.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct UpdateVectorRequest {
    pub vector: Option<Vec<f32>>,
    pub metadata: Option<HashMap<String, serde_json::Value>>,
    pub document: Option<String>,
    pub upsert: bool,
}

/// Search request.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct SearchRequest {
    pub vector: Vec<f32>,
    pub k: usize,
    pub include_metadata: bool,
    pub include_documents: bool,
    pub include_energy: bool,
    pub filter: Option<HashMap<String, serde_json::Value>>,
    pub options: Option<HashMap<String, serde_json::Value>>,
}

/// Query request (text-based).
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct QueryRequest {
    pub query: String,
    pub k: usize,
    pub include_metadata: bool,
}

/// Energy query request.
#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct EnergyRequest {
    pub vector: Vec<f32>,
    pub temperature: f64,
    pub steps: usize,
}

// ─── Response Types ───

/// API response wrapper.
#[derive(Debug, Clone, Serialize)]
pub struct ApiResponse<T: Serialize> {
    pub success: bool,
    pub data: Option<T>,
    pub error: Option<String>,
    pub request_id: String,
}

/// Search result item.
#[derive(Debug, Clone, Serialize)]
pub struct SearchResultItem {
    pub id: String,
    pub score: f32,
    pub metadata: Option<HashMap<String, serde_json::Value>>,
    pub document: Option<String>,
    pub energy: Option<f64>,
}

/// Search response.
#[derive(Debug, Clone, Serialize)]
pub struct SearchResponse {
    pub results: Vec<SearchResultItem>,
    pub total: usize,
    pub latency_ms: f64,
}

/// Collection info.
#[derive(Debug, Clone, Serialize)]
pub struct CollectionInfo {
    pub name: String,
    pub dimension: usize,
    pub mode: String,
    pub vector_count: usize,
    pub enable_ebm: bool,
}

/// Health response.
#[derive(Debug, Clone, Serialize)]
pub struct HealthResponse {
    pub status: String,
    pub version: String,
    pub uptime_secs: f64,
    pub memory_mb: f64,
}

/// Stats response.
#[derive(Debug, Clone, Serialize)]
pub struct StatsResponse {
    pub collections: usize,
    pub total_vectors: usize,
    pub total_queries: usize,
    pub avg_latency_ms: f64,
    pub memory_mb: f64,
}

// ─── Edge API Handler ───

/// Edge API: handles REST operations for a vector database edge node.
pub struct EdgeApi {
    collections: HashMap<String, CollectionState>,
    total_queries: usize,
    total_latency_ms: f64,
    start_time: f64,
    version: String,
}

#[derive(Debug)]
struct CollectionState {
    name: String,
    dimension: usize,
    mode: String,
    enable_ebm: bool,
    vectors: Vec<Vec<f32>>,
    vector_ids: Vec<String>,
    vector_metadata: Vec<HashMap<String, serde_json::Value>>,
    vector_documents: Vec<Option<String>>,
}

impl CollectionState {
    fn vector_count(&self) -> usize {
        self.vectors.len()
    }

    /// Simple brute-force search.
    fn search(&self, query: &[f32], k: usize) -> Vec<(usize, f32)> {
        let k = k.min(self.vectors.len());
        if k == 0 || self.dimension == 0 {
            return Vec::new();
        }
        if query.len() != self.dimension {
            return Vec::new();
        }

        let mut scored: Vec<(usize, f32)> = self
            .vectors
            .iter()
            .enumerate()
            .map(|(i, vec)| {
                let sim = cosine_sim(query, vec);
                (i, sim)
            })
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        scored
    }
}

impl EdgeApi {
    pub fn new(version: &str) -> Self {
        Self {
            collections: HashMap::new(),
            total_queries: 0,
            total_latency_ms: 0.0,
            start_time: now_secs(),
            version: version.to_string(),
        }
    }

    // ─── Health & Stats ───

    pub fn health(&self) -> HealthResponse {
        HealthResponse {
            status: "ok".into(),
            version: self.version.clone(),
            uptime_secs: now_secs() - self.start_time,
            memory_mb: 0.0,
        }
    }

    pub fn stats(&self) -> StatsResponse {
        let total_vectors: usize = self.collections.values().map(|c| c.vector_count()).sum();
        StatsResponse {
            collections: self.collections.len(),
            total_vectors,
            total_queries: self.total_queries,
            avg_latency_ms: if self.total_queries > 0 {
                self.total_latency_ms / self.total_queries as f64
            } else {
                0.0
            },
            memory_mb: 0.0,
        }
    }

    // ─── Collections CRUD ───

    pub fn list_collections(&self) -> Vec<CollectionInfo> {
        self.collections
            .values()
            .map(|c| CollectionInfo {
                name: c.name.clone(),
                dimension: c.dimension,
                mode: c.mode.clone(),
                vector_count: c.vector_count(),
                enable_ebm: c.enable_ebm,
            })
            .collect()
    }

    pub fn create_collection(
        &mut self,
        req: &CreateCollectionRequest,
    ) -> Result<CollectionInfo, String> {
        if self.collections.contains_key(&req.name) {
            return Err(format!("Collection '{}' already exists", req.name));
        }
        if req.dimension < 1 || req.dimension > 65536 {
            return Err("dimension must be between 1 and 65536".into());
        }
        let collection = CollectionState {
            name: req.name.clone(),
            dimension: req.dimension,
            mode: req.mode.clone(),
            enable_ebm: req.enable_ebm,
            vectors: Vec::new(),
            vector_ids: Vec::new(),
            vector_metadata: Vec::new(),
            vector_documents: Vec::new(),
        };
        self.collections.insert(req.name.clone(), collection);
        Ok(CollectionInfo {
            name: req.name.clone(),
            dimension: req.dimension,
            mode: req.mode.clone(),
            vector_count: 0,
            enable_ebm: req.enable_ebm,
        })
    }

    pub fn get_collection(&self, name: &str) -> Result<CollectionInfo, String> {
        self.collections
            .get(name)
            .map(|c| CollectionInfo {
                name: c.name.clone(),
                dimension: c.dimension,
                mode: c.mode.clone(),
                vector_count: c.vector_count(),
                enable_ebm: c.enable_ebm,
            })
            .ok_or_else(|| format!("Collection '{}' not found", name))
    }

    pub fn delete_collection(&mut self, name: &str) -> Result<(), String> {
        self.collections
            .remove(name)
            .map(|_| ())
            .ok_or_else(|| format!("Collection '{}' not found", name))
    }

    // ─── Vectors CRUD ───

    pub fn insert_vectors(
        &mut self,
        collection: &str,
        req: &InsertVectorsRequest,
    ) -> Result<usize, String> {
        let col = self
            .collections
            .get_mut(collection)
            .ok_or_else(|| format!("Collection '{}' not found", collection))?;
        let start = now_secs();
        for (i, vec) in req.vectors.iter().enumerate() {
            if vec.len() != col.dimension {
                return Err(format!(
                    "Vector {} has dimension {}, expected {}",
                    i,
                    vec.len(),
                    col.dimension
                ));
            }
            col.vectors.push(vec.clone());
            col.vector_ids.push(
                req.ids
                    .as_ref()
                    .and_then(|ids| ids.get(i).cloned())
                    .unwrap_or_else(|| format!("vec_{}", col.vectors.len())),
            );
            col.vector_metadata.push(
                req.metadata
                    .as_ref()
                    .and_then(|m| m.get(i).cloned())
                    .unwrap_or_default(),
            );
            col.vector_documents
                .push(req.documents.as_ref().and_then(|d| d.get(i).cloned()));
        }
        let inserted = req.vectors.len();
        self.total_queries += 1;
        self.total_latency_ms += (now_secs() - start) * 1000.0;
        Ok(inserted)
    }

    pub fn get_vector(&self, collection: &str, id: &str) -> Result<SearchResultItem, String> {
        let col = self
            .collections
            .get(collection)
            .ok_or_else(|| format!("Collection '{}' not found", collection))?;
        let idx = col
            .vector_ids
            .iter()
            .position(|vid| vid == id)
            .ok_or_else(|| format!("Vector '{}' not found", id))?;
        Ok(SearchResultItem {
            id: col.vector_ids[idx].clone(),
            score: 1.0,
            metadata: Some(col.vector_metadata[idx].clone()),
            document: col.vector_documents[idx].clone(),
            energy: None,
        })
    }

    pub fn update_vector(
        &mut self,
        collection: &str,
        id: &str,
        req: &UpdateVectorRequest,
    ) -> Result<(), String> {
        let col = self
            .collections
            .get_mut(collection)
            .ok_or_else(|| format!("Collection '{}' not found", collection))?;
        let idx = col
            .vector_ids
            .iter()
            .position(|vid| vid == id)
            .ok_or_else(|| format!("Vector '{}' not found", id))?;

        if let Some(ref vec) = req.vector {
            if vec.len() != col.dimension {
                return Err("Dimension mismatch".into());
            }
            col.vectors[idx] = vec.clone();
        }
        if let Some(ref meta) = req.metadata {
            col.vector_metadata[idx] = meta.clone();
        }
        if let Some(ref doc) = req.document {
            col.vector_documents[idx] = Some(doc.clone());
        }
        Ok(())
    }

    pub fn delete_vector(&mut self, collection: &str, id: &str) -> Result<(), String> {
        let col = self
            .collections
            .get_mut(collection)
            .ok_or_else(|| format!("Collection '{}' not found", collection))?;
        let idx = col
            .vector_ids
            .iter()
            .position(|vid| vid == id)
            .ok_or_else(|| format!("Vector '{}' not found", id))?;
        col.vectors.remove(idx);
        col.vector_ids.remove(idx);
        col.vector_metadata.remove(idx);
        col.vector_documents.remove(idx);
        Ok(())
    }

    // ─── Search ───

    pub fn search(
        &mut self,
        collection: &str,
        req: &SearchRequest,
    ) -> Result<SearchResponse, String> {
        let col = self
            .collections
            .get(collection)
            .ok_or_else(|| format!("Collection '{}' not found", collection))?;
        let start = now_secs();
        let scored = col.search(&req.vector, req.k);
        let results: Vec<SearchResultItem> = scored
            .iter()
            .map(|(idx, score)| SearchResultItem {
                id: col.vector_ids[*idx].clone(),
                score: *score,
                metadata: if req.include_metadata {
                    Some(col.vector_metadata[*idx].clone())
                } else {
                    None
                },
                document: if req.include_documents {
                    col.vector_documents[*idx].clone()
                } else {
                    None
                },
                energy: None,
            })
            .collect();
        let total = results.len();
        self.total_queries += 1;
        self.total_latency_ms += (now_secs() - start) * 1000.0;
        Ok(SearchResponse {
            results,
            total,
            latency_ms: (now_secs() - start) * 1000.0,
        })
    }
}

fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let na: f32 = a.iter().map(|&v| v * v).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|&v| v * v).sum::<f32>().sqrt();
    let denom = na * nb;
    if denom < 1e-8 {
        0.0
    } else {
        dot / denom
    }
}

fn now_secs() -> f64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_collection_crud() {
        let mut api = EdgeApi::new("2.1.0");
        let info = api
            .create_collection(&CreateCollectionRequest {
                name: "test".into(),
                dimension: 3,
                mode: "standard".into(),
                enable_ebm: false,
                storage_path: None,
                metadata_schema: None,
            })
            .unwrap();
        assert_eq!(info.vector_count, 0);

        assert!(api
            .create_collection(&CreateCollectionRequest {
                name: "test".into(),
                dimension: 3,
                mode: "standard".into(),
                enable_ebm: false,
                storage_path: None,
                metadata_schema: None,
            })
            .is_err());

        let list = api.list_collections();
        assert_eq!(list.len(), 1);

        api.delete_collection("test").unwrap();
        assert!(api.list_collections().is_empty());
    }

    #[test]
    fn test_insert_and_search() {
        let mut api = EdgeApi::new("2.1.0");
        api.create_collection(&CreateCollectionRequest {
            name: "test".into(),
            dimension: 3,
            mode: "standard".into(),
            enable_ebm: false,
            storage_path: None,
            metadata_schema: None,
        })
        .unwrap();

        api.insert_vectors(
            "test",
            &InsertVectorsRequest {
                vectors: vec![vec![1.0, 0.0, 0.0], vec![0.0, 1.0, 0.0]],
                ids: Some(vec!["v1".into(), "v2".into()]),
                metadata: None,
                documents: None,
            },
        )
        .unwrap();

        let resp = api
            .search(
                "test",
                &SearchRequest {
                    vector: vec![1.0, 0.0, 0.0],
                    k: 1,
                    include_metadata: false,
                    include_documents: false,
                    include_energy: false,
                    filter: None,
                    options: None,
                },
            )
            .unwrap();
        assert_eq!(resp.results[0].id, "v1");
        assert!(resp.results[0].score > 0.99);
    }

    #[test]
    fn test_vector_update_delete() {
        let mut api = EdgeApi::new("2.1.0");
        api.create_collection(&CreateCollectionRequest {
            name: "test".into(),
            dimension: 2,
            mode: "standard".into(),
            enable_ebm: false,
            storage_path: None,
            metadata_schema: None,
        })
        .unwrap();
        api.insert_vectors(
            "test",
            &InsertVectorsRequest {
                vectors: vec![vec![1.0, 0.0]],
                ids: Some(vec!["v1".into()]),
                metadata: None,
                documents: None,
            },
        )
        .unwrap();

        api.update_vector(
            "test",
            "v1",
            &UpdateVectorRequest {
                vector: Some(vec![0.0, 1.0]),
                metadata: None,
                document: None,
                upsert: false,
            },
        )
        .unwrap();

        api.delete_vector("test", "v1").unwrap();
        let stats = api.stats();
        assert_eq!(stats.total_vectors, 0);
    }

    #[test]
    fn test_health_stats() {
        let api = EdgeApi::new("2.1.0");
        let health = api.health();
        assert_eq!(health.status, "ok");
        let stats = api.stats();
        assert_eq!(stats.collections, 0);
    }
}
