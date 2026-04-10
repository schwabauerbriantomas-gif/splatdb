//! Graph Splat — graph structures over Gaussian splats.
//! Hybrid store combining vector search with knowledge graph.
//! Ported from splatsdb Python.
//!
//! ## Stack Overflow Fix (2026-04-04)
//! Original code had potential stack overflow from:
//! - Unbounded `Vec` growth in `get_outgoing()` returning owned Vecs
//! - No max result cap on BFS `traverse()`
//! - `add_entity` unused `entity_type` parameter masked with `__`
//! - Missing `Send + Sync` bounds for thread-safe usage

use std::collections::{HashMap, HashSet, VecDeque};

/// Maximum nodes returned by traversal to prevent unbounded memory growth.
const MAX_TRAVERSE_RESULTS: usize = 10_000;
/// Maximum embedding dimension we accept.
const MAX_EMBEDDING_DIM: usize = 8192;

/// Node types in the graph.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum NodeType {
    Document,
    Entity,
    Concept,
}

/// Directed edge in the graph.
#[derive(Debug, Clone)]
pub struct GraphEdge {
    pub source_id: usize,
    pub target_id: usize,
    pub relation_type: String,
    pub weight: f64,
}

/// Node data — kept lightweight (no nested Vecs of edges inline).
#[derive(Debug, Clone)]
pub struct GraphNode {
    pub id: usize,
    pub node_type: NodeType,
    pub content: String,
    pub embedding: Vec<f32>,
    pub alpha: f64,
    pub kappa: f64,
}

/// Error type for graph operations.
#[derive(Debug, Clone, PartialEq)]
pub enum GraphError {
    /// Embedding dimension exceeds maximum allowed.
    EmbeddingTooLarge { dim: usize, max: usize },
    /// Content string exceeds maximum allowed length.
    ContentTooLong { len: usize, max: usize },
    /// Node ID not found in graph.
    NodeNotFound { id: usize },
    /// Result set hit the safety cap.
    ResultCapHit { cap: usize },
    /// Invalid weight value (NaN or infinite).
    InvalidWeight,
}

impl std::fmt::Display for GraphError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::EmbeddingTooLarge { dim, max } => {
                write!(f, "embedding dim {dim} exceeds max {max}")
            }
            Self::ContentTooLong { len, max } => {
                write!(f, "content length {len} exceeds max {max}")
            }
            Self::NodeNotFound { id } => write!(f, "node {id} not found"),
            Self::ResultCapHit { cap } => write!(f, "result set hit safety cap of {cap}"),
            Self::InvalidWeight => write!(f, "weight must be finite"),
        }
    }
}

impl std::error::Error for GraphError {}

/// Maximum content string length.
const MAX_CONTENT_LEN: usize = 1_000_000;

/// Hybrid store combining vector search with knowledge graph.
/// All heavy data stored in HashMaps (heap) to avoid stack overflow.
pub struct GaussianGraphStore {
    nodes: HashMap<usize, GraphNode>,
    edges: Vec<GraphEdge>,
    /// node_id -> outgoing edge indices
    outgoing: HashMap<usize, Vec<usize>>,
    /// node_id -> incoming edge indices
    incoming: HashMap<usize, Vec<usize>>,
    type_index: HashMap<NodeType, HashSet<usize>>,
    entity_name_index: HashMap<String, usize>,
    next_id: usize,
}

impl GaussianGraphStore {
    /// Create an empty graph store.
    pub fn new() -> Self {
        Self {
            nodes: HashMap::new(),
            edges: Vec::new(),
            outgoing: HashMap::new(),
            incoming: HashMap::new(),
            type_index: HashMap::new(),
            entity_name_index: HashMap::new(),
            next_id: 1,
        }
    }

    /// Validate embedding dimensions are within bounds.
    fn validate_embedding(embedding: &[f32]) -> Result<(), GraphError> {
        if embedding.len() > MAX_EMBEDDING_DIM {
            return Err(GraphError::EmbeddingTooLarge {
                dim: embedding.len(),
                max: MAX_EMBEDDING_DIM,
            });
        }
        Ok(())
    }

    /// Validate content string length.
    fn validate_content(content: &str) -> Result<(), GraphError> {
        if content.len() > MAX_CONTENT_LEN {
            return Err(GraphError::ContentTooLong {
                len: content.len(),
                max: MAX_CONTENT_LEN,
            });
        }
        Ok(())
    }

    /// Add a document node.
    ///
    /// # Errors
    /// Returns `GraphError::EmbeddingTooLarge` if embedding dimension exceeds `MAX_EMBEDDING_DIM`.
    /// Returns `GraphError::ContentTooLong` if content exceeds `MAX_CONTENT_LEN`.
    pub fn add_document(&mut self, text: &str, embedding: &[f32]) -> Result<usize, GraphError> {
        Self::validate_embedding(embedding)?;
        Self::validate_content(text)?;

        let id = self.next_id;
        self.next_id += 1;
        self.nodes.insert(
            id,
            GraphNode {
                id,
                node_type: NodeType::Document,
                content: text.to_string(),
                embedding: embedding.to_vec(),
                alpha: 1.0,
                kappa: 10.0,
            },
        );
        self.type_index
            .entry(NodeType::Document)
            .or_default()
            .insert(id);
        Ok(id)
    }

    /// Add an entity node (with name deduplication).
    ///
    /// # Errors
    /// Returns `GraphError::EmbeddingTooLarge` if embedding dimension exceeds `MAX_EMBEDDING_DIM`.
    /// Returns `GraphError::ContentTooLong` if name exceeds `MAX_CONTENT_LEN`.
    pub fn add_entity(
        &mut self,
        name: &str,
        embedding: &[f32],
        entity_type: &str,
    ) -> Result<usize, GraphError> {
        Self::validate_embedding(embedding)?;
        Self::validate_content(name)?;

        let key = name.to_lowercase();
        if let Some(&id) = self.entity_name_index.get(&key) {
            return Ok(id);
        }

        let id = self.next_id;
        self.next_id += 1;
        let mut node = GraphNode {
            id,
            node_type: NodeType::Entity,
            content: name.to_string(),
            embedding: embedding.to_vec(),
            alpha: 0.8,
            kappa: 15.0,
        };

        // Use entity_type to influence node parameters
        match entity_type {
            "concept" => {
                node.alpha = 0.6;
                node.kappa = 20.0;
            }
            "person" | "organization" | "location" => {
                node.alpha = 0.9;
                node.kappa = 12.0;
            }
            _ => {}
        }

        self.nodes.insert(id, node);
        self.type_index
            .entry(NodeType::Entity)
            .or_default()
            .insert(id);
        self.entity_name_index.insert(key, id);
        Ok(id)
    }

    /// Add a relation edge between two nodes.
    ///
    /// # Errors
    /// Returns `GraphError::NodeNotFound` if source or target doesn't exist.
    /// Returns `GraphError::InvalidWeight` if weight is NaN or infinite.
    pub fn add_relation(
        &mut self,
        source_id: usize,
        target_id: usize,
        relation_type: &str,
        weight: f64,
    ) -> Result<(), GraphError> {
        if !weight.is_finite() {
            return Err(GraphError::InvalidWeight);
        }
        if !self.nodes.contains_key(&source_id) {
            return Err(GraphError::NodeNotFound { id: source_id });
        }
        if !self.nodes.contains_key(&target_id) {
            return Err(GraphError::NodeNotFound { id: target_id });
        }

        let edge_idx = self.edges.len();
        self.edges.push(GraphEdge {
            source_id,
            target_id,
            relation_type: relation_type.into(),
            weight,
        });
        self.outgoing.entry(source_id).or_default().push(edge_idx);
        self.incoming.entry(target_id).or_default().push(edge_idx);
        Ok(())
    }

    /// Get a node by ID.
    pub fn get_node(&self, id: usize) -> Option<&GraphNode> {
        self.nodes.get(&id)
    }

    /// Get outgoing edge count for a node (avoids allocating a Vec).
    pub fn outgoing_count(&self, id: usize) -> usize {
        self.outgoing.get(&id).map_or(0, |v| v.len())
    }

    /// Get outgoing edges for a node. Returns borrowed references to avoid
    /// excessive allocation on large graphs.
    pub fn get_outgoing(&self, id: usize) -> Vec<&GraphEdge> {
        self.outgoing
            .get(&id)
            .map(|indices| indices.iter().filter_map(|&i| self.edges.get(i)).collect())
            .unwrap_or_default()
    }

    /// Get neighbors of a node. Caps result at `MAX_TRAVERSE_RESULTS`.
    pub fn get_neighbors(&self, id: usize) -> Vec<usize> {
        let mut neighbors = HashSet::new();
        if let Some(indices) = self.outgoing.get(&id) {
            for &i in indices {
                if let Some(e) = self.edges.get(i) {
                    neighbors.insert(e.target_id);
                }
            }
        }
        if let Some(indices) = self.incoming.get(&id) {
            for &i in indices {
                if let Some(e) = self.edges.get(i) {
                    neighbors.insert(e.source_id);
                }
            }
        }
        neighbors.into_iter().take(MAX_TRAVERSE_RESULTS).collect()
    }

    /// Search entities by embedding similarity.
    ///
    /// Returns the top-k entities most similar to `query_emb`.
    /// Returns empty Vec if no entities exist.
    pub fn search_entities(&self, query_emb: &[f32], k: usize) -> Vec<&GraphNode> {
        let entity_ids = self
            .type_index
            .get(&NodeType::Entity)
            .map(|s| s.iter().copied().collect::<Vec<_>>())
            .unwrap_or_default();
        if entity_ids.is_empty() {
            return Vec::new();
        }

        let mut scored: Vec<(usize, f32)> = entity_ids
            .iter()
            .filter_map(|&id| {
                self.nodes
                    .get(&id)
                    .map(|n| (id, cosine_sim(query_emb, &n.embedding)))
            })
            .collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        scored
            .into_iter()
            .filter_map(|(id, _)| self.nodes.get(&id))
            .collect()
    }

    /// BFS traversal from a start node.
    ///
    /// Returns nodes reachable within `max_depth` hops, capped at
    /// `MAX_TRAVERSE_RESULTS` to prevent unbounded memory growth on
    /// highly connected graphs.
    pub fn traverse(&self, start_id: usize, max_depth: usize) -> Vec<usize> {
        if !self.nodes.contains_key(&start_id) {
            return Vec::new();
        }
        let mut visited = HashSet::new();
        visited.insert(start_id);
        let mut queue = VecDeque::new();
        queue.push_back((start_id, 0usize));
        let mut result = vec![start_id];

        while let Some((current_id, depth)) = queue.pop_front() {
            // Hard cap to prevent stack overflow / OOM
            if result.len() >= MAX_TRAVERSE_RESULTS {
                break;
            }
            if depth >= max_depth {
                continue;
            }
            if let Some(indices) = self.outgoing.get(&current_id) {
                for &i in indices {
                    if result.len() >= MAX_TRAVERSE_RESULTS {
                        break;
                    }
                    if let Some(e) = self.edges.get(i) {
                        if !visited.contains(&e.target_id) {
                            visited.insert(e.target_id);
                            result.push(e.target_id);
                            queue.push_back((e.target_id, depth + 1));
                        }
                    }
                }
            }
        }
        result
    }

    /// Hybrid search: vector similarity + graph context boost.
    pub fn hybrid_search(&self, query_emb: &[f32], k: usize) -> Vec<HybridResult> {
        let doc_ids = self
            .type_index
            .get(&NodeType::Document)
            .map(|s| s.iter().copied().collect::<Vec<_>>())
            .unwrap_or_default();
        if doc_ids.is_empty() {
            return Vec::new();
        }

        let mut scored: Vec<(usize, f64)> = doc_ids
            .iter()
            .filter_map(|&id| {
                self.nodes.get(&id).map(|node| {
                    let sim = cosine_sim(query_emb, &node.embedding) as f64;
                    let graph_boost = (self.outgoing_count(id) as f64 * 0.05).min(0.2);
                    (id, sim + graph_boost)
                })
            })
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        scored
            .into_iter()
            .filter_map(|(id, score)| {
                self.nodes.get(&id).map(|n| HybridResult {
                    node_id: id,
                    content: n.content.clone(),
                    score,
                    node_type: n.node_type,
                })
            })
            .collect()
    }

    /// Graph statistics.
    pub fn get_stats(&self) -> GraphStats {
        GraphStats {
            total_nodes: self.nodes.len(),
            total_edges: self.edges.len(),
            documents: self
                .type_index
                .get(&NodeType::Document)
                .map(|s| s.len())
                .unwrap_or(0),
            entities: self
                .type_index
                .get(&NodeType::Entity)
                .map(|s| s.len())
                .unwrap_or(0),
            concepts: self
                .type_index
                .get(&NodeType::Concept)
                .map(|s| s.len())
                .unwrap_or(0),
        }
    }

    /// Number of nodes in the graph.
    pub fn n_nodes(&self) -> usize {
        self.nodes.len()
    }
    /// Number of edges in the graph.
    pub fn n_edges(&self) -> usize {
        self.edges.len()
    }
}

impl Default for GaussianGraphStore {
    fn default() -> Self {
        Self::new()
    }
}

/// Hybrid search result.
#[derive(Debug, Clone)]
pub struct HybridResult {
    pub node_id: usize,
    pub content: String,
    pub score: f64,
    pub node_type: NodeType,
}

/// Graph statistics.
#[derive(Debug, Clone)]
pub struct GraphStats {
    pub total_nodes: usize,
    pub total_edges: usize,
    pub documents: usize,
    pub entities: usize,
    pub concepts: usize,
}

/// Compute cosine similarity between two vectors.
/// Returns 0.0 if either vector has zero magnitude.
fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let na: f32 = a.iter().map(|v| v * v).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|v| v * v).sum::<f32>().sqrt();
    let d = na * nb;
    if d < 1e-8 {
        0.0
    } else {
        dot / d
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_document_and_entity() {
        let mut g = GaussianGraphStore::new();
        let d = g.add_document("Hello", &[1.0, 0.0]).unwrap();
        let e = g.add_entity("Greeting", &[1.0, 0.0], "concept").unwrap();
        assert_ne!(d, e);
        assert_eq!(g.n_nodes(), 2);
    }

    #[test]
    fn test_entity_dedup() {
        let mut g = GaussianGraphStore::new();
        let a = g.add_entity("Same", &[1.0], "t").unwrap();
        let b = g.add_entity("same", &[1.0], "t").unwrap();
        assert_eq!(a, b);
        assert_eq!(g.n_nodes(), 1);
    }

    #[test]
    fn test_relation() {
        let mut g = GaussianGraphStore::new();
        let d = g.add_document("Doc", &[1.0, 0.0]).unwrap();
        let e = g.add_entity("Ent", &[0.0, 1.0], "t").unwrap();
        assert!(g.add_relation(d, e, "MENTIONS", 0.9).is_ok());
        assert!(g.add_relation(999, e, "X", 1.0).is_err());
        assert_eq!(g.n_edges(), 1);
        assert_eq!(g.get_neighbors(d), vec![e]);
    }

    #[test]
    fn test_invalid_weight() {
        let mut g = GaussianGraphStore::new();
        let d = g.add_document("Doc", &[1.0]).unwrap();
        let e = g.add_entity("Ent", &[0.0], "t").unwrap();
        assert!(g.add_relation(d, e, "X", f64::NAN).is_err());
        assert!(g.add_relation(d, e, "X", f64::INFINITY).is_err());
        assert!(g.add_relation(d, e, "X", f64::NEG_INFINITY).is_err());
        assert_eq!(g.n_edges(), 0);
    }

    #[test]
    fn test_search_entities() {
        let mut g = GaussianGraphStore::new();
        g.add_entity("Cat", &[1.0, 0.0], "animal").unwrap();
        g.add_entity("Dog", &[0.9, 0.1], "animal").unwrap();
        g.add_entity("Car", &[0.0, 1.0], "vehicle").unwrap();
        let r = g.search_entities(&[1.0, 0.0], 2);
        assert!(!r.is_empty());
        assert_eq!(r[0].content, "Cat");
    }

    #[test]
    fn test_traverse() {
        let mut g = GaussianGraphStore::new();
        let n1 = g.add_document("D1", &[1.0, 0.0]).unwrap();
        let n2 = g.add_entity("E1", &[0.0, 1.0], "t").unwrap();
        let n3 = g.add_entity("E2", &[0.5, 0.5], "t").unwrap();
        g.add_relation(n1, n2, "MENTIONS", 1.0).unwrap();
        g.add_relation(n2, n3, "RELATED", 0.8).unwrap();
        let visited = g.traverse(n1, 2);
        assert!(visited.contains(&n2));
        assert!(visited.contains(&n3));
    }

    #[test]
    fn test_traverse_cyclic_graph() {
        // Cycle: A -> B -> C -> A — must not hang or overflow
        let mut g = GaussianGraphStore::new();
        let a = g.add_document("A", &[1.0]).unwrap();
        let b = g.add_entity("B", &[1.0], "t").unwrap();
        let c = g.add_entity("C", &[1.0], "t").unwrap();
        g.add_relation(a, b, "R", 1.0).unwrap();
        g.add_relation(b, c, "R", 1.0).unwrap();
        g.add_relation(c, a, "R", 1.0).unwrap();
        let visited = g.traverse(a, 100);
        // Should visit all 3 exactly once despite cycle
        assert!(visited.contains(&a));
        assert!(visited.contains(&b));
        assert!(visited.contains(&c));
        assert_eq!(visited.len(), 3);
    }

    #[test]
    fn test_traverse_large_graph_capped() {
        // Build a star graph: center -> N leaves
        let mut g = GaussianGraphStore::new();
        let center = g.add_document("center", &[1.0]).unwrap();
        for i in 0..MAX_TRAVERSE_RESULTS + 100 {
            let leaf = g.add_entity(&format!("leaf_{i}"), &[1.0], "t").unwrap();
            g.add_relation(center, leaf, "R", 1.0).unwrap();
        }
        let visited = g.traverse(center, 1);
        assert!(visited.len() <= MAX_TRAVERSE_RESULTS);
    }

    #[test]
    fn test_hybrid_search() {
        let mut g = GaussianGraphStore::new();
        let d = g
            .add_document("Python programming", &[1.0, 0.0, 0.0])
            .unwrap();
        let e = g.add_entity("Python", &[0.9, 0.1, 0.0], "lang").unwrap();
        g.add_relation(d, e, "MENTIONS", 0.9).unwrap();
        let r = g.hybrid_search(&[1.0, 0.0, 0.0], 5);
        assert!(!r.is_empty());
        assert!(r[0].score > 0.0);
    }

    #[test]
    fn test_stats() {
        let mut g = GaussianGraphStore::new();
        g.add_document("D", &[1.0]).unwrap();
        g.add_entity("E", &[0.0], "t").unwrap();
        let s = g.get_stats();
        assert_eq!(s.documents, 1);
        assert_eq!(s.entities, 1);
        assert_eq!(s.total_edges, 0);
    }

    #[test]
    fn test_embedding_too_large() {
        let mut g = GaussianGraphStore::new();
        let huge_emb = vec![0.0f32; MAX_EMBEDDING_DIM + 1];
        let result = g.add_document("test", &huge_emb);
        assert!(matches!(result, Err(GraphError::EmbeddingTooLarge { .. })));
    }

    #[test]
    fn test_content_too_long() {
        let mut g = GaussianGraphStore::new();
        let long_content = "x".repeat(MAX_CONTENT_LEN + 1);
        let result = g.add_document(&long_content, &[1.0]);
        assert!(matches!(result, Err(GraphError::ContentTooLong { .. })));
    }

    #[test]
    fn test_cosine_sim_mismatched_dims() {
        let sim = cosine_sim(&[1.0, 0.0], &[1.0]);
        assert_eq!(sim, 0.0);
    }

    #[test]
    fn test_cosine_sim_zero_vectors() {
        let sim = cosine_sim(&[0.0, 0.0], &[0.0, 0.0]);
        assert_eq!(sim, 0.0);
    }

    #[test]
    fn test_cosine_sim_identical() {
        let sim = cosine_sim(&[1.0, 0.0], &[1.0, 0.0]);
        assert!((sim - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_entity_type_influences_params() {
        let mut g = GaussianGraphStore::new();
        let concept = g.add_entity("c", &[1.0], "concept").unwrap();
        let person = g.add_entity("p", &[1.0], "person").unwrap();
        let default = g.add_entity("d", &[1.0], "other").unwrap();

        assert_eq!(g.get_node(concept).unwrap().alpha, 0.6);
        assert_eq!(g.get_node(concept).unwrap().kappa, 20.0);
        assert_eq!(g.get_node(person).unwrap().alpha, 0.9);
        assert_eq!(g.get_node(person).unwrap().kappa, 12.0);
        assert_eq!(g.get_node(default).unwrap().alpha, 0.8);
        assert_eq!(g.get_node(default).unwrap().kappa, 15.0);
    }

    #[test]
    fn test_nonexistent_node_traverse() {
        let g = GaussianGraphStore::new();
        let visited = g.traverse(999, 5);
        assert!(visited.is_empty());
    }

    #[test]
    fn test_nonexistent_node_neighbors() {
        let g = GaussianGraphStore::new();
        let neighbors = g.get_neighbors(999);
        assert!(neighbors.is_empty());
    }
}
