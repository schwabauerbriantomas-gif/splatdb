//! Graph Splat — graph structures over Gaussian splats.
//! Hybrid store combining vector search with knowledge graph.
//! Ported from splatdb Python.

use std::collections::{HashMap, HashSet, VecDeque};

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

    /// Add a document node.
    pub fn add_document(&mut self, text: &str, embedding: &[f32]) -> usize {
        let id = self.next_id;
        self.next_id += 1;
        self.nodes.insert(id, GraphNode {
            id, node_type: NodeType::Document, content: text.to_string(),
            embedding: embedding.to_vec(), alpha: 1.0, kappa: 10.0,
        });
        self.type_index.entry(NodeType::Document).or_default().insert(id);
        id
    }

    /// Add an entity node (with name deduplication).
    pub fn add_entity(&mut self, name: &str, embedding: &[f32], __entity_type: &str) -> usize {
        let key = name.to_lowercase();
        if let Some(&id) = self.entity_name_index.get(&key) { return id; }

        let id = self.next_id;
        self.next_id += 1;
        self.nodes.insert(id, GraphNode {
            id, node_type: NodeType::Entity, content: name.to_string(),
            embedding: embedding.to_vec(), alpha: 0.8, kappa: 15.0,
        });
        self.type_index.entry(NodeType::Entity).or_default().insert(id);
        self.entity_name_index.insert(key, id);
        id
    }

    /// Add a relation edge between two nodes.
    pub fn add_relation(&mut self, source_id: usize, target_id: usize, relation_type: &str, weight: f64) -> bool {
        if !self.nodes.contains_key(&source_id) || !self.nodes.contains_key(&target_id) {
            return false;
        }
        let edge_idx = self.edges.len();
        self.edges.push(GraphEdge {
            source_id, target_id, relation_type: relation_type.into(), weight,
        });
        self.outgoing.entry(source_id).or_default().push(edge_idx);
        self.incoming.entry(target_id).or_default().push(edge_idx);
        true
    }

    /// Get a node by ID.
    pub fn get_node(&self, id: usize) -> Option<&GraphNode> { self.nodes.get(&id) }

    /// Get outgoing edges for a node.
    pub fn get_outgoing(&self, id: usize) -> Vec<&GraphEdge> {
        self.outgoing.get(&id).map(|indices| {
            indices.iter().filter_map(|&i| self.edges.get(i)).collect()
        }).unwrap_or_default()
    }

    /// Get neighbors of a node.
    pub fn get_neighbors(&self, id: usize) -> Vec<usize> {
        let mut neighbors = HashSet::new();
        if let Some(indices) = self.outgoing.get(&id) {
            for &i in indices { if let Some(e) = self.edges.get(i) { neighbors.insert(e.target_id); } }
        }
        if let Some(indices) = self.incoming.get(&id) {
            for &i in indices { if let Some(e) = self.edges.get(i) { neighbors.insert(e.source_id); } }
        }
        neighbors.into_iter().collect()
    }

    /// Search entities by embedding similarity.
    pub fn search_entities(&self, query_emb: &[f32], k: usize) -> Vec<&GraphNode> {
        let entity_ids = self.type_index.get(&NodeType::Entity)
            .map(|s| s.iter().copied().collect::<Vec<_>>()).unwrap_or_default();
        if entity_ids.is_empty() { return Vec::new(); }

        let mut scored: Vec<(usize, f32)> = entity_ids.iter().filter_map(|&id| {
            self.nodes.get(&id).map(|n| (id, cosine_sim(query_emb, &n.embedding)))
        }).collect();
        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        scored.into_iter().filter_map(|(id, _)| self.nodes.get(&id)).collect()
    }

    /// BFS traversal from a start node.
    pub fn traverse(&self, start_id: usize, max_depth: usize) -> Vec<usize> {
        if !self.nodes.contains_key(&start_id) { return Vec::new(); }
        let mut visited = HashSet::new();
        visited.insert(start_id);
        let mut queue = VecDeque::new();
        queue.push_back((start_id, 0usize));
        let mut result = vec![start_id];

        while let Some((current_id, depth)) = queue.pop_front() {
            if depth >= max_depth { continue; }
            for edge in self.get_outgoing(current_id) {
                if !visited.contains(&edge.target_id) {
                    visited.insert(edge.target_id);
                    result.push(edge.target_id);
                    queue.push_back((edge.target_id, depth + 1));
                }
            }
        }
        result
    }

    /// Hybrid search: vector similarity + graph context boost.
    pub fn hybrid_search(&self, query_emb: &[f32], k: usize) -> Vec<HybridResult> {
        let doc_ids = self.type_index.get(&NodeType::Document)
            .map(|s| s.iter().copied().collect::<Vec<_>>()).unwrap_or_default();
        if doc_ids.is_empty() { return Vec::new(); }

        let mut scored: Vec<(usize, f64)> = doc_ids.iter().filter_map(|&id| {
            self.nodes.get(&id).map(|node| {
                let sim = cosine_sim(query_emb, &node.embedding) as f64;
                let graph_boost = (self.get_outgoing(id).len() as f64 * 0.05).min(0.2);
                (id, sim + graph_boost)
            })
        }).collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);
        scored.into_iter().filter_map(|(id, score)| {
            self.nodes.get(&id).map(|n| HybridResult {
                node_id: id, content: n.content.clone(), score, node_type: n.node_type,
            })
        }).collect()
    }

    /// Graph statistics.
    pub fn get_stats(&self) -> GraphStats {
        GraphStats {
            total_nodes: self.nodes.len(),
            total_edges: self.edges.len(),
            documents: self.type_index.get(&NodeType::Document).map(|s| s.len()).unwrap_or(0),
            entities: self.type_index.get(&NodeType::Entity).map(|s| s.len()).unwrap_or(0),
            concepts: self.type_index.get(&NodeType::Concept).map(|s| s.len()).unwrap_or(0),
        }
    }

    /// Number of nodes in the graph.
    pub fn n_nodes(&self) -> usize { self.nodes.len() }
    /// Number of edges in the graph.
    pub fn n_edges(&self) -> usize { self.edges.len() }
}

impl Default for GaussianGraphStore { fn default() -> Self { Self::new() } }

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

fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(&x, &y)| x * y).sum();
    let na: f32 = a.iter().map(|v| v * v).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|v| v * v).sum::<f32>().sqrt();
    let d = na * nb;
    if d < 1e-8 { 0.0 } else { dot / d }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add_document_and_entity() {
        let mut g = GaussianGraphStore::new();
        let d = g.add_document("Hello", &[1.0, 0.0]);
        let e = g.add_entity("Greeting", &[1.0, 0.0], "concept");
        assert_ne!(d, e);
        assert_eq!(g.n_nodes(), 2);
    }

    #[test]
    fn test_entity_dedup() {
        let mut g = GaussianGraphStore::new();
        let a = g.add_entity("Same", &[1.0], "t");
        let b = g.add_entity("same", &[1.0], "t");
        assert_eq!(a, b);
        assert_eq!(g.n_nodes(), 1);
    }

    #[test]
    fn test_relation() {
        let mut g = GaussianGraphStore::new();
        let d = g.add_document("Doc", &[1.0, 0.0]);
        let e = g.add_entity("Ent", &[0.0, 1.0], "t");
        assert!(g.add_relation(d, e, "MENTIONS", 0.9));
        assert!(!g.add_relation(999, e, "X", 1.0));
        assert_eq!(g.n_edges(), 1);
        assert_eq!(g.get_neighbors(d), vec![e]);
    }

    #[test]
    fn test_search_entities() {
        let mut g = GaussianGraphStore::new();
        g.add_entity("Cat", &[1.0, 0.0], "animal");
        g.add_entity("Dog", &[0.9, 0.1], "animal");
        g.add_entity("Car", &[0.0, 1.0], "vehicle");
        let r = g.search_entities(&[1.0, 0.0], 2);
        assert!(r.len() >= 1);
        assert_eq!(r[0].content, "Cat");
    }

    #[test]
    fn test_traverse() {
        let mut g = GaussianGraphStore::new();
        let n1 = g.add_document("D1", &[1.0, 0.0]);
        let n2 = g.add_entity("E1", &[0.0, 1.0], "t");
        let n3 = g.add_entity("E2", &[0.5, 0.5], "t");
        g.add_relation(n1, n2, "MENTIONS", 1.0);
        g.add_relation(n2, n3, "RELATED", 0.8);
        let visited = g.traverse(n1, 2);
        assert!(visited.contains(&n2));
        assert!(visited.contains(&n3));
    }

    #[test]
    fn test_hybrid_search() {
        let mut g = GaussianGraphStore::new();
        let d = g.add_document("Python programming", &[1.0, 0.0, 0.0]);
        let e = g.add_entity("Python", &[0.9, 0.1, 0.0], "lang");
        g.add_relation(d, e, "MENTIONS", 0.9);
        let r = g.hybrid_search(&[1.0, 0.0, 0.0], 5);
        assert!(!r.is_empty());
        assert!(r[0].score > 0.0);
    }

    #[test]
    fn test_stats() {
        let mut g = GaussianGraphStore::new();
        g.add_document("D", &[1.0]);
        g.add_entity("E", &[0.0], "t");
        let s = g.get_stats();
        assert_eq!(s.documents, 1);
        assert_eq!(s.entities, 1);
        assert_eq!(s.total_edges, 0);
    }
}
