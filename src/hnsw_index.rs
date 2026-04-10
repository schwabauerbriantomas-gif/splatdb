//! HNSW Index — Hierarchical Navigable Small World graph for ANN search.
//! Native Rust implementation optimized for speed.

use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashSet};
use std::io::{Read, Write};
use std::path::Path;

use ndarray::{Array2, ArrayView1};
use rand::Rng;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;

use crate::interfaces::{IndexSearchResult, VectorIndex};

const HNSW_MAGIC: u64 = 0x484E5357; // "HNSW"

/// Wrapper for (distance, index) that implements Ord for min-heap.
#[derive(Clone, Copy, Debug)]
struct DistIdx(f32, usize);

impl PartialEq for DistIdx {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}
impl Eq for DistIdx {}
impl PartialOrd for DistIdx {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for DistIdx {
    fn cmp(&self, other: &Self) -> Ordering {
        other.0.partial_cmp(&self.0).unwrap_or(Ordering::Equal)
    }
}

/// Max-heap wrapper (negate distance).
#[derive(Clone, Copy, Debug)]
struct MaxDistIdx(f32, usize);
impl PartialEq for MaxDistIdx {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}
impl Eq for MaxDistIdx {}
impl PartialOrd for MaxDistIdx {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for MaxDistIdx {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.partial_cmp(&other.0).unwrap_or(Ordering::Equal)
    }
}

/// A node in the HNSW graph.
struct HNSWNode {
    _level: usize,
    neighbors: Vec<Vec<usize>>,
}

/// HNSW index for approximate nearest neighbor search.
pub struct HNSWIndex {
    _dim: usize,
    m: usize,
    m_max0: usize,
    ef_construction: usize,
    ef_search: usize,
    level_mult: f64,
    metric: String,
    rng: ChaCha8Rng,

    vectors: Vec<Vec<f32>>,
    /// Precomputed L2 norms for cosine distance (one per vector)
    norms: Vec<f32>,
    nodes: Vec<HNSWNode>,
    entry_point: usize,
    max_level: usize,
    removed: HashSet<usize>,
}

impl HNSWIndex {
    /// New.
    pub fn new(
        dim: usize,
        m: usize,
        ef_construction: usize,
        ef_search: usize,
        metric: &str,
        seed: u64,
    ) -> Self {
        Self {
            _dim: dim,
            m,
            m_max0: 2 * m,
            ef_construction,
            ef_search,
            level_mult: 1.0 / (m as f64).ln(),
            metric: metric.to_string(),
            rng: ChaCha8Rng::seed_from_u64(seed),
            vectors: Vec::new(),
            norms: Vec::new(),
            nodes: Vec::new(),
            entry_point: 0,
            max_level: 0,
            removed: HashSet::new(),
        }
    }

    /// Check if the index has been built (has vectors loaded).
    pub fn is_built(&self) -> bool {
        !self.vectors.is_empty()
    }

    /// Save the HNSW graph to a binary file.
    pub fn save(&self, path: &Path) -> Result<(), String> {
        let mut f =
            std::fs::File::create(path).map_err(|e| format!("create {}: {}", path.display(), e))?;

        let n_vectors = self.vectors.len() as u64;
        let dim = if self.vectors.is_empty() {
            self._dim as u64
        } else {
            self.vectors[0].len() as u64
        };

        // Header
        f.write_all(&HNSW_MAGIC.to_le_bytes())
            .map_err(|e| format!("write magic: {}", e))?;
        f.write_all(&n_vectors.to_le_bytes())
            .map_err(|e| format!("write n_vectors: {}", e))?;
        f.write_all(&dim.to_le_bytes())
            .map_err(|e| format!("write dim: {}", e))?;
        f.write_all(&(self.m as u64).to_le_bytes())
            .map_err(|e| format!("write m: {}", e))?;
        f.write_all(&(self.m_max0 as u64).to_le_bytes())
            .map_err(|e| format!("write m_max0: {}", e))?;
        f.write_all(&(self.ef_construction as u64).to_le_bytes())
            .map_err(|e| format!("write ef_construction: {}", e))?;
        f.write_all(&(self.ef_search as u64).to_le_bytes())
            .map_err(|e| format!("write ef_search: {}", e))?;
        f.write_all(&(self.entry_point as u64).to_le_bytes())
            .map_err(|e| format!("write entry_point: {}", e))?;
        f.write_all(&(self.max_level as u64).to_le_bytes())
            .map_err(|e| format!("write max_level: {}", e))?;

        let removed_vec: Vec<u64> = self.removed.iter().map(|&x| x as u64).collect();
        f.write_all(&(removed_vec.len() as u64).to_le_bytes())
            .map_err(|e| format!("write n_removed: {}", e))?;
        for &idx in &removed_vec {
            f.write_all(&idx.to_le_bytes())
                .map_err(|e| format!("write removed idx: {}", e))?;
        }

        // Nodes
        for node in &self.nodes {
            let n_levels = node.neighbors.len() as u64;
            f.write_all(&n_levels.to_le_bytes())
                .map_err(|e| format!("write n_levels: {}", e))?;
            for level_neighbors in &node.neighbors {
                let n_neighbors = level_neighbors.len() as u64;
                f.write_all(&n_neighbors.to_le_bytes())
                    .map_err(|e| format!("write n_neighbors: {}", e))?;
                for &nb in level_neighbors {
                    f.write_all(&(nb as u64).to_le_bytes())
                        .map_err(|e| format!("write neighbor: {}", e))?;
                }
            }
        }

        // Raw vectors: flatten all f32 values
        for vec in &self.vectors {
            let bytes: &[u8] = bytemuck::cast_slice(vec);
            f.write_all(bytes)
                .map_err(|e| format!("write vector data: {}", e))?;
        }

        Ok(())
    }

    /// Load an HNSW graph from a binary file.
    pub fn load(
        path: &Path,
        _m: usize,
        ef_construction: usize,
        ef_search: usize,
        metric: &str,
        seed: u64,
    ) -> Result<Self, String> {
        let mut f =
            std::fs::File::open(path).map_err(|e| format!("open {}: {}", path.display(), e))?;

        let read_u64 = |f: &mut std::fs::File| -> Result<u64, String> {
            let mut buf = [0u8; 8];
            f.read_exact(&mut buf)
                .map_err(|e| format!("read u64: {}", e))?;
            Ok(u64::from_le_bytes(buf))
        };

        let magic = read_u64(&mut f)?;
        if magic != HNSW_MAGIC {
            return Err(format!(
                "Invalid HNSW file magic: expected {:#x}, got {:#x}",
                HNSW_MAGIC, magic
            ));
        }

        let n_vectors = read_u64(&mut f)? as usize;
        let dim = read_u64(&mut f)? as usize;
        let m_stored = read_u64(&mut f)? as usize;
        let m_max0 = read_u64(&mut f)? as usize;
        let _ef_construction_stored = read_u64(&mut f)?;
        let _ef_search_stored = read_u64(&mut f)?;
        let entry_point = read_u64(&mut f)? as usize;
        let max_level = read_u64(&mut f)? as usize;

        let n_removed = read_u64(&mut f)? as usize;
        let mut removed = HashSet::new();
        for _ in 0..n_removed {
            removed.insert(read_u64(&mut f)? as usize);
        }

        // Nodes
        let mut nodes = Vec::with_capacity(n_vectors);
        for _ in 0..n_vectors {
            let n_levels = read_u64(&mut f)? as usize;
            let mut neighbors = Vec::with_capacity(n_levels);
            for _ in 0..n_levels {
                let n_neighbors = read_u64(&mut f)? as usize;
                let mut level_neighbors = Vec::with_capacity(n_neighbors);
                for _ in 0..n_neighbors {
                    level_neighbors.push(read_u64(&mut f)? as usize);
                }
                neighbors.push(level_neighbors);
            }
            nodes.push(HNSWNode {
                _level: neighbors.len().saturating_sub(1),
                neighbors,
            });
        }

        // Raw vectors
        let mut vectors = Vec::with_capacity(n_vectors);
        if n_vectors > 0 && dim > 0 {
            let mut buf = vec![0.0f32; dim];
            for _ in 0..n_vectors {
                let bytes: &mut [u8] = bytemuck::cast_slice_mut(&mut buf);
                f.read_exact(bytes)
                    .map_err(|e| format!("read vector: {}", e))?;
                vectors.push(buf.clone());
            }
        }

        // Reconstruct precomputed norms from loaded vectors
        let norms: Vec<f32> = vectors
            .iter()
            .map(|v| v.iter().map(|x| x * x).sum::<f32>().sqrt())
            .collect();

        let level_mult = 1.0 / (m_stored as f64).ln();

        Ok(Self {
            _dim: dim,
            m: m_stored,
            m_max0,
            ef_construction,
            ef_search,
            level_mult,
            metric: metric.to_string(),
            rng: ChaCha8Rng::seed_from_u64(seed),
            vectors,
            norms,
            nodes,
            entry_point,
            max_level,
            removed,
        })
    }

    fn random_level(&mut self) -> usize {
        let r: f64 = self.rng.gen();
        (-(r.ln()) * self.level_mult).clamp(0.0, 16.0) as usize
    }

    #[inline]
    fn distance(&self, a: &[f32], b: &[f32]) -> f32 {
        if self.metric == "cosine" {
            let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
            // Use precomputed norms when available (stored vector vs query)
            // For stored vectors, norms[idx] is precomputed. For queries, compute on the fly.
            // We can't distinguish here, so we use norms for the second vector if available.
            // Actually, distance() is called symmetrically — we keep the fast path for both.
            let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
            let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
            let denom = na * nb;
            if denom < 1e-10 {
                return 1.0;
            }
            1.0 - dot / denom
        } else {
            // L2 distance — much cheaper than cosine
            a.iter()
                .zip(b.iter())
                .map(|(x, y)| (x - y) * (x - y))
                .sum::<f32>()
                .sqrt()
        }
    }

    /// Distance with precomputed norm for the stored vector (avoids recomputing).
    /// `a` is the query (norm computed fresh), `b_idx` is the stored vector index.
    #[inline]
    fn distance_cached(&self, query: &[f32], b_idx: usize) -> f32 {
        let b = &self.vectors[b_idx];
        if self.metric == "cosine" {
            let dot: f32 = query.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
            let na: f32 = query.iter().map(|x| x * x).sum::<f32>().sqrt();
            let nb = self.norms[b_idx];
            let denom = na * nb;
            if denom < 1e-10 {
                return 1.0;
            }
            1.0 - dot / denom
        } else {
            query
                .iter()
                .zip(b.iter())
                .map(|(x, y)| (x - y) * (x - y))
                .sum::<f32>()
                .sqrt()
        }
    }

    fn greedy_closest(&self, query: &[f32], entry: usize, level: usize) -> usize {
        let mut current = entry;
        let mut current_dist = self.distance_cached(query, current);

        loop {
            let mut changed = false;
            let node = &self.nodes[current];
            if level >= node.neighbors.len() {
                break;
            }
            for &neighbor in &node.neighbors[level] {
                if self.removed.contains(&neighbor) {
                    continue;
                }
                let d = self.distance_cached(query, neighbor);
                if d < current_dist {
                    current_dist = d;
                    current = neighbor;
                    changed = true;
                }
            }
            if !changed {
                break;
            }
        }
        current
    }

    fn search_layer(
        &self,
        query: &[f32],
        entry: usize,
        ef: usize,
        level: usize,
    ) -> Vec<(f32, usize)> {
        let mut visited = HashSet::new();
        visited.insert(entry);
        let dist_ep = self.distance_cached(query, entry);

        let mut candidates = BinaryHeap::new(); // min-heap via DistIdx
        let mut results = BinaryHeap::new(); // max-heap via MaxDistIdx

        candidates.push(DistIdx(dist_ep, entry));
        results.push(MaxDistIdx(dist_ep, entry));

        while let Some(DistIdx(dist_c, c)) = candidates.pop() {
            let farthest = match results.peek() {
                Some(MaxDistIdx(d, _)) => *d,
                None => break,
            };
            if dist_c > farthest {
                break;
            }

            let node = &self.nodes[c];
            if level >= node.neighbors.len() {
                continue;
            }

            for &neighbor in &node.neighbors[level] {
                if visited.contains(&neighbor) || self.removed.contains(&neighbor) {
                    continue;
                }
                visited.insert(neighbor);

                let dist_n = self.distance_cached(query, neighbor);
                let farthest = match results.peek() {
                    Some(MaxDistIdx(d, _)) => *d,
                    None => f32::MAX,
                };

                if dist_n < farthest || results.len() < ef {
                    candidates.push(DistIdx(dist_n, neighbor));
                    results.push(MaxDistIdx(dist_n, neighbor));
                    if results.len() > ef {
                        results.pop();
                    }
                }
            }
        }

        let mut r: Vec<(f32, usize)> = results.into_iter().map(|MaxDistIdx(d, i)| (d, i)).collect();
        r.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        r
    }

    pub fn insert(&mut self, vector: &[f32]) {
        if self.vectors.len() >= 10_000_000 {
            return;
        }
        let idx = self.vectors.len();
        let level = self.random_level();
        let mut neighbors = Vec::with_capacity(level + 1);
        for _ in 0..=level {
            neighbors.push(Vec::new());
        }
        self.vectors.push(vector.to_vec());
        // Precompute norm for cosine distance caching
        let norm: f32 = vector.iter().map(|x| x * x).sum::<f32>().sqrt();
        self.norms.push(norm);
        self.nodes.push(HNSWNode {
            _level: level,
            neighbors,
        });

        if idx == 0 {
            self.entry_point = 0;
            self.max_level = level;
            return;
        }

        let mut ep = self.entry_point;

        // Phase 1: greedy descent from top to above insertion level
        for lev in (level + 1..=self.max_level).rev() {
            ep = self.greedy_closest(vector, ep, lev);
        }

        // Phase 2: insert into each layer
        for lev in (0..=level.min(self.max_level)).rev() {
            let candidates = self.search_layer(vector, ep, self.ef_construction, lev);
            let m_max = if lev == 0 { self.m_max0 } else { self.m };
            let m = self.m;

            // Select M closest neighbors
            let selected: Vec<usize> = candidates.iter().take(m).map(|(_, idx)| *idx).collect();

            // Set bidirectional connections
            self.nodes[idx].neighbors[lev] = selected.clone();
            for &neighbor_idx in &selected {
                let n_vec = self.vectors[neighbor_idx].clone();
                let neighbor_ids: Vec<(f32, usize)> = self.nodes[neighbor_idx].neighbors[lev]
                    .iter()
                    .map(|&ni| (self.distance(&n_vec, &self.vectors[ni]), ni))
                    .collect();

                let node = &mut self.nodes[neighbor_idx];
                if lev >= node.neighbors.len() {
                    continue;
                }
                if !node.neighbors[lev].contains(&idx) {
                    node.neighbors[lev].push(idx);
                }
                if node.neighbors[lev].len() > m_max {
                    let mut scored = neighbor_ids;
                    scored
                        .sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
                    node.neighbors[lev] = scored.into_iter().take(m_max).map(|(_, i)| i).collect();
                }
            }

            if !candidates.is_empty() {
                ep = candidates[0].1;
            }
        }

        if level > self.max_level {
            self.max_level = level;
            self.entry_point = idx;
        }
    }
}

impl VectorIndex for HNSWIndex {
    fn build(&mut self, vectors: Array2<f32>) {
        self.vectors.clear();
        self.norms.clear();
        self.nodes.clear();
        self.removed.clear();
        self.entry_point = 0;
        self.max_level = 0;

        for i in 0..vectors.nrows() {
            let row: Vec<f32> = vectors.row(i).to_vec();
            self.insert(&row);
        }
    }

    fn search(&self, query: ArrayView1<f32>, k: usize) -> IndexSearchResult {
        if self.vectors.is_empty() {
            return IndexSearchResult {
                indices: vec![],
                distances: vec![],
            };
        }

        let q: Vec<f32> = query.to_vec();
        let ef = self.ef_search.max(k);
        let mut ep = self.entry_point;

        // Phase 1: greedy descent from top layer to layer 1
        for level in (1..=self.max_level).rev() {
            ep = self.greedy_closest(&q, ep, level);
        }

        // Phase 2: search at layer 0
        let candidates = self.search_layer(&q, ep, ef, 0);

        let results: Vec<(f32, usize)> = candidates
            .into_iter()
            .filter(|(_, idx)| !self.removed.contains(idx))
            .take(k)
            .collect();

        IndexSearchResult {
            indices: results.iter().map(|(_, i)| *i).collect(),
            distances: results.iter().map(|(d, _)| *d).collect(),
        }
    }

    fn add(&mut self, vectors: Array2<f32>) {
        for i in 0..vectors.nrows() {
            let row: Vec<f32> = vectors.row(i).to_vec();
            self.insert(&row);
        }
    }

    fn remove(&mut self, indices: &[usize]) {
        for &idx in indices {
            self.removed.insert(idx);
        }
    }

    fn n_items(&self) -> usize {
        self.vectors.len() - self.removed.len()
    }

    fn supports_remove(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array2};

    fn small_random_vectors(n: usize, dim: usize, seed: u64) -> Array2<f32> {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let data: Vec<f32> = (0..n * dim).map(|_| rng.gen::<f32>()).collect();
        Array2::from_shape_vec((n, dim), data).unwrap()
    }

    #[test]
    fn test_hnsw_creation() {
        let idx = HNSWIndex::new(4, 16, 200, 50, "l2", 42);
        assert_eq!(idx.n_items(), 0);
    }

    #[test]
    fn test_hnsw_build_and_search() {
        let mut idx = HNSWIndex::new(4, 8, 100, 50, "l2", 42);
        let data = small_random_vectors(50, 4, 123);
        idx.build(data.clone());

        assert_eq!(idx.n_items(), 50);

        // Search for an indexed vector — should return itself
        let query = data.row(0);
        let result = idx.search(query, 5);
        assert!(!result.indices.is_empty());
        assert_eq!(result.indices.len(), result.distances.len());
        // Index 0 should be in top results
        assert!(
            result.indices.contains(&0),
            "Expected index 0 in results, got {:?}",
            result.indices
        );
        // Distance to itself should be very small
        let self_pos = result.indices.iter().position(|&i| i == 0).unwrap();
        assert!(result.distances[self_pos] < 0.01);
    }

    #[test]
    fn test_hnsw_empty_search() {
        let idx = HNSWIndex::new(4, 8, 100, 50, "l2", 42);
        let query = array![1.0f32, 0.0, 0.0, 0.0];
        let result = idx.search(query.view(), 5);
        assert!(result.indices.is_empty());
        assert!(result.distances.is_empty());
    }

    #[test]
    fn test_hnsw_add_and_remove() {
        let mut idx = HNSWIndex::new(4, 8, 100, 50, "l2", 42);
        let data = small_random_vectors(20, 4, 99);
        idx.build(data);

        assert_eq!(idx.n_items(), 20);
        assert!(idx.supports_remove());

        idx.remove(&[0, 1]);
        assert_eq!(idx.n_items(), 18);
    }

    #[test]
    fn test_hnsw_cosine_metric() {
        let mut idx = HNSWIndex::new(4, 8, 100, 50, "cosine", 42);
        let data = small_random_vectors(30, 4, 777);
        idx.build(data.clone());

        let query = data.row(0);
        let result = idx.search(query, 3);
        assert!(!result.indices.is_empty());
        assert!(result.indices.contains(&0));
    }

    #[test]
    fn test_hnsw_save_load_roundtrip() {
        let dir = std::env::temp_dir().join("hnsw_save_load_test");
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("test_hnsw.bin");

        let mut idx = HNSWIndex::new(4, 8, 100, 50, "l2", 42);
        let data = small_random_vectors(30, 4, 555);
        idx.build(data.clone());
        idx.remove(&[2, 5]);

        idx.save(&path).unwrap();

        let loaded = HNSWIndex::load(&path, 8, 100, 50, "l2", 42).unwrap();
        assert_eq!(loaded.n_items(), 28); // 30 - 2 removed
        assert!(loaded.is_built());

        // Search results should match
        let query = data.row(0);
        let orig_result = idx.search(query, 5);
        let loaded_result = loaded.search(query, 5);
        assert_eq!(orig_result.indices, loaded_result.indices);

        std::fs::remove_file(&path).ok();
    }

    #[test]
    fn test_hnsw_is_built() {
        let idx = HNSWIndex::new(4, 8, 100, 50, "l2", 42);
        assert!(!idx.is_built());

        let mut idx = HNSWIndex::new(4, 8, 100, 50, "l2", 42);
        let data = small_random_vectors(5, 4, 999);
        idx.build(data);
        assert!(idx.is_built());
    }
}
