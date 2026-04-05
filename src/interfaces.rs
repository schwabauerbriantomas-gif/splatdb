//! VectorIndex trait and BruteForceIndex implementation.

use ndarray::{Array2, ArrayView1};

/// Result of a vector search operation.
#[derive(Clone, Debug)]
pub struct IndexSearchResult {
    pub indices: Vec<usize>,
    pub distances: Vec<f32>,
}

/// Trait for vector search backends.
pub trait VectorIndex: Send + Sync {
    fn build(&mut self, vectors: Array2<f32>);
    fn search(&self, query: ArrayView1<f32>, k: usize) -> IndexSearchResult;
    fn add(&mut self, vectors: Array2<f32>);
    fn remove(&mut self, indices: &[usize]);
    fn n_items(&self) -> usize;
    fn supports_remove(&self) -> bool;
}

/// Brute-force linear scan index — exact search, O(N) per query.
pub struct BruteForceIndex {
    vectors: Option<Array2<f32>>,
    metric: String,
}

impl BruteForceIndex {
    /// New.
    pub fn new(metric: &str) -> Self {
        Self {
            vectors: None,
            metric: metric.to_string(),
        }
    }

    fn top_k_indices(&self, mut scores: Vec<(usize, f32)>, k: usize) -> (Vec<usize>, Vec<f32>) {
        let k = k.min(scores.len());
        if k == 0 {
            return (vec![], vec![]);
        }
        // Partial sort: use select_nth_unstable for O(n) average
        scores.select_nth_unstable_by(k - 1, |a, b| {
            a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
        });
        scores.truncate(k);
        scores.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        let indices: Vec<usize> = scores.iter().map(|(i, _)| *i).collect();
        let distances: Vec<f32> = scores.iter().map(|(_, d)| *d).collect();
        (indices, distances)
    }
}

impl VectorIndex for BruteForceIndex {
    fn build(&mut self, vectors: Array2<f32>) {
        self.vectors = Some(vectors);
    }

    fn search(&self, query: ArrayView1<f32>, k: usize) -> IndexSearchResult {
        let vectors = match &self.vectors {
            Some(v) => v,
            None => {
                return IndexSearchResult {
                    indices: vec![],
                    distances: vec![],
                }
            }
        };
        let n = vectors.nrows();
        let k = k.min(n);
        if k == 0 {
            return IndexSearchResult {
                indices: vec![],
                distances: vec![],
            };
        }

        if self.metric == "cosine" {
            let q_norm = query.dot(&query).sqrt().max(1e-10);
            let scores: Vec<(usize, f32)> = (0..n)
                .map(|i| {
                    let row = vectors.row(i);
                    let r_norm = row.dot(&row).sqrt().max(1e-10);
                    let sim = row.dot(&query) / (r_norm * q_norm);
                    (i, 1.0 - sim)
                })
                .collect();
            let (indices, distances) = self.top_k_indices(scores, k);
            IndexSearchResult { indices, distances }
        } else {
            // Euclidean — use squared distance for ranking, sqrt only top-k
            let scores: Vec<(usize, f32)> = (0..n)
                .map(|i| {
                    let row = vectors.row(i);
                    let diff = &row - &query;
                    (i, diff.dot(&diff))
                })
                .collect();
            let k = k.min(scores.len());
            let mut scores = scores;
            scores.select_nth_unstable_by(k - 1, |a, b| {
                a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
            });
            scores.truncate(k);
            scores.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
            let indices: Vec<usize> = scores.iter().map(|(i, _)| *i).collect();
            let distances: Vec<f32> = scores.iter().map(|(_, d)| d.sqrt()).collect();
            IndexSearchResult { indices, distances }
        }
    }

    fn add(&mut self, vectors: Array2<f32>) {
        match &mut self.vectors {
            Some(existing) => {
                let nrows = existing.nrows() + vectors.nrows();
                let dim = existing.ncols();
                let mut data = Vec::with_capacity(nrows * dim);
                data.extend_from_slice(existing.as_slice().unwrap_or(&[]));
                data.extend_from_slice(vectors.as_slice().unwrap_or(&[]));
                self.vectors = Some(ndarray::Array2::from_shape_vec((nrows, dim), data).unwrap());
            }
            None => self.vectors = Some(vectors),
        }
    }

    fn remove(&mut self, indices: &[usize]) {
        if let Some(vectors) = &self.vectors {
            let remove_set: std::collections::HashSet<usize> = indices.iter().copied().collect();
            let keep: Vec<usize> = (0..vectors.nrows())
                .filter(|i| !remove_set.contains(i))
                .collect();
            if keep.is_empty() {
                self.vectors = None;
            } else {
                let dim = vectors.ncols();
                let mut data = Vec::with_capacity(keep.len() * dim);
                for &old_i in &keep {
                    data.extend_from_slice(vectors.row(old_i).as_slice().unwrap_or(&[]));
                }
                self.vectors = Some(Array2::from_shape_vec((keep.len(), dim), data).unwrap());
            }
        }
    }

    fn n_items(&self) -> usize {
        self.vectors.as_ref().map_or(0, |v| v.nrows())
    }

    fn supports_remove(&self) -> bool {
        true
    }
}

/// Result of index strategy selection.
#[derive(Clone, Debug)]
pub struct IndexSelectionResult {
    pub recommended: String,
    pub n_vectors: usize,
    pub dim: usize,
    pub reason: String,
}

/// Select the best index strategy based on data size.
pub fn select_index_strategy(n_vectors: usize, dim: usize) -> IndexSelectionResult {
    if n_vectors < 15_000 {
        IndexSelectionResult {
            recommended: "bruteforce".into(),
            n_vectors,
            dim,
            reason: format!("{} < 15K -> linear scan sufficient", n_vectors),
        }
    } else if n_vectors < 100_000 {
        IndexSelectionResult {
            recommended: "hrm2".into(),
            n_vectors,
            dim,
            reason: format!("{} in [15K, 100K) -> HRM2 hierarchical", n_vectors),
        }
    } else {
        IndexSelectionResult {
            recommended: "hnsw".into(),
            n_vectors,
            dim,
            reason: format!("{} >= 100K -> HNSW for scalability", n_vectors),
        }
    }
}
