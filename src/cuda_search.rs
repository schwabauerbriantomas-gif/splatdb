//! CUDA Brute-Force Search — high-performance k-NN search.
//! CPU implementation using ndarray (fallback when CUDA unavailable).
//! Interface ready for GPU backend via cudarc/cust.
//! Ported from splatsdb Python.

use ndarray::{Array1, Array2, ArrayView1, ArrayView2};

/// Search metric type.
#[derive(Debug, Clone, Copy)]
pub enum Metric {
    Cosine,
    L2,
}

/// Search result.
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub indices: Vec<usize>,
    pub distances: Vec<f32>,
}

/// High-performance brute-force k-NN search.
/// Precomputes norms for efficient cosine/L2 distance calculation.
pub struct BruteForceSearcher {
    index: Array2<f32>,    // [N, D] — stored row-major
    norms_sq: Array1<f64>, // [N] — precomputed ||x_i||^2
    metric: Metric,
}

impl BruteForceSearcher {
    /// Create a new searcher from embeddings.
    pub fn new(embeddings: ArrayView2<f32>, metric: Metric) -> Self {
        let index = embeddings.to_owned();
        let norms_sq = precompute_norms_sq(index.view());
        Self {
            index,
            norms_sq,
            metric,
        }
    }

    /// Number of indexed vectors.
    pub fn n_vectors(&self) -> usize {
        self.index.nrows()
    }

    /// Embedding dimension.
    pub fn dim(&self) -> usize {
        self.index.ncols()
    }

    /// Search for k nearest neighbors.
    pub fn search(&self, query: ArrayView1<f32>, k: usize) -> SearchResult {
        let k = k.min(self.n_vectors());
        if k == 0 {
            return SearchResult {
                indices: vec![],
                distances: vec![],
            };
        }

        match self.metric {
            Metric::Cosine => self.cosine_search(query, k),
            Metric::L2 => self.l2_search(query, k),
        }
    }

    /// Batch search for multiple queries.
    pub fn search_batch(&self, queries: ArrayView2<f32>, k: usize) -> Vec<SearchResult> {
        (0..queries.nrows())
            .map(|i| self.search(queries.row(i), k))
            .collect()
    }

    /// Rebuild index with new embeddings.
    pub fn rebuild(&mut self, embeddings: ArrayView2<f32>) {
        self.index = embeddings.to_owned();
        self.norms_sq = precompute_norms_sq(self.index.view());
    }

    fn cosine_search(&self, query: ArrayView1<f32>, k: usize) -> SearchResult {
        let n = self.n_vectors();
        // Compute dot products with all index vectors
        let dots: Vec<f64> = (0..n)
            .map(|i| {
                let row = self.index.row(i);
                dot_product_f64(query, row)
            })
            .collect();

        let q_norm = query
            .iter()
            .map(|&v| v as f64 * v as f64)
            .sum::<f64>()
            .sqrt();
        let x_norms: Vec<f64> = self.norms_sq.iter().map(|&n| n.sqrt()).collect();

        // Cosine similarity, then convert to distance (1 - sim)
        let scored: Vec<(usize, f32)> = (0..n)
            .map(|i| {
                let denom = q_norm * x_norms[i].max(1e-8);
                let sim = (dots[i] / denom) as f32;
                (i, 1.0 - sim)
            })
            .collect();

        top_k_by_distance(scored, k)
    }

    fn l2_search(&self, query: ArrayView1<f32>, k: usize) -> SearchResult {
        let n = self.n_vectors();
        let q_sq: f64 = query.iter().map(|&v| v as f64 * v as f64).sum();

        let scored: Vec<(usize, f32)> = (0..n)
            .map(|i| {
                let row = self.index.row(i);
                let dot = dot_product_f64(query, row);
                let dist_sq = (q_sq + self.norms_sq[i] - 2.0 * dot).max(0.0);
                (i, dist_sq.sqrt() as f32)
            })
            .collect();

        top_k_by_distance(scored, k)
    }
}

/// Multi-start search with majority voting.
/// Runs multiple perturbed queries and aggregates by vote count.
pub struct MultiStartSearcher {
    searcher: BruteForceSearcher,
    n_starts: usize,
    noise_scale: f32,
}

impl MultiStartSearcher {
    /// New.
    pub fn new(
        embeddings: ArrayView2<f32>,
        n_starts: usize,
        noise_scale: f32,
        metric: Metric,
    ) -> Self {
        Self {
            searcher: BruteForceSearcher::new(embeddings, metric),
            n_starts,
            noise_scale,
        }
    }

    /// Search.
    pub fn search(&self, query: ArrayView1<f32>, k: usize) -> SearchResult {
        let mut vote_counts: std::collections::HashMap<usize, usize> =
            std::collections::HashMap::new();
        let mut vote_dists: std::collections::HashMap<usize, f64> =
            std::collections::HashMap::new();

        for start in 0..self.n_starts {
            let perturbed = if start == 0 {
                query.to_owned()
            } else {
                let mut perturbed = query.to_owned();
                for v in perturbed.iter_mut() {
                    *v += (rand_pseudo(start as u64, v.to_bits()) % 1000) as f32 / 1000.0
                        * self.noise_scale
                        * 2.0
                        - self.noise_scale;
                }
                perturbed
            };

            let result = self.searcher.search(perturbed.view(), k);
            for (i, idx) in result.indices.iter().enumerate() {
                *vote_counts.entry(*idx).or_insert(0) += 1;
                *vote_dists.entry(*idx).or_insert(0.0) += result.distances[i] as f64;
            }
        }

        if vote_counts.is_empty() {
            return SearchResult {
                indices: vec![],
                distances: vec![],
            };
        }

        let mut candidates: Vec<(usize, f64, f32)> = vote_counts
            .into_iter()
            .map(|(idx, count)| {
                let avg_dist = vote_dists[&idx] / count as f64;
                (idx, avg_dist, count as f32)
            })
            .collect();
        // Sort by votes desc, then avg distance asc
        candidates.sort_by(|a, b| {
            b.2.partial_cmp(&a.2)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        });

        let k = k.min(candidates.len());
        SearchResult {
            indices: candidates[..k].iter().map(|(i, _, _)| *i).collect(),
            distances: candidates[..k].iter().map(|(_, d, _)| *d as f32).collect(),
        }
    }

    /// Rebuild.
    pub fn rebuild(&mut self, embeddings: ArrayView2<f32>) {
        self.searcher.rebuild(embeddings);
    }

    /// N vectors.
    pub fn n_vectors(&self) -> usize {
        self.searcher.n_vectors()
    }
}

/// CUDA search facade — wraps BruteForceSearcher with CUDA-detection interface.
pub struct CudaSearch {
    searcher: Option<BruteForceSearcher>,
    metric: Metric,
}

impl CudaSearch {
    /// New.
    pub fn new(_dim: usize, metric: Metric) -> Self {
        Self {
            searcher: None,
            metric,
        }
    }

    /// Is available.
    pub fn is_available(&self) -> bool {
        self.searcher.is_some()
    }

    /// Index vectors (stores on CPU; GPU path would upload here).
    pub fn index(&mut self, vectors: ArrayView2<f32>) -> Result<(), String> {
        self.searcher = Some(BruteForceSearcher::new(vectors, self.metric));
        Ok(())
    }

    /// Search k nearest neighbors.
    pub fn search(&self, query: ArrayView1<f32>, k: usize) -> Result<SearchResult, String> {
        match &self.searcher {
            Some(s) => Ok(s.search(query, k)),
            None => Err("No index loaded".into()),
        }
    }

    /// Batch search.
    pub fn search_batch(
        &self,
        queries: ArrayView2<f32>,
        k: usize,
    ) -> Result<Vec<SearchResult>, String> {
        match &self.searcher {
            Some(s) => Ok(s.search_batch(queries, k)),
            None => Err("No index loaded".into()),
        }
    }

    /// N vectors.
    pub fn n_vectors(&self) -> usize {
        self.searcher.as_ref().map(|s| s.n_vectors()).unwrap_or(0)
    }
}

// --- Helpers ---

fn precompute_norms_sq(data: ArrayView2<f32>) -> Array1<f64> {
    Array1::from_shape_vec(data.nrows(), {
        (0..data.nrows())
            .map(|i| data.row(i).iter().map(|&v| v as f64 * v as f64).sum())
            .collect()
    })
    .unwrap()
}

fn dot_product_f64(a: ArrayView1<f32>, b: ArrayView1<f32>) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(&x, &y)| x as f64 * y as f64)
        .sum()
}

fn top_k_by_distance(mut scored: Vec<(usize, f32)>, k: usize) -> SearchResult {
    scored.select_nth_unstable_by(k.saturating_sub(1), |a, b| {
        a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
    });
    scored.truncate(k);
    scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    SearchResult {
        indices: scored.iter().map(|(i, _)| *i).collect(),
        distances: scored.iter().map(|(_, d)| *d).collect(),
    }
}

/// Simple deterministic pseudo-random for reproducible noise.
fn rand_pseudo(seed: u64, input: u32) -> u64 {
    let mut h = seed.wrapping_add(input as u64);
    h = h.wrapping_mul(0x517cc1b727220a95);
    h ^= h >> 33;
    h = h.wrapping_mul(0x51af3807);
    h ^= h >> 33;
    h
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::array;

    fn make_data() -> Array2<f32> {
        array![
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
        ]
        .into_shape_with_order((5, 3))
        .unwrap()
    }

    #[test]
    fn test_cosine_search() {
        let data = make_data();
        let searcher = BruteForceSearcher::new(data.view(), Metric::Cosine);
        let query = array![1.0, 0.0, 0.0];
        let result = searcher.search(query.view(), 2);
        assert_eq!(result.indices[0], 0); // exact match
        assert!(result.distances[0] < 0.01);
    }

    #[test]
    fn test_l2_search() {
        let data = make_data();
        let searcher = BruteForceSearcher::new(data.view(), Metric::L2);
        let query = array![0.9, 0.0, 0.0];
        let result = searcher.search(query.view(), 2);
        assert_eq!(result.indices[0], 0);
    }

    #[test]
    fn test_batch_search() {
        let data = make_data();
        let searcher = BruteForceSearcher::new(data.view(), Metric::Cosine);
        let queries = array![[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]];
        let results = searcher.search_batch(queries.view(), 1);
        assert_eq!(results[0].indices[0], 0);
        assert_eq!(results[1].indices[0], 1);
    }

    #[test]
    fn test_cuda_facade() {
        let data = make_data();
        let mut cuda = CudaSearch::new(3, Metric::Cosine);
        assert!(cuda.index(data.view()).is_ok());
        let query = array![1.0, 0.0, 0.0];
        let result = cuda.search(query.view(), 2).unwrap();
        assert_eq!(result.indices[0], 0);
    }

    #[test]
    fn test_multistart_search() {
        let data = make_data();
        let ms = MultiStartSearcher::new(data.view(), 3, 0.01, Metric::Cosine);
        let query = array![1.0, 0.0, 0.0];
        let result = ms.search(query.view(), 2);
        assert_eq!(result.indices[0], 0);
    }

    #[test]
    fn test_rebuild() {
        let data = make_data();
        let mut searcher = BruteForceSearcher::new(data.view(), Metric::Cosine);
        assert_eq!(searcher.n_vectors(), 5);
        let small = array![[1.0, 0.0], [0.0, 1.0]]
            .into_shape_with_order((2, 2))
            .unwrap();
        searcher.rebuild(small.view());
        assert_eq!(searcher.n_vectors(), 2);
    }
}
