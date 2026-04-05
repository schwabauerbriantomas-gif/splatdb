//! LSH Index — Cross-Polytope Locality-Sensitive Hashing.
//! Optimized native Rust implementation.

use ndarray::{Array2, ArrayView1};
use rand::Rng;
use rand_chacha::ChaCha8Rng;
use rand::SeedableRng;
use std::collections::HashMap;

/// LSH configuration.
#[derive(Clone, Debug)]
pub struct LSHConfig {
    pub dim: usize,
    pub n_tables: usize,
    pub n_bits: usize,
    pub n_probes: usize,
    pub n_candidates: usize,
    pub seed: u64,
}

impl Default for LSHConfig {
    fn default() -> Self {
        Self {
            dim: 640,
            n_tables: 15,
            n_bits: 18,
            n_probes: 50,
            n_candidates: 500,
            seed: 42,
        }
    }
}

/// Cross-Polytope LSH index.
pub struct CrossPolytopeLSH {
    config: LSHConfig,
    k: usize,
    rotations: Vec<Vec<Array2<f32>>>,
    tables: Vec<HashMap<Vec<i64>, Vec<usize>>>,
    vectors: Option<Array2<f32>>,
    n_vectors: usize,
}

impl CrossPolytopeLSH {
    /// New.
    pub fn new(config: LSHConfig) -> Self {
        let k = (config.n_bits as f64 / (2.0 * config.dim as f64).log2()).ceil().max(1.0).min(10000.0) as usize;
        let n_tables = config.n_tables;
        let dim = config.dim;
        let seed = config.seed;
        let mut rng = ChaCha8Rng::seed_from_u64(seed);

        let rotations: Vec<Vec<Array2<f32>>> = (0..n_tables)
            .map(|_| (0..k).map(|_| random_rotation(dim, &mut rng)).collect())
            .collect();

        Self {
            config,
            k,
            rotations,
            tables: vec![HashMap::new(); n_tables],
            vectors: None,
            n_vectors: 0,
        }
    }

    /// Index.
    pub fn index(&mut self, vectors: Array2<f32>) {
        let n = vectors.nrows();
        let mut normalized = vectors.to_owned();
        for i in 0..n {
            let mut row = normalized.row_mut(i);
            let norm = row.dot(&row).sqrt().max(1e-10);
            row.mapv_inplace(|v| v / norm);
        }

        self.vectors = Some(normalized.clone());
        self.n_vectors = n;
        self.tables = vec![HashMap::new(); self.config.n_tables];

        for t in 0..self.config.n_tables {
            for idx in 0..n {
                let hash = self.compute_hash(&normalized.row(idx).to_vec(), t);
                self.tables[t].entry(hash).or_default().push(idx);
            }
        }
    }

    /// Query.
    pub fn query(&self, query: &ArrayView1<f32>, k: usize) -> (Vec<usize>, Vec<f32>) {
        let vectors = match &self.vectors {
            Some(v) => v,
            None => return (vec![], vec![]),
        };

        let norm = query.dot(query).sqrt().max(1e-10);
        let q: Vec<f32> = query.iter().map(|&v| v / norm).collect();

        let mut candidate_set = std::collections::HashSet::new();
        for t in 0..self.config.n_tables {
            let probes = self.multi_probe_hashes(&q, t);
            for hash in probes.iter().take(self.config.n_probes) {
                if let Some(indices) = self.tables[t].get(hash) {
                    candidate_set.extend(indices.iter().copied());
                }
            }
        }

        let mut candidates: Vec<usize> = candidate_set.into_iter().collect();
        if candidates.len() < k {
            let extra: Vec<usize> = (0..self.n_vectors).filter(|i| !candidates.contains(i)).take(k * 10).collect();
            candidates.extend(extra);
        }
        candidates.truncate(self.config.n_candidates);

        let mut scored: Vec<(f32, usize)> = candidates.iter().map(|&idx| {
            let row = vectors.row(idx);
            let dist: f32 = row.iter().zip(q.iter()).map(|(a, b)| (a - b) * (a - b)).sum::<f32>().sqrt();
            (dist, idx)
        }).collect();

        scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(k);

        (scored.iter().map(|(_, i)| *i).collect(), scored.iter().map(|(d, _)| *d).collect())
    }

    /// Get recall.
    pub fn get_recall(&self, queries: &Array2<f32>, ground_truth: &[Vec<usize>], k: usize) -> f32 {
        let n = queries.nrows();
        let mut total_found = 0usize;
        let total_expected = n * k;

        for (i, gt) in ground_truth.iter().enumerate().take(n) {
            let (predicted, _) = self.query(&queries.row(i), k);
            let gt_set: std::collections::HashSet<usize> = gt.iter().copied().collect();
            for &p in &predicted {
                if gt_set.contains(&p) {
                    total_found += 1;
                }
            }
        }

        if total_expected == 0 { return 0.0; }
        total_found as f32 / total_expected as f32
    }

    fn compute_hash(&self, vector: &[f32], table_idx: usize) -> Vec<i64> {
        self.rotations[table_idx].iter().map(|rotation| {
            let rotated: Vec<f32> = mat_vec_mul(rotation, vector);
            let (max_idx, max_val) = rotated.iter().enumerate()
                .max_by(|a, b| a.1.abs().partial_cmp(&b.1.abs()).unwrap_or(std::cmp::Ordering::Equal))
                .expect("rotated vector should not be empty");
            let sign_bit = if *max_val >= 0.0 { 0i64 } else { 1i64 };
            max_idx as i64 * 2 + sign_bit
        }).collect()
    }

    fn multi_probe_hashes(&self, vector: &[f32], table_idx: usize) -> Vec<Vec<i64>> {
        let m = 2.max((self.config.n_probes as f64).powf(1.0 / self.k as f64).ceil() as usize * 2);

        let parts: Vec<Vec<(i64, f32)>> = self.rotations[table_idx].iter().map(|rotation| {
            let rotated = mat_vec_mul(rotation, vector);

            let mut indexed: Vec<(usize, f32)> = rotated.iter().enumerate()
                .map(|(i, &v)| (i, v.abs())).collect();
            indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            indexed.truncate(m);

            indexed.iter().map(|&(idx, _)| {
                let val = rotated[idx];
                let sign_bit = if val >= 0.0 { 0i64 } else { 1i64 };
                (idx as i64 * 2 + sign_bit, rotated[idx].abs())
            }).collect()
        }).collect();

        // Enumerate cartesian product combinations (limited)
        let mut probe_candidates: Vec<(f32, Vec<i64>)> = Vec::new();
        let mut current = vec![0usize; self.k];

        loop {
            let hash: Vec<i64> = current.iter().enumerate()
                .map(|(part_i, &idx)| parts[part_i][idx.min(parts[part_i].len() - 1)].0)
                .collect();
            let score: f32 = current.iter().enumerate()
                .map(|(part_i, &idx)| parts[part_i][idx.min(parts[part_i].len() - 1)].1)
                .sum();

            probe_candidates.push((score, hash));

            let mut carry = true;
            for i in (0..self.k).rev() {
                if carry {
                    current[i] += 1;
                    if current[i] >= parts[i].len() {
                        current[i] = 0;
                    } else {
                        carry = false;
                    }
                }
            }
            if carry || probe_candidates.len() >= self.config.n_probes * 10 { break; }
        }

        probe_candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        probe_candidates.truncate(self.config.n_probes);
        probe_candidates.into_iter().map(|(_, h)| h).collect()
    }
}

fn mat_vec_mul(mat: &Array2<f32>, vec: &[f32]) -> Vec<f32> {
    let rows = mat.nrows();
    let mut result = Vec::with_capacity(rows);
    for i in 0..rows {
        let row = mat.row(i);
        let dot: f32 = row.iter().zip(vec.iter()).map(|(a, b)| a * b).sum();
        result.push(dot);
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{array, Array2};

    fn small_random_data(n: usize, dim: usize, seed: u64) -> Array2<f32> {
        let mut rng = ChaCha8Rng::seed_from_u64(seed);
        let data: Vec<f32> = (0..n * dim).map(|_| rng.gen::<f32>()).collect();
        Array2::from_shape_vec((n, dim), data).unwrap()
    }

    #[test]
    fn test_lsh_config_default() {
        let cfg = LSHConfig::default();
        assert_eq!(cfg.dim, 640);
        assert_eq!(cfg.n_tables, 15);
        assert_eq!(cfg.n_bits, 18);
        assert_eq!(cfg.seed, 42);
    }

    #[test]
    fn test_lsh_creation() {
        let cfg = LSHConfig { dim: 4, n_tables: 3, n_bits: 4, n_probes: 5, n_candidates: 50, seed: 42 };
        let lsh = CrossPolytopeLSH::new(cfg);
        assert_eq!(lsh.n_vectors, 0);
    }

    #[test]
    fn test_index_and_query() {
        let cfg = LSHConfig { dim: 4, n_tables: 5, n_bits: 4, n_probes: 10, n_candidates: 100, seed: 42 };
        let mut lsh = CrossPolytopeLSH::new(cfg);
        let data = small_random_data(50, 4, 123);

        // Index the data
        lsh.index(data.clone());
        assert_eq!(lsh.n_vectors, 50);

        // Query: searching for an indexed vector should return it near the top
        let query = data.row(0);
        let (indices, distances) = lsh.query(&query, 5);
        assert!(!indices.is_empty());
        assert_eq!(indices.len(), distances.len());
        // The query vector itself (idx 0) should appear in results
        assert!(indices.contains(&0));
        // Distance to itself should be very small
        let self_pos = indices.iter().position(|&i| i == 0).unwrap();
        assert!(distances[self_pos] < 0.1);
    }

    #[test]
    fn test_query_without_index() {
        let cfg = LSHConfig { dim: 4, n_tables: 3, n_bits: 4, n_probes: 5, n_candidates: 50, seed: 42 };
        let lsh = CrossPolytopeLSH::new(cfg);
        let query = array![1.0f32, 0.0, 0.0, 0.0];
        let (indices, distances) = lsh.query(&query.view(), 5);
        assert!(indices.is_empty());
        assert!(distances.is_empty());
    }

    #[test]
    fn test_recall() {
        let cfg = LSHConfig { dim: 8, n_tables: 8, n_bits: 6, n_probes: 20, n_candidates: 200, seed: 42 };
        let mut lsh = CrossPolytopeLSH::new(cfg);
        let data = small_random_data(100, 8, 456);
        lsh.index(data.clone());

        let queries = data.slice(ndarray::s![0..10, ..]).to_owned();
        let ground_truth: Vec<Vec<usize>> = (0..10).map(|i| {
            let q = queries.row(i);
            let mut scored: Vec<(f32, usize)> = (0..100).map(|j| {
                let r = data.row(j);
                let dist: f32 = r.iter().zip(q.iter()).map(|(a, b)| (a - b) * (a - b)).sum::<f32>().sqrt();
                (dist, j)
            }).collect();
            scored.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            scored.into_iter().take(5).map(|(_, i)| i).collect()
        }).collect();

        let recall = lsh.get_recall(&queries, &ground_truth, 5);
        assert!(recall > 0.0, "Recall should be > 0, got {}", recall);
    }
}

fn random_rotation(dim: usize, rng: &mut ChaCha8Rng) -> Array2<f32> {
    let mut a = Array2::zeros((dim, dim));
    for i in 0..dim {
        for j in 0..dim {
            a[[i, j]] = rng.gen::<f32>() * 2.0 - 1.0;
        }
    }

    let mut q = Array2::zeros((dim, dim));
    for j in 0..dim {
        for i in 0..dim { q[[i, j]] = a[[i, j]]; }
        for k in 0..j {
            let dot: f32 = (0..dim).map(|i| q[[i, j]] * q[[i, k]]).sum();
            for i in 0..dim { q[[i, j]] -= dot * q[[i, k]]; }
        }
        let norm: f32 = (0..dim).map(|i| q[[i, j]] * q[[i, j]]).sum::<f32>().sqrt();
        if norm > 1e-10 {
            for i in 0..dim { q[[i, j]] /= norm; }
        }
    }
    q
}
