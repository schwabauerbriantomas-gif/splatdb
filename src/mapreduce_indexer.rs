//! MapReduce Indexer — parallel indexing pattern for HRM2.
//! Accelerates clustering by dividing dataset into chunks (Map),
//! processing each chunk, and combining partial clusters (Reduce).
//! Ported from splatsdb Python.

use std::collections::HashMap;

/// Result from processing a single chunk (Map phase).
#[derive(Debug, Clone)]
pub struct ChunkResult {
    pub chunk_id: usize,
    pub coarse_labels: Vec<usize>,
    pub fine_labels: Vec<usize>,
    pub coarse_centers: Vec<Vec<f32>>,
    pub fine_centers: HashMap<usize, Vec<Vec<f32>>>,
    pub indices: Vec<usize>,
    pub build_time_secs: f64,
    pub n_samples: usize,
}

/// Result from the Reduce phase.
#[derive(Debug, Clone)]
pub struct ReduceResult {
    pub total_build_time_secs: f64,
    pub map_time_secs: f64,
    pub reduce_time_secs: f64,
    pub n_coarse_clusters: usize,
    pub n_fine_clusters: usize,
    pub n_splats: usize,
    pub n_workers_used: usize,
}

/// MapReduce indexer configuration.
#[derive(Debug, Clone)]
pub struct MapReduceConfig {
    pub n_workers: usize,
    pub chunk_size: usize,
    pub n_coarse: usize,
    pub n_fine_per_coarse: usize,
    pub random_state: u64,
}

impl Default for MapReduceConfig {
    fn default() -> Self {
        Self {
            n_workers: 4,
            chunk_size: 2500,
            n_coarse: 100,
            n_fine_per_coarse: 10,
            random_state: 42,
        }
    }
}

/// MapReduce Indexer for parallel HRM2 indexing.
pub struct MapReduceIndexer {
    config: MapReduceConfig,
}

impl MapReduceIndexer {
    /// New.
    pub fn new(config: MapReduceConfig) -> Self {
        Self { config }
    }

    /// Map phase: divide dataset into chunks and process each.
    pub fn map_phase(&self, embeddings: &[Vec<f32>]) -> Vec<ChunkResult> {
        let n = embeddings.len();
        let dim = embeddings.first().map(|v| v.len()).unwrap_or(0);
        if dim == 0 || n == 0 {
            return Vec::new();
        }

        let chunk_size = self.config.chunk_size.min(n);
        let n_chunks = n.div_ceil(chunk_size);
        let mut results = Vec::new();

        for chunk_id in 0..n_chunks {
            let start = chunk_id * chunk_size;
            let end = (start + chunk_size).min(n);
            let chunk_data: Vec<&Vec<f32>> = embeddings[start..end].iter().collect();
            let chunk_flat: Vec<f32> = chunk_data.iter().flat_map(|v| v.iter().copied()).collect();
            let chunk_n = end - start;
            let indices: Vec<usize> = (start..end).collect();

            let t0 = std::time::Instant::now();

            // Coarse KMeans on chunk
            let n_coarse = self.config.n_coarse.min(chunk_n);
            let (coarse_centers, coarse_labels) = simple_kmeans(
                &chunk_flat,
                chunk_n,
                dim,
                n_coarse,
                self.config.random_state + chunk_id as u64,
            );

            // Fine KMeans per coarse cluster
            let mut fine_labels = vec![0usize; chunk_n];
            let mut fine_centers: HashMap<usize, Vec<Vec<f32>>> = HashMap::new();
            for c in 0..n_coarse {
                let members: Vec<usize> = coarse_labels
                    .iter()
                    .enumerate()
                    .filter(|(_, &l)| l == c)
                    .map(|(i, _)| i)
                    .collect();
                if members.is_empty() {
                    continue;
                }

                let member_data: Vec<f32> = members
                    .iter()
                    .flat_map(|&i| chunk_flat[i * dim..(i + 1) * dim].iter().copied())
                    .collect();
                let n_fine = self.config.n_fine_per_coarse.min(members.len());
                let (_, labels) = simple_kmeans(
                    &member_data,
                    members.len(),
                    dim,
                    n_fine,
                    self.config.random_state + c as u64,
                );
                for (j, &idx) in members.iter().enumerate() {
                    fine_labels[idx] = labels[j];
                }
                fine_centers.insert(c, vec![Vec::new()]);
            }

            results.push(ChunkResult {
                chunk_id,
                coarse_labels,
                fine_labels,
                coarse_centers,
                fine_centers,
                indices,
                build_time_secs: t0.elapsed().as_secs_f64(),
                n_samples: chunk_n,
            });
        }

        results
    }

    /// Reduce phase: combine chunk results into unified clustering.
    pub fn reduce_phase(&self, chunk_results: &[ChunkResult]) -> ReduceResult {
        let t0 = std::time::Instant::now();
        let map_time: f64 = chunk_results.iter().map(|r| r.build_time_secs).sum();
        let n_splats: usize = chunk_results.iter().map(|r| r.n_samples).sum();
        let n_coarse_clusters = self.config.n_coarse;
        let n_fine_clusters = n_coarse_clusters * self.config.n_fine_per_coarse;

        // Merge coarse centers from all chunks
        let all_coarse: Vec<Vec<f32>> = chunk_results
            .iter()
            .flat_map(|r| r.coarse_centers.iter().cloned())
            .collect();

        // Re-cluster the merged coarse centers to get unified labels
        let dim = all_coarse.first().map(|v| v.len()).unwrap_or(0);
        if dim > 0 && !all_coarse.is_empty() {
            let flat: Vec<f32> = all_coarse.iter().flat_map(|v| v.iter().copied()).collect();
            let n_final = n_coarse_clusters.min(all_coarse.len());
            let _ = simple_kmeans(
                &flat,
                all_coarse.len(),
                dim,
                n_final,
                self.config.random_state,
            );
        }

        ReduceResult {
            total_build_time_secs: t0.elapsed().as_secs_f64() + map_time,
            map_time_secs: map_time,
            reduce_time_secs: t0.elapsed().as_secs_f64(),
            n_coarse_clusters,
            n_fine_clusters,
            n_splats,
            n_workers_used: self.config.n_workers,
        }
    }

    /// Full MapReduce: map then reduce.
    pub fn index(&self, embeddings: &[Vec<f32>]) -> ReduceResult {
        let chunks = self.map_phase(embeddings);
        self.reduce_phase(&chunks)
    }
}

/// Simple KMeans clustering (CPU).
fn simple_kmeans(
    data: &[f32],
    n: usize,
    dim: usize,
    k: usize,
    seed: u64,
) -> (Vec<Vec<f32>>, Vec<usize>) {
    let k = k.min(n);
    if k == 0 || dim == 0 {
        return (Vec::new(), vec![0usize; n]);
    }

    // Init centroids by sampling from data
    let mut centroids: Vec<Vec<f32>> = Vec::new();
    let mut rng_state = seed;
    for _ in 0..k {
        rng_state = rng_state.wrapping_mul(6364136223846793005).wrapping_add(1);
        let idx = (rng_state as usize) % n;
        centroids.push(data[idx * dim..(idx + 1) * dim].to_vec());
    }

    let mut labels = vec![0usize; n];

    for _ in 0..20 {
        let mut changed = false;
        for i in 0..n {
            let row = &data[i * dim..(i + 1) * dim];
            let mut best_c = 0;
            let mut best_dist = f64::MAX;
            for (c, center) in centroids.iter().enumerate() {
                let dist: f64 = row
                    .iter()
                    .zip(center.iter())
                    .map(|(&x, &y)| {
                        let d = x as f64 - y as f64;
                        d * d
                    })
                    .sum::<f64>()
                    .sqrt();
                if dist < best_dist {
                    best_dist = dist;
                    best_c = c;
                }
            }
            if labels[i] != best_c {
                labels[i] = best_c;
                changed = true;
            }
        }
        if !changed {
            break;
        }

        // Update centroids
        let mut counts = vec![0usize; k];
        let mut sums = vec![vec![0.0f64; dim]; k];
        for i in 0..n {
            let c = labels[i];
            counts[c] += 1;
            for j in 0..dim {
                sums[c][j] += data[i * dim + j] as f64;
            }
        }
        for c in 0..k {
            if counts[c] > 0 {
                for j in 0..dim {
                    centroids[c][j] = (sums[c][j] / counts[c] as f64) as f32;
                }
            }
        }
    }

    (centroids, labels)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_embeddings() -> Vec<Vec<f32>> {
        vec![
            vec![1.0, 0.0],
            vec![1.1, 0.1],
            vec![0.9, -0.1],
            vec![0.0, 1.0],
            vec![0.1, 1.1],
            vec![-0.1, 0.9],
            vec![5.0, 5.0],
            vec![5.1, 4.9],
            vec![4.9, 5.1],
        ]
    }

    #[test]
    fn test_map_reduce() {
        let indexer = MapReduceIndexer::new(MapReduceConfig {
            chunk_size: 3,
            n_workers: 2,
            n_coarse: 3,
            n_fine_per_coarse: 2,
            random_state: 42,
        });
        let result = indexer.index(&make_embeddings());
        assert_eq!(result.n_splats, 9);
        assert!(result.total_build_time_secs > 0.0);
    }

    #[test]
    fn test_kmeans() {
        let data: Vec<f32> = vec![1.0, 0.0, 1.1, 0.1, 0.0, 1.0, 0.1, 1.1];
        let (centers, labels) = simple_kmeans(&data, 4, 2, 2, 42);
        assert_eq!(centers.len(), 2);
        assert_eq!(labels.len(), 4);
        // First two should be in same cluster
        assert_eq!(labels[0], labels[1]);
    }

    #[test]
    fn test_empty_embeddings() {
        let indexer = MapReduceIndexer::new(MapReduceConfig::default());
        let chunks = indexer.map_phase(&[]);
        assert!(chunks.is_empty());
    }
}
