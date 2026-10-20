//! GPU acceleration module.
//!
//! When compiled with `--features cuda`, provides real CUDA operations
//! via cudarc. Falls back to CPU (ndarray) when CUDA is unavailable.
//! CPU path is always available — CUDA is purely additive acceleration.

#[cfg(feature = "cuda")]
pub mod cuda_kernel;

#[cfg(feature = "cuda")]
pub mod cuda_extended;

/// Check if CUDA is available at runtime.
#[cfg(feature = "cuda")]
pub fn is_cuda_available() -> bool {
    crate::gpu::with_silent_panic(|| {
        use cudarc::driver::CudaContext;
        CudaContext::device_count().map(|c| c > 0).unwrap_or(false)
    })
    .unwrap_or(false)
}

/// Run a cudarc initialization closure without letting its internal panic
/// escape: cudarc panics when `libcuda` is absent (runner/machine without a
/// GPU driver). Returns None on panic; the panic message is silenced.
#[cfg(feature = "cuda")]
pub fn with_silent_panic<T>(f: impl FnOnce() -> T + std::panic::UnwindSafe) -> Option<T> {
    let prev = std::panic::take_hook();
    std::panic::set_hook(Box::new(|_| {}));
    let result = std::panic::catch_unwind(f).ok();
    std::panic::set_hook(prev);
    result
}

/// Check if CUDA is available at runtime.
#[cfg(not(feature = "cuda"))]
pub fn is_cuda_available() -> bool {
    false
}

/// GPU-accelerated batch L2 distance computation.
///
/// CUDA path: uploads dataset to GPU memory, computes distances on device.
/// CPU fallback: ndarray-based computation (always available).
pub fn batch_l2_distances(query: &[f32], dataset: &[f32], n_rows: usize, dim: usize) -> Vec<f32> {
    #[cfg(feature = "cuda")]
    {
        if is_cuda_available() {
            if let Some(result) = cuda_kernel::gpu_l2_distances(query, dataset, n_rows, dim) {
                return result;
            }
        }
    }
    cpu_l2_distances(query, dataset, n_rows, dim)
}

/// GPU-accelerated batch cosine distance computation.
pub fn batch_cosine_distances(
    query: &[f32],
    dataset: &[f32],
    n_rows: usize,
    dim: usize,
) -> Vec<f32> {
    #[cfg(feature = "cuda")]
    {
        if is_cuda_available() {
            if let Some(result) = cuda_kernel::gpu_cosine_distances(query, dataset, n_rows, dim) {
                return result;
            }
        }
    }
    cpu_cosine_distances(query, dataset, n_rows, dim)
}

/// GPU-accelerated batch search: multiple queries against dataset.
///
/// Returns (indices, distances) for top-k per query.
/// Uses GPU when available, CPU otherwise.
pub fn batch_search(
    queries: &[f32],
    n_queries: usize,
    dataset: &[f32],
    n_vectors: usize,
    dim: usize,
    k: usize,
    metric: &str,
) -> Vec<(Vec<usize>, Vec<f32>)> {
    #[cfg(feature = "cuda")]
    {
        if is_cuda_available() {
            if let Some(result) = cuda_kernel::gpu_batch_search(
                queries, n_queries, dataset, n_vectors, dim, k, metric,
            ) {
                return result;
            }
        }
    }
    cpu_batch_search(queries, n_queries, dataset, n_vectors, dim, k, metric)
}

/// Get GPU info if CUDA is available.
#[cfg(feature = "cuda")]
pub fn gpu_info() -> Option<String> {
    use cudarc::driver::CudaContext;
    let ctx = CudaContext::new(0).ok()?;
    let name = ctx.name().unwrap_or_else(|_| "Unknown".to_string());
    let (major, minor) = ctx.compute_capability().unwrap_or((0, 0));
    let total_mem = ctx.total_mem().unwrap_or(0);
    let free_mem = ctx.mem_get_info().map(|(free, _)| free).unwrap_or(0);
    Some(format!(
        "{} (sm_{}{}), {:.1} GB total, {:.1} GB free",
        name,
        major,
        minor,
        total_mem as f64 / 1e9,
        free_mem as f64 / 1e9,
    ))
}

/// Get GPU info if CUDA is available.
#[cfg(not(feature = "cuda"))]
pub fn gpu_info() -> Option<String> {
    None
}

/// Gaussian top-k search over the splat energy landscape:
/// score_i = alpha_i * exp(-kappa_i * ||q - mu_i||^2) — HIGHER is better.
///
/// No 1/2 factor in the exponent (convention of energy.rs e_splats).
/// Top-K is over the FINAL score: per-item kappa makes it NOT equivalent to
/// ascending d^2 ordering. Returns per-query (index, score) sorted descending;
/// exact score ties break toward the smaller index (CPU stable sort; the GPU
/// kernel matches within a thread — cross-thread exact ties are negligible for
/// continuous data). Falls back to an exact CPU reference (natural exp, same
/// ordering) when CUDA is unavailable.
#[allow(clippy::too_many_arguments)]
pub fn gaussian_topk_scores(
    queries: &[f32],
    dataset: &[f32],
    alpha: &[f32],
    kappa: &[f32],
    n_vectors: usize,
    dim: usize,
    n_queries: usize,
    k: usize,
) -> Option<Vec<Vec<(usize, f32)>>> {
    if dim == 0 || n_vectors == 0 || n_queries == 0 || k == 0 {
        return None;
    }
    if dataset.len() != n_vectors.checked_mul(dim)? {
        return None;
    }
    if queries.len() != n_queries.checked_mul(dim)? {
        return None;
    }
    if alpha.len() != n_vectors || kappa.len() != n_vectors {
        return None;
    }

    #[cfg(feature = "cuda")]
    {
        if is_cuda_available() {
            if let Some(result) = cuda_kernel::gpu_gaussian_topk(
                queries, n_queries, dataset, alpha, kappa, n_vectors, dim, k,
            ) {
                return Some(result);
            }
        }
    }
    Some(cpu_gaussian_topk(
        queries, dataset, alpha, kappa, n_vectors, dim, n_queries, k,
    ))
}

/// Exact CPU reference: natural exponential, descending score, ties → lower index.
#[allow(clippy::too_many_arguments)]
pub(crate) fn cpu_gaussian_topk(
    queries: &[f32],
    dataset: &[f32],
    alpha: &[f32],
    kappa: &[f32],
    n_vectors: usize,
    dim: usize,
    n_queries: usize,
    k: usize,
) -> Vec<Vec<(usize, f32)>> {
    let k_eff = k.min(n_vectors);
    (0..n_queries)
        .map(|q| {
            let q_slice = &queries[q * dim..(q + 1) * dim];
            let mut scored: Vec<(usize, f32)> = (0..n_vectors)
                .map(|i| {
                    let row = &dataset[i * dim..(i + 1) * dim];
                    let d2: f32 = q_slice
                        .iter()
                        .zip(row.iter())
                        .map(|(a, b)| (a - b) * (a - b))
                        .sum();
                    (i, alpha[i] * (-kappa[i] * d2).exp())
                })
                .collect();
            // Stable sort descending by score → ties keep smaller index first.
            scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
            scored.truncate(k_eff);
            scored
        })
        .collect()
}

// ============================================================================
// Extended GPU operations (quantization, clustering, geometry, LSH)
// ============================================================================

/// GPU-accelerated batch rotation: y[i] = R · x[i] for N vectors.
/// `rotation` is [D, D] row-major, `vectors` is [N, D], returns [N, D].
#[allow(unused_variables)]
pub fn rotation_gemv(vectors: &[f32], rotation: &[f32], n: usize, dim: usize) -> Option<Vec<f32>> {
    #[cfg(feature = "cuda")]
    {
        if is_cuda_available() {
            let ext = cuda_extended::GpuExtended::new()?;
            return ext.rotation_gemv(vectors, rotation, n, dim);
        }
    }
    None
}

/// GPU-accelerated batch inverse rotation:
#[allow(unused_variables)]
pub fn rotation_gemv_inverse(
    vectors: &[f32],
    rotation: &[f32],
    n: usize,
    dim: usize,
) -> Option<Vec<f32>> {
    #[cfg(feature = "cuda")]
    {
        if is_cuda_available() {
            let ext = cuda_extended::GpuExtended::new()?;
            return ext.rotation_gemv_inverse(vectors, rotation, n, dim);
        }
    }
    None
}

/// GPU-accelerated batch QJL sketch: sign(G · x) for N vectors.
#[allow(unused_variables)]
pub fn qjl_batch_sketch(
    vectors: &[f32],
    projections: &[f32],
    n: usize,
    dim: usize,
    g: usize,
) -> Option<Vec<i8>> {
    #[cfg(feature = "cuda")]
    {
        if is_cuda_available() {
            let ext = cuda_extended::GpuExtended::new()?;
            return ext.qjl_batch_sketch(vectors, projections, n, dim, g);
        }
    }
    None
}

/// GPU-accelerated batch QJL inner product
#[allow(unused_variables)]
pub fn qjl_batch_ip_estimate(
    projections: &[f32],
    signs: &[i8],
    query: &[f32],
    n: usize,
    dim: usize,
    g: usize,
) -> Option<Vec<f32>> {
    #[cfg(feature = "cuda")]
    {
        if is_cuda_available() {
            let ext = cuda_extended::GpuExtended::new()?;
            return ext.qjl_batch_ip_estimate(projections, signs, query, n, dim, g);
        }
    }
    None
}

/// GPU-accelerated KMeans assignment: assign N points to nearest centroid.
/// GPU-accelerated KMeans assignment:
#[allow(unused_variables)]
pub fn kmeans_assign(
    points: &[f32],
    centroids: &[f32],
    n: usize,
    k: usize,
    dim: usize,
) -> Option<(Vec<i32>, Vec<f32>)> {
    #[cfg(feature = "cuda")]
    {
        if is_cuda_available() {
            let ext = cuda_extended::GpuExtended::new()?;
            return ext.kmeans_assign(points, centroids, n, k, dim);
        }
    }
    None
}

/// GPU-accelerated all-pairs cosine similarity matrix.
#[allow(unused_variables)]
pub fn cosine_similarity_matrix(vectors: &[f32], n: usize, dim: usize) -> Option<Vec<f32>> {
    #[cfg(feature = "cuda")]
    {
        if is_cuda_available() {
            let ext = cuda_extended::GpuExtended::new()?;
            return ext.cosine_similarity_matrix(vectors, n, dim);
        }
    }
    None
}

/// GPU-accelerated batch geodesic distance between paired vectors.
#[allow(unused_variables)]
pub fn batch_geodesic(x: &[f32], y: &[f32], n: usize, dim: usize) -> Option<Vec<f32>> {
    #[cfg(feature = "cuda")]
    {
        if is_cuda_available() {
            let ext = cuda_extended::GpuExtended::new()?;
            return ext.batch_geodesic(x, y, n, dim);
        }
    }
    None
}

/// GPU-accelerated batch LSH hash computation.
#[allow(unused_variables)]
pub fn lsh_batch_hash(
    vectors: &[f32],
    projections: &[f32],
    n: usize,
    dim: usize,
    t: usize,
    k: usize,
) -> Option<Vec<i8>> {
    #[cfg(feature = "cuda")]
    {
        if is_cuda_available() {
            let ext = cuda_extended::GpuExtended::new()?;
            return ext.lsh_batch_hash(vectors, projections, n, dim, t, k);
        }
    }
    None
}

// --- CPU implementations (always available) ---

fn cpu_l2_distances(query: &[f32], dataset: &[f32], n_rows: usize, dim: usize) -> Vec<f32> {
    let mut distances = Vec::with_capacity(n_rows);
    for i in 0..n_rows {
        let row = &dataset[i * dim..(i + 1) * dim];
        let dist: f32 = query
            .iter()
            .zip(row.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum();
        distances.push(dist);
    }
    distances
}

fn cpu_cosine_distances(query: &[f32], dataset: &[f32], n_rows: usize, dim: usize) -> Vec<f32> {
    let q_norm: f32 = query.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-10);
    let mut distances = Vec::with_capacity(n_rows);
    for i in 0..n_rows {
        let row = &dataset[i * dim..(i + 1) * dim];
        let dot: f32 = query.iter().zip(row.iter()).map(|(a, b)| a * b).sum();
        let r_norm: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-10);
        distances.push(1.0 - dot / (q_norm * r_norm));
    }
    distances
}

fn cpu_batch_search(
    queries: &[f32],
    n_queries: usize,
    dataset: &[f32],
    n_vectors: usize,
    dim: usize,
    k: usize,
    metric: &str,
) -> Vec<(Vec<usize>, Vec<f32>)> {
    (0..n_queries)
        .map(|q| {
            let q_slice = &queries[q * dim..(q + 1) * dim];
            let distances = if metric == "cosine" {
                cpu_cosine_distances(q_slice, dataset, n_vectors, dim)
            } else {
                cpu_l2_distances(q_slice, dataset, n_vectors, dim)
            };
            let mut indices: Vec<usize> = (0..n_vectors).collect();
            indices.sort_by(|a, b| distances[*a].partial_cmp(&distances[*b]).unwrap());
            indices.truncate(k);
            let dists = indices.iter().map(|&i| distances[i]).collect();
            (indices, dists)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cpu_l2_distances() {
        let query = vec![1.0f32, 0.0];
        let dataset = vec![1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let dists = cpu_l2_distances(&query, &dataset, 3, 2);
        assert!((dists[0] - 0.0).abs() < 1e-6);
        assert!((dists[1] - 2.0).abs() < 1e-6);
        assert!((dists[2] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cpu_cosine_distances() {
        let query = vec![1.0f32, 0.0];
        let dataset = vec![1.0, 0.0, 0.0, 1.0];
        let dists = cpu_cosine_distances(&query, &dataset, 2, 2);
        assert!((dists[0] - 0.0).abs() < 1e-6);
        assert!((dists[1] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cpu_batch_search() {
        let queries = vec![1.0f32, 0.0];
        let dataset = vec![0.0, 1.0, 1.0, 0.0, 0.5, 0.5];
        let results = cpu_batch_search(&queries, 1, &dataset, 3, 2, 2, "l2");
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].0[0], 1);
    }

    #[test]
    fn test_batch_l2_distances_public_api() {
        let query = vec![1.0f32, 0.0];
        let dataset = vec![1.0, 0.0, 0.0, 1.0];
        let dists = batch_l2_distances(&query, &dataset, 2, 2);
        assert_eq!(dists.len(), 2);
        assert!((dists[0] - 0.0).abs() < 1e-6);
    }

    #[test]
    fn test_batch_search_public_api() {
        let queries = vec![1.0f32, 0.0];
        let dataset = vec![1.0, 0.0, 0.0, 1.0];
        let results = batch_search(&queries, 1, &dataset, 2, 2, 2, "l2");
        assert_eq!(results.len(), 1);
    }

    /// Test that CUDA pipeline actually works on this machine (GPU upload/download/verify).
    /// Only runs when compiled with --features cuda.
    #[cfg(feature = "cuda")]
    #[test]
    fn test_cuda_gpu_pipeline() {
        if !is_cuda_available() {
            return; // skip on machines without GPU
        }

        // Large enough to matter for GPU transfer
        let dim = 640;
        let n = 1000;
        let mut dataset = vec![0.0f32; n * dim];
        for i in 0..n {
            dataset[i * dim] = i as f32 / n as f32;
        }
        let query = vec![0.5f32; dim];

        // L2 via GPU pipeline
        let gpu_dists = batch_l2_distances(&query, &dataset, n, dim);
        assert_eq!(gpu_dists.len(), n);

        // Cosine via GPU pipeline
        let gpu_cos = batch_cosine_distances(&query, &dataset, n, dim);
        assert_eq!(gpu_cos.len(), n);

        // Batch search via GPU pipeline
        let queries_flat = vec![0.5f32; 2 * dim];
        let results = batch_search(&queries_flat, 2, &dataset, n, dim, 5, "l2");
        assert_eq!(results.len(), 2);
        assert_eq!(results[0].0.len(), 5);
        assert_eq!(results[1].0.len(), 5);

        // GPU info
        let info = gpu_info().expect("GPU info should be available");
        assert!(info.contains("GB"));
    }

    /// Test that custom PTX kernels produce identical results to CPU computation.
    /// This validates the GPU compute path (not just upload/download).
    #[cfg(feature = "cuda")]
    #[test]
    fn test_ptx_kernel_correctness() {
        use cuda_kernel::GpuIndex;

        if !is_cuda_available() {
            return;
        }

        let dim = 640;
        let n = 500;

        // Generate deterministic dataset
        let mut dataset = vec![0.0f32; n * dim];
        let mut state = 42u64;
        for val in dataset.iter_mut() {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *val = ((state >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0;
        }
        let query = vec![0.5f32; dim];

        // CPU ground truth
        let cpu_l2: Vec<f32> = (0..n)
            .map(|i| {
                let row = &dataset[i * dim..(i + 1) * dim];
                query
                    .iter()
                    .zip(row.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum()
            })
            .collect();

        let q_norm: f32 = query.iter().map(|x| x * x).sum::<f32>().sqrt();
        let cpu_cos: Vec<f32> = (0..n)
            .map(|i| {
                let row = &dataset[i * dim..(i + 1) * dim];
                let dot: f32 = query.iter().zip(row.iter()).map(|(a, b)| a * b).sum();
                let r_norm: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt();
                1.0 - dot / (q_norm * r_norm)
            })
            .collect();

        // GPU via GpuIndex (should use PTX kernels if available)
        let mut idx = GpuIndex::new().expect("CUDA context failed");
        assert!(idx.upload_dataset(&dataset, n, dim));

        // Test single L2
        let gpu_l2 = idx.l2_distances(&query).expect("L2 distances failed");
        assert_eq!(gpu_l2.len(), cpu_l2.len());

        // Verify GPU matches CPU within floating point tolerance
        for (gpu, cpu) in gpu_l2.iter().zip(cpu_l2.iter()) {
            let diff = (gpu - cpu).abs();
            assert!(
                diff < 1e-2,
                "L2 mismatch: GPU={} CPU={} diff={}",
                gpu,
                cpu,
                diff
            );
        }

        // Test cosine
        let gpu_cos = idx
            .cosine_distances(&query)
            .expect("Cosine distances failed");
        for (gpu, cpu) in gpu_cos.iter().zip(cpu_cos.iter()) {
            let diff = (gpu - cpu).abs();
            assert!(
                diff < 1e-2,
                "Cosine mismatch: GPU={} CPU={} diff={}",
                gpu,
                cpu,
                diff
            );
        }
    }

    /// Gaussian top-k: exact CPU reference against hand-computed values.
    /// score = alpha * exp(-kappa * ||q-mu||^2), no 1/2 factor, higher better.
    #[test]
    fn test_gaussian_topk_cpu_reference_exact() {
        // mu1 = origin (d2=0 -> score=alpha1), mu2 = (3,4) (d2=25 -> score=alpha2*e^-25)
        let dataset = vec![0.0f32, 0.0, 3.0, 4.0];
        let alpha = vec![0.5f32, 2.0];
        let kappa = vec![1.0f32, 1.0];
        let queries = vec![0.0f32, 0.0];
        let out = cpu_gaussian_topk(&queries, &dataset, &alpha, &kappa, 2, 2, 1, 2);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].len(), 2);
        assert_eq!(out[0][0].0, 0);
        assert!((out[0][0].1 - 0.5).abs() < 1e-6, "got {}", out[0][0].1);
        let expected2 = 2.0f32 * (-25.0f32).exp();
        assert_eq!(out[0][1].0, 1);
        assert!(
            (out[0][1].1 - expected2).abs() < 1e-6,
            "got {}",
            out[0][1].1
        );
    }

    /// Gaussian top-k: kappa per item makes ordering differ from ascending d2.
    #[test]
    fn test_gaussian_topk_not_d2_ordering() {
        // mu1 at d2=1 (score 1.0*exp(-1)=0.368), mu2 at d2=4 but kappa tiny
        // (score 1.0*exp(-4*0.01)=0.961) -> mu2 must rank FIRST.
        let dataset = vec![1.0f32, 0.0, 2.0, 0.0];
        let alpha = vec![1.0f32, 1.0];
        let kappa = vec![1.0f32, 0.01];
        let queries = vec![0.0f32, 0.0];
        let out = gaussian_topk_scores(&queries, &dataset, &alpha, &kappa, 2, 2, 1, 2)
            .expect("valid input");
        assert_eq!(out[0][0].0, 1, "kappa-weighted score must beat smaller d2");
        assert!(out[0][0].1 > out[0][1].1);
    }

    /// Gaussian top-k GPU vs exact CPU reference: random 500x64D, Q=4, K=10.
    /// Tolerance 1e-5 on scores, identical indices. Runs only with a GPU.
    #[cfg(feature = "cuda")]
    #[test]
    fn test_gaussian_topk_gpu_matches_cpu() {
        if !is_cuda_available() {
            return; // silent skip on GPU-less machines
        }

        let n = 500;
        let dim = 64; // divisible by 4 -> float4 path
        let n_queries = 4;
        let k = 10;

        let (dataset, alpha, kappa, queries) = gaussian_test_data(n, dim, n_queries, 42);

        let cpu = cpu_gaussian_topk(&queries, &dataset, &alpha, &kappa, n, dim, n_queries, k);

        // Direct GPU kernel path (None => PTX/GPU kernel unavailable -> skip).
        let gpu = cuda_kernel::gpu_gaussian_topk(
            &queries, n_queries, &dataset, &alpha, &kappa, n, dim, k,
        );
        if let Some(gpu) = gpu {
            assert_eq!(gpu.len(), n_queries);
            for q in 0..n_queries {
                assert_eq!(gpu[q].len(), k, "query {} missing results", q);
                for (rank, ((gi, gs), (ci, cs))) in gpu[q].iter().zip(cpu[q].iter()).enumerate() {
                    assert_eq!(
                        gi, ci,
                        "query {} rank {}: GPU idx {} != CPU idx {}",
                        q, rank, gi, ci
                    );
                    let diff = (gs - cs).abs();
                    assert!(
                        diff < 1e-5,
                        "query {} rank {} idx {}: GPU score {} vs CPU {} (diff {})",
                        q,
                        rank,
                        gi,
                        gs,
                        cs,
                        diff
                    );
                }
            }
            eprintln!(
                "[gaussian-topk test] 500x64D Q=4 K=10 GPU==CPU OK; top score q0: {:.6e}",
                gpu[0][0].1
            );
        } else {
            eprintln!("[gaussian-topk test] GPU kernel unavailable -> CPU-only check");
            // Public wrapper must still return the CPU reference.
            let via_wrapper =
                gaussian_topk_scores(&queries, &dataset, &alpha, &kappa, n, dim, n_queries, k)
                    .expect("wrapper must fall back to CPU");
            assert_eq!(via_wrapper, cpu);
        }
    }

    /// Gaussian top-k GPU vs CPU: D NOT divisible by 4 (scalar path) and K=1.
    #[cfg(feature = "cuda")]
    #[test]
    fn test_gaussian_topk_gpu_non_multiple4_k1() {
        if !is_cuda_available() {
            return;
        }

        let n = 500;
        let dim = 63; // NOT divisible by 4
        let n_queries = 3;
        let k = 1;

        let (dataset, alpha, kappa, queries) = gaussian_test_data(n, dim, n_queries, 7);

        let cpu = cpu_gaussian_topk(&queries, &dataset, &alpha, &kappa, n, dim, n_queries, k);
        let gpu = cuda_kernel::gpu_gaussian_topk(
            &queries, n_queries, &dataset, &alpha, &kappa, n, dim, k,
        );
        if let Some(gpu) = gpu {
            for q in 0..n_queries {
                assert_eq!(gpu[q].len(), 1);
                assert_eq!(
                    gpu[q][0].0, cpu[q][0].0,
                    "query {}: GPU idx {} != CPU idx {}",
                    q, gpu[q][0].0, cpu[q][0].0
                );
                let diff = (gpu[q][0].1 - cpu[q][0].1).abs();
                assert!(diff < 1e-5, "query {}: diff {}", q, diff);
            }
            eprintln!(
                "[gaussian-topk test] D=63 K=1 GPU==CPU OK; q0 best idx {} score {:.6e}",
                gpu[0][0].0, gpu[0][0].1
            );
        }
    }

    /// Large-batch shape-only smoke test: N=20k, D=64, Q=256, K=16.
    /// GPU path when available; silent skip otherwise (CPU would take too long).
    #[cfg(feature = "cuda")]
    #[test]
    fn test_gaussian_topk_large_batch_shapes() {
        if !is_cuda_available() {
            return; // silent skip
        }

        let n = 20_000;
        let dim = 64;
        let n_queries = 256;
        let k = 16;

        let (dataset, alpha, kappa, queries) = gaussian_test_data(n, dim, n_queries, 99);

        let result = cuda_kernel::gpu_gaussian_topk(
            &queries, n_queries, &dataset, &alpha, &kappa, n, dim, k,
        )
        .expect("GPU kernel must handle 20k x 64 x 256 x 16");

        assert_eq!(result.len(), n_queries);
        for (q, row) in result.iter().enumerate() {
            assert_eq!(row.len(), k, "query {}", q);
            for (rank, &(i, s)) in row.iter().enumerate() {
                assert!(i < n, "query {} rank {} idx {} >= N", q, rank, i);
                if rank > 0 {
                    assert!(
                        row[rank - 1].1 >= s,
                        "query {}: scores not descending at rank {}",
                        q,
                        rank
                    );
                }
            }
        }
        eprintln!("[gaussian-topk test] 20k x 64D Q=256 K=16 shape check OK");
    }

    /// Deterministic pseudo-random test data (same LCG as existing tests).
    /// Returns (dataset [N*D], alpha [N] in [0.1,1], kappa [N] in [0.05,1],
    /// queries [Q*D]).
    #[cfg_attr(not(feature = "cuda"), allow(dead_code))]
    fn gaussian_test_data(
        n: usize,
        dim: usize,
        n_queries: usize,
        seed: u64,
    ) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let mut state = seed;
        let next = |state: &mut u64| -> f32 {
            *state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((*state >> 33) as f32) / (u32::MAX as f32)
        };
        let dataset: Vec<f32> = (0..n * dim).map(|_| next(&mut state) * 2.0 - 1.0).collect();
        let alpha: Vec<f32> = (0..n).map(|_| 0.1 + 0.9 * next(&mut state)).collect();
        let kappa: Vec<f32> = (0..n).map(|_| 0.05 + 0.95 * next(&mut state)).collect();
        let queries: Vec<f32> = (0..n_queries * dim)
            .map(|_| next(&mut state) * 2.0 - 1.0)
            .collect();
        (dataset, alpha, kappa, queries)
    }

    /// Benchmark: PTX kernel vs CPU-only for 10K vectors.
    /// Only meaningful on machines with GPU.
    #[cfg(feature = "cuda")]
    #[test]
    fn test_ptx_kernel_performance() {
        use cuda_kernel::GpuIndex;
        use std::time::Instant;

        if !is_cuda_available() {
            return;
        }

        let dim = 640;
        let n = 10_000;
        let n_queries = 100;

        let mut dataset = vec![0.0f32; n * dim];
        let mut state = 12345u64;
        for val in dataset.iter_mut() {
            state = state
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            *val = ((state >> 33) as f32) / (u32::MAX as f32) * 2.0 - 1.0;
        }
        let queries = vec![0.3f32; n_queries * dim];

        // CPU benchmark
        let t0 = Instant::now();
        for q in 0..n_queries {
            let q_slice = &queries[q * dim..(q + 1) * dim];
            let mut _dists: Vec<f32> = (0..n)
                .map(|i| {
                    let row = &dataset[i * dim..(i + 1) * dim];
                    q_slice
                        .iter()
                        .zip(row.iter())
                        .map(|(a, b)| (a - b).powi(2))
                        .sum()
                })
                .collect();
        }
        let cpu_ms = t0.elapsed().as_millis();

        // GPU benchmark with persistent index
        let mut idx = GpuIndex::new().expect("CUDA context failed");
        assert!(idx.upload_dataset(&dataset, n, dim));

        let t1 = Instant::now();
        let gpu_batch = idx
            .batch_l2_distances(&queries, n_queries)
            .expect("batch L2 failed");
        let gpu_ms = t1.elapsed().as_millis();
        assert_eq!(gpu_batch.len(), n_queries);
        assert_eq!(gpu_batch[0].len(), n);

        eprintln!(
            "[splatsdb benchmark] 10K x 640D, {} queries | CPU: {}ms | GPU: {}ms | speedup: {:.2}x",
            n_queries,
            cpu_ms,
            gpu_ms,
            cpu_ms as f64 / gpu_ms.max(1) as f64
        );
    }
}
