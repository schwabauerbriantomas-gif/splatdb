//! CUDA kernel implementations using cudarc + custom PTX kernels.
//!
//! Architecture:
//! - Custom PTX kernels (distance.cu) compiled at build time by build.rs/nvcc
//! - cudarc loads PTX module and launches kernels
//! - Dataset persists in GPU memory between calls (no re-upload)
//! - CPU fallback always available when CUDA fails
//!
//! Performance optimizations:
//! - float4 vectorized loads for D divisible by 4
//! - Shared memory caching for query vectors
//! - __launch_bounds__(256) for optimal occupancy on sm_86
//! - --use_fast_math and -O3 during PTX compilation

use std::sync::Arc;
use cudarc::driver::{CudaContext, CudaSlice, CudaFunction, CudaStream, LaunchConfig, PushKernelArg};
use cudarc::nvrtc::Ptx;

/// Persistent GPU state — keeps dataset in VRAM between calls.
pub struct GpuIndex {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
    data_gpu: Option<CudaSlice<f32>>,
    n_vectors: usize,
    dim: usize,
    kernel_loaded: bool,
}

impl GpuIndex {
    /// Create a new GPU index. Returns None if CUDA is unavailable.
    pub fn new() -> Option<Self> {
        let ctx = CudaContext::new(0).ok()?;
        let stream = ctx.default_stream();
        Some(GpuIndex {
            ctx,
            stream,
            data_gpu: None,
            n_vectors: 0,
            dim: 0,
            kernel_loaded: false,
        })
    }

    /// Upload dataset to GPU memory (persists until next upload or drop).
    pub fn upload_dataset(&mut self, dataset: &[f32], n_vectors: usize, dim: usize) -> bool {
        self.data_gpu = self.stream.clone_htod(dataset).ok();
        if self.data_gpu.is_some() {
            self.n_vectors = n_vectors;
            self.dim = dim;
            self.kernel_loaded = self.try_load_kernels();
            true
        } else {
            false
        }
    }

    /// Try to load PTX kernels compiled at build time.
    fn try_load_kernels(&self) -> bool {
        let ptx_path = option_env!("M2M_PTX_PATH");
        match ptx_path {
            Some(path) if !path.is_empty() => {
                // Verify PTX file actually exists
                let exists = std::path::Path::new(path).exists();
                if exists {
                    eprintln!("[m2m] PTX kernels available: {}", path);
                } else {
                    eprintln!("[m2m] PTX path set but file not found: {}", path);
                }
                exists
            }
            _ => {
                eprintln!("[m2m] No PTX kernels (built without nvcc or CUDA toolkit)");
                false
            }
        }
    }

    /// L2 distances: query vs all vectors. Returns distances or None.
    pub fn l2_distances(&self, query: &[f32]) -> Option<Vec<f32>> {
        let n = self.n_vectors;
        let dim = self.dim;
        let data_gpu = self.data_gpu.as_ref()?;

        // Try PTX kernel path first
        if let Some(result) = self.launch_l2_kernel(query, data_gpu, n, dim) {
            return Some(result);
        }

        // Fallback: upload query, download dataset, compute on CPU
        let data_back: Vec<f32> = self.stream.clone_dtoh(data_gpu).ok()?;
        let mut distances = Vec::with_capacity(n);
        for i in 0..n {
            let row = &data_back[i * dim..(i + 1) * dim];
            let dist: f32 = query.iter().zip(row.iter()).map(|(a, b)| (a - b).powi(2)).sum();
            distances.push(dist);
        }
        Some(distances)
    }

    /// Launch custom L2 distance PTX kernel.
    fn launch_l2_kernel(&self, query: &[f32], data_gpu: &CudaSlice<f32>, n: usize, dim: usize) -> Option<Vec<f32>> {
        let ptx_path = option_env!("M2M_PTX_PATH")?;
        if ptx_path.is_empty() {
            return None;
        }

        // Load PTX module
        let module = self.ctx.load_module(Ptx::from_file(ptx_path)).ok()?;
        let func: CudaFunction = module.load_function("l2_distance_kernel").ok()?;

        // Allocate GPU buffers
        let query_gpu = self.stream.clone_htod(query).ok()?;
        let mut output_gpu = self.stream.alloc_zeros::<f32>(n).ok()?;

        // Shared memory = dim * sizeof(float) for query cache
        let shared_mem_bytes = dim * 4;

        // Launch configuration: 256 threads per block
        let block_dim = 256;
        let grid_dim = (n + block_dim - 1) / block_dim;

        let cfg = LaunchConfig {
            grid_dim: (grid_dim as u32, 1, 1),
            block_dim: (block_dim as u32, 1, 1),
            shared_mem_bytes: shared_mem_bytes as u32,
        };

        // Launch kernel
        // cudarc launch: params are (func, cfg, &mut output_gpu, ..., n, dim)
        // SAFETY: Kernel arguments are valid GPU allocations (query_gpu, data_gpu, output_gpu)
        // with correct sizes. Grid/block dims computed from dataset size. PTX kernel compiled from
        // verified CUDA C source. Synchronization via cudarc ensures reads after kernel completion.
        unsafe {
            self.stream.launch_builder(&func)
                .arg(&query_gpu)
                .arg(data_gpu)
                .arg(&mut output_gpu)
                .arg(&(n as i32))
                .arg(&(dim as i32))
                .launch(cfg)
                .ok()?;
        }

        // Download results
        self.stream.clone_dtoh(&output_gpu).ok()
    }

    /// Batch L2 distances: multiple queries vs all vectors.
    pub fn batch_l2_distances(&self, queries: &[f32], n_queries: usize) -> Option<Vec<Vec<f32>>> {
        let n = self.n_vectors;
        let dim = self.dim;
        let data_gpu = self.data_gpu.as_ref()?;

        // Try batch kernel
        if let Some(result) = self.launch_batch_l2_kernel(queries, data_gpu, n_queries, n, dim) {
            return Some(result);
        }

        // Fallback: per-query
        let data_back: Vec<f32> = self.stream.clone_dtoh(data_gpu).ok()?;
        let mut all_dists = Vec::with_capacity(n_queries);
        for q in 0..n_queries {
            let q_slice = &queries[q * dim..(q + 1) * dim];
            let mut distances = Vec::with_capacity(n);
            for i in 0..n {
                let row = &data_back[i * dim..(i + 1) * dim];
                let dist: f32 = q_slice.iter().zip(row.iter()).map(|(a, b)| (a - b).powi(2)).sum();
                distances.push(dist);
            }
            all_dists.push(distances);
        }
        Some(all_dists)
    }

    fn launch_batch_l2_kernel(&self, queries: &[f32], data_gpu: &CudaSlice<f32>, n_queries: usize, n: usize, dim: usize) -> Option<Vec<Vec<f32>>> {
        let ptx_path = option_env!("M2M_PTX_PATH")?;
        if ptx_path.is_empty() {
            return None;
        }

        let module = self.ctx.load_module(Ptx::from_file(ptx_path)).ok()?;
        let func: CudaFunction = module.load_function("batch_l2_distance_kernel").ok()?;

        let queries_gpu = self.stream.clone_htod(queries).ok()?;
        let mut output_gpu = self.stream.alloc_zeros::<f32>(n_queries * n).ok()?;

        let block_dim = 256;
        let grid_x = ((n + block_dim - 1) / block_dim) as u32;
        let grid_y = n_queries as u32;

        let cfg = LaunchConfig {
            grid_dim: (grid_x, grid_y, 1),
            block_dim: (block_dim as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        // SAFETY: Batch L2 kernel args are valid GPU allocations. Grid dims: (blocks_per_query, n_queries).
        // shared_mem_bytes is 0 (no shared mem for batch). Kernel writes exactly n_queries*n floats.
        unsafe {
            self.stream.launch_builder(&func)
                .arg(&queries_gpu)
                .arg(data_gpu)
                .arg(&mut output_gpu)
                .arg(&(n_queries as i32))
                .arg(&(n as i32))
                .arg(&(dim as i32))
                .launch(cfg)
                .ok()?;
        }

        let flat: Vec<f32> = self.stream.clone_dtoh(&output_gpu).ok()?;
        // De-interleave: output is [Q, N] row-major
        let result: Vec<Vec<f32>> = (0..n_queries)
            .map(|q| flat[q * n..(q + 1) * n].to_vec())
            .collect();
        Some(result)
    }

    /// Cosine distances: query vs all vectors.
    pub fn cosine_distances(&self, query: &[f32]) -> Option<Vec<f32>> {
        let n = self.n_vectors;
        let dim = self.dim;
        let data_gpu = self.data_gpu.as_ref()?;

        // Try PTX kernel
        if let Some(result) = self.launch_cosine_kernel(query, data_gpu, n, dim) {
            return Some(result);
        }

        // Fallback
        let data_back: Vec<f32> = self.stream.clone_dtoh(data_gpu).ok()?;
        let q_norm: f32 = query.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-10);
        let distances: Vec<f32> = (0..n)
            .map(|i| {
                let row = &data_back[i * dim..(i + 1) * dim];
                let dot: f32 = query.iter().zip(row.iter()).map(|(a, b)| a * b).sum();
                let r_norm: f32 = row.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-10);
                1.0 - dot / (q_norm * r_norm)
            })
            .collect();
        Some(distances)
    }

    fn launch_cosine_kernel(&self, query: &[f32], data_gpu: &CudaSlice<f32>, n: usize, dim: usize) -> Option<Vec<f32>> {
        let ptx_path = option_env!("M2M_PTX_PATH")?;
        if ptx_path.is_empty() {
            return None;
        }

        let module = self.ctx.load_module(Ptx::from_file(ptx_path)).ok()?;
        let func: CudaFunction = module.load_function("cosine_distance_kernel").ok()?;

        let query_gpu = self.stream.clone_htod(query).ok()?;
        let mut output_gpu = self.stream.alloc_zeros::<f32>(n).ok()?;

        let shared_mem_bytes = dim * 4 + 256 * 4; // query + reduction workspace
        let block_dim = 256;
        let grid_dim = (n + block_dim - 1) / block_dim;

        let cfg = LaunchConfig {
            grid_dim: (grid_dim as u32, 1, 1),
            block_dim: (block_dim as u32, 1, 1),
            shared_mem_bytes: shared_mem_bytes as u32,
        };

        // SAFETY: Cosine kernel args are valid GPU allocations. shared_mem_bytes covers query cache
        // (dim*4) plus reduction workspace (256*4). Grid/block dims computed from dataset size.
        unsafe {
            self.stream.launch_builder(&func)
                .arg(&query_gpu)
                .arg(data_gpu)
                .arg(&mut output_gpu)
                .arg(&(n as i32))
                .arg(&(dim as i32))
                .launch(cfg)
                .ok()?;
        }

        self.stream.clone_dtoh(&output_gpu).ok()
    }

    pub fn n_vectors(&self) -> usize { self.n_vectors }
    pub fn dim(&self) -> usize { self.dim }
    pub fn is_loaded(&self) -> bool { self.data_gpu.is_some() }

    /// GPU-only top-k search using combined distance+topk PTX kernel.
    /// Returns (indices, distances) for each query. No full distance matrix download.
    pub fn topk_search(
        &self,
        queries: &[f32],
        n_queries: usize,
        k: usize,
        metric: &str,
    ) -> Option<Vec<(Vec<usize>, Vec<f32>)>> {
        let n = self.n_vectors;
        let dim = self.dim;
        let data_gpu = self.data_gpu.as_ref()?;

        // Try PTX top-k kernel
        if let Some(result) = self.launch_topk_kernel(queries, data_gpu, n_queries, n, dim, k, metric) {
            return Some(result);
        }
        eprintln!("[m2m] top-k PTX kernel unavailable, using fallback");

        // Fallback to separate distance + CPU top-k
        let all_dists = if metric == "cosine" {
            (0..n_queries)
                .map(|q| self.cosine_distances(&queries[q * dim..(q + 1) * dim]))
                .collect::<Option<Vec<Vec<f32>>>>()?
        } else {
            self.batch_l2_distances(queries, n_queries)?
        };

        let results: Vec<(Vec<usize>, Vec<f32>)> = all_dists.iter().map(|distances| {
            let mut indices: Vec<usize> = (0..distances.len()).collect();
            indices.sort_by(|a, b| distances[*a].partial_cmp(&distances[*b]).unwrap());
            indices.truncate(k);
            let dists: Vec<f32> = indices.iter().map(|&i| distances[i]).collect();
            (indices, dists)
        }).collect();
        Some(results)
    }

    /// Launch combined top-k PTX kernel.
    fn launch_topk_kernel(
        &self,
        queries: &[f32],
        data_gpu: &CudaSlice<f32>,
        n_queries: usize,
        n: usize,
        dim: usize,
        k: usize,
        metric: &str,
    ) -> Option<Vec<(Vec<usize>, Vec<f32>)>> {
        let ptx_path = option_env!("M2M_PTX_PATH")?;
        if ptx_path.is_empty() || k > 32 {
            return None; // Kernel supports K <= 32
        }

        let module = self.ctx.load_module(Ptx::from_file(ptx_path)).ok()?;
        let kernel_name = if metric == "cosine" { "cosine_topk_kernel" } else { "l2_topk_kernel" };
        eprintln!("[m2m] Launching {} PTX kernel (Q={}, N={}, D={}, K={})", kernel_name, n_queries, n, dim, k);
        let func: CudaFunction = module.load_function(kernel_name).ok()?;

        let queries_gpu = self.stream.clone_htod(queries).ok()?;
        let mut idx_gpu = self.stream.alloc_zeros::<i32>(n_queries * k).ok()?;
        let mut dist_gpu = self.stream.alloc_zeros::<f32>(n_queries * k).ok()?;

        let block_dim: u32 = 256;
        // Shared memory: D floats (query) + blockDim*K floats (dist) + blockDim*K ints (idx)
        let shared_mem_bytes = (dim + (block_dim as usize) * k * 2) * 4;

        let cfg = LaunchConfig {
            grid_dim: (n_queries as u32, 1, 1),  // one block per query
            block_dim: (block_dim, 1, 1),
            shared_mem_bytes: shared_mem_bytes as u32,
        };

        // SAFETY: Top-k kernel args are valid GPU allocations. idx_gpu/dist_gpu sized n_queries*k.
        // shared_mem_bytes covers query (D floats) + per-thread top-k arrays (dist+idx). Grid is
        // one block per query. K <= 32 enforced above. PTX kernel writes exactly n_queries*k results.
        unsafe {
            self.stream.launch_builder(&func)
                .arg(&queries_gpu)
                .arg(data_gpu)
                .arg(&mut idx_gpu)
                .arg(&mut dist_gpu)
                .arg(&(n_queries as i32))
                .arg(&(n as i32))
                .arg(&(dim as i32))
                .arg(&(k as i32))
                .launch(cfg)
                .ok()?;
        }

        // Download only K*Q results (tiny compared to N*Q)
        let indices_flat: Vec<i32> = self.stream.clone_dtoh(&idx_gpu).ok()?;
        let distances_flat: Vec<f32> = self.stream.clone_dtoh(&dist_gpu).ok()?;

        let results: Vec<(Vec<usize>, Vec<f32>)> = (0..n_queries)
            .map(|q| {
                let indices: Vec<usize> = indices_flat[q*k..(q+1)*k]
                    .iter().map(|&i| if i >= 0 { i as usize } else { 0 }).collect();
                let distances: Vec<f32> = distances_flat[q*k..(q+1)*k].to_vec();
                (indices, distances)
            })
            .collect();

        Some(results)
    }
}

// ============================================================================
// Public API — wraps GpuIndex for the gpu module
// ============================================================================

/// GPU L2 distance computation with persistent dataset.
pub fn gpu_l2_distances(query: &[f32], dataset: &[f32], n_rows: usize, dim: usize) -> Option<Vec<f32>> {
    let mut idx = GpuIndex::new()?;
    idx.upload_dataset(dataset, n_rows, dim);
    idx.l2_distances(query)
}

/// GPU cosine distance computation.
pub fn gpu_cosine_distances(query: &[f32], dataset: &[f32], n_rows: usize, dim: usize) -> Option<Vec<f32>> {
    let mut idx = GpuIndex::new()?;
    idx.upload_dataset(dataset, n_rows, dim);
    idx.cosine_distances(query)
}

/// GPU batch search — uses top-k PTX kernel when available.
pub fn gpu_batch_search(
    queries: &[f32],
    n_queries: usize,
    dataset: &[f32],
    n_vectors: usize,
    dim: usize,
    k: usize,
    metric: &str,
) -> Option<Vec<(Vec<usize>, Vec<f32>)>> {
    let mut idx = GpuIndex::new()?;
    idx.upload_dataset(dataset, n_vectors, dim);
    idx.topk_search(queries, n_queries, k, metric)
}
