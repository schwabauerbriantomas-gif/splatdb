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

use cudarc::driver::{
    CudaContext, CudaFunction, CudaModule, CudaSlice, CudaStream, LaunchConfig, PushKernelArg,
};
use cudarc::nvrtc::Ptx;
use std::sync::Arc;

/// Persistent GPU state — keeps dataset in VRAM between calls.
pub struct GpuIndex {
    stream: Arc<CudaStream>,
    data_gpu: Option<CudaSlice<f32>>,
    n_vectors: usize,
    dim: usize,
    /// PTX module loaded ONCE at construction — module JIT used to run on every
    /// call (cuda_kernel.rs:135,228,319,430 pattern), dominating launch cost.
    kernel_module: Option<Arc<CudaModule>>,
    /// Per-splat amplitudes [N] (see upload_alpha_kappa).
    alpha_gpu: Option<CudaSlice<f32>>,
    /// Per-splat concentrations [N] (see upload_alpha_kappa).
    kappa_gpu: Option<CudaSlice<f32>>,
}

impl GpuIndex {
    /// Create a new GPU index. Returns None if CUDA is unavailable.
    pub fn new() -> Option<Self> {
        let ctx = match CudaContext::new(0) {
            Ok(c) => c,
            Err(e) => {
                eprintln!("[splatsdb] CUDA context creation failed: {:?}", e);
                return None;
            }
        };
        eprintln!("[splatsdb] CUDA context created successfully");
        let stream = ctx.default_stream();

        // Load the PTX module once and keep it alive for the index lifetime.
        let kernel_module = Self::load_ptx_module(&ctx);

        Some(GpuIndex {
            stream,
            data_gpu: None,
            n_vectors: 0,
            dim: 0,
            kernel_module,
            alpha_gpu: None,
            kappa_gpu: None,
        })
    }

    /// Load the distance.cu PTX module. None if M2M_PTX_PATH is unset/empty,
    /// the file is missing, or cuModuleLoad fails.
    fn load_ptx_module(ctx: &Arc<CudaContext>) -> Option<Arc<CudaModule>> {
        let ptx_path = option_env!("M2M_PTX_PATH")?;
        if ptx_path.is_empty() {
            eprintln!("[splatsdb] No PTX kernels (built without nvcc or CUDA toolkit)");
            return None;
        }
        if !std::path::Path::new(ptx_path).exists() {
            eprintln!("[splatsdb] PTX path set but file not found: {}", ptx_path);
            return None;
        }
        match ctx.load_module(Ptx::from_file(ptx_path)) {
            Ok(m) => {
                eprintln!("[splatsdb] PTX module loaded (cached): {}", ptx_path);
                Some(m)
            }
            Err(e) => {
                eprintln!("[splatsdb] PTX module load failed: {:?}", e);
                None
            }
        }
    }

    /// Fetch a kernel function from the cached PTX module.
    fn kernel_fn(&self, name: &str) -> Option<CudaFunction> {
        let module = self.kernel_module.as_ref()?;
        match module.load_function(name) {
            Ok(f) => Some(f),
            Err(e) => {
                eprintln!(
                    "[splatsdb] Kernel '{}' not found in PTX module: {:?}",
                    name, e
                );
                None
            }
        }
    }

    /// Upload dataset to GPU memory (persists until next upload or drop).
    pub fn upload_dataset(&mut self, dataset: &[f32], n_vectors: usize, dim: usize) -> bool {
        // Validate dimensions match dataset size
        if dim == 0 || n_vectors == 0 {
            return false;
        }
        let expected = n_vectors.checked_mul(dim);
        if expected != Some(dataset.len()) {
            return false;
        }
        self.data_gpu = self.stream.clone_htod(dataset).ok();
        if self.data_gpu.is_some() {
            self.n_vectors = n_vectors;
            self.dim = dim;
            true
        } else {
            false
        }
    }

    /// Upload per-splat Gaussian parameters: amplitudes alpha [N] and
    /// concentrations kappa [N]. Both persist in VRAM until the next upload.
    /// `n` must match the uploaded dataset size; returns false otherwise.
    pub fn upload_alpha_kappa(&mut self, alpha: &[f32], kappa: &[f32]) -> bool {
        if alpha.len() != self.n_vectors || kappa.len() != self.n_vectors {
            eprintln!(
                "[splatsdb] upload_alpha_kappa size mismatch: alpha={}, kappa={}, n_vectors={}",
                alpha.len(),
                kappa.len(),
                self.n_vectors
            );
            return false;
        }
        let alpha_gpu = self.stream.clone_htod(alpha).ok();
        let kappa_gpu = self.stream.clone_htod(kappa).ok();
        match (alpha_gpu, kappa_gpu) {
            (Some(a), Some(k)) => {
                self.alpha_gpu = Some(a);
                self.kappa_gpu = Some(k);
                true
            }
            _ => {
                self.alpha_gpu = None;
                self.kappa_gpu = None;
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
            let dist: f32 = query
                .iter()
                .zip(row.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum();
            distances.push(dist);
        }
        Some(distances)
    }

    /// Launch custom L2 distance PTX kernel.
    fn launch_l2_kernel(
        &self,
        query: &[f32],
        data_gpu: &CudaSlice<f32>,
        n: usize,
        dim: usize,
    ) -> Option<Vec<f32>> {
        let ptx_path = option_env!("M2M_PTX_PATH");
        if ptx_path.is_none_or(|p| p.is_empty()) {
            return None;
        }

        let func: CudaFunction = self.kernel_fn("l2_distance_kernel")?;

        // Bounds checks before i32 casts
        if n > i32::MAX as usize {
            return None;
        }
        if dim > i32::MAX as usize {
            return None;
        }

        // Allocate GPU buffers
        let query_gpu = self.stream.clone_htod(query).ok()?;
        let mut output_gpu = self.stream.alloc_zeros::<f32>(n).ok()?;

        // Shared memory = dim * sizeof(float) for query cache
        let shared_mem_bytes = dim * 4;

        // Launch configuration: 256 threads per block
        let block_dim = 256;
        let grid_dim = n.div_ceil(block_dim);

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
            self.stream
                .launch_builder(&func)
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
                let dist: f32 = q_slice
                    .iter()
                    .zip(row.iter())
                    .map(|(a, b)| (a - b).powi(2))
                    .sum();
                distances.push(dist);
            }
            all_dists.push(distances);
        }
        Some(all_dists)
    }

    fn launch_batch_l2_kernel(
        &self,
        queries: &[f32],
        data_gpu: &CudaSlice<f32>,
        n_queries: usize,
        n: usize,
        dim: usize,
    ) -> Option<Vec<Vec<f32>>> {
        let ptx_path = option_env!("M2M_PTX_PATH");
        if ptx_path.is_none_or(|p| p.is_empty()) {
            return None;
        }

        let func: CudaFunction = self.kernel_fn("batch_l2_distance_kernel")?;

        let queries_gpu = self.stream.clone_htod(queries).ok()?;
        let total_output = n_queries.checked_mul(n)?;
        let mut output_gpu = self.stream.alloc_zeros::<f32>(total_output).ok()?;

        // Bounds checks before i32/u32 casts
        if n > i32::MAX as usize {
            return None;
        }
        if dim > i32::MAX as usize {
            return None;
        }
        if n_queries > i32::MAX as usize {
            return None;
        }
        if n_queries > u32::MAX as usize {
            return None;
        }

        let block_dim = 256;
        let grid_x = (n.div_ceil(block_dim)) as u32;
        let grid_y = n_queries as u32;

        let cfg = LaunchConfig {
            grid_dim: (grid_x, grid_y, 1),
            block_dim: (block_dim as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        // SAFETY: Batch L2 kernel args are valid GPU allocations. Grid dims: (blocks_per_query, n_queries).
        // shared_mem_bytes is 0 (no shared mem for batch). Kernel writes exactly n_queries*n floats.
        unsafe {
            self.stream
                .launch_builder(&func)
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

    fn launch_cosine_kernel(
        &self,
        query: &[f32],
        data_gpu: &CudaSlice<f32>,
        n: usize,
        dim: usize,
    ) -> Option<Vec<f32>> {
        let ptx_path = option_env!("M2M_PTX_PATH");
        if ptx_path.is_none_or(|p| p.is_empty()) {
            return None;
        }

        let func: CudaFunction = self.kernel_fn("cosine_distance_kernel")?;

        let query_gpu = self.stream.clone_htod(query).ok()?;
        let mut output_gpu = self.stream.alloc_zeros::<f32>(n).ok()?;

        // Bounds checks before i32/u32 casts
        if n > i32::MAX as usize {
            return None;
        }
        if dim > i32::MAX as usize {
            return None;
        }

        let shared_mem_bytes = dim * 4 + 256 * 4; // query + reduction workspace
        let block_dim = 256;
        let grid_dim = n.div_ceil(block_dim);

        let cfg = LaunchConfig {
            grid_dim: (grid_dim as u32, 1, 1),
            block_dim: (block_dim as u32, 1, 1),
            shared_mem_bytes: shared_mem_bytes as u32,
        };

        // SAFETY: Cosine kernel args are valid GPU allocations. shared_mem_bytes covers query cache
        // (dim*4) plus reduction workspace (256*4). Grid/block dims computed from dataset size.
        unsafe {
            self.stream
                .launch_builder(&func)
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

    pub fn n_vectors(&self) -> usize {
        self.n_vectors
    }
    pub fn dim(&self) -> usize {
        self.dim
    }
    pub fn is_loaded(&self) -> bool {
        self.data_gpu.is_some()
    }

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
        if let Some(result) =
            self.launch_topk_kernel(queries, data_gpu, n_queries, n, dim, k, metric)
        {
            return Some(result);
        }
        eprintln!("[splatsdb] top-k PTX kernel unavailable, using fallback");

        // Fallback to separate distance + CPU top-k
        let all_dists = if metric == "cosine" {
            (0..n_queries)
                .map(|q| self.cosine_distances(&queries[q * dim..(q + 1) * dim]))
                .collect::<Option<Vec<Vec<f32>>>>()?
        } else {
            self.batch_l2_distances(queries, n_queries)?
        };

        let results: Vec<(Vec<usize>, Vec<f32>)> = all_dists
            .iter()
            .map(|distances| {
                let mut indices: Vec<usize> = (0..distances.len()).collect();
                indices.sort_by(|a, b| distances[*a].partial_cmp(&distances[*b]).unwrap());
                indices.truncate(k);
                let dists: Vec<f32> = indices.iter().map(|&i| distances[i]).collect();
                (indices, dists)
            })
            .collect();
        Some(results)
    }

    /// Launch combined top-k PTX kernel.
    #[allow(clippy::too_many_arguments)]
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
        let ptx_path = option_env!("M2M_PTX_PATH");
        if ptx_path.is_none_or(|p| p.is_empty()) || k > 32 {
            return None; // Kernel supports K <= 32
        }

        let kernel_name = if metric == "cosine" {
            "cosine_topk_kernel"
        } else {
            "l2_topk_kernel"
        };
        eprintln!(
            "[splatsdb] Launching {} PTX kernel (Q={}, N={}, D={}, K={})",
            kernel_name, n_queries, n, dim, k
        );
        let func: CudaFunction = self.kernel_fn(kernel_name)?;

        // Bounds checks before i32/u32 casts
        if n_queries > i32::MAX as usize {
            return None;
        }
        if n_queries > u32::MAX as usize {
            return None;
        }
        if n > i32::MAX as usize {
            return None;
        }
        if dim > i32::MAX as usize {
            return None;
        }
        if k > i32::MAX as usize {
            return None;
        }

        let queries_gpu = self.stream.clone_htod(queries).ok()?;
        let nq_k = n_queries.checked_mul(k)?;
        let mut idx_gpu = self.stream.alloc_zeros::<i32>(nq_k).ok()?;
        let mut dist_gpu = self.stream.alloc_zeros::<f32>(nq_k).ok()?;

        let block_dim: u32 = 256;
        // Shared memory: D floats (query) + blockDim*K floats (dist) + blockDim*K ints (idx)
        let shared_mem_bytes = dim
            .saturating_add((block_dim as usize).saturating_mul(k).saturating_mul(2))
            .saturating_mul(4);

        let cfg = LaunchConfig {
            grid_dim: (n_queries as u32, 1, 1), // one block per query
            block_dim: (block_dim, 1, 1),
            shared_mem_bytes: shared_mem_bytes as u32,
        };

        // SAFETY: Top-k kernel args are valid GPU allocations. idx_gpu/dist_gpu sized n_queries*k.
        // shared_mem_bytes covers query (D floats) + per-thread top-k arrays (dist+idx). Grid is
        // one block per query. K <= 32 enforced above. PTX kernel writes exactly n_queries*k results.
        unsafe {
            self.stream
                .launch_builder(&func)
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
                let indices: Vec<usize> = indices_flat[q * k..(q + 1) * k]
                    .iter()
                    .map(|&i| if i >= 0 { i as usize } else { 0 })
                    .collect();
                let distances: Vec<f32> = distances_flat[q * k..(q + 1) * k].to_vec();
                (indices, distances)
            })
            .collect();

        Some(results)
    }

    /// GPU-only Gaussian top-k search:
    /// score_i = alpha_i * exp(-kappa_i * ||q - mu_i||^2), higher is better.
    /// Requires dataset + alpha/kappa uploaded. Returns (index, score) pairs
    /// per query, sorted by descending score. No CPU fallback here — the
    /// module-level wrapper (gpu::gaussian_topk_scores) provides it.
    pub fn gaussian_topk_search(
        &self,
        queries: &[f32],
        n_queries: usize,
        k: usize,
    ) -> Option<Vec<Vec<(usize, f32)>>> {
        let n = self.n_vectors;
        let dim = self.dim;
        let data_gpu = self.data_gpu.as_ref()?;
        let alpha_gpu = self.alpha_gpu.as_ref()?;
        let kappa_gpu = self.kappa_gpu.as_ref()?;

        if queries.len() != n_queries.checked_mul(dim)? {
            return None;
        }

        self.launch_gaussian_topk_kernel(
            queries, data_gpu, alpha_gpu, kappa_gpu, n_queries, n, dim, k,
        )
    }

    /// Launch the gaussian_topk PTX kernel. Mirrors launch_topk_kernel.
    #[allow(clippy::too_many_arguments)]
    fn launch_gaussian_topk_kernel(
        &self,
        queries: &[f32],
        data_gpu: &CudaSlice<f32>,
        alpha_gpu: &CudaSlice<f32>,
        kappa_gpu: &CudaSlice<f32>,
        n_queries: usize,
        n: usize,
        dim: usize,
        k: usize,
    ) -> Option<Vec<Vec<(usize, f32)>>> {
        if option_env!("M2M_PTX_PATH").is_none_or(|p| p.is_empty()) || k > 32 || k == 0 {
            return None; // Kernel supports 1 <= K <= 32
        }

        let func: CudaFunction = self.kernel_fn("gaussian_topk_kernel")?;

        // Bounds checks before i32/u32 casts
        if n_queries > i32::MAX as usize || n_queries > u32::MAX as usize {
            return None;
        }
        if n > i32::MAX as usize || dim > i32::MAX as usize || k > i32::MAX as usize {
            return None;
        }

        let queries_gpu = self.stream.clone_htod(queries).ok()?;
        let nq_k = n_queries.checked_mul(k)?;
        let mut idx_gpu = self.stream.alloc_zeros::<i32>(nq_k).ok()?;
        let mut score_gpu = self.stream.alloc_zeros::<f32>(nq_k).ok()?;

        let block_dim: u32 = 256;
        // Shared memory: D floats (query) + blockDim*K floats (score) + blockDim*K ints (idx)
        let shared_mem_bytes = dim
            .saturating_add((block_dim as usize).saturating_mul(k).saturating_mul(2))
            .saturating_mul(4);

        // K=32 needs (64 + 256*32*2)*4 = 65,536 bytes on D=64 — above the
        // 48KB default. Opt in to extended dynamic shared memory (sm_86
        // supports up to 99KB per block) or fall back to the CPU path.
        if shared_mem_bytes > 48 * 1024 {
            use cudarc::driver::sys::{CUfunction_attribute, CUfunction_attribute_enum};
            let attr: CUfunction_attribute_enum =
                CUfunction_attribute::CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES;
            if func.set_attribute(attr, shared_mem_bytes as i32).is_err() {
                eprintln!(
                    "[splatsdb] gaussian_topk_kernel: opt-in extended shared memory \
                     ({} bytes) refused by driver — using CPU fallback",
                    shared_mem_bytes
                );
                return None;
            }
        }

        let cfg = LaunchConfig {
            grid_dim: (n_queries as u32, 1, 1), // one block per query
            block_dim: (block_dim, 1, 1),
            shared_mem_bytes: shared_mem_bytes as u32,
        };

        eprintln!(
            "[splatsdb] Launching gaussian_topk_kernel PTX kernel (Q={}, N={}, D={}, K={})",
            n_queries, n, dim, k
        );

        // SAFETY: gaussian_topk_kernel args are valid GPU allocations sized
        // [Q,D]/[N,D]/[N]/[N]/[Q,K]/[Q,K]. shared_mem_bytes covers query cache
        // (D floats) + per-thread top-k arrays (blockDim*K floats + ints).
        // Grid is one block per query; 1 <= K <= 32 enforced above. Kernel
        // writes exactly n_queries*k (idx, score) pairs.
        unsafe {
            self.stream
                .launch_builder(&func)
                .arg(&queries_gpu)
                .arg(data_gpu)
                .arg(alpha_gpu)
                .arg(kappa_gpu)
                .arg(&mut idx_gpu)
                .arg(&mut score_gpu)
                .arg(&(n_queries as i32))
                .arg(&(n as i32))
                .arg(&(dim as i32))
                .arg(&(k as i32))
                .launch(cfg)
                .ok()?;
        }

        // Download only K*Q results
        let indices_flat: Vec<i32> = self.stream.clone_dtoh(&idx_gpu).ok()?;
        let scores_flat: Vec<f32> = self.stream.clone_dtoh(&score_gpu).ok()?;

        let results: Vec<Vec<(usize, f32)>> = (0..n_queries)
            .map(|q| {
                indices_flat[q * k..(q + 1) * k]
                    .iter()
                    .zip(scores_flat[q * k..(q + 1) * k].iter())
                    .filter(|(&i, _)| i >= 0)
                    .map(|(&i, &s)| (i as usize, s))
                    .collect()
            })
            .collect();

        Some(results)
    }
}

// ============================================================================
// Public API — wraps GpuIndex for the gpu module
// ============================================================================

/// GPU L2 distance computation with persistent dataset.
pub fn gpu_l2_distances(
    query: &[f32],
    dataset: &[f32],
    n_rows: usize,
    dim: usize,
) -> Option<Vec<f32>> {
    let mut idx = GpuIndex::new()?;
    idx.upload_dataset(dataset, n_rows, dim);
    idx.l2_distances(query)
}

/// GPU cosine distance computation.
pub fn gpu_cosine_distances(
    query: &[f32],
    dataset: &[f32],
    n_rows: usize,
    dim: usize,
) -> Option<Vec<f32>> {
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

/// GPU Gaussian top-k search — uses gaussian_topk PTX kernel when available.
/// score_i = alpha_i * exp(-kappa_i * ||q - mu_i||^2), descending.
#[allow(clippy::too_many_arguments)]
pub fn gpu_gaussian_topk(
    queries: &[f32],
    n_queries: usize,
    dataset: &[f32],
    alpha: &[f32],
    kappa: &[f32],
    n_vectors: usize,
    dim: usize,
    k: usize,
) -> Option<Vec<Vec<(usize, f32)>>> {
    let mut idx = GpuIndex::new()?;
    if !idx.upload_dataset(dataset, n_vectors, dim) {
        return None;
    }
    if !idx.upload_alpha_kappa(alpha, kappa) {
        return None;
    }
    idx.gaussian_topk_search(queries, n_queries, k)
}
