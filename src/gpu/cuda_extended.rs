//! Extended GPU operations — quantization, clustering, geometry, LSH.
//!
//! Uses PTX kernels compiled from `kernels/extended_kernels.cu`.
//! Each method follows the same pattern as the distance kernels in cuda_kernel.rs:
//! - Load PTX module from build-time path
//! - Upload data to GPU
//! - Launch kernel
//! - Download results
//!
//! All methods have CPU fallbacks (return None on failure, caller falls back).

use cudarc::driver::{CudaContext, CudaFunction, CudaSlice, CudaStream, LaunchConfig, Ptx};
use std::sync::Arc;

/// Extended GPU operations. Owns a CUDA context and can load extended PTX kernels.
/// Unlike GpuIndex which holds a dataset, this is a stateless launcher —
/// callers upload data per-call (extended ops have different datasets each time).
pub struct GpuExtended {
    ctx: Arc<CudaContext>,
    stream: Arc<CudaStream>,
}

impl GpuExtended {
    /// Create a new extended GPU context. Returns None if CUDA unavailable.
    pub fn new() -> Option<Self> {
        let ctx = CudaContext::new(0).ok()?;
        let stream = ctx.default_stream();
        Some(GpuExtended { ctx, stream })
    }

    fn load_ext_kernel(&self, name: &str) -> Option<CudaFunction> {
        let ptx_path = option_env!("M2M_EXTENDED_PTX_PATH")?;
        if ptx_path.is_empty() {
            return None;
        }
        if !std::path::Path::new(ptx_path).exists() {
            return None;
        }
        let module = self.ctx.load_module(Ptx::from_file(ptx_path)).ok()?;
        module.load_function(name).ok()
    }

    // ========================================================================
    // 1. Rotation GEMV: batch y = R · x for N vectors
    // ========================================================================

    /// Batch rotation forward: output[i] = R · input[i] for i in 0..N.
    /// `rotation` is [D, D] row-major, `vectors` is [N, D].
    pub fn rotation_gemv(
        &self,
        vectors: &[f32],
        rotation: &[f32],
        n: usize,
        dim: usize,
    ) -> Option<Vec<f32>> {
        let func = self.load_ext_kernel("rotation_gemv_kernel")?;

        let vecs_gpu = self.stream.clone_htod(vectors).ok()?;
        let rot_gpu = self.stream.clone_htod(rotation).ok()?;
        let total = n.checked_mul(dim)?;
        let mut out_gpu = self.stream.alloc_zeros::<f32>(total).ok()?;

        let shared_mem = dim * 4; // cache input vector
        let cfg = LaunchConfig {
            grid_dim: (n as u32, 1, 1), // one block per vector
            block_dim: (256, 1, 1),
            shared_mem_bytes: shared_mem as u32,
        };

        unsafe {
            self.stream
                .launch_builder(&func)
                .arg(&vecs_gpu)
                .arg(&rot_gpu)
                .arg(&mut out_gpu)
                .arg(&(n as i32))
                .arg(&(dim as i32))
                .launch(cfg)
                .ok()?;
        }

        self.stream.clone_dtoh(&out_gpu).ok()
    }

    /// Batch rotation inverse: output[i] = R^T · input[i].
    pub fn rotation_gemv_inverse(
        &self,
        vectors: &[f32],
        rotation: &[f32],
        n: usize,
        dim: usize,
    ) -> Option<Vec<f32>> {
        let func = self.load_ext_kernel("rotation_gemv_inverse_kernel")?;

        let vecs_gpu = self.stream.clone_htod(vectors).ok()?;
        let rot_gpu = self.stream.clone_htod(rotation).ok()?;
        let total = n.checked_mul(dim)?;
        let mut out_gpu = self.stream.alloc_zeros::<f32>(total).ok()?;

        let shared_mem = dim * 4;
        let cfg = LaunchConfig {
            grid_dim: (n as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: shared_mem as u32,
        };

        unsafe {
            self.stream
                .launch_builder(&func)
                .arg(&vecs_gpu)
                .arg(&rot_gpu)
                .arg(&mut out_gpu)
                .arg(&(n as i32))
                .arg(&(dim as i32))
                .launch(cfg)
                .ok()?;
        }

        self.stream.clone_dtoh(&out_gpu).ok()
    }

    // ========================================================================
    // 2. QJL Batch Sketch + IP Estimate
    // ========================================================================

    /// Batch QJL sketch: compute sign(G · x) for N vectors.
    /// `projections` is [G, D], returns [N, G] flat i8 signs.
    pub fn qjl_batch_sketch(
        &self,
        vectors: &[f32],
        projections: &[f32],
        n: usize,
        dim: usize,
        g: usize,
    ) -> Option<Vec<i8>> {
        let func = self.load_ext_kernel("qjl_batch_sketch_kernel")?;

        let vecs_gpu = self.stream.clone_htod(vectors).ok()?;
        let proj_gpu = self.stream.clone_htod(projections).ok()?;
        let total = n.checked_mul(g)?;
        let mut signs_gpu = self.stream.alloc_zeros::<i8>(total).ok()?;

        let shared_mem = dim * 4;
        let cfg = LaunchConfig {
            grid_dim: (n as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: shared_mem as u32,
        };

        unsafe {
            self.stream
                .launch_builder(&func)
                .arg(&vecs_gpu)
                .arg(&proj_gpu)
                .arg(&mut signs_gpu)
                .arg(&(n as i32))
                .arg(&(dim as i32))
                .arg(&(g as i32))
                .launch(cfg)
                .ok()?;
        }

        self.stream.clone_dtoh(&signs_gpu).ok()
    }

    /// Batch QJL inner product estimate from sketches.
    /// `projections` is [G, D], `signs` is [N, G] flat i8, `query` is [D].
    /// Returns [N] float estimates.
    pub fn qjl_batch_ip_estimate(
        &self,
        projections: &[f32],
        signs: &[i8],
        query: &[f32],
        n: usize,
        dim: usize,
        g: usize,
    ) -> Option<Vec<f32>> {
        let func = self.load_ext_kernel("qjl_batch_ip_estimate_kernel")?;

        let proj_gpu = self.stream.clone_htod(projections).ok()?;
        let signs_gpu = self.stream.clone_htod(signs).ok()?;
        let query_gpu = self.stream.clone_htod(query).ok()?;
        let mut out_gpu = self.stream.alloc_zeros::<f32>(n).ok()?;

        let scale = std::f32::consts::PI / (2.0 * g as f32);
        let shared_mem = g * 4; // precomputed proj·query

        let block_dim = 256u32;
        let grid_dim = n.div_ceil(block_dim as usize) as u32;

        let cfg = LaunchConfig {
            grid_dim: (grid_dim, 1, 1),
            block_dim: (block_dim, 1, 1),
            shared_mem_bytes: shared_mem as u32,
        };

        unsafe {
            self.stream
                .launch_builder(&func)
                .arg(&proj_gpu)
                .arg(&signs_gpu)
                .arg(&query_gpu)
                .arg(&mut out_gpu)
                .arg(&(n as i32))
                .arg(&(dim as i32))
                .arg(&(g as i32))
                .arg(&scale)
                .launch(cfg)
                .ok()?;
        }

        self.stream.clone_dtoh(&out_gpu).ok()
    }

    // ========================================================================
    // 3. KMeans Assignment
    // ========================================================================

    /// KMeans assignment: assign N points to nearest of K centroids.
    /// Returns (assignments[N], distances[N]) where distance is squared L2.
    pub fn kmeans_assign(
        &self,
        points: &[f32],
        centroids: &[f32],
        n: usize,
        k: usize,
        dim: usize,
    ) -> Option<(Vec<i32>, Vec<f32>)> {
        let func = self.load_ext_kernel("kmeans_assign_kernel")?;

        let pts_gpu = self.stream.clone_htod(points).ok()?;
        let cent_gpu = self.stream.clone_htod(centroids).ok()?;
        let mut assign_gpu = self.stream.alloc_zeros::<i32>(n).ok()?;
        let mut dist_gpu = self.stream.alloc_zeros::<f32>(n).ok()?;

        let block_dim = 256u32;
        let grid_dim = n.div_ceil(block_dim as usize) as u32;

        let cfg = LaunchConfig {
            grid_dim: (grid_dim, 1, 1),
            block_dim: (block_dim, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.stream
                .launch_builder(&func)
                .arg(&pts_gpu)
                .arg(&cent_gpu)
                .arg(&mut assign_gpu)
                .arg(&mut dist_gpu)
                .arg(&(n as i32))
                .arg(&(k as i32))
                .arg(&(dim as i32))
                .launch(cfg)
                .ok()?;
        }

        let assignments = self.stream.clone_dtoh(&assign_gpu).ok()?;
        let distances = self.stream.clone_dtoh(&dist_gpu).ok()?;
        Some((assignments, distances))
    }

    // ========================================================================
    // 4. Cosine Similarity Matrix
    // ========================================================================

    /// All-pairs cosine similarity matrix for N normalized vectors.
    /// Returns [N, N] flat row-major. Uses tile-based upper triangle kernel.
    pub fn cosine_similarity_matrix(
        &self,
        vectors: &[f32],
        n: usize,
        dim: usize,
    ) -> Option<Vec<f32>> {
        let func = self.load_ext_kernel("cosine_similarity_kernel")?;

        let vecs_gpu = self.stream.clone_htod(vectors).ok()?;
        let total = n.checked_mul(n)?;
        let mut out_gpu = self.stream.alloc_zeros::<f32>(total).ok()?;

        let tile = 16u32;
        let grid_x = n.div_ceil(tile as usize) as u32;
        let grid_y = grid_x;

        let cfg = LaunchConfig {
            grid_dim: (grid_x, grid_y, 1),
            block_dim: (tile, tile, 1), // 16x16 = 256 threads
            shared_mem_bytes: 0,
        };

        unsafe {
            self.stream
                .launch_builder(&func)
                .arg(&vecs_gpu)
                .arg(&mut out_gpu)
                .arg(&(n as i32))
                .arg(&(dim as i32))
                .launch(cfg)
                .ok()?;
        }

        self.stream.clone_dtoh(&out_gpu).ok()
    }

    // ========================================================================
    // 5. Batch Geodesic Distance
    // ========================================================================

    /// Pairwise geodesic distance between corresponding vector pairs.
    /// `x` and `y` are [N, D], returns [N] distances.
    pub fn batch_geodesic(&self, x: &[f32], y: &[f32], n: usize, dim: usize) -> Option<Vec<f32>> {
        let func = self.load_ext_kernel("batch_geodesic_kernel")?;

        let x_gpu = self.stream.clone_htod(x).ok()?;
        let y_gpu = self.stream.clone_htod(y).ok()?;
        let mut out_gpu = self.stream.alloc_zeros::<f32>(n).ok()?;

        let block_dim = 256u32;
        let grid_dim = n.div_ceil(block_dim as usize) as u32;

        let cfg = LaunchConfig {
            grid_dim: (grid_dim, 1, 1),
            block_dim: (block_dim, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            self.stream
                .launch_builder(&func)
                .arg(&x_gpu)
                .arg(&y_gpu)
                .arg(&mut out_gpu)
                .arg(&(n as i32))
                .arg(&(dim as i32))
                .launch(cfg)
                .ok()?;
        }

        self.stream.clone_dtoh(&out_gpu).ok()
    }

    // ========================================================================
    // 6. LSH Batch Hash
    // ========================================================================

    /// Batch LSH hash: compute hash sign bits for N vectors across T tables with K projections each.
    /// `projections` is [T*K, D], returns [N, T*K] flat i8 signs.
    pub fn lsh_batch_hash(
        &self,
        vectors: &[f32],
        projections: &[f32],
        n: usize,
        dim: usize,
        t: usize,
        k: usize,
    ) -> Option<Vec<i8>> {
        let func = self.load_ext_kernel("lsh_batch_hash_kernel")?;

        let vecs_gpu = self.stream.clone_htod(vectors).ok()?;
        let proj_gpu = self.stream.clone_htod(projections).ok()?;
        let total = n.checked_mul(t)?.checked_mul(k)?;
        let mut signs_gpu = self.stream.alloc_zeros::<i8>(total).ok()?;

        let shared_mem = dim * 4; // cache vector
        let cfg = LaunchConfig {
            grid_dim: (n as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: shared_mem as u32,
        };

        unsafe {
            self.stream
                .launch_builder(&func)
                .arg(&vecs_gpu)
                .arg(&proj_gpu)
                .arg(&mut signs_gpu)
                .arg(&(n as i32))
                .arg(&(dim as i32))
                .arg(&(t as i32))
                .arg(&(k as i32))
                .launch(cfg)
                .ok()?;
        }

        self.stream.clone_dtoh(&signs_gpu).ok()
    }
}
