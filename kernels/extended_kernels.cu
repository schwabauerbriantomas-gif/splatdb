// SplatsDB Extended CUDA Kernels — Quantization, Clustering, Geometry, LSH
// Optimized for sm_86 (RTX 3090), CUDA 12.4
// All kernels assume row-major [N, D] layout
//
// Kernels:
//   1. rotation_gemv_kernel         — Batch R·x (forward rotation)
//   2. rotation_gemv_inverse_kernel — Batch R^T·x (inverse rotation)
//   3. qjl_batch_sketch_kernel      — Batch QJL sign sketch
//   4. qjl_batch_ip_estimate_kernel — Batch QJL inner product estimate
//   5. kmeans_assign_kernel         — KMeans++ assignment step
//   6. cosine_similarity_kernel     — All-pairs cosine similarity (upper triangle)
//   7. batch_geodesic_kernel        — Pairwise geodesic distance
//   8. lsh_batch_hash_kernel        — Batch LSH hash computation

#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <math.h>

// ============================================================================
// 1. Rotation GEMV: batch R · x for N vectors
// ============================================================================
// Each block handles one vector. Shared memory caches the input vector.
// Each thread computes ceil(D/blockDim) output elements (dot of one row of R with input).

extern "C" __launch_bounds__(256)
__global__ void rotation_gemv_kernel(
    const float* __restrict__ vectors,  // [N, D]
    const float* __restrict__ rotation, // [D, D] row-major
    float* __restrict__ output,         // [N, D]
    int N,
    int D
) {
    extern __shared__ float s_vec[];

    int vec_idx = blockIdx.x;
    if (vec_idx >= N) return;

    // Cooperative load of input vector into shared memory
    for (int j = threadIdx.x; j < D; j += blockDim.x) {
        s_vec[j] = vectors[vec_idx * D + j];
    }
    __syncthreads();

    // Each thread computes output elements striding by blockDim
    for (int j = threadIdx.x; j < D; j += blockDim.x) {
        const float* row = rotation + j * D;
        float sum = 0.0f;

        if (D % 4 == 0) {
            const float4* row4 = reinterpret_cast<const float4*>(row);
            const float4* vec4 = reinterpret_cast<const float4*>(s_vec);
            int D4 = D / 4;
            #pragma unroll 4
            for (int k = 0; k < D4; k++) {
                float4 r = row4[k];
                float4 v = vec4[k];
                sum += r.x*v.x + r.y*v.y + r.z*v.z + r.w*v.w;
            }
        } else {
            #pragma unroll 8
            for (int k = 0; k < D; k++) {
                sum += row[k] * s_vec[k];
            }
        }

        output[vec_idx * D + j] = sum;
    }
}

// ============================================================================
// 2. Rotation GEMV Inverse: batch R^T · x for N vectors
// ============================================================================
// R^T means output[j] = sum_k R[k*D + j] * input[k] (column access).
// We tile the computation: each thread handles one output element j,
// accumulating sum over k with strided column access.

extern "C" __launch_bounds__(256)
__global__ void rotation_gemv_inverse_kernel(
    const float* __restrict__ vectors,  // [N, D]
    const float* __restrict__ rotation, // [D, D] row-major
    float* __restrict__ output,         // [N, D]
    int N,
    int D
) {
    extern __shared__ float s_vec[];

    int vec_idx = blockIdx.x;
    if (vec_idx >= N) return;

    // Cooperative load of input vector into shared memory
    for (int j = threadIdx.x; j < D; j += blockDim.x) {
        s_vec[j] = vectors[vec_idx * D + j];
    }
    __syncthreads();

    // output[j] = sum_k R[k][j] * input[k] = sum_k rotation[k*D + j] * s_vec[k]
    for (int j = threadIdx.x; j < D; j += blockDim.x) {
        float sum = 0.0f;
        #pragma unroll 8
        for (int k = 0; k < D; k++) {
            sum += rotation[k * D + j] * s_vec[k];
        }
        output[vec_idx * D + j] = sum;
    }
}

// ============================================================================
// 3. QJL Batch Sketch: compute sign(G · x) for N vectors
// ============================================================================
// Input: N vectors [N, D], G projection rows [G, D]
// Output: signs [N, G] as int8 (+1 or -1)
// Each block handles one vector. Shared memory caches the vector.
// Each thread computes ceil(G/blockDim) projection signs.

extern "C" __launch_bounds__(256)
__global__ void qjl_batch_sketch_kernel(
    const float* __restrict__ vectors,     // [N, D]
    const float* __restrict__ projections, // [G, D]
    int8_t* __restrict__ signs,            // [N, G]
    int N,
    int D,
    int G
) {
    extern __shared__ float s_vec[];

    int vec_idx = blockIdx.x;
    if (vec_idx >= N) return;

    // Cache input vector in shared memory
    for (int j = threadIdx.x; j < D; j += blockDim.x) {
        s_vec[j] = vectors[vec_idx * D + j];
    }
    __syncthreads();

    // Each thread computes signs for its assigned projections
    for (int g = threadIdx.x; g < G; g += blockDim.x) {
        const float* proj = projections + g * D;
        float dot = 0.0f;

        if (D % 4 == 0) {
            const float4* proj4 = reinterpret_cast<const float4*>(proj);
            const float4* vec4 = reinterpret_cast<const float4*>(s_vec);
            int D4 = D / 4;
            #pragma unroll 4
            for (int k = 0; k < D4; k++) {
                float4 p = proj4[k];
                float4 v = vec4[k];
                dot += p.x*v.x + p.y*v.y + p.z*v.z + p.w*v.w;
            }
        } else {
            #pragma unroll 8
            for (int k = 0; k < D; k++) {
                dot += proj[k] * s_vec[k];
            }
        }

        signs[vec_idx * G + g] = (dot >= 0.0f) ? 1 : -1;
    }
}

// ============================================================================
// 4. QJL Batch IP Estimate: estimate <x_i, query> from sketches
// ============================================================================
// For each sketch i, compute: scale * sum_g signs[i,g] * (projections[g] · query)
// Optimization: precompute projections[g] · query in shared memory first.
// Grid: ceil(N / blockDim) blocks, each thread handles one sketch.

extern "C" __launch_bounds__(256)
__global__ void qjl_batch_ip_estimate_kernel(
    const float* __restrict__ projections, // [G, D]
    const int8_t* __restrict__ signs,      // [N, G]
    const float* __restrict__ query,       // [D]
    float* __restrict__ output,            // [N]
    int N,
    int D,
    int G,
    float scale                           // PI / (2 * G)
) {
    extern __shared__ float s_proj_query[]; // [G] precomputed projection·query

    // Cooperative precompute: projections[g] · query for all g
    for (int g = threadIdx.x; g < G; g += blockDim.x) {
        const float* proj = projections + g * D;
        float dot = 0.0f;
        if (D % 4 == 0) {
            const float4* proj4 = reinterpret_cast<const float4*>(proj);
            const float4* query4 = reinterpret_cast<const float4*>(query);
            int D4 = D / 4;
            #pragma unroll 4
            for (int k = 0; k < D4; k++) {
                float4 p = proj4[k];
                float4 q = query4[k];
                dot += p.x*q.x + p.y*q.y + p.z*q.z + p.w*q.w;
            }
        } else {
            #pragma unroll 8
            for (int k = 0; k < D; k++) {
                dot += proj[k] * query[k];
            }
        }
        s_proj_query[g] = dot;
    }
    __syncthreads();

    // Each thread estimates IP for one sketch
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < N; i += gridDim.x * blockDim.x) {
        float estimate = 0.0f;
        const int8_t* sketch = signs + i * G;
        for (int g = 0; g < G; g++) {
            estimate += (float)sketch[g] * s_proj_query[g];
        }
        output[i] = scale * estimate;
    }
}

// ============================================================================
// 5. KMeans Assignment: assign N points to nearest of K centroids
// ============================================================================
// Each thread handles one point. Computes L2 distance to all K centroids.
// Grid: ceil(N / blockDim) blocks.

extern "C" __launch_bounds__(256)
__global__ void kmeans_assign_kernel(
    const float* __restrict__ points,    // [N, D]
    const float* __restrict__ centroids, // [K, D]
    int* __restrict__ assignments,       // [N]
    float* __restrict__ distances,       // [N] (squared L2, can be NULL)
    int N,
    int K,
    int D
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    const float* point = points + i * D;
    float best_dist = 1e30f;
    int best_k = 0;

    for (int c = 0; c < K; c++) {
        const float* cent = centroids + c * D;
        float sum = 0.0f;

        if (D % 4 == 0) {
            const float4* p4 = reinterpret_cast<const float4*>(point);
            const float4* c4 = reinterpret_cast<const float4*>(cent);
            int D4 = D / 4;
            #pragma unroll 4
            for (int j = 0; j < D4; j++) {
                float4 p = p4[j];
                float4 cc = c4[j];
                float dx = p.x - cc.x;
                float dy = p.y - cc.y;
                float dz = p.z - cc.z;
                float dw = p.w - cc.w;
                sum += dx*dx + dy*dy + dz*dz + dw*dw;
            }
        } else {
            #pragma unroll 8
            for (int j = 0; j < D; j++) {
                float diff = point[j] - cent[j];
                sum += diff * diff;
            }
        }

        if (sum < best_dist) {
            best_dist = sum;
            best_k = c;
        }
    }

    assignments[i] = best_k;
    if (distances) {
        distances[i] = best_dist;
    }
}

// ============================================================================
// 6. Cosine Similarity Matrix: upper triangle of N×N matrix
// ============================================================================
// Each block computes a tile of the upper triangle.
// Grid: (ceil(N/TILE) , ceil(N/TILE), 1), block: (TILE, TILE, 1)
// TILE=16 → 256 threads per block (good occupancy).

extern "C" __launch_bounds__(256)
__global__ void cosine_similarity_kernel(
    const float* __restrict__ vectors, // [N, D] (assumed normalized)
    float* __restrict__ output,        // [N, N]
    int N,
    int D
) {
    // Tile-based approach: each block computes a TILE×TILE patch
    const int TILE = 16;
    int row_start = blockIdx.x * TILE;
    int col_start = blockIdx.y * TILE;

    // Only compute upper triangle (row_start <= col_start)
    if (row_start > col_start + TILE - 1) return;

    int local_row = threadIdx.x;
    int local_col = threadIdx.y;
    int i = row_start + local_row;
    int j = col_start + local_col;

    if (i >= N || j >= N) return;

    // Only upper triangle
    if (i > j) return;

    const float* vec_i = vectors + i * D;
    const float* vec_j = vectors + j * D;
    float dot = 0.0f;

    if (D % 4 == 0) {
        const float4* vi4 = reinterpret_cast<const float4*>(vec_i);
        const float4* vj4 = reinterpret_cast<const float4*>(vec_j);
        int D4 = D / 4;
        #pragma unroll 4
        for (int k = 0; k < D4; k++) {
            float4 a = vi4[k];
            float4 b = vj4[k];
            dot += a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w;
        }
    } else {
        #pragma unroll 8
        for (int k = 0; k < D; k++) {
            dot += vec_i[k] * vec_j[k];
        }
    }

    output[i * N + j] = dot;
    if (i != j) {
        output[j * N + i] = dot;
    }
}

// ============================================================================
// 7. Batch Geodesic Distance: pairwise arccos(dot(x_i, y_i))
// ============================================================================
// Computes geodesic distance between corresponding pairs.
// Grid: ceil(N / blockDim) blocks, each thread handles one pair.

extern "C" __launch_bounds__(256)
__global__ void batch_geodesic_kernel(
    const float* __restrict__ x, // [N, D]
    const float* __restrict__ y, // [N, D]
    float* __restrict__ output,  // [N]
    int N,
    int D
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;

    const float* xi = x + i * D;
    const float* yi = y + i * D;

    float dot = 0.0f;
    if (D % 4 == 0) {
        const float4* x4 = reinterpret_cast<const float4*>(xi);
        const float4* y4 = reinterpret_cast<const float4*>(yi);
        int D4 = D / 4;
        #pragma unroll 4
        for (int k = 0; k < D4; k++) {
            float4 a = x4[k];
            float4 b = y4[k];
            dot += a.x*b.x + a.y*b.y + a.z*b.z + a.w*b.w;
        }
    } else {
        #pragma unroll 8
        for (int k = 0; k < D; k++) {
            dot += xi[k] * yi[k];
        }
    }

    // Early exit for near-identical vectors
    if (dot > 1.0f - 1e-5f) {
        output[i] = 0.0f;
        return;
    }

    // Clip and acos
    float clipped = fmaxf(-1.0f + 1e-7f, fminf(1.0f - 1e-7f, dot));
    output[i] = acosf(clipped);
}

// ============================================================================
// 8. LSH Batch Hash: compute hash codes for N vectors across T tables
// ============================================================================
// Each hash is computed by: for each of k projections in table t,
// project the vector and take the sign → hash component.
// Output: flat array [N * T * k] of int64 (each hash component is a sign bit index).
// Simplified: output [N * T * k] as int8 signs that the CPU assembles into hash keys.
//
// Grid: (N blocks, 1, 1), each block handles one vector.
// Block: 256 threads, each handles ceil(T*k / 256) hash components.

extern "C" __launch_bounds__(256)
__global__ void lsh_batch_hash_kernel(
    const float* __restrict__ vectors,     // [N, D]
    const float* __restrict__ projections, // [T * k, D] all projections for all tables
    int8_t* __restrict__ hash_signs,       // [N, T * k]
    int N,
    int D,
    int T,
    int K
) {
    extern __shared__ float s_vec[];

    int vec_idx = blockIdx.x;
    if (vec_idx >= N) return;

    // Cache vector in shared memory
    for (int j = threadIdx.x; j < D; j += blockDim.x) {
        s_vec[j] = vectors[vec_idx * D + j];
    }
    __syncthreads();

    int total_hashes = T * K;

    for (int h = threadIdx.x; h < total_hashes; h += blockDim.x) {
        int t = h / K;
        int kk = h % K;
        int proj_idx = t * K + kk;

        const float* proj = projections + proj_idx * D;
        float dot = 0.0f;

        if (D % 4 == 0) {
            const float4* proj4 = reinterpret_cast<const float4*>(proj);
            const float4* vec4 = reinterpret_cast<const float4*>(s_vec);
            int D4 = D / 4;
            #pragma unroll 4
            for (int j = 0; j < D4; j++) {
                float4 p = proj4[j];
                float4 v = vec4[j];
                dot += p.x*v.x + p.y*v.y + p.z*v.z + p.w*v.w;
            }
        } else {
            #pragma unroll 8
            for (int j = 0; j < D; j++) {
                dot += proj[j] * s_vec[j];
            }
        }

        hash_signs[vec_idx * total_hashes + h] = (dot >= 0.0f) ? 1 : -1;
    }
}
