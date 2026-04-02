// M2M Vector Search — CUDA Kernels
// Optimized for sm_86 (RTX 3090), CUDA 12.4
// All kernels assume row-major [N, D] layout

#include <cuda_fp16.h>
#include <cuda_runtime.h>

// ============================================================================
// L2 Distance: single query vs N vectors
// ============================================================================
// Each block processes a tile of vectors. Query is cached in shared memory.
// Uses float4 vectorized loads for D divisible by 4.

extern "C" __launch_bounds__(256)
__global__ void l2_distance_kernel(
    const float* __restrict__ query,    // [D]
    const float* __restrict__ dataset,  // [N, D]
    float* __restrict__ output,         // [N]
    int N,
    int D
) {
    // Cache query in shared memory (up to 2048 floats = 8KB)
    extern __shared__ float s_query[];
    
    // Cooperative load of query into shared memory
    for (int j = threadIdx.x; j < D; j += blockDim.x) {
        s_query[j] = query[j];
    }
    __syncthreads();
    
    // Each thread computes L2 distance for one vector
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    
    const float* vec = dataset + i * D;
    float sum = 0.0f;
    
    // Vectorized path for D divisible by 4
    if (D % 4 == 0) {
        const float4* vec4 = reinterpret_cast<const float4*>(vec);
        const float4* query4 = reinterpret_cast<const float4*>(s_query);
        int D4 = D / 4;
        
        #pragma unroll 4
        for (int j = 0; j < D4; j++) {
            float4 v = vec4[j];
            float4 q = query4[j];
            float dx = q.x - v.x;
            float dy = q.y - v.y;
            float dz = q.z - v.z;
            float dw = q.w - v.w;
            sum += dx*dx + dy*dy + dz*dz + dw*dw;
        }
    } else {
        // Scalar fallback
        #pragma unroll 8
        for (int j = 0; j < D; j++) {
            float diff = s_query[j] - vec[j];
            sum += diff * diff;
        }
    }
    
    output[i] = sum;
}

// ============================================================================
// Batch L2 Distance: Q queries vs N vectors
// ============================================================================
// 2D grid: x = vector index (blockDim.x per block), y = query index
// Each thread computes one (query, vector) distance pair

extern "C" __launch_bounds__(256)
__global__ void batch_l2_distance_kernel(
    const float* __restrict__ queries,  // [Q, D]
    const float* __restrict__ dataset,  // [N, D]
    float* __restrict__ output,         // [Q, N]
    int Q,
    int N,
    int D
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;  // vector index
    int q = blockIdx.y;                               // query index
    
    if (i >= N || q >= Q) return;
    
    const float* query = queries + q * D;
    const float* vec = dataset + i * D;
    
    float sum = 0.0f;
    
    if (D % 4 == 0) {
        const float4* vec4 = reinterpret_cast<const float4*>(vec);
        const float4* query4 = reinterpret_cast<const float4*>(query);
        int D4 = D / 4;
        
        #pragma unroll 4
        for (int j = 0; j < D4; j++) {
            float4 v = vec4[j];
            float4 qu = query4[j];
            float dx = qu.x - v.x;
            float dy = qu.y - v.y;
            float dz = qu.z - v.z;
            float dw = qu.w - v.w;
            sum += dx*dx + dy*dy + dz*dz + dw*dw;
        }
    } else {
        #pragma unroll 8
        for (int j = 0; j < D; j++) {
            float diff = query[j] - vec[j];
            sum += diff * diff;
        }
    }
    
    output[q * N + i] = sum;
}

// ============================================================================
// Cosine Distance: single query vs N vectors
// ============================================================================
// Computes 1 - (q . v_i) / (||q|| * ||v_i||)
// Uses shared memory reduction for query norm.

extern "C" __launch_bounds__(256)
__global__ void cosine_distance_kernel(
    const float* __restrict__ query,    // [D]
    const float* __restrict__ dataset,  // [N, D]
    float* __restrict__ output,         // [N]
    int N,
    int D
) {
    extern __shared__ float s_buf[];  // [D + 256] = query + reduction workspace
    
    float* s_query = s_buf;
    float* s_reduce = s_buf + D;
    
    // Cooperative load query into shared memory
    for (int j = threadIdx.x; j < D; j += blockDim.x) {
        s_query[j] = query[j];
    }
    __syncthreads();
    
    // Reduction: compute sum of query^2 across all threads
    float q_part = 0.0f;
    for (int j = threadIdx.x; j < D; j += blockDim.x) {
        q_part += s_query[j] * s_query[j];
    }
    s_reduce[threadIdx.x] = q_part;
    __syncthreads();
    
    // Tree reduction for query norm
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            s_reduce[threadIdx.x] += s_reduce[threadIdx.x + s];
        }
        __syncthreads();
    }
    float q_norm = sqrtf(max(s_reduce[0], 1e-20f));
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    
    const float* vec = dataset + i * D;
    float dot = 0.0f;
    float v_norm_sq = 0.0f;
    
    if (D % 4 == 0) {
        const float4* vec4 = reinterpret_cast<const float4*>(vec);
        const float4* query4 = reinterpret_cast<const float4*>(s_query);
        int D4 = D / 4;
        
        #pragma unroll 4
        for (int j = 0; j < D4; j++) {
            float4 v = vec4[j];
            float4 q = query4[j];
            dot += q.x*v.x + q.y*v.y + q.z*v.z + q.w*v.w;
            v_norm_sq += v.x*v.x + v.y*v.y + v.z*v.z + v.w*v.w;
        }
    } else {
        #pragma unroll 8
        for (int j = 0; j < D; j++) {
            dot += s_query[j] * vec[j];
            v_norm_sq += vec[j] * vec[j];
        }
    }
    
    float denom = q_norm * sqrtf(max(v_norm_sq, 1e-20f));
    output[i] = 1.0f - dot / max(denom, 1e-10f);
}

// ============================================================================
// Batch Cosine Distance: Q queries vs N vectors
// ============================================================================

extern "C" __launch_bounds__(256)
__global__ void batch_cosine_distance_kernel(
    const float* __restrict__ queries,  // [Q, D]
    const float* __restrict__ dataset,  // [N, D]
    float* __restrict__ output,         // [Q, N]
    int Q,
    int N,
    int D
) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int q = blockIdx.y;
    
    if (i >= N || q >= Q) return;
    
    const float* query = queries + q * D;
    const float* vec = dataset + i * D;
    
    float dot = 0.0f;
    float q_norm_sq = 0.0f;
    float v_norm_sq = 0.0f;
    
    if (D % 4 == 0) {
        const float4* vec4 = reinterpret_cast<const float4*>(vec);
        const float4* query4 = reinterpret_cast<const float4*>(query);
        int D4 = D / 4;
        
        #pragma unroll 4
        for (int j = 0; j < D4; j++) {
            float4 v = vec4[j];
            float4 qu = query4[j];
            dot += qu.x*v.x + qu.y*v.y + qu.z*v.z + qu.w*v.w;
            q_norm_sq += qu.x*qu.x + qu.y*qu.y + qu.z*qu.z + qu.w*qu.w;
            v_norm_sq += v.x*v.x + v.y*v.y + v.z*v.z + v.w*v.w;
        }
    } else {
        #pragma unroll 8
        for (int j = 0; j < D; j++) {
            dot += query[j] * vec[j];
            q_norm_sq += query[j] * query[j];
            v_norm_sq += vec[j] * vec[j];
        }
    }
    
    float denom = sqrtf(max(q_norm_sq, 1e-20f)) * sqrtf(max(v_norm_sq, 1e-20f));
    output[q * N + i] = 1.0f - dot / max(denom, 1e-10f);
}

// ============================================================================
// Combined L2 Distance + Top-K Selection
// ============================================================================
// One block per query. Each thread processes N/blockDim vectors.
// Thread-local top-k → shared memory merge → final output.
// Eliminates the N*Q distance download — only K*Q results come back.
//
// Shared memory layout:
//   s_query[D]          — cached query vector
//   s_dist[blockDim*K]  — thread-local top-k distances
//   s_idx[blockDim*K]   — thread-local top-k indices
// For D=640, K=10, blockDim=256: ~23KB (fits in 48KB sm_86 shared mem)

extern "C" __launch_bounds__(256)
__global__ void l2_topk_kernel(
    const float* __restrict__ queries,  // [Q, D]
    const float* __restrict__ dataset,  // [N, D]
    int* __restrict__ out_indices,      // [Q, K]
    float* __restrict__ out_distances,  // [Q, K]
    int Q,
    int N,
    int D,
    int K
) {
    extern __shared__ float s_data[];
    float* s_query = s_data;
    float* s_dist = s_query + D;
    int* s_idx = reinterpret_cast<int*>(s_dist + blockDim.x * K);

    int q = blockIdx.x;
    if (q >= Q) return;

    // Cooperative load of query into shared memory
    for (int j = threadIdx.x; j < D; j += blockDim.x) {
        s_query[j] = queries[q * D + j];
    }
    __syncthreads();

    // Thread-local top-k (max K=32)
    float local_dist[32];
    int local_idx[32];
    for (int i = 0; i < K; i++) { local_dist[i] = 1e30f; local_idx[i] = -1; }

    // Process assigned vectors (strided by blockDim)
    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        const float* vec = dataset + i * D;
        float sum = 0.0f;

        if (D % 4 == 0) {
            const float4* vec4 = reinterpret_cast<const float4*>(vec);
            const float4* q4 = reinterpret_cast<const float4*>(s_query);
            int D4 = D / 4;
            #pragma unroll 4
            for (int j = 0; j < D4; j++) {
                float4 v = vec4[j]; float4 qq = q4[j];
                float dx=qq.x-v.x, dy=qq.y-v.y, dz=qq.z-v.z, dw=qq.w-v.w;
                sum += dx*dx + dy*dy + dz*dz + dw*dw;
            }
        } else {
            #pragma unroll 8
            for (int j = 0; j < D; j++) {
                float diff = s_query[j] - vec[j];
                sum += diff * diff;
            }
        }

        // Insert into sorted thread-local top-k
        if (sum < local_dist[K - 1]) {
            local_dist[K - 1] = sum;
            local_idx[K - 1] = i;
            // Bubble up to maintain sorted order
            for (int j = K - 1; j > 0 && local_dist[j] < local_dist[j-1]; j--) {
                float td = local_dist[j]; local_dist[j] = local_dist[j-1]; local_dist[j-1] = td;
                int ti = local_idx[j]; local_idx[j] = local_idx[j-1]; local_idx[j-1] = ti;
            }
        }
    }

    // Write thread-local results to shared memory
    for (int i = 0; i < K; i++) {
        s_dist[threadIdx.x * K + i] = local_dist[i];
        s_idx[threadIdx.x * K + i] = local_idx[i];
    }
    __syncthreads();

    // Thread 0 merges all blockDim*K candidates into final top-K
    if (threadIdx.x == 0) {
        float best_dist[32];
        int best_idx[32];
        for (int i = 0; i < K; i++) { best_dist[i] = 1e30f; best_idx[i] = -1; }

        int total = blockDim.x * K;
        for (int i = 0; i < total; i++) {
            float d = s_dist[i];
            if (d < best_dist[K - 1]) {
                best_dist[K - 1] = d;
                best_idx[K - 1] = s_idx[i];
                for (int j = K - 1; j > 0 && best_dist[j] < best_dist[j-1]; j--) {
                    float td = best_dist[j]; best_dist[j] = best_dist[j-1]; best_dist[j-1] = td;
                    int ti = best_idx[j]; best_idx[j] = best_idx[j-1]; best_idx[j-1] = ti;
                }
            }
        }

        for (int i = 0; i < K; i++) {
            out_indices[q * K + i] = best_idx[i];
            out_distances[q * K + i] = best_dist[i];
        }
    }
}

// ============================================================================
// Combined Cosine Distance + Top-K Selection
// ============================================================================
// Same structure as l2_topk_kernel but computes cosine distance = 1 - cos(q,v)

extern "C" __launch_bounds__(256)
__global__ void cosine_topk_kernel(
    const float* __restrict__ queries,  // [Q, D]
    const float* __restrict__ dataset,  // [N, D]
    int* __restrict__ out_indices,      // [Q, K]
    float* __restrict__ out_distances,  // [Q, K]
    int Q,
    int N,
    int D,
    int K
) {
    extern __shared__ float s_data[];
    float* s_query = s_data;
    float* s_dist = s_query + D;
    int* s_idx = reinterpret_cast<int*>(s_dist + blockDim.x * K);

    int q = blockIdx.x;
    if (q >= Q) return;

    for (int j = threadIdx.x; j < D; j += blockDim.x) {
        s_query[j] = queries[q * D + j];
    }
    __syncthreads();

    // Precompute query norm
    float q_norm_sq = 0.0f;
    if (D % 4 == 0) {
        const float4* q4 = reinterpret_cast<const float4*>(s_query);
        for (int j = 0; j < D/4; j++) {
            float4 qq = q4[j];
            q_norm_sq += qq.x*qq.x + qq.y*qq.y + qq.z*qq.z + qq.w*qq.w;
        }
    } else {
        for (int j = 0; j < D; j++) q_norm_sq += s_query[j] * s_query[j];
    }
    float q_norm = sqrtf(max(q_norm_sq, 1e-20f));

    float local_dist[32];
    int local_idx[32];
    for (int i = 0; i < K; i++) { local_dist[i] = 1e30f; local_idx[i] = -1; }

    for (int i = threadIdx.x; i < N; i += blockDim.x) {
        const float* vec = dataset + i * D;
        float dot = 0.0f;
        float v_norm_sq = 0.0f;

        if (D % 4 == 0) {
            const float4* vec4 = reinterpret_cast<const float4*>(vec);
            const float4* q4 = reinterpret_cast<const float4*>(s_query);
            int D4 = D / 4;
            #pragma unroll 4
            for (int j = 0; j < D4; j++) {
                float4 v = vec4[j]; float4 qq = q4[j];
                dot += qq.x*v.x + qq.y*v.y + qq.z*v.z + qq.w*v.w;
                v_norm_sq += v.x*v.x + v.y*v.y + v.z*v.z + v.w*v.w;
            }
        } else {
            #pragma unroll 8
            for (int j = 0; j < D; j++) {
                dot += s_query[j] * vec[j];
                v_norm_sq += vec[j] * vec[j];
            }
        }

        float v_norm = sqrtf(max(v_norm_sq, 1e-20f));
        float cosine_dist = 1.0f - dot / (q_norm * v_norm);

        if (cosine_dist < local_dist[K - 1]) {
            local_dist[K - 1] = cosine_dist;
            local_idx[K - 1] = i;
            for (int j = K - 1; j > 0 && local_dist[j] < local_dist[j-1]; j--) {
                float td = local_dist[j]; local_dist[j] = local_dist[j-1]; local_dist[j-1] = td;
                int ti = local_idx[j]; local_idx[j] = local_idx[j-1]; local_idx[j-1] = ti;
            }
        }
    }

    for (int i = 0; i < K; i++) {
        s_dist[threadIdx.x * K + i] = local_dist[i];
        s_idx[threadIdx.x * K + i] = local_idx[i];
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float best_dist[32];
        int best_idx[32];
        for (int i = 0; i < K; i++) { best_dist[i] = 1e30f; best_idx[i] = -1; }

        int total = blockDim.x * K;
        for (int i = 0; i < total; i++) {
            float d = s_dist[i];
            if (d < best_dist[K - 1]) {
                best_dist[K - 1] = d;
                best_idx[K - 1] = s_idx[i];
                for (int j = K - 1; j > 0 && best_dist[j] < best_dist[j-1]; j--) {
                    float td = best_dist[j]; best_dist[j] = best_dist[j-1]; best_dist[j-1] = td;
                    int ti = best_idx[j]; best_idx[j] = best_idx[j-1]; best_idx[j-1] = ti;
                }
            }
        }

        for (int i = 0; i < K; i++) {
            out_indices[q * K + i] = best_idx[i];
            out_distances[q * K + i] = best_dist[i];
        }
    }
}
