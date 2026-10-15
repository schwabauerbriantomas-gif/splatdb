# O2-B1 — Kernel CUDA gaussian_topk + launcher Rust + tests numéricos

**Estado: COMPLETO. Tests reales en RTX 3090 (sm_86): 13/13 gpu:: verdes, 0 failed.**
Fecha: 2026-10-15 · Hijo B1 (kernel CUDA) · Repo: /mnt/d/splatdb · Sin commits git (según contrato).

## Qué se implementó

### 1. `kernels/distance.cu` — kernel `gaussian_topk_kernel` (append, ~líneas 464-577)
- Firma exacta del contrato: `gaussian_topk_kernel(const float* queries[Q,D], const float* dataset[N,D], const float* alpha[N], const float* kappa[N], int* out_idx[Q,K], float* out_scores[Q,K], int Q, int N, int D, int K)`.
- `score_i = alpha_i * __expf(-kappa_i * d2)`, `d2 = ||q-mu_i||²`, **sin factor ½** (convención `src/energy.rs:5-6`).
- **Top-k sobre el SCORE FINAL, descendente** (mayor = mejor). Kappa por-item ⇒ NO equivalente a d² ascendente — hay test que lo demuestra (`test_gaussian_topk_not_d2_ordering`).
- Patrón EXACTO de `l2_topk_kernel` (distance.cu:245+): 1 block/query (`grid=(Q,1,1)`), query en smem, top-K thread-local con arrays `[32]` (K≤32), merge por thread 0 en shared, `__launch_bounds__(256)`, float4 si `D%4==0` con fallback escalar.
- Smem: `D floats + 256·K floats + 256·K ints` (D=64,K=16 → ~33KB, OK sm_86).
- Sentido de ordenación invertido respecto de l2 (max-score vs min-dist); comparaciones estrictas `>` preservan el insert más temprano en empates intra-thread.

### 2. `src/gpu/cuda_kernel.rs` — launcher + caché PTX
- **`GpuIndex::upload_alpha_kappa(&mut self, alpha, kappa) -> bool`**: valida `len == n_vectors`, sube dos `CudaSlice<f32>` persistentes (`alpha_gpu`/`kappa_gpu`).
- **`GpuIndex::gaussian_topk_search(&self, queries, n_queries, k) -> Option<Vec<Vec<(usize,f32)>>>`**: requiere dataset+alpha/kappa subidos, valida `queries.len == n_queries*dim`, delega al launcher.
- `launch_gaussian_topk_kernel`: patrón copiado de `launch_topk_kernel` (mismo archivo) — bounds checks i32/u32, `alloc_zeros` de idx/score `[Q,K]`, smem idéntico, `launch_builder` con 10 args en orden del kernel, download solo `Q*K` pares, filtro `i >= 0`. K∈[1,32].
- **Caché PTX arreglado**: `GpuIndex::new()` carga el módulo UNA vez (`load_ptx_module`) y lo guarda en `kernel_module: Option<Arc<CudaModule>>`; los 4 launchers viejos (l2/batch/cosine/topk) ahora usan `kernel_fn(name)` en vez de `ctx.load_module` por llamada (antes: JIT de PTX en cada llamada — cuda_kernel.rs:135,228,319,430). Eliminados `kernel_loaded` y `try_load_kernels` (muertos) y el campo `ctx` (dead_code tras el refactor).

### 3. `src/gpu/mod.rs` — wrapper público + CPU reference
- **`pub fn gaussian_topk_scores(queries, dataset, alpha, kappa, n, d, q, k) -> Option<Vec<Vec<(usize,f32)>>>`**: valida shapes, dispatch GPU si `is_cuda_available()`, fallback CPU exacto. Patrón de `batch_search`.
- **`cpu_gaussian_topk`**: score exacto con `exp` natural, orden descendente por score, stable sort (empates → índice menor), `k_eff = k.min(n)`.

### 4. Tests (src/gpu/mod.rs #[cfg(test)], patrón test_ptx_kernel_correctness)
| Test | Qué verifica |
|---|---|
| `test_gaussian_topk_cpu_reference_exact` | CPU vs valores calculados a mano (d²=0 y d²=25, sin ½) |
| `test_gaussian_topk_not_d2_ordering` | kappa pequeño en d² mayor GANA a d² menor — top-k ≠ orden d² |
| `test_gaussian_topk_gpu_matches_cpu` | GPU vs CPU: 500×64D, Q=4, K=10 — **mismos índices, tol 1e-5 en score** |
| `test_gaussian_topk_gpu_non_multiple4_k1` | D=63 (no divisible por 4, path escalar) + K=1, GPU==CPU |
| `test_gaussian_topk_large_batch_shapes` | 20k×64D, Q=256, K=16 — shapes, idx<N, scores descendentes |
| + helper `gaussian_test_data` | LCG determinista (mismo patrón que tests existentes) |

## Evidencia real (output de `cargo test --release --lib --features cuda gpu:: -- --nocapture --test-threads=1`, RTX 3090)

```
test gpu::tests::test_cpu_cosine_distances ... ok
test gpu::tests::test_cpu_l2_distances ... ok
test gpu::tests::test_cuda_gpu_pipeline ... ok
test gpu::tests::test_gaussian_topk_cpu_reference_exact ... ok
test gpu::tests::test_gaussian_topk_gpu_matches_cpu ... [splatsdb] Launching gaussian_topk_kernel PTX kernel (Q=4, N=500, D=64, K=10)
[gaussian-topk test] 500x64D Q=4 K=10 GPU==CPU OK; top score q0: 3.166830e-1
ok
test gpu::tests::test_gaussian_topk_gpu_non_multiple4_k1 ... [splatsdb] Launching gaussian_topk_kernel PTX kernel (Q=3, N=500, D=63, K=1)
[gaussian-topk test] D=63 K=1 GPU==CPU OK; q0 best idx 65 score 3.287609e-1
ok
test gpu::tests::test_gaussian_topk_large_batch_shapes ... [splatsdb] Launching gaussian_topk_kernel PTX kernel (Q=256, N=20000, D=64, K=16)
[gaussian-topk test] 20k x 64D Q=256 K=16 shape check OK
ok
test gpu::tests::test_gaussian_topk_not_d2_ordering ... ok
test gpu::tests::test_ptx_kernel_correctness ... ok
test gpu::tests::test_ptx_kernel_performance ... [splatsdb benchmark] 10K x 640D, 100 queries | CPU: 559ms | GPU: 13ms | speedup: 43.00x
ok

test result: ok. 13 passed; 0 failed; 0 ignored; 0 measured; 315 filtered out; finished in 3.78s
```

Filtro `gaussian` (incluye tests de otros módulos que matchean): `12 passed; 0 failed` (api_server + splats + splat_types + gpu).

## Compilación
- PTX: `~/.local/cuda124/bin/nvcc --ptx -arch=sm_86 -O3 --use_fast_math --extra-device-vectorization -o ptx/distance.ptx kernels/distance.cu` → OK (104KB, contiene `gaussian_topk_kernel`).
- Build: `M2M_PTX_PATH=... M2M_EXTENDED_PTX_PATH=... cargo build --release --features cuda` → OK.

## Desviaciones / notas para el padre
1. **Tests corridos en copia aislada del repo** (`~/.hermes/profiles/deepharness/cache/scratch/splatdb-b1`, ext4 nativo, fuera del árbol compartido): `src/candidate_backend.rs` (untracked, trabajo de OTRO hijo) tiene un error de compilación en SU test (E0308, línea 217: pattern `(idx, _)` contra `&usize` en `test_hnsw_backend_parity_with_vector_index`) que rompe `cargo test --lib` de todo el repo. En mi copia lo parcheé localmente para poder compilar; **el original en /mnt/d/splatdb quedó INTACTO** — B2 (o quien sea dueño) debe arreglarlo. Mis 3 archivos en /mnt/d/splatdb son los definitivos.
2. **Sin cambios de firma vs contrato**; lo único fuera del literal: eliminé el campo muerto `ctx` de `GpuIndex` (warning dead_code tras quitar los load_module por llamada) y los métodos muertos `kernel_loaded`/`try_load_kernels` — parte del refactor del caché pedido.
3. `--use_fast_math` + `__expf`: la tolerancia 1e-5 del test GPU vs CPU se cumple holgada en los casos probados (scores O(0.1-1), d²≤~250 en 64D con datos [-1,1]).
4. `/mnt/d` estuvo muy lento durante la tarea (los `du`/`rsync` agotaban timeouts) — probablemente otros hijos compilando en paralelo; por eso la verificación corrió en ext4 local.

## Archivos modificados (ownership respetado)
- `kernels/distance.cu` (+115 líneas, solo append del kernel)
- `src/gpu/cuda_kernel.rs` (+260/−37 refactor + launcher nuevo)
- `src/gpu/mod.rs` (+283 líneas: wrapper, CPU ref, tests)
- `ptx/distance.ptx` (regenerado con nvcc sm_86)
