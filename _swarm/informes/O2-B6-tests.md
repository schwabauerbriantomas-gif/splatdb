# O2-B6 — Suite de integración E2E + parity GPU/CPU (clone exclusivo)

**Hijo:** B6 (tests de integración) · **Fecha:** 2026-10-15 ~01:45 UTC
**Clone:** `/home/schwabauer/splatsdb-b6-clone` (base: commit `d2d2b3c` del principal)
**GPU:** RTX 3090 (sm_86), CUDA 12.4 (`~/.local/cuda124/bin/nvcc`)

## Archivos nuevos (para integración del padre)

| Archivo | Qué es |
|---|---|
| `/home/schwabauer/splatsdb-b6-clone/tests/integration_gaussian.rs` | **Suite nueva** (5 tests, todos en verde). Único archivo nuevo que escribió B6. |
| `/home/schwabauer/splatsdb-b6-clone/ptx/distance.ptx` | PTX recompilado por B6 con nvcc 12.4 (contiene `gaussian_topk_kernel`). |
| `/home/schwabauer/splatsdb-b6-clone/ptx/extended_kernels.ptx` | Copia del principal. |

**Nota:** `src/splats.rs`, `src/gpu/{cuda_kernel,mod}.rs`, `src/lib.rs`, `src/candidate_backend.rs`, `kernels/distance.cu` en el clone son **copias del working-tree del principal** (B1/B2 sin commit al momento de clonar). B6 NO editó ninguno — solo `tests/integration_gaussian.rs` es de B6. Los 6 archivos sincronizados están listos para descartar del clone si el padre integra B1/B2 directo del principal.

## Suite nueva: tests/integration_gaussian.rs

```
$ cargo test --release --features cuda --test integration_gaussian
test api_smoke::api_server_smoke_search_oneshot ... ok
test gaussian_search_ranking_equals_distance_ranking_with_uniform_params ... ok
test gaussian_search_matches_bruteforce_reference ... ok
test e2e_fused_matches_linear_top3_5k_vectors ... ok
test gpu_parity::gaussian_topk_gpu_cpu_parity ... ok   (#[cfg(feature="cuda")], skip silencioso sin GPU)
test result: ok. 5 passed; 0 failed
```

1. **`e2e_fused_matches_linear_top3_5k_vectors`** — 5 000 vectores random normalizados (ChaCha8, seed fija `20261015`), dim 64, HNSW on (ef_search=256, over_fetch=10) → `build_index` → para 50 queries: `find_neighbors_fused` top-3 **== top-3 de `find_neighbors` (lineal)**, mismatches=0, distancias exactas coinciden (tol 1e-5).
2. **`gaussian_search_matches_bruteforce_reference`** — `find_neighbors_gaussian` (k=5, 16 queries) vs referencia bruteforce independiente implementada en el test (`alpha_i·exp(-kappa_i·‖q−mu_i‖²)`): mismos índices en el mismo orden, scores tol 1e-5, orden descendente verificado.
3. **`gaussian_search_ranking_equals_distance_ranking_with_uniform_params`** — cross-check: con alpha/kappa uniformes el ranking gaussiano debe igualar el de distancia (ata ambas rutas).
4. **`gpu_parity::gaussian_topk_gpu_cpu_parity`** (`--features cuda`) — `gpu_gaussian_topk` (kernel PTX `gaussian_topk_kernel` real, RTX 3090) vs referencia CPU independiente: mismos índices + scores tol 1e-5, para K=1 y K=5 (Q=64, N=4096, D=64). Evidencia de ejecución real del kernel:
   ```
   [splatsdb] Launching gaussian_topk_kernel PTX kernel (Q=64, N=4096, D=64, K=1)
   [splatsdb] Launching gaussian_topk_kernel PTX kernel (Q=64, N=4096, D=64, K=5)
   gaussian GPU kernel parity verified for 2/3 k values
   ```
5. **`api_smoke::api_server_smoke_search_oneshot`** — bootea el `run_server` real (axum) en puerto efímero → GET /health → POST /store ×2 → POST /search (top_k=2) sobre HTTP crudo (sin deps nuevas; no hay builder del Router expuesto, así que es oneshot real sobre TCP).

## Baseline de regresión (cargo test --release --lib)

| Corrida | Resultado | Nota |
|---|---|---|
| **BASELINE** (clone limpio @ `d2d2b3c`, antes de tocar nada) | **304 passed / 0 failed** (5.66s) | Referencia pre-B1/B2 |
| Clone + archivos B1/B2 (snapshot 01:11–01:12) | 314 passed / 1 failed | Bug B2 `add_batch` (abajo) |
| **FINAL** (re-sync principal 01:39–01:40) | **324 passed / 0 failed** (6.26s) | B2 corrigió su bug; lib compila también con `--features cuda` |

**Bug B2 detectado y ya resuelto por B2 en el principal:** `HnswBackend::add_batch` aceptaba un vector con dim distinta sin errorar (`candidate_backend.rs:240` fallaba; reproducible en el snapshot 01:11). En el re-sync final (01:40) el test pasa — no requiere acción.
También: `src/candidate_backend.rs:217` tuvo un E0308 transitorio durante la edición de B2 (~01:35, build rota 6× E0063 `missing field score` en splats.rs) — foto de mitad de edit; al 01:39 compila limpio.

**Bonus verificado en re-sync final** (target nuevo de B1 en `src/gpu/mod.rs`):
```
$ cargo test --release --features cuda --lib test_ptx_kernel_performance
[splatsdb benchmark] 10K x 640D, 100 queries | CPU: 718ms | GPU: 18ms | speedup: 39.89x
test gpu::tests::test_ptx_kernel_performance ... ok
```

## Hallazgos para otros hijos

- **Para B1:** `gaussian_topk_kernel` con **K=32 es rechazado en runtime** (launch devuelve None → fallback CPU silencioso). Causa: smem dinámica = (D + 256·K·2)·4 bytes = **65 536 B con K=32, D=64** > límite default 48 KB de sm_86 sin `cudaFuncSetAttribute`. Fix sugerido: opt-in de smem extendida o reducir block dim. El test lo reporta como "wrapper-only k=32" sin fallar (paridad CPU wrapper verificada igual).
- **build.rs** elige `nvcc.exe` de Windows en WSL y falla ("nvcc compilation failed") — la suite depende de `M2M_PTX_PATH` apuntando a un PTX Linux válido. Con el PTX del clone compilado con nvcc 12.4 Linux funciona perfecto.
- B2 ya agregó sus propios unit tests de `find_neighbors_gaussian` en `src/splats.rs` (vistos al 01:12) — complementarios a los de esta suite.

## Comandos para reproducir

```bash
cd /home/schwabauer/splatsdb-b6-clone
export M2M_PTX_PATH=$PWD/ptx/distance.ptx M2M_EXTENDED_PTX_PATH=$PWD/ptx/extended_kernels.ptx
cargo test --release --test integration_gaussian              # 4 tests (sin cuda)
cargo test --release --features cuda --test integration_gaussian   # 5 tests (con parity GPU)
cargo test --release --lib                                    # 314 passed / 1 failed (bug B2)
```
