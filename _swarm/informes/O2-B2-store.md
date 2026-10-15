# O2-B2 — Trait CandidateBackend + SplatStore::find_neighbors_gaussian

Fecha: 2026-10-15 · Repo: /mnt/d/splatdb · Estado: **COMPLETO, sin commit** (según contrato).

## Resumen

| Entregable | Estado |
|---|---|
| (1) `src/candidate_backend.rs` — trait `CandidateBackend` + `BackendCaps` | ✅ |
| (2) Refactor `splats.rs` → HNSW detrás de `Box<dyn CandidateBackend>` | ✅ sin cambio observable |
| (3) `find_neighbors_gaussian` (CPU exacto, overfetch ×8, re-score top-k) | ✅ |
| (4) Hook GPU `gaussian_topk_scores` (feature `cuda`, fallback CPU) | ✅ activo (B1 ya commiteó el wrapper) |
| Tests gaussianos (ordering, kappa-case, round-trip score) | ✅ 7 nuevos |

## (1) Trait — src/candidate_backend.rs (nuevo)

Contrato congelado del padre implementado textualmente:

```rust
pub trait CandidateBackend: Send + Sync {
    fn search(&self, q: &[f32], k_fetch: usize) -> Vec<(usize, f32)>;
    fn add_batch(&mut self, vectors: &[Vec<f32>]) -> Result<(), String>;
    fn len(&self) -> usize;
    fn capabilities(&self) -> BackendCaps;   // { supports_delete, incremental }
}
pub struct BackendCaps { pub supports_delete: bool, pub incremental: bool }
```

Superficie extendida siguiendo el informe R8 §2 (adaptada a lo que SplatStore
necesita y no se puede recuperar desde `Box<dyn>`): `name()`, `build(&Array2<f32>)`,
`is_built()`, `save(&Path)`, default `is_empty()`.

`HnswBackend` envuelve el HNSW casero. Paridad garantizada: `search` devuelve
EXACTAMENTE los índices que devolvía `HNSWIndex::search` (test de paridad contra
la impl `VectorIndex` directa). `add_batch` valida dim contra la dim del backend
(no contra `vectors[0]` — bug encontrado por test: un batch de 1 vector malo se
autovalidaba) e inserta uno a uno.

## (2) Refactor splats.rs — comportamiento preservado

- Campo `hnsw: Option<HNSWIndex>` → `candidates: Option<Box<dyn CandidateBackend>>` (~splats.rs:40).
- Factoría en `new()` (~:95): por ahora solo HNSW según `enable_hnsw`; CAGRA/u otros entran ahí sin tocar el store.
- `build_index_with_save` / `build_index_if_needed` / `try_load_hnsw` / `save_hnsw` /
  `hnsw_sync_incremental` / `insert_with_hnsw` / `add_batch_with_hnsw` /
  `find_neighbors_fused` (estrategia 1) / `has_hnsw` / `hnsw_is_built` — todos
  pasan por el trait. Load sigue despachando por formato HNSW (magic `0x484E5357`).
- Backends no-incrementales (CAGRA futuro): `hnsw_sync_incremental` y compañía
  hacen rebuild completo vía `capabilities().incremental == false` (política R8).
- APIs públicas `has_hnsw`/`hnsw_is_built`/`hnsw_indexed_count`/`hnsw_needs_sync`
  se mantienen — CLI/MCP/api_server compilan sin cambios (verificados en build de lib).
- Nuevo accessor `backend_caps() -> Option<BackendCaps>`.

## (3) find_neighbors_gaussian — splats.rs

`score_i = alpha_i * exp(-kappa_i * ||q-mu_i||²)` **SIN ½** (energy.rs:5-6), mayor = mejor.

Pipeline CPU: `find_neighbors_fused(q, min(k*8, n))` → re-score exacto con
alpha/kappa del store (helper `SplatStore::gaussian_score`, asociado y público) →
top-k por score descendente → `NeighborResult { distance: L2 exacto (no
cuadrado), score }`.

`NeighborResult` ganó campo `score: f32`, poblado en TODOS los paths que lo
construyen (find_neighbors, fused estrategias 1 y 2, filtered, fast) con la misma
fórmula — verificado por test que recorre los 5 paths.

## (4) Hook GPU — ACTIVO

`crate::gpu::gaussian_topk_scores` ya existe (B1 lo commiteó durante esta tarea;
firma: `(queries, dataset, alpha, kappa, n_vectors, dim, n_queries, k) ->
Option<Vec<Vec<(usize,f32)>>>`, mismo score sin ½). El hook en
`find_neighbors_gaussian` está **cableado y compilado** bajo
`#[cfg(feature = "cuda")]` + `is_cuda_available()`: hace top-k exhaustivo del
score sobre el dataset persistente (mu/alpha/kappa del store) y devuelve
`NeighborResult` con distancia exacta recomputada. Fallback CPU transparente si
no hay GPU o el wrapper devuelve `None`.

**Path activo verificado**: en esta máquina WSL NO hay GPU visible
(`nvidia-smi` falla), así que ambos builds corrieron el path CPU. El path GPU
compila (`--features cuda`, test suite verde) pero quedó sin ejercicio runtime
— probar en máquina con GPU.

## Tests — output real

```
$ cargo test --release --lib splats -- --test-threads=2
test result: ok. 31 passed; 0 failed; 0 ignored; 0 measured; 293 filtered out

$ cargo test --release --lib candidate_backend -- --test-threads=2
test result: ok. 2 passed; 0 failed; 0 ignored

$ cargo test --release --features cuda --lib splats -- --test-threads=2
test result: ok. 31 passed; 0 failed; 0 ignored; 0 measured; 299 filtered out

$ cargo test --release --lib -- --test-threads=2
test result: ok. 324 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
```

Tests nuevos (9):
- `test_gaussian_basic_ordering_and_self_query` — scores no-crecientes, self-query score == alpha.
- `test_gaussian_low_kappa_beats_close_high_kappa` — **caso explícito del contrato**: d=2.0/kappa=0.05 (score 0.8187) vence a d=1.0/kappa=5.0 (score 0.0067); el orden por distancia es el inverso.
- `test_gaussian_matches_bruteforce_reference` — 40 vectores, k=5 → fetch cubre TODO el store → top-k idéntico al scan exhaustivo (determinista, no depende del recall del HNSW).
- `test_gaussian_empty_and_k_zero` — vacío, k=0, k>n clampa a n.
- `test_gaussian_without_hnsw_linear_fallback` — sin HNSW: fused→linear, gaussiano exacto igual.
- `test_neighbor_result_score_populated_all_paths` — score consistente con alpha·exp(−κ·d²) en 5 paths.
- `test_score_roundtrip_through_save_load` — save/load del store + índice HNSW; scores y distancias bit-idénticos post-reload.
- `test_hnsw_backend_parity_with_vector_index` — adapter devuelve mismo orden que HNSWIndex::search directo.
- `test_hnsw_backend_add_batch_incremental` — incremental + validación de dim + corrupto rechazado.

Los 24 tests preexistentes de splats.rs (incremental HNSW, persistencia, etc.)
siguen verdes — cero cambio observable del refactor.

## Desvíos / notas

- **Interfaces.rs**: terminé SIN tocarlo — `VectorIndex` sigue igual y el trait
  nuevo vive en su propio archivo (el padre listaba interfaces.rs como ownership
  opcional por si el trait iba ahí; el informe R8 y el contrato congelado ponen
  el trait en candidate_backend.rs).
- **Carreras con hermanos**: durante la tarea hubo 2 ventanas donde la lib no
  compilaba por ediciones concurrentes de api_server.rs (tower dep) y gpu/mod.rs
  (DatasetSnapshot) de otros hijos; ambas se resolvieron solas. Mis runs finales
  (arriba) son posteriores y limpios.
- `find_neighbors_fused` usa `query.as_slice().unwrap_or(&[])` para pasar el view
  al trait (`&[f32]`): ArrayView1 estándar es contiguous, no hay caso real de None.
- NO modifiqué cli/, api_server.rs, gpu/, kernels/, README, hnsw_index.rs.
  Único archivo de otro tocado: `src/lib.rs` (+2 líneas: `pub mod candidate_backend;`
  reordenado alfabético) — imprescindible para registrar el módulo.
- Git: sin commit, según contrato.
