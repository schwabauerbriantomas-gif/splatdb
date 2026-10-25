# Changelog

## [v2.6.1] - 2026-10-25

### Added
- Two-phase IVF over the splat field: `field-search --nprobe N` (CLI) and `/field-search {"mode":"two-phase","nprobe":N}` (HTTP) — phase 1 selects top-N clusters by d(q,μ), phase 2 re-ranks their members exactly; `--gpu` runs phase 1 on the PTX kernel (`f32bc87`)
- `/search` results now carry `topic` + `confidence` per hit (1 − d(x,μ_topic)/R_topic) when a `field/` is persisted; dead-weight splats pruned during ingest (`5cf7e09`)
- `semantic-hub/faststart.sh`: one-command stack — embeddinggemma-2 embedder (:8860) + exact GPU hub (:8871) + end-to-end smoke (`5cf7e09`)

### Fixed
- `serve --gpu`: the `/search` GPU index now searches the full chunk corpus (`field.raw`) instead of the cluster centroids μ, so `topic`/`confidence` always refer to real chunks (`cf38316`)
- CI: `gpu_info()` no longer panics when `cudarc` finds no driver (`5c12ee5`); cudarc-without-driver panic guarded in the rest of the GPU path and clippy 1.99 lints fixed (`96ddbd2`)

### Removed
- `training` preset and its subsystems (data lake, `train_embeddings`, training config fields) as unvalidated dead weight — ~1,250 lines deleted (`5cf7e09`)

## [v2.6.0] - 2026-10-20

### Added
- `field-search` CLI: búsqueda exacta por cotas sobre el campo splat (celdas con radio + desigualdad triangular); radios y membresías CSR persistidos por cada ingest (`field/`, shard `raw`) — recall 0.99 vs GT exacto
- `serve --gpu`: índice exacto en VRAM (kernel PTX `l2_topk_kernel` fused, herencia m2m); `/search` con embedding corre GPU y marca `"backend":"gpu"` — p50 11.4 ms E2E en 26k×768, recall 1.0
- Endpoint `/field-search`: sweep exacto server-side con métricas de poda reales
- Feature `cuda` ON por defecto (build.rs degrada sin nvcc; runtime sin driver cae a CPU)
- `bench-latent/`: GT exacto 30 queries + comparativa de modos — HNSW 0.997, campo exacto 0.99, scoring de energía gaussiana descartado (recall 0.00-0.05, alpha domina = popularidad)

### Fixed
- `ingest-hierarchical` no persistía splats: calculaba, imprimía éxito y descartaba todo
- `simple_kmeans`: asignación por GEMM (init LCG + update f64 intactos) — flat 30-60+ min → 63 s, hierarchical 38,5 → 9,2 s
- Poda por cotas: `kth` solo es cota válida con el top-k completo (recall 0.837 → 0.99)

### Changed
- CI: matriz de features (cpu-only + default-cuda), runtime smoke sin GPU, clippy en ambas configs

## [v2.5.0] - 2026-07-04

### Added
- GitHub Actions CI workflow (test + lint)
- PyPI Trusted Publisher auto-release workflow
- 256 tests passing (CPU mode)

### Fixed
- Various CI/CD improvements
