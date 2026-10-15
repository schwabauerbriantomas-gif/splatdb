# O2-B3 — API server: fused path + micro-batching + rescore gaussiano opcional

Estado: **COMPLETO**. 9/9 tests propios, 324/324 de la suite `--lib` completa, `cargo check --all-targets` OK, clippy limpio en `api_server.rs`. NO commiteado (por contrato).

## Archivos tocados

| Archivo | Cambio | Ownership |
|---|---|---|
| `src/api_server.rs` | Reescrito (361 → 1276 líneas) | Mío (exclusivo) |
| `Cargo.toml` | +2 líneas: `tower = { version = "0.5", features = ["util"] }` en `[dev-dependencies]` | **Fuera de contrato** — ver abajo |

**Único cambio fuera de ownership**: `[dev-dependencies] tower` (tower 0.5.3 ya era transitiva de axum 0.7 con feature `util` activo, verificado con `cargo tree`; sólo la hice directa para que los tests usen `tower::ServiceExt::oneshot` como pedía el contrato). Cero impacto en el build de producción (`dev-dependencies` no entra en binarios). Si el padre prefiere revertirlo, los tests de integración del server no compilan sin él.

No hizo falta wiring de `main.rs`: `run_server` mantiene su firma `(addr, port)` y `cli/data_cmds.rs:489` la llama igual.

## (1) Config con HNSW + find_neighbors_fused

- `server_config()` (api_server.rs:712): reemplaza `SplatsDBConfig::default()` por struct-update con `enable_hnsw: true, hnsw_m: 32, hnsw_ef_construction: 200, hnsw_ef_search: 100, over_fetch: 5`. Antes HNSW estaba OFF → `find_neighbors_fused` degeneraba a scan lineal (R9 §5.3, config/mod.rs:268).
- CPU path: `cpu_fused_batch` llama `store.find_neighbors_fused` (splats.rs:456) — Strategy 1 HNSW + re-rank exacto. Ya no se llama `find_neighbors` lineal en ningún path del server.

## (2) Hash embedding O(1)

- `simple_hash_embedding` (api_server.rs:626): hashea el texto **UNA** vez y deriva las D dimensiones con SplitMix64 (`state |= 1` de seed impar para buena distribución desde el paso 0). Antes: `DefaultHasher` × D por request (R9 §5.1). Test `test_hash_embedding_properties` cubre determinismo, norma unitaria, cota |x|≤1, distinción por texto.

## (3) Micro-batching (mpsc + batch_worker)

Piezas nuevas (sketch R9 §2 implementado):
- `BatchRequest { query, k, rescore_gaussian, resp: oneshot::Sender }` (api_server.rs:66).
- `AppState.batch_tx: mpsc::Sender<BatchRequest>` (cap 256) + `batches_served: Arc<AtomicUsize>` para observabilidad/tests.
- `batch_worker` (api_server.rs:370): `recv()` bloqueante del primero, ventana de coalescing `timeout_at(now + 2ms)` drenando hasta `MAX_BATCH=64`.
- `serve_batch`: clona queries, `spawn_blocking` → **una sola toma del Mutex por batch** (adentro CPU-fused o GPU path), reparto por oneshot.
- **El hot path de `/search` ya no toma el Mutex**: valida (lock breve de stats), computa embedding, `batch_tx.send().await` + `rx.await`. El worker es el único que bloquea el store para buscar.
- CPU: `cpu_fused_batch` — lock único, `find_neighbors_fused` por query del batch, y `touch(&indices)` (R9 §5.3) aprovechando que ya tiene el lock (el server nunca tocaba frecuencias).
- GPU (si `is_cuda_available()`): `gpu_batch_path` — `DatasetSnapshot` cacheado (mu+alpha+kappa planos vía `get_mu/get_alpha/get_kappa`) refrescado **solo si cambió `n_active`**, lock breve sólo para el refresh, luego `gpu::batch_search` (un top-k kernel para todo el batch, fallback CPU adentro). Nunca re-uploadea por request (R9 §1 issue 4 resuelto a nivel server).

## (4) `?rescore=gaussian`

- `SearchQuery { rescore }` via `Query<SearchQuery>` en `/search` (api_server.rs:249). `rescore=gaussian` activa el modo; cualquier otro valor se ignora (200).
- **B2 y B1 aterrizaron mientras tanto** — integrado el rescore REAL, no el fallback con TODO:
  - CPU: `store.find_neighbors_gaussian(&view, k)` (splats.rs:634, over-fetch ×8 + re-score exacto).
  - GPU: batch partido en dos — queries rescored → `gpu::gaussian_topk_scores` (gpu/mod.rs:119, wrapper de B1 con kernel GPU + fallback CPU exacto), queries plain → `gpu::batch_search`; reensamblado en orden original.
- Semántica de respuesta: con `rescore=gaussian`, `score` = score gaussiano `alpha·exp(-kappa·d²)` (**mayor = mejor**, DESC); sin él, `score` = L2 exacto (**menor = mejor**, ASC) — igual que antes.

## Tests (#[cfg(test)] en src/api_server.rs — tests/ es de otro hijo)

Con `tower::ServiceExt::oneshot`, como pedía el contrato:
1. `test_health_endpoint` — 200 + shape `{"status","version"}`.
2. `test_store_and_search_roundtrip` — 3 stores → status `n_active=3, dimension=32, has_hnsw=true`; search exacto devuelve top_k=2 con índice correcto y distancia ~0; query por texto (hash embedding) OK.
3. `test_search_empty_store` — 200 con `results: []`.
4. `test_search_rescore_gaussian_param_accepted` — `?rescore=gaussian` → 200 + shape; `?rescore=bogus` tolerado.
5. `test_rescore_gaussian_changes_score_semantics` — sin rescore scores ASC (L2); con rescore DESC y en (0,1]; mismo ganador con alpha/kappa uniformes.
6. `test_embedding_dim_mismatch_rejected` — 400 en `/search` y `/store` con dim 31 vs 32 (la validación ahora corre ANTES del early-return de store vacío — bug de orden que introduje y arreglé).
7. `test_concurrent_searches_coalesce_into_batches` — 8 searches concurrentes → **sirvieron en menos de 8 batches** (observado: 1; el assert exige `<8`). Verifica que el coalescing funciona de verdad vía el contador `batches_served`.
8. `test_gpu_batch_path_snapshot_and_rescore_split` — ejerce `gpu_batch_path` vía `spawn_blocking` (como `serve_batch`): batch mixto L2+rescore, snapshot cacheado con `version` estable, invalidación al crecer el store (6→7).
9. `test_hash_embedding_properties` — ver (2).

## Output real

```
$ cargo test --release --lib api_server -- --test-threads=2
running 9 tests
test api_server::tests::test_embedding_dim_mismatch_rejected ... ok
test api_server::tests::test_gpu_batch_path_snapshot_and_rescore_split ... ok
test api_server::tests::test_concurrent_searches_coalesce_into_batches ... ok
test api_server::tests::test_hash_embedding_properties ... ok
test api_server::tests::test_health_endpoint ... ok
test api_server::tests::test_search_empty_store ... ok
test api_server::tests::test_rescore_gaussian_changes_score_semantics ... ok
test api_server::tests::test_search_rescore_gaussian_param_accepted ... ok
test api_server::tests::test_store_and_search_roundtrip ... ok
test result: ok. 9 passed; 0 failed; 0 ignored; 0 measured; 315 filtered out

$ cargo test --release --lib -- --test-threads=2   (suite completa, post-cambio)
test result: ok. 324 passed; 0 failed; 0 ignored; 0 measured

$ cargo check --release --all-targets  → sin errores
$ cargo clippy --release --lib         → 0 warnings en api_server.rs
```

## Notas / deudas

- `n_active` como versión del snapshot es suficiente porque el server sólo agrega vía `/store` (append-only). Si algún hijo agrega delete/compact al server, hay que cambiar `DatasetSnapshot.version` a un contador de escrituras.
- `cpu_fused_batch` hace el `touch` post-search en el mismo lock del batch (gratis). `gpu_batch_path` toma un segundo lock breve para el touch — aceptable, no está en el hot path del query.
- Coalescing observado en test: 8 requests → 1 batch (current-thread runtime). Bajo carga real multi-worker la ventana de 2ms puede producir batches de 1 con tráfico esporádico — comportamiento correcto (degrada a path anterior sin lock global).
- El `POST /status` sigue tomando el lock async — es un endpoint frío, no se tocó.
