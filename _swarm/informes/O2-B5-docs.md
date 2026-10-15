# O2-B5 — Honestidad de docs: README/BENCHMARKS con números trazables

**Fecha**: 2026-10-15 · **Repo**: `/mnt/d/splatdb` (rama `wip/v2.6.0-local`) · **Ownership**: README.md, BENCHMARKS.md, docs/ · Sin tocar código, sin commits.

## Archivos modificados/creados

| Archivo | Acción |
|---|---|
| `README.md` | Reescrita la sección Benchmarks + 5 correcciones puntuales (ver abajo) |
| `BENCHMARKS.md` | **Reemplazo total** (el v1 del 2026-03-29 contenía los números untraced) |
| `docs/README.md` | **Nuevo** — índice de docs + política de benchmarks (4 reglas) |

## Qué se hizo (README.md)

1. **"Honest note" (línea 66)**: eliminado *"beats Faiss CPU at 12,195 QPS"*. Nueva formulación: GPU = brute-force exacto k=10 12,195 QPS VRAM-persistente; Faiss HNSW CPU = ANN k=64 9,758 QPS recall 0.9926; **regímenes distintos, no victoria like-for-like**.
2. **Tabla Faiss comparison**: cada fila ahora lleva `k`, tipo de búsqueda (exact/ANN) y recall@k. La fila GPU quedó marcada k=10/exacto (antes se mostraba dentro de una tabla "same k=64"). Agregada columna GPU+upload (10,101 QPS). Takeaways reescritos: primera viñeta = "Different regimes, not a head-to-head win". Eliminadas las imágenes `splatsdb-faiss-comparison.png` y `splatsdb-benchmarks.png` del body (verificado con visión: **la primera mete la fila GPU k=10 en un chart rotulado k=64 y lleva el claim "beats… 1.25× faster"**); quedó un `<details>` que documenta que los PNGs están desactualizados y deben regenerarse.
3. **Claims REMOVIDOS documentados**: bloque "Removed claims" (1,667 QPS 640D y 20,000 QPS linear-100-splats — sin fuente, `a2908b1` los marcó untraced). El 1,667 también se quitó de la sección GPU Acceleration, reemplazado por el microbenchmark reproducible de hoy.
4. **20.93× de HOY agregado** (2 lugares): sección GPU Top-K (subsección "GPU vs CPU kernel benchmark (2026-10-15)") y sección GPU Acceleration. Con comando exacto de reproducción (`cargo test --release --features cuda test_ptx_kernel_performance`), condiciones (10K×640D, 100 queries, seed 12345, batch L2 sin top-k) y advertencia de que es microbenchmark de kernel, no benchmark end-to-end.
5. **Nueva sección "Gaussian Re-scoring"** (How It Works): score = α·exp(−κ·‖q−μ‖²), vínculo con el paisaje de energía E(x) = −log(Σ α·exp(−κ·‖x−μ‖²)) (`src/energy.rs`, `src/ebm/`), y posicionamiento honesto: **PCME/PCME++ ya hacen scoring probabilístico en académico — la novedad es a nivel producto/DB** (κ por-item como feature de primera clase en search/MCP/energy), no un claim de novedad absoluta.
6. **LongMemEval**: caveat agregado — 96.6% es **retrieval-only**, el 86.2% del paper es **end-to-end**; tareas distintas, sin claim de victoria sistémica. Corregido también en "Where SplatsDB Wins" (línea 1391).
7. **Hallazgo extra (proactividad)**: la tabla "HNSW Search (v2.5)" (1,054/579 QPS, recall 0.998/0.995) **tampoco tiene JSON fuente en esta rama** — `hnsw_validated_results.json` no existe aquí, y el propio autor en `main` (`a2908b1`) la reemplazó por 789/242 QPS recall 0.861 (datos sintéticos). Agregado **provenance caveat** en README y BENCHMARKS sin borrar los números (no estaba en mi mandato eliminarlos; quedó documentado y marcado "indicative until re-run").
8. **Integrity pledge** reescrito: de slogans ("No gaming. No cherry-picking.") a regla verificable: todo número declara k, recall y fuente; lo sin fuente se remueve y se documenta.

## Qué se hizo (BENCHMARKS.md nuevo)

Solo tablas regenerables; cada tabla con **Source (archivo JSON o comando) + fecha + hardware + condiciones (k, tipo de búsqueda)**:

1. GPU brute-force vs CPU (SIFT-128, k=10) — del JSON hardware 2026-04-10, con nota de que el run original no dejó log y el JSON fue transcripción manual.
2. Faiss vs SplatsDB HNSW (k=64) — JSON + `bench_windows.py`/`faiss_benchmark.py`, con advertencia explícita de NO cruzar con la tabla 1.
3. HNSW standalone (k=10) — con provenance caveat (ver punto 7).
4. Kernel microbenchmark 640D (2026-10-15) — 565ms vs 27ms = 20.93×, el único número 100% reproducible hoy con comando sin placeholder.
5. LongMemEval — con caveat retrieval-only vs end-to-end.
6. **Sección "Removed claims"**: catálogo de 5 remociones (20,000 QPS, 1,667 QPS, 269/214 QPS CPU 640D, "beats Faiss", "beats all published systems") con motivo de cada una.
7. **"Pending regeneration (TODO-B4)"**: harness unificado, re-medición 640D end-to-end, recall GPU vs GT explícito, regeneración de PNGs.

Comandos de reproducción: los verificados contra el código real (`bench-gpu -n 100000 -d 128 -q 1000 -k 10 -m l2` existe en `src/cli/mod.rs:203`; `bench-hnsw --train … --queries … --gt … --dim 128 --k 64` en `mod.rs:305`) marcados `[COMANDO: …]` + **TODO-B4** para que B4 los consolide.

## Verificación

- `grep` de claims: no queda ningún "beats"/1,667/20,000 activo en README (solo en los bloques "Removed claims").
- Validación programática de tablas markdown (conteo de pipes header vs filas): README OK, BENCHMARKS OK.
- `git diff --stat`: solo README.md y BENCHMARKS.md modificados por mí; `docs/` nuevo. Los otros archivos modified en `git status` (`src/splats.rs`, `kernels/distance.cu`, etc.) son de otros hijos del swarm — no los toqué.
- Sin commits (mandato cumplido).

## Issues encontrados

1. **El test `test_ptx_kernel_performance` NO compila hoy en esta rama** (E0063 `missing field 'score'` en `src/splats.rs` — refactor a medias de otro hijo + nvcc falla en WSL). El 20.93× queda citado como **evidencia verificada por el padre hoy en esta máquina**, no como verificación propia. Cuando el árbol compile, el comando del doc lo reproduce.
2. **PNGs de benchmarks obsoletos** (`assets/splatsdb-{benchmarks,faiss-comparison}.png`): mezclan la fila k=10 en chart "k=64" y llevan el claim "beats" — los saqué del flujo del README y documenté la deuda en BENCHMARKS.md (TODO-B4 regenerar). No los borré (assets no es mi ownership explícito, y el padre puede preferir regenerarlos).
3. `BENCHMARKS_V2.md` (micro-optimizaciones del criterion viejo, sin script de regeneración): quedó referenciado en BENCHMARKS.md como "superseded, no es fuente de claims", sin borrarlo.
4. Las tablas "HNSW Search (v2.5)" y "Comparison: HNSW vs Linear Scan" del README (1,054/579 QPS) no estaban en la lista de evidencia verificada ni en las fuentes no defendibles; se verificó que no tienen JSON en disco → se dejaron con caveat de provenance en vez de removerlas (decisión conservadora; el padre puede ordenar remoção total).
